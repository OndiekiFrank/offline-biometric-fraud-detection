"""
Face-to-Phone demo (offline)
- Enroll a user (save face template)
- Live webcam verification: accepts or rejects based on embedding distance
- Simulated transaction stream + IsolationForest anomaly detection
- Local encrypted logging of events
- Simple PIN fallback for inclusion (simulated)
"""

import os
import time
import threading
import queue
import cv2
import numpy as np
import face_recognition
from sklearn.ensemble import IsolationForest
from cryptography.fernet import Fernet
import json
from datetime import datetime

ENROLL_DIR = "enroll"
ENROLL_IMG = os.path.join(ENROLL_DIR, "owner.jpg")
LOG_FILE = "alerts.log"
KEY_FILE = "secret.key"

# Parameters
VERIFICATION_THRESHOLD = 0.45  # cosine distance threshold (tuneable)
ANOMALY_WINDOW = 50            # transactions to train on initially
ANOMALY_RUN_FREQ = 1.0         # seconds between synthetic transactions
DEVICE_ID = "Device-42"

# --- Utility: encryption key (simple device-local secret) ---
def ensure_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
    else:
        with open(KEY_FILE, "rb") as f:
            key = f.read()
    return Fernet(key)

fernet = ensure_key()

def encrypted_log(obj):
    txt = json.dumps(obj, default=str).encode("utf-8")
    token = fernet.encrypt(txt)
    with open(LOG_FILE, "ab") as f:
        f.write(token + b"\n")

# --- Enrollment: capture a single image from webcam (run once) ---
def enroll_from_webcam():
    os.makedirs(ENROLL_DIR, exist_ok=True)
    print("[ENROLL] Press 'e' to capture enrollment photo from webcam. Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ENROLL] No camera available")
            break
        cv2.imshow("Enrollment - Press e to save", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('e'):
            cv2.imwrite(ENROLL_IMG, frame)
            print(f"[ENROLL] saved {ENROLL_IMG}")
            break
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# --- Face encoding helpers ---
def load_enrolled_encoding():
    if not os.path.exists(ENROLL_IMG):
        return None
    img = face_recognition.load_image_file(ENROLL_IMG)
    encs = face_recognition.face_encodings(img)
    if len(encs) == 0:
        print("[ENROLL] No face found in enrollment image.")
        return None
    return encs[0]

def live_verify(known_encoding, stop_event):
    """
    Start webcam, verify live faces continuously.
    On mismatch or success, push events to logger.
    """
    cap = cv2.VideoCapture(0)
    print("[VERIFY] Starting live verification. Press 'q' in the window to stop.")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb = small[:, :, ::-1]
        faces = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, faces)
        status = "NoFace"
        if len(encodings) > 0:
            # use first face found
            dist = face_recognition.face_distance([known_encoding], encodings[0])[0]
            status = "Match" if dist <= VERIFICATION_THRESHOLD else "Mismatch"
            print(f"[VERIFY] dist={dist:.3f} => {status}")
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "device": DEVICE_ID,
                "event": "biometric_check",
                "status": status,
                "distance": float(dist)
            }
            encrypted_log(event)
            # overlay
            label = f"{status} ({dist:.2f})"
            cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0) if status=="Match" else (0,0,255), 2)
        else:
            cv2.putText(frame, "No face", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        cv2.imshow("Face Verification (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[VERIFY] stopped.")

# --- Simulated transactions generator + anomaly detector ---
def transaction_generator(tx_queue, stop_event):
    """
    Put simulated transaction dicts into tx_queue periodically.
    Features: amount, hour_of_day, num_prev_tx_24h (simple features).
    Occasionally inject an anomalous transaction (large amount).
    """
    import random
    tx_id = 0
    while not stop_event.is_set():
        tx_id += 1
        base_amount = random.gauss(300, 80)  # typical small transfers
        # every 20th transaction is a large spike (simulate fraud)
        if tx_id % 20 == 0:
            amount = abs(random.gauss(2000, 300))
        else:
            amount = max(1, abs(base_amount))
        tx = {
            "tx_id": tx_id,
            "amount": float(round(amount,2)),
            "hour": datetime.now().hour,
            "prev24h": random.randint(0,10)
        }
        tx_queue.put(tx)
        time.sleep(ANOMALY_RUN_FREQ)

def run_anomaly_detector(tx_queue, stop_event):
    """
    Lightweight IsolationForest model that re-fits on a rolling window.
    For demo: collect initial window, then predict.
    """
    from collections import deque
    window = deque(maxlen=ANOMALY_WINDOW)
    model = None
    while not stop_event.is_set():
        try:
            tx = tx_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        features = [tx["amount"], tx["hour"], tx["prev24h"]]
        window.append(features)
        # train model when we have enough samples
        if len(window) >= ANOMALY_WINDOW and model is None:
            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit(np.array(window))
            print("[ANOM] model trained on initial window")
        elif model is not None:
            score = model.decision_function([features])[0]  # higher is more normal
            pred = model.predict([features])[0]             # -1 anomaly, 1 normal
            print(f"[ANOM] tx {tx['tx_id']} amt={tx['amount']:.2f} pred={pred} score={score:.4f}")
            if pred == -1:
                alert = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "device": DEVICE_ID,
                    "event": "transaction_anomaly",
                    "tx": tx,
                    "score": float(score)
                }
                encrypted_log(alert)
                # Immediate alert to console (replace with UI/notification)
                print("!!! ALERT: anomalous transaction detected:", tx)
        else:
            print(f"[ANOM] warming up ({len(window)}/{ANOMALY_WINDOW}) - tx {tx['tx_id']} collected")

# --- Simple PIN fallback (simulate) ---
def pin_fallback():
    saved_pin = "1234"  # demo; in real app store hashed and salted
    pin = input("Enter fallback PIN: ")
    if pin == saved_pin:
        print("[FALLBACK] PIN accepted")
        encrypted_log({"timestamp": datetime.utcnow().isoformat(),"device":DEVICE_ID,"event":"pin_used","status":"success"})
        return True
    else:
        print("[FALLBACK] PIN rejected")
        encrypted_log({"timestamp": datetime.utcnow().isoformat(),"device":DEVICE_ID,"event":"pin_used","status":"fail"})
        return False

# --- Main demo orchestration ---
def main():
    print("=== Face-to-Phone Offline Demo ===")
    # Step 1: ensure enrollment
    if not os.path.exists(ENROLL_IMG):
        print("No enrollment found. Running enrollment.")
        enroll_from_webcam()

    known_enc = load_enrolled_encoding()
    if known_enc is None:
        print("Enrollment failed. Exiting.")
        return

    # Start transaction generator + anomaly detector in background
    tx_q = queue.Queue()
    stop_event = threading.Event()
    tgen = threading.Thread(target=transaction_generator, args=(tx_q, stop_event), daemon=True)
    tad = threading.Thread(target=run_anomaly_detector, args=(tx_q, stop_event), daemon=True)
    tgen.start(); tad.start()

    # Start live verification - blocks until user quits
    try:
        live_verify(known_enc, stop_event)
    except KeyboardInterrupt:
        stop_event.set()

    # Shutdown
    stop_event.set()
    tgen.join(timeout=1)
    tad.join(timeout=1)
    print("Demo finished. Encrypted logs are in", LOG_FILE)

if __name__ == "__main__":
    main()
