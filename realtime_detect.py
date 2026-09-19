"""
realtime_detect.py
--------------------
Loads the trained model and runs LIVE defect detection using your webcam.
On each frame, it predicts GOOD / DEFECTIVE and overlays the result.
When a defect is detected, it prints a SIMULATED PLC signal to the console
(as if a real PLC/relay were being triggered to activate a reject mechanism).

USAGE:
    python realtime_detect.py

Press 'q' to quit.

If no webcam is available, the script exits cleanly with a clear message
instead of crashing -- use test_on_images.py in that case.
"""

import os
import sys
import time

try:
    import cv2
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print("ERROR: Required library missing ->", e)
    print("Run:  pip install tensorflow opencv-python numpy --break-system-packages")
    sys.exit(1)

MODEL_PATH = os.path.join("model", "defect_model.h5")
IMG_SIZE = (160, 160)
CONFIDENCE_THRESHOLD = 0.5  # >0.5 -> defective (matches class order good=0, defective=1)


def load_model_safely():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at '{MODEL_PATH}'.")
        print("Run train_model.py first to create the model.")
        sys.exit(1)
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        print("ERROR: Failed to load model ->", e)
        sys.exit(1)


def simulate_plc_signal(is_defective: bool):
    """
    Placeholder for real PLC communication.
    In a real deployment this would send a signal over Modbus/OPC-UA/GPIO
    to trigger a reject arm or stop the conveyor. Here we simulate it
    with a console log so the logic/flow is clearly demonstrated.
    """
    if is_defective:
        print("[PLC SIGNAL] -> RELAY_ON  | REJECT_ARM_ACTIVATE | CONVEYOR_PAUSE")
    # (no signal needed for good parts; belt just continues)


def preprocess_frame(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def main():
    model = load_model_safely()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not access webcam (no camera found or it's in use by another app).")
        print("You can still test the model on static images using test_on_images.py")
        sys.exit(1)

    print("Webcam started. Press 'q' to quit.")

    total_scanned = 0
    total_rejected = 0
    last_pred_time = 0
    PRED_INTERVAL = 0.5  # predict twice a second, not every single frame (avoids CPU overload)

    label_text = "Scanning..."
    box_color = (200, 200, 200)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame from webcam. Stopping.")
                break

            now = time.time()
            if now - last_pred_time > PRED_INTERVAL:
                last_pred_time = now
                try:
                    processed = preprocess_frame(frame)
                    pred = model.predict(processed, verbose=0)[0][0]
                    is_defective = pred > CONFIDENCE_THRESHOLD
                    total_scanned += 1

                    if is_defective:
                        total_rejected += 1
                        label_text = f"DEFECT DETECTED ({pred:.2f})"
                        box_color = (0, 0, 255)  # red
                        simulate_plc_signal(True)
                    else:
                        label_text = f"OK ({1 - pred:.2f})"
                        box_color = (0, 200, 0)  # green
                except Exception as e:
                    label_text = "Prediction error"
                    print("WARNING: prediction failed on this frame:", e)

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (10, 10), (w - 10, h - 10), box_color, 3)
            cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
            cv2.putText(frame, f"Scanned: {total_scanned}  Rejected: {total_rejected}",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Industrial Defect Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession summary -> Scanned: {total_scanned}, Rejected: {total_rejected}")


if __name__ == "__main__":
    main()
