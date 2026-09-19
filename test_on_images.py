"""
test_on_images.py
--------------------
Fallback for demo/interview if a webcam is not available or not convenient.
Runs the trained model on all images inside 'test_images/' and saves
annotated results (with predicted label) into 'output/'.

USAGE:
    Put some images in test_images/ then run:
    python test_on_images.py
"""

import os
import sys

try:
    import cv2
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print("ERROR: Required library missing ->", e)
    print("Run:  pip install tensorflow opencv-python numpy --break-system-packages")
    sys.exit(1)

MODEL_PATH = os.path.join("model", "defect_model.h5")
TEST_DIR = "test_images"
OUTPUT_DIR = "output"
IMG_SIZE = (160, 160)


def simulate_plc_signal(is_defective: bool, filename: str):
    if is_defective:
        print(f"[PLC SIGNAL] {filename} -> RELAY_ON | REJECT_ARM_ACTIVATE")
    else:
        print(f"[PLC SIGNAL] {filename} -> PASS | CONVEYOR_CONTINUE")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at '{MODEL_PATH}'. Run train_model.py first.")
        sys.exit(1)

    if not os.path.isdir(TEST_DIR):
        print(f"ERROR: '{TEST_DIR}' folder not found. Create it and add some test images.")
        sys.exit(1)

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(valid_ext)]
    if not files:
        print(f"ERROR: No images found in '{TEST_DIR}'. Add some .jpg/.png files and re-run.")
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Running inference on {len(files)} image(s)...\n")

    for fname in files:
        path = os.path.join(TEST_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"WARNING: Could not read '{fname}', skipping.")
            continue

        resized = cv2.resize(img, IMG_SIZE)
        input_arr = np.expand_dims(resized.astype("float32") / 255.0, axis=0)

        pred = model.predict(input_arr, verbose=0)[0][0]
        is_defective = pred > 0.5
        label = f"DEFECT ({pred:.2f})" if is_defective else f"OK ({1 - pred:.2f})"
        color = (0, 0, 255) if is_defective else (0, 200, 0)

        annotated = img.copy()
        cv2.rectangle(annotated, (5, 5), (img.shape[1] - 5, img.shape[0] - 5), color, 4)
        cv2.putText(annotated, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        out_path = os.path.join(OUTPUT_DIR, f"result_{fname}")
        cv2.imwrite(out_path, annotated)

        simulate_plc_signal(is_defective, fname)
        print(f"  {fname}: {label}  -> saved to {out_path}\n")

    print("Done. Check the 'output' folder for annotated results.")


if __name__ == "__main__":
    main()
