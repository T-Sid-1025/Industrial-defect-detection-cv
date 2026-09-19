"""
train_model.py
----------------
Trains a defect-classification model using transfer learning (MobileNetV2).
Works reliably even on small datasets (as few as 20-30 images per class)
and runs fine on CPU (no GPU required).

FOLDER STRUCTURE REQUIRED:
    dataset/
        good/         -> put images of GOOD / non-defective products here
        defective/    -> put images of DEFECTIVE products here

USAGE:
    python train_model.py

OUTPUT:
    model/defect_model.h5   -> trained model, used later by realtime_detect.py / test_on_images.py
"""

import os
import sys

# ---- Basic environment sanity checks (fail with a clear message, never a raw crash) ----
try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError as e:
    print("ERROR: Required library missing ->", e)
    print("Run:  pip install tensorflow opencv-python numpy pillow --break-system-packages")
    sys.exit(1)

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset"
GOOD_DIR = os.path.join(DATASET_DIR, "good")
DEFECT_DIR = os.path.join(DATASET_DIR, "defective")
MODEL_OUT = os.path.join("model", "defect_model.h5")
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 15
# -----------------------------------------


def sanity_check_dataset():
    """Make sure dataset folders exist and actually contain images before we waste time."""
    if not os.path.isdir(GOOD_DIR) or not os.path.isdir(DEFECT_DIR):
        print(f"ERROR: Expected folders '{GOOD_DIR}' and '{DEFECT_DIR}' were not found.")
        print("Create them and put images inside before running training.")
        sys.exit(1)

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    good_count = len([f for f in os.listdir(GOOD_DIR) if f.lower().endswith(valid_ext)])
    defect_count = len([f for f in os.listdir(DEFECT_DIR) if f.lower().endswith(valid_ext)])

    print(f"Found {good_count} 'good' images and {defect_count} 'defective' images.")

    if good_count < 5 or defect_count < 5:
        print("ERROR: You need at least 5 images in EACH class to train (ideally 30+ for decent accuracy).")
        print("Add more images to dataset/good and dataset/defective, then re-run.")
        sys.exit(1)

    return good_count, defect_count


def build_model():
    """Build a MobileNetV2-based transfer learning model for binary classification.
    Tries to use ImageNet pretrained weights (best accuracy on small datasets).
    If the weights can't be downloaded (no internet / blocked network), it falls
    back to training from scratch instead of crashing.
    """
    pretrained_loaded = True
    try:
        base_model = MobileNetV2(
            input_shape=IMG_SIZE + (3,),
            include_top=False,
            weights="imagenet"
        )
        print("Loaded ImageNet pretrained weights (transfer learning mode).")
    except Exception as e:
        pretrained_loaded = False
        print("WARNING: Could not download pretrained ImageNet weights ->", e)
        print("Falling back to training MobileNetV2 from scratch (weights=None).")
        print("Tip: if this keeps happening, check your internet connection and retry.")
        base_model = MobileNetV2(
            input_shape=IMG_SIZE + (3,),
            include_top=False,
            weights=None
        )

    # Only freeze the base if we actually have useful pretrained features.
    # If training from scratch, the base must stay trainable or the model can never learn.
    base_model.trainable = not pretrained_loaded

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x)  # binary: 0 = good, 1 = defective

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    print("=" * 60)
    print("Industrial Defect Detection - Model Training")
    print("=" * 60)

    good_count, defect_count = sanity_check_dataset()

    # Use a validation split only if we have enough images; otherwise train on everything
    total_images = good_count + defect_count
    val_split = 0.2 if total_images >= 20 else 0.0

    datagen_kwargs = dict(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
    )
    if val_split > 0:
        datagen_kwargs["validation_split"] = val_split

    datagen = ImageDataGenerator(**datagen_kwargs)

    try:
        train_gen = datagen.flow_from_directory(
            DATASET_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            subset="training" if val_split > 0 else None,
            classes=["good", "defective"],  # ensures good=0, defective=1 consistently
        )

        val_gen = None
        if val_split > 0:
            val_gen = datagen.flow_from_directory(
                DATASET_DIR,
                target_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                class_mode="binary",
                subset="validation",
                classes=["good", "defective"],
            )
    except Exception as e:
        print("ERROR while loading dataset:", e)
        sys.exit(1)

    print("Class mapping:", train_gen.class_indices)

    model = build_model()
    model.summary()

    callbacks = [EarlyStopping(monitor="loss", patience=4, restore_best_weights=True)]

    print("\nStarting training... (this may take a few minutes on CPU)\n")
    try:
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )
    except Exception as e:
        print("ERROR during training:", e)
        sys.exit(1)

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_OUT)

    # Save training metrics so the web dashboard can display them (accuracy, dataset size, etc.)
    try:
        history_dict = model.history.history if hasattr(model, "history") else {}
        final_val_acc = history_dict.get("val_accuracy", [None])[-1]
        final_val_loss = history_dict.get("val_loss", [None])[-1]
        metrics = {
            "architecture": "MobileNetV2 (transfer learning)",
            "image_size": list(IMG_SIZE),
            "good_images": good_count,
            "defective_images": defect_count,
            "total_images": good_count + defect_count,
            "epochs_trained": len(history_dict.get("loss", [])),
            "final_val_accuracy": round(final_val_acc, 4) if final_val_acc is not None else None,
            "final_val_loss": round(final_val_loss, 4) if final_val_loss is not None else None,
            "trained_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.join("model", "metrics.json"), "w") as f:
            __import__("json").dump(metrics, f, indent=2)
    except Exception as e:
        print("WARNING: could not save metrics.json (non-fatal) ->", e)

    print(f"\nSUCCESS: Model saved to '{MODEL_OUT}'")
    print("You can now run realtime_detect.py or test_on_images.py")


if __name__ == "__main__":
    main()
