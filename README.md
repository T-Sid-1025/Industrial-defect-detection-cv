# Real-Time Industrial Defect Detection System with Simulated PLC Integration

A computer vision system that inspects products on an assembly line, classifies
them as **GOOD** or **DEFECTIVE** in real time using a deep learning model, and
simulates the automation signal that would be sent to a PLC to trigger a
reject mechanism / stop the conveyor.

## Problem Statement
Manual visual quality inspection on production lines is slow, inconsistent,
and expensive. This project automates that inspection using computer vision,
and demonstrates how the decision output can be wired into an industrial
automation system (PLC) for a closed-loop reject mechanism.

## Tech Stack
- **Python 3**
- **OpenCV** — image preprocessing, webcam capture, real-time overlay
- **TensorFlow / Keras** — MobileNetV2 transfer-learning model for classification
- **NumPy**

## How It Works
1. `train_model.py` trains a binary classifier (good vs. defective) on top of
   MobileNetV2 (pretrained on ImageNet, fine-tuned on your product images).
2. `realtime_detect.py` opens your webcam, runs the model on live frames,
   overlays the prediction (green box = OK, red box = DEFECT), and prints a
   simulated PLC signal (`RELAY_ON | REJECT_ARM_ACTIVATE`) whenever a defect
   is found — this is the hook where a real deployment would talk to a PLC
   over Modbus/OPC-UA/GPIO.
3. `test_on_images.py` is a fallback that runs the same model on a folder of
   static images (useful for demos if a webcam isn't available).

## Project Structure
```
cv_defect_project/
├── dataset/
│   ├── good/          <- put non-defective product images here
│   └── defective/     <- put defective product images here
├── test_images/       <- a few images for the static-image demo
├── model/              <- trained model gets saved here (defect_model.h5)
├── output/             <- annotated results from test_on_images.py
├── templates/
│   └── index.html      <- web dashboard frontend
├── app.py               <- Flask web dashboard (recommended for demos)
├── train_model.py
├── realtime_detect.py
├── test_on_images.py
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Getting a Dataset (fastest option)
You don't need to collect your own images. Use a ready-made Kaggle dataset,
for example:
- "Casting Product Image Data for Quality Inspection" (metal casting parts,
  defective vs. ok — very close to a real industrial QC use case)
- "NEU Surface Defect Database" (steel surface defects)

Download it, then sort images into `dataset/good/` and `dataset/defective/`.
(Even 20-30 images per class is enough to get a working demo thanks to
transfer learning + data augmentation used in `train_model.py`.)

If you'd rather use your own product/part, just take ~25 photos of it in
good condition and ~25 with an induced defect (scratch, dent, missing piece,
paint mark, etc.) under consistent lighting.

## Usage

```bash
# 1. Train
python train_model.py

# 2. Web dashboard (recommended for demos/interviews)
python app.py
# then open http://127.0.0.1:5000 in your browser
# drag & drop a product image and click "Run Inspection"

# 3. OR run live detection via webcam (terminal window)
python realtime_detect.py
# press 'q' to quit

# 4. OR test on a folder of static images (no webcam/browser needed)
python test_on_images.py
```

### About the Web Dashboard (`app.py`)
This is a small Flask app that wraps the same trained model in a proper
browser UI instead of a terminal:
- Drag-and-drop (or click to browse) image upload
- Instant PASS / DEFECT result with a confidence meter
- A live "PLC Signal Log" panel showing the simulated automation signal for
  every inspection, plus running pass/reject counters
- A status indicator at the top confirms the model loaded correctly, and
  every error (bad file, no model, etc.) is shown clearly in the UI instead
  of crashing the page or the server

## Notes on Robustness
- The training script checks that the dataset folders exist and contain
  enough images before starting, and falls back to training from scratch if
  pretrained weights can't be downloaded (e.g. restricted network), instead
  of crashing.
- The real-time script checks whether a webcam is actually available and
  exits with a clear message (instead of crashing) if not, pointing you to
  the static-image fallback.
- Both inference scripts wrap prediction in error handling so a single bad
  frame/image can't stop the whole run.

## Future Improvements (good talking points for interview)
- Replace the simulated PLC signal with a real Modbus TCP / OPC-UA write to
  an actual PLC register.
- Add a REST API layer so the inspection station can log results to a
  central MES/SCADA system.
- Expand from binary (good/defective) to multi-class (defect type
  classification: scratch, dent, crack, missing part).
- Swap MobileNetV2 for a YOLOv8 detector to get localized defect bounding
  boxes instead of whole-image classification.

## Resume Bullet (ready to use)
"Built a real-time computer vision system (Python, OpenCV, TensorFlow/
MobileNetV2) that classifies products as defective/non-defective from live
video and simulates a PLC reject-signal trigger, demonstrating an
end-to-end automated visual quality inspection pipeline for industrial
production lines."
