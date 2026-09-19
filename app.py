"""
app.py
--------
Flask web app for the Industrial Defect Detection project.
Gives you a proper browser-based dashboard instead of a plain terminal demo:
- Drag & drop / click-to-upload an image -> get instant prediction
- Optional live webcam mode (uses the browser's camera, not OpenCV's)
- Simulated PLC signal log shown right in the UI

USAGE:
    python app.py
Then open the printed URL (usually http://127.0.0.1:5000) in your browser.
"""

import os
import sys
import io
import base64
from datetime import datetime

try:
    from flask import Flask, request, jsonify, render_template
    import numpy as np
    import cv2
    import tensorflow as tf
except ImportError as e:
    print("ERROR: Required library missing ->", e)
    print("Run:  pip install -r requirements.txt")
    sys.exit(1)

MODEL_PATH = os.path.join("model", "defect_model.h5")
METRICS_PATH = os.path.join("model", "metrics.json")
IMG_SIZE = (160, 160)

app = Flask(__name__)
model = None  # loaded lazily so the server can still start and show a clear error


def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. Run 'python train_model.py' first."
            )
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


def predict_image(bgr_image):
    """Takes an OpenCV BGR image, returns (label, confidence, is_defective)."""
    resized = cv2.resize(bgr_image, IMG_SIZE)
    arr = np.expand_dims(resized.astype("float32") / 255.0, axis=0)
    m = get_model()
    pred = float(m.predict(arr, verbose=0)[0][0])
    is_defective = pred > 0.5
    confidence = pred if is_defective else (1 - pred)
    label = "DEFECT" if is_defective else "OK"
    return label, confidence, is_defective


def make_gradcam_overlay(bgr_image, last_conv_layer_name="out_relu"):
    """
    Generates a Grad-CAM heatmap showing which region of the image most
    influenced the model's decision, overlaid on the original image.
    Returns a base64-encoded PNG data URI, or None if it can't be computed
    (never raises -- explainability is a bonus, not something that should
    break a prediction).
    """
    try:
        m = get_model()
        resized = cv2.resize(bgr_image, IMG_SIZE)
        arr = np.expand_dims(resized.astype("float32") / 255.0, axis=0)

        grad_model = tf.keras.models.Model(
            inputs=m.inputs,
            outputs=[m.get_layer(last_conv_layer_name).output, m.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(arr)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, IMG_SIZE)
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        base_resized = cv2.resize(bgr_image, IMG_SIZE)
        overlay = cv2.addWeighted(base_resized, 0.55, heatmap_color, 0.45, 0)

        success, buffer = cv2.imencode(".png", overlay)
        if not success:
            return None
        b64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    except Exception as e:
        print("WARNING: Grad-CAM generation failed (non-fatal) ->", e)
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Quick check the frontend can use to confirm the model is loaded, without crashing the page."""
    try:
        get_model()
        return jsonify({"status": "ok", "model_loaded": True})
    except Exception as e:
        return jsonify({"status": "error", "model_loaded": False, "message": str(e)}), 500


@app.route("/model-info")
def model_info():
    """Returns training metrics for the dashboard's info panel. Never crashes if the file is missing."""
    import json
    if not os.path.exists(METRICS_PATH):
        return jsonify({"available": False})
    try:
        with open(METRICS_PATH) as f:
            data = json.load(f)
        data["available"] = True
        return jsonify(data)
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Could not decode image. Please upload a valid JPG/PNG."}), 400

        label, confidence, is_defective = predict_image(img)
        heatmap_data_uri = make_gradcam_overlay(img)

        plc_message = (
            "RELAY_ON | REJECT_ARM_ACTIVATE | CONVEYOR_PAUSE"
            if is_defective
            else "PASS | CONVEYOR_CONTINUE"
        )

        return jsonify({
            "label": label,
            "confidence": round(confidence * 100, 1),
            "is_defective": is_defective,
            "plc_message": plc_message,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "heatmap": heatmap_data_uri,  # may be None if Grad-CAM couldn't be computed
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("Industrial Defect Detection - Web Dashboard")
    print("=" * 60)
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: '{MODEL_PATH}' not found yet.")
        print("The server will still start, but predictions will fail until you run train_model.py.")
    print("\nOpen this in your browser once the server starts:  http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
