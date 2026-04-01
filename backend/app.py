"""Flask API for chest X-ray disease detection."""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

import model as mdl

app = Flask(__name__)
CORS(app)

# Load model on startup
mdl.init_model()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if not file or not _allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Accepted: PNG, JPG, JPEG"}), 400

    try:
        pil_image = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "Could not read image file"}), 400

    patient_info = {
        "age": request.form.get("age", ""),
        "gender": request.form.get("gender", ""),
        "smoking_status": request.form.get("smoking_status", ""),
        "surgical_history": request.form.get("surgical_history", ""),
        "existing_conditions": request.form.get("existing_conditions", ""),
    }

    try:
        result = mdl.predict(pil_image, patient_info)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/gradcam", methods=["POST"])
def api_gradcam():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    class_idx = request.form.get("class_idx", type=int)

    if class_idx is None:
        return jsonify({"error": "class_idx is required"}), 400

    if class_idx < 0 or class_idx >= len(mdl._label_cols):
        return jsonify({"error": f"class_idx must be 0-{len(mdl._label_cols)-1}"}), 400

    try:
        pil_image = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "Could not read image file"}), 400

    try:
        result = mdl.predict_gradcam(pil_image, class_idx)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Grad-CAM failed: {str(e)}"}), 500


@app.route("/api/test-image", methods=["GET"])
def test_image():
    """Serve a sample X-ray for testing."""
    import os
    img_dir = os.path.join(os.path.dirname(__file__), "..", "NIH", "images")
    sample = "00000001_000.png"
    return app.send_static_file("") if False else \
        __import__("flask").send_from_directory(os.path.abspath(img_dir), sample)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": mdl._label_cols is not None})


if __name__ == "__main__":
    app.run(debug=False, port=5001)
