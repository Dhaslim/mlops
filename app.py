# app.py
from flask import Flask, request, jsonify
import joblib
from pathlib import Path


app = Flask(__name__)
MODEL_PATH = Path("artifacts/model.pkl")


# Auto-train if model missing
if not MODEL_PATH.exists():
    import train as _train
    _train.main()


# Load model ONCE at startup (not on every request)
model = joblib.load("artifacts/model.pkl")


# Endpoint 1: Browser UI
@app.route("/", methods=["GET"])
def home():
    return """... HTML form for students to test in browser ..."""


# Endpoint 2: Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# Endpoint 3: Prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "send JSON with key 'features'"}), 400
    try:
        pred = model.predict([data["features"]])
        return jsonify({"prediction": int(pred[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
