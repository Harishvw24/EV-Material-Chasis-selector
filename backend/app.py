import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

DEFAULT_MODEL_PATH = Path("model/material_gbc.joblib")

app = Flask(__name__)
CORS(app)


def load_model_bundle():
    model_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {model_path}. Run train_model.py first."
        )
    bundle = joblib.load(model_path)
    model = bundle.get("model")
    feature_columns = bundle.get("feature_columns")
    if model is None or feature_columns is None:
        raise ValueError("Invalid model bundle. Expected model and feature_columns.")
    return model, feature_columns


MODEL, FEATURE_COLUMNS = load_model_bundle()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON payload."}), 400

    missing = [key for key in FEATURE_COLUMNS if key not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        inputs = {key: float(payload[key]) for key in FEATURE_COLUMNS}
    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be numeric."}), 400

    df = pd.DataFrame([inputs])
    prediction = MODEL.predict(df)[0]
    response = {
        "prediction": int(prediction),
        "usable": bool(prediction),
    }

    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(df)[0]
        response["probability"] = float(proba[1])

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
