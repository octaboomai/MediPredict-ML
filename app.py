"""
=============================================================================
College Mini-Project: Heart Disease Prediction Web Application
File: app.py
Description: Flask web server that loads the locally trained Scikit-learn
             model artifact (.pkl) and provides web endpoints for real-time
             predictions, model performance metrics, and batch CSV analysis.
100% Local & Self-Contained - No External Cloud APIs Used.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates", static_folder="static")

MODEL_FILE = "heart_disease_model.pkl"
DATASET_FILE = "dataset.csv"
model_artifact = None

def load_or_train_model():
    global model_artifact
    if not os.path.exists(MODEL_FILE):
        print(f"[WARN] Model file '{MODEL_FILE}' not found. Auto-running training script...")
        try:
            import train_model
            train_model.train_and_export_model(DATASET_FILE, MODEL_FILE)
        except Exception as e:
            print(f"[ERROR] Failed to auto-train model: {e}")
            return False

    try:
        model_artifact = joblib.load(MODEL_FILE)
        print(f"[SUCCESS] Loaded machine learning model from '{MODEL_FILE}'")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model artifact: {e}")
        return False

# Initialize model
load_or_train_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model_artifact
    if model_artifact is None:
        if not load_or_train_model():
            return jsonify({"status": "error", "message": "Model could not be initialized."}), 500

    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        feature_names = model_artifact["features"]
        feature_values = []

        for feat in feature_names:
            if feat not in data:
                return jsonify({"status": "error", "message": f"Missing required feature: '{feat}'"}), 400
            feature_values.append(float(data[feat]))

        input_array = np.array([feature_values])
        scaler = model_artifact["scaler"]
        input_scaled = scaler.transform(input_array)

        model = model_artifact["model"]
        prediction_class = int(model.predict(input_scaled)[0])
        prob_heart_disease = float(model.predict_proba(input_scaled)[0][1])
        percentage = round(prob_heart_disease * 100, 1)

        recommendations = []
        if float(data.get("trestbps", 120)) >= 140:
            recommendations.append("Resting blood pressure indicates Stage 2 hypertension. Advise medical review.")
        if float(data.get("chol", 200)) >= 240:
            recommendations.append("Serum cholesterol is elevated (>240 mg/dl). Recommend full lipid panel.")
        if int(data.get("exang", 0)) == 1:
            recommendations.append("Exercise angina reported. Recommend stress echocardiogram.")
        if not recommendations:
            recommendations.append("Cardiovascular metrics are within acceptable baseline ranges. Maintain active routine.")

        return jsonify({
            "status": "success",
            "prediction": prediction_class,
            "result_label": "High Risk of Heart Disease" if prediction_class == 1 else "Low Risk / Healthy Profile",
            "probability": prob_heart_disease,
            "risk_percentage": percentage,
            "risk_tier": "High" if percentage >= 65 else ("Moderate" if percentage >= 30 else "Low"),
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/metrics", methods=["GET"])
def get_metrics():
    global model_artifact
    if model_artifact and "metrics" in model_artifact:
        return jsonify({
            "status": "success",
            "metrics": model_artifact["metrics"],
            "algorithm": "RandomForestClassifier(n_estimators=100, max_depth=6)"
        })
    return jsonify({"status": "error", "message": "Metrics not found."}), 404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
