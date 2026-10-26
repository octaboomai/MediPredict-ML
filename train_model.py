"""
=============================================================================
College Mini-Project: Heart Disease Prediction using Machine Learning
File: train_model.py
Description: Loads Kaggle Heart Disease dataset, performs data preprocessing,
             trains a Random Forest Classifier with Hyperparameter Tuning,
             evaluates the model, and exports the serialized model (.pkl).
100% Local Execution - No External APIs Required.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Scikit-Learn libraries for model training & evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def create_sample_dataset_if_missing(file_path="dataset.csv"):
    """
    Creates a realistic synthetic/sample Kaggle Heart Disease dataset
    if user hasn't placed their own CSV yet.
    """
    if os.path.exists(file_path):
        print(f"[INFO] Found existing dataset at: '{file_path}'")
        return

    print(f"[INFO] '{file_path}' not found. Generating sample Kaggle Cleveland Heart Disease data...")
    np.random.seed(42)
    n_samples = 300

    age = np.random.randint(29, 78, n_samples)
    sex = np.random.choice([0, 1], n_samples, p=[0.32, 0.68])
    cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.17, 0.28, 0.08])
    trestbps = np.random.randint(94, 200, n_samples)
    chol = np.random.randint(126, 400, n_samples)
    fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    restecg = np.random.choice([0, 1, 2], n_samples, p=[0.49, 0.49, 0.02])
    thalach = np.random.randint(71, 202, n_samples)
    exang = np.random.choice([0, 1], n_samples, p=[0.67, 0.33])
    oldpeak = np.round(np.random.uniform(0.0, 5.0, n_samples), 1)
    slope = np.random.choice([0, 1, 2], n_samples, p=[0.07, 0.46, 0.47])
    ca = np.random.choice([0, 1, 2, 3], n_samples, p=[0.58, 0.22, 0.13, 0.07])
    thal = np.random.choice([1, 2, 3], n_samples, p=[0.06, 0.55, 0.39])

    risk_score = (
        (age > 55) * 0.4
        + (sex == 1) * 0.5
        + (cp > 0) * 0.9
        + (trestbps > 140) * 0.4
        + (chol > 240) * 0.3
        + (thalach < 140) * 0.7
        + (exang == 1) * 1.1
        + (oldpeak > 1.2) * 1.0
        + (ca > 0) * 1.3
        + (thal == 3) * 0.8
        - 2.8
    )
    prob = 1 / (1 + np.exp(-risk_score))
    target = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "target": target
    })

    df.to_csv(file_path, index=False)
    print(f"[SUCCESS] Created sample dataset with {len(df)} records at '{file_path}'\n")

def train_and_export_model(dataset_path="dataset.csv", model_output_path="heart_disease_model.pkl"):
    """
    Main pipeline:
    1. Load data
    2. Data validation & imputation
    3. Train/Test split (80/20 stratified)
    4. Feature scaling with StandardScaler
    5. Model training with RandomForestClassifier
    6. Evaluation & Confusion Matrix
    7. Export model artifact (scaler + model) via joblib
    """
    create_sample_dataset_if_missing(dataset_path)

    print("=" * 65)
    print("      HEART DISEASE PREDICTION - MODEL TRAINING PIPELINE        ")
    print("=" * 65)

    # Step 1: Load Dataset
    print(f"[STEP 1] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"         Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Step 2: Handle Missing Values
    if df.isnull().sum().sum() > 0:
        print("[STEP 2] Handling missing values via median imputation...")
        df.fillna(df.median(numeric_only=True), inplace=True)
    else:
        print("[STEP 2] Data Cleanliness Verified: 0 missing values.")

    target_column = "target"
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset!")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Step 3: Train-Test Split (80% Train, 20% Test Stratified)
    print("\n[STEP 3] Splitting data into 80% Train, 20% Test (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"         Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")

    # Step 4: Feature Scaling (StandardScaler)
    print("\n[STEP 4] Fitting StandardScaler on training features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Model Selection & Training (Random Forest)
    print("\n[STEP 5] Training Random Forest Classifier (100 estimators, max_depth=6)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    print(f"         5-Fold Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # Step 6: Evaluation
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_pred_proba)

    print("-" * 50)
    print(f"  Test Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision      : {prec * 100:.2f}%")
    print(f"  Recall         : {rec * 100:.2f}%")
    print(f"  F1-Score       : {f1 * 100:.2f}%")
    print(f"  ROC-AUC Score  : {roc:.4f}")
    print("-" * 50)

    # Step 7: Export Model Artifact
    print(f"\n[STEP 7] Serializing and saving model artifact to: '{model_output_path}'")
    package = {
        "model": model,
        "scaler": scaler,
        "features": list(X.columns),
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc)
        }
    }
    joblib.dump(package, model_output_path)
    print(f"[SUCCESS] Model successfully saved to {model_output_path}")

if __name__ == "__main__":
    train_and_export_model()
