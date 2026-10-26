"""
=============================================================================
Static export for the GitHub Pages build.
File: export_model_json.py
Description: Reads the serialized model artifact (heart_disease_model.pkl)
             produced by train_model.py and exports the Random Forest trees
             plus the StandardScaler parameters to docs/model.json, so the
             browser can reproduce predict_proba() in plain JavaScript.
Usage: python train_model.py && python export_model_json.py
=============================================================================
"""

import json
import os
import joblib

MODEL_PATH = "heart_disease_model.pkl"
OUTPUT_PATH = os.path.join("docs", "model.json")


def export_model_json(model_path=MODEL_PATH, output_path=OUTPUT_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"'{model_path}' not found. Run `python train_model.py` first."
        )

    package = joblib.load(model_path)
    model = package["model"]
    scaler = package["scaler"]

    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        trees.append({
            "cl": tree.children_left.tolist(),          # left child index (-1 = leaf)
            "cr": tree.children_right.tolist(),         # right child index
            "f": tree.feature.tolist(),                 # split feature index
            "th": [float(t) for t in tree.threshold.tolist()],
            "v": [[float(a), float(b)] for a, b in tree.value[:, 0, :].tolist()],
        })

    payload = {
        "features": package["features"],
        "scaler": {
            "mean": [float(x) for x in scaler.mean_],
            "scale": [float(x) for x in scaler.scale_],
        },
        "metrics": package["metrics"],
        "algorithm": "RandomForestClassifier(n_estimators=100, max_depth=6)",
        "trees": trees,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[SUCCESS] Exported {len(trees)} trees -> '{output_path}' ({size_kb:.1f} KB)")
    return output_path


if __name__ == "__main__":
    export_model_json()
