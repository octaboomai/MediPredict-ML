# Heart Disease Prediction System (College Mini-Project)
**Course:** Machine Learning / Data Science Mini-Project  
**Domain:** Healthcare AI & Predictive Modeling  
**Tech Stack:** Python 3.9+, Scikit-Learn, Flask, Pandas, NumPy, Joblib, HTML5, Tailwind CSS  
**Constraint:** 100% Self-Contained Local Deployment (Zero External APIs, Zero Cloud Dependencies)

---

## 1. Project Directory Structure
```text
heart_disease_prediction/
│
├── dataset.csv                 # Kaggle Cleveland Heart Disease dataset (CSV)
├── train_model.py              # ML training, preprocessing, cross-validation & .pkl export
├── app.py                      # Flask local backend server & inference endpoints
├── heart_disease_model.pkl     # Generated serialized model + StandardScaler pipeline
├── requirements.txt            # Python library dependencies
├── README.md                   # Project documentation & run guide
├── viva_defense_guide.md       # Top 25 Viva Questions & Answers for College Evaluation
│
└── templates/
    └── index.html              # Responsive web UI with real-time AJAX inference
```

---

## 2. Step-by-Step Execution Guide (Localhost)

### Step 1: Navigate to project
```bash
cd heart_disease_prediction
```

### Step 2: Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the ML Model
```bash
python train_model.py
```

### Step 5: Start the local Flask web application
```bash
python app.py
```
Visit http://127.0.0.1:5000 in your browser!

---

## 7. Live Demo (GitHub Pages)

A static build of the app is published from the `docs/` folder:

**https://octaboomai.github.io/MediPredict-ML/**

GitHub Pages can only serve static files, so this build has no Flask backend. Instead,
`train_model.py`'s trained Random Forest (100 trees, max depth 6) and its `StandardScaler`
are exported to `docs/model.json`, and the tree traversal is re-implemented in JavaScript so
predictions run entirely in the visitor's browser. It uses the same UI, presets, thresholds
and recommendation logic as the Flask app.

To regenerate the static build after retraining the model:

```bash
python train_model.py          # produces heart_disease_model.pkl
python export_model_json.py    # writes docs/model.json from the .pkl
```

The published site is a single-page app with five tabs — **Live Predictor**, **Kaggle Dataset & EDA**,
**Model Evaluation**, **Source Code & Files** and **College Viva Guide**. EDA figures, the confusion
matrix, the ROC curve and the feature-importance chart are all generated from `dataset.csv` and the
trained model, so they stay in sync after a retrain. Styling is compiled Tailwind (`docs/tailwind.css`),
so the page has no runtime CDN dependency.

The Flask version in `app.py` is unchanged and remains the reference implementation for
local use, viva demonstrations and the `POST /predict` + `GET /metrics` API.
