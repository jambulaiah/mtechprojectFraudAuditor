# Fraud Auditor Project

This repository contains a machine-learning fraud detection project with two entry points:

- `Main.py` — a **Streamlit web app** for training and predicting fraud.
- `fraud_detection.py` — a **console script** for training/evaluating models and predicting from terminal input.

It also includes pre-generated model artifacts:

- `fraud_detection_model.pkl`
- `label_encoder.pkl`

## Quick start (recommended)

### 1) Clone and enter the project

```bash
git clone <your-repo-url>
cd mtechprojectFraudAuditor
```

### 2) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Run the Streamlit app

```bash
streamlit run Main.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

---

## Dataset expectations

The project expects a CSV with columns similar to:

- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFlaggedFraud`
- `isFraud` (target)

In the app, upload this CSV, choose a model, and click **Train Model**.

## Running the console script (optional)

The legacy script reads `Financial.csv` from the project root:

```bash
python fraud_detection.py
```

Make sure `Financial.csv` exists before running.

## Troubleshooting

- If imports fail, confirm your virtual environment is active and run:
  ```bash
  pip install -r requirements.txt
  ```
- If Streamlit command is missing:
  ```bash
  python -m streamlit run Main.py
  ```
- If prediction fails for unseen category text values in the app (`type`, `nameOrig`, `nameDest`), retrain with data that contains those categories.

## Project structure

- `Main.py` — Streamlit training + prediction UI.
- `fraud_detection.py` — terminal-based model training/comparison script.
- `requirements.txt` — Python dependencies needed to run this project.
- `fraud_detection_model.pkl` / `label_encoder.pkl` — saved model artifacts.
