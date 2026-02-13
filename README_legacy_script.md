# Legacy Script Notes (`fraud_detection.py`)

This script is kept for CLI-based experimentation and model comparison.

## What it does

- Loads `Financial.csv`
- Cleans missing data
- Encodes categorical columns (`type`, `nameOrig`, `nameDest`)
- Trains three models:
  - Random Forest
  - KNN
  - AdaBoost
- Prints metrics and confusion matrices
- Accepts manual terminal input for single-record prediction

## Important assumptions

- The script expects `Financial.csv` in the repository root.
- Manual prediction input uses numeric values only.
- It is not wired for production serving; use `Main.py` for a web interface.

## Run

```bash
python fraud_detection.py
```
