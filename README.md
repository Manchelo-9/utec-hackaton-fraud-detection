# UTEC Hackaton — IEEE Fraud Detection

Ensemble model for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) challenge.
Combines **LightGBM**, **XGBoost**, and **Random Forest** with AUC-weighted averaging.

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install pandas numpy lightgbm xgboost scikit-learn
```

### 3. Download the dataset

Download the dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) and place the CSV files in the `data/` folder:

```
data/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
├── test_identity.csv
└── sample_submission.csv
```

## Usage

```bash
python first_glance_model.py
```

This will:
1. Load and merge transaction + identity data
2. Engineer features (UID, frequency encoding, group aggregations, time features)
3. Train three models with time-based validation (80/20 split)
4. Save individual and ensemble submissions to `output/`

## Output

| File | Description |
|------|-------------|
| `submission_lightgbm.csv` | LightGBM predictions |
| `submission_xgboost.csv` | XGBoost predictions |
| `submission_randomforest.csv` | Random Forest predictions |
| `submission_ensemble.csv` | Weighted ensemble (primary submission) |

## Project Structure

```
├── first_glance_model.py   # Main training script
├── data/                   # Dataset CSVs (not tracked in git)
└── output/                 # Generated submissions (not tracked in git)
```

## Team

UTEC AI Hackaton team.
