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
pip install -r requirements.txt
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
python main.py
```

This will:
1. Load and merge transaction + identity data
2. Engineer features (UID, frequency encoding, group aggregations, time features)
3. Train three models with time-based validation (80/20 split)
4. Generate evaluation plots to `output/plots/`
5. Save individual and ensemble submissions to `output/`

## Project Structure

```
├── main.py                        # Entry point — runs the full pipeline
├── baseline_model.py              # Original single-file version (reference)
├── fraud_detection/               # Core package
│   ├── __init__.py                # Public API and module docs
│   ├── config.py                  # Paths, hyperparameters, feature settings
│   ├── data_loader.py             # Dataset ingestion, merging, memory optimization
│   ├── features.py                # Feature engineering and chronological split
│   ├── models.py                  # LightGBM, XGBoost, Random Forest, ensemble
│   ├── visualization.py           # Evaluation plots (ROC, PR, confusion, etc.)
│   └── submission.py              # Export Kaggle submission CSVs
├── data/                          # Dataset CSVs (not tracked in git)
├── output/                        # Generated submissions and plots (not tracked)
│   └── plots/                     # ROC curves, confusion matrices, etc.
└── requirements.txt
```

## Evaluation Plots

After running the pipeline, the following plots are generated in `output/plots/`:

| Plot | Description |
|------|-------------|
| `roc_curves.png` | ROC curves for all models + ensemble |
| `precision_recall.png` | Precision-Recall curves |
| `confusion_matrices.png` | Side-by-side confusion matrices |
| `prediction_distributions.png` | Fraud vs legit probability distributions |
| `model_comparison.png` | Bar chart comparing validation AUC |
| `feature_importance_*.png` | Top-20 features per model |

## Team

UTEC AI Hackaton team.
