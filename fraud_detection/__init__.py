"""
fraud_detection — IEEE-CIS Fraud Detection ensemble pipeline.

Modules:
    config          Centralized paths, hyperparameters, and feature settings.
    data_loader     Dataset ingestion, merging, and memory optimization.
    features        Feature engineering and chronological train/validation split.
    models          Model training (LightGBM, XGBoost, Random Forest) and ensemble.
    visualization   Evaluation charts: ROC, PR, confusion, distributions, importance.
    submission      Export prediction CSVs for Kaggle submission.
"""

from .config import DATA_DIR, OUTPUT_DIR, PLOTS_DIR
from .data_loader import load_data
from .features import engineer_features, time_based_split
from .models import build_ensemble, train_lightgbm, train_random_forest, train_xgboost
from .submission import save_submissions
from .visualization import generate_all_plots

__all__ = [
    "DATA_DIR", "OUTPUT_DIR", "PLOTS_DIR",
    "load_data",
    "engineer_features", "time_based_split",
    "train_lightgbm", "train_xgboost", "train_random_forest", "build_ensemble",
    "save_submissions",
    "generate_all_plots",
]
