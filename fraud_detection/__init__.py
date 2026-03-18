"""
fraud_detection — IEEE-CIS Fraud Detection ensemble pipeline.

Modules:
    config          Centralized paths, hyperparameters, and feature settings.
    data_loader     Dataset ingestion, merging, memory optimization, and NA detection.
    features        Leak-free feature engineering, imputation, and stratified split.
    models          Model training (LightGBM, XGBoost, Random Forest) and ensemble.
    visualization   Evaluation charts: ROC, PR, confusion, distributions, importance.
"""

from .config import DATA_DIR, OUTPUT_DIR, PLOTS_DIR
from .data_loader import detect_high_na_cols, load_data
from .features import impute_for_rf, prepare_data
from .models import build_ensemble, train_lightgbm, train_random_forest, train_xgboost
from .visualization import generate_all_plots

__all__ = [
    "DATA_DIR", "OUTPUT_DIR", "PLOTS_DIR",
    "load_data", "detect_high_na_cols",
    "prepare_data", "impute_for_rf",
    "train_lightgbm", "train_xgboost", "train_random_forest", "build_ensemble",
    "generate_all_plots",
]
