"""Centralized configuration for paths, hyperparameters, and feature settings."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
PLOTS_DIR = OUTPUT_DIR / "plots"

# ── General ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.20
NA_THRESHOLD = 0.20
MAX_BOOST_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 100

# ── Model hyperparameters ────────────────────────────────────────────────
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 256,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "n_jobs": -1,
    "seed": RANDOM_SEED,
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.01,
    "max_depth": 8,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "verbose": 1,
}

# ── Feature engineering ──────────────────────────────────────────────────
FREQ_ENCODE_COLS = [
    "card1", "card2", "card3", "card5",
    "addr1", "addr2", "uid", "P_emaildomain",
]
AGG_COLS = ["TransactionAmt", "D9", "D15"]
AGG_STATS = ["mean", "std"]
DROP_COLS = ["isFraud", "TransactionID", "TransactionDT"]
IDENTITY_MARKER_COL = "id_01"
SECONDS_PER_HOUR = 3600
