"""
IEEE Fraud Detection — Ensemble Model (LightGBM + XGBoost + Random Forest)

Trains three models on the IEEE-CIS fraud detection dataset and produces
an AUC-weighted ensemble submission.
"""

import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

# ── Hyperparameters ──────────────────────────────────────────────────────
RANDOM_SEED = 42
TRAIN_SPLIT_RATIO = 0.80
MAX_BOOST_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 100

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

# ── Feature engineering config ───────────────────────────────────────────
FREQ_ENCODE_COLS = ["card1", "card2", "card3", "card5", "addr1", "addr2",
                    "uid", "P_emaildomain"]
AGG_COLS = ["TransactionAmt", "D9", "D15"]
AGG_STATS = ["mean", "std"]
DROP_COLS = ["isFraud", "TransactionID", "TransactionDT"]

SECONDS_PER_HOUR = 3600


# ── Helpers ──────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to the smallest viable dtype."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        dtype = df[col].dtype
        if np.issubdtype(dtype, np.integer):
            c_min, c_max = df[col].min(), df[col].max()
            for candidate in [np.int8, np.int16, np.int32]:
                info = np.iinfo(candidate)
                if c_min >= info.min and c_max <= info.max:
                    df[col] = df[col].astype(candidate)
                    break
        elif np.issubdtype(dtype, np.floating):
            c_min, c_max = df[col].min(), df[col].max()
            finfo = np.finfo(np.float32)
            if c_min >= finfo.min and c_max <= finfo.max:
                df[col] = df[col].astype(np.float32)
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem else 0
        print(f"Mem. usage decreased to {end_mem:5.2f} Mb ({reduction:.1f}% reduction)")
    return df


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge transaction + identity CSVs."""
    print_section("Loading Data")
    train_txn = pd.read_csv(data_dir / "train_transaction.csv")
    train_id = pd.read_csv(data_dir / "train_identity.csv")
    test_txn = pd.read_csv(data_dir / "test_transaction.csv")
    test_id = pd.read_csv(data_dir / "test_identity.csv")

    train = pd.merge(train_txn, train_id, on="TransactionID", how="left")
    test = pd.merge(test_txn, test_id, on="TransactionID", how="left")
    del train_txn, train_id, test_txn, test_id
    gc.collect()

    print(f"Train Shape: {train.shape}, Test Shape: {test.shape}")
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    return train, test


def engineer_features(train: pd.DataFrame,
                      test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series,
                                                    pd.DataFrame, pd.DataFrame]:
    """Create features on combined train+test, then split back."""
    print_section("Engineering Features")
    len_train = len(train)
    df = pd.concat([train, test], axis=0, sort=False)
    del train, test
    gc.collect()

    # UID: identifies clients by card + address + email domain
    df["uid"] = (df["card1"].astype(str) + "_"
                 + df["addr1"].astype(str) + "_"
                 + df["P_emaildomain"].astype(str))

    # Frequency encoding
    for col in FREQ_ENCODE_COLS:
        df[col + "_count"] = df[col].map(df[col].value_counts(dropna=False))

    # Group aggregations (cache the groupby object)
    grouped = df.groupby("uid")
    for col in AGG_COLS:
        if col in df.columns:
            for stat in AGG_STATS:
                df[f"{col}_to_uid_{stat}"] = grouped[col].transform(stat)

    # Time features
    df["hour"] = (df["TransactionDT"] / SECONDS_PER_HOUR) % 24

    # Decimal part of transaction amount
    df["TransactionAmt_decimal"] = ((df["TransactionAmt"] % 1) * 1000).astype(int)

    # Label-encode categorical columns
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Split back
    X_train = df.iloc[:len_train].drop(DROP_COLS, axis=1)
    y_train = df.iloc[:len_train]["isFraud"]
    X_test = df.iloc[len_train:].drop(DROP_COLS, axis=1)
    submission = pd.DataFrame({"TransactionID": df.iloc[len_train:]["TransactionID"]})
    del df
    gc.collect()

    return X_train, y_train, X_test, submission


def time_based_split(X: pd.DataFrame, y: pd.Series,
                     ratio: float = TRAIN_SPLIT_RATIO):
    """Split chronologically — train on first `ratio`, validate on rest."""
    idx = int(len(X) * ratio)
    X_tr, X_val = X.iloc[:idx].copy(), X.iloc[idx:].copy()
    y_tr, y_val = y.iloc[:idx].copy(), y.iloc[idx:].copy()
    del X
    gc.collect()
    return X_tr, y_tr, X_val, y_val


def train_lightgbm(X_tr, y_tr, X_val, y_val):
    print_section("Training LightGBM")
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        LGB_PARAMS, train_data,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                   lgb.log_evaluation(period=100)],
    )
    val_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, val_pred)
    print(f"LightGBM Validation AUC: {auc:.6f}")
    return model, val_pred, auc


def train_xgboost(X_tr, y_tr, X_val, y_val, X_test):
    print_section("Training XGBoost")
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        XGB_PARAMS, dtrain,
        num_boost_round=MAX_BOOST_ROUNDS,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=100,
    )
    val_pred = model.predict(dval)
    auc = roc_auc_score(y_val, val_pred)
    print(f"XGBoost Validation AUC: {auc:.6f}")
    test_pred = model.predict(xgb.DMatrix(X_test))
    del dtrain, dval
    gc.collect()
    return model, val_pred, test_pred, auc


def train_random_forest(X_tr, y_tr, X_val, y_val):
    print_section("Training Random Forest")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_tr, y_tr)
    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print(f"Random Forest Validation AUC: {auc:.6f}")
    return model, val_pred, auc


def ensemble_and_save(submission, predictions, aucs, output_dir):
    """Create weighted-average ensemble and save all submissions."""
    print_section("Creating Ensemble Predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual model submissions
    for name, preds in predictions.items():
        submission["isFraud"] = preds
        submission.to_csv(output_dir / f"submission_{name}.csv", index=False)

    # Weighted ensemble
    names = list(predictions.keys())
    weights = np.array([aucs[n] for n in names])
    weights = weights / weights.sum()

    print("\nEnsemble Weights:")
    for name, w in zip(names, weights):
        print(f"  {name}: {w:.4f}")

    test_preds = np.column_stack([predictions[n] for n in names])
    ensemble_pred = test_preds @ weights

    submission["isFraud"] = ensemble_pred
    submission.to_csv(output_dir / "submission_ensemble.csv", index=False)

    return ensemble_pred, dict(zip(names, weights))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    train, test = load_data(DATA_DIR)
    X_train, y_train, X_test, submission = engineer_features(train, test)
    X_tr, y_tr, X_val, y_val = time_based_split(X_train, y_train)

    # Train models
    lgb_model, lgb_val, lgb_auc = train_lightgbm(X_tr, y_tr, X_val, y_val)
    _, xgb_val, xgb_test, xgb_auc = train_xgboost(X_tr, y_tr, X_val, y_val, X_test)
    rf_model, rf_val, rf_auc = train_random_forest(X_tr, y_tr, X_val, y_val)

    # Test predictions for LGB and RF
    lgb_test = lgb_model.predict(X_test)
    rf_test = rf_model.predict_proba(X_test)[:, 1]

    # Ensemble validation AUC
    aucs = {"lightgbm": lgb_auc, "xgboost": xgb_auc, "randomforest": rf_auc}
    w = np.array(list(aucs.values()))
    w = w / w.sum()
    ensemble_val = np.column_stack([lgb_val, xgb_val, rf_val]) @ w
    print(f"\nEnsemble Validation AUC: {roc_auc_score(y_val, ensemble_val):.6f}")

    # Save
    predictions = {"lightgbm": lgb_test, "xgboost": xgb_test, "randomforest": rf_test}
    ensemble_and_save(submission, predictions, aucs, OUTPUT_DIR)

    # Summary
    print_section("SUMMARY")
    for name, auc in aucs.items():
        print(f"{name:20s} AUC: {auc:.6f}")


if __name__ == "__main__":
    main()
