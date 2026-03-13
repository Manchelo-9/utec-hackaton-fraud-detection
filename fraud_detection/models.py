"""Model training: LightGBM, XGBoost, Random Forest."""

import gc

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from .config import (EARLY_STOPPING_ROUNDS, LGB_PARAMS, MAX_BOOST_ROUNDS,
                     RF_PARAMS, XGB_PARAMS)


def train_lightgbm(X_tr, y_tr, X_val, y_val):
    """Train LightGBM and return (model, val_predictions, auc)."""
    print("Training LightGBM...")
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
    """Train XGBoost and return (model, val_predictions, test_predictions, auc)."""
    print("Training XGBoost...")
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
    """Train Random Forest and return (model, val_predictions, auc)."""
    print("Training Random Forest...")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_tr, y_tr)
    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print(f"Random Forest Validation AUC: {auc:.6f}")
    return model, val_pred, auc


def build_ensemble(val_preds: dict, test_preds: dict, aucs: dict, y_val):
    """Compute AUC-weighted ensemble predictions.

    Returns (ensemble_val, ensemble_test, weights_dict).
    """
    names = list(aucs.keys())
    weights = np.array([aucs[n] for n in names])
    weights = weights / weights.sum()

    val_stack = np.column_stack([val_preds[n] for n in names])
    test_stack = np.column_stack([test_preds[n] for n in names])

    ensemble_val = val_stack @ weights
    ensemble_test = test_stack @ weights
    ensemble_auc = roc_auc_score(y_val, ensemble_val)

    print(f"\nEnsemble Validation AUC: {ensemble_auc:.6f}")
    print("Ensemble weights:")
    for name, w in zip(names, weights):
        print(f"  {name}: {w:.4f}")

    return ensemble_val, ensemble_test, dict(zip(names, weights)), ensemble_auc
