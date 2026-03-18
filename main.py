"""IEEE-CIS Fraud Detection — Full pipeline.

Usage:
    python main.py
"""

from fraud_detection.config import PLOTS_DIR
from fraud_detection.data_loader import load_data
from fraud_detection.features import impute_for_rf, prepare_data
from fraud_detection.models import (build_ensemble, train_lightgbm,
                                    train_random_forest, train_xgboost)
from fraud_detection.visualization import generate_all_plots


def main():
    # 1. Load train data only
    raw = load_data()

    # 2. Clean, split, then engineer features (leak-free)
    X_train, X_test, y_train, y_test = prepare_data(raw)

    # 3. Median-impute for RandomForest (LGB/XGB handle NaN natively)
    X_train_imp, X_test_imp = impute_for_rf(X_train, X_test)

    # 4. Train models
    lgb_model, lgb_pred, lgb_auc = train_lightgbm(X_train, y_train, X_test, y_test)
    xgb_model, xgb_pred, xgb_auc = train_xgboost(X_train, y_train, X_test, y_test)
    rf_model, rf_pred, rf_auc = train_random_forest(
        X_train_imp, y_train, X_test_imp, y_test)

    # 5. Ensemble
    preds = {"lightgbm": lgb_pred, "xgboost": xgb_pred, "randomforest": rf_pred}
    aucs = {"lightgbm": lgb_auc, "xgboost": xgb_auc, "randomforest": rf_auc}
    ensemble_pred, weights, ensemble_auc = build_ensemble(preds, aucs, y_test)
    aucs["ensemble"] = ensemble_auc

    # 6. Visualizations
    models = {"lightgbm": lgb_model, "randomforest": rf_model}
    generate_all_plots(y_test, preds, ensemble_pred, aucs, models, PLOTS_DIR,
                       X_train=X_train)

    # 7. Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, auc in aucs.items():
        print(f"  {name:20s} AUC: {auc:.6f}")
    print(f"\nPlots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
