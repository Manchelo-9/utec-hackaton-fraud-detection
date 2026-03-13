"""IEEE-CIS Fraud Detection — Full pipeline.

Usage:
    python main.py
"""

from fraud_detection.config import OUTPUT_DIR, PLOTS_DIR
from fraud_detection.data_loader import load_data
from fraud_detection.features import engineer_features, time_based_split
from fraud_detection.models import (build_ensemble, train_lightgbm,
                                    train_random_forest, train_xgboost)
from fraud_detection.submission import save_submissions
from fraud_detection.visualization import generate_all_plots


def main():
    # 1. Load data
    train, test = load_data()

    # 2. Feature engineering
    X_train, y_train, X_test, submission = engineer_features(train, test)
    X_tr, y_tr, X_val, y_val = time_based_split(X_train, y_train)

    # 3. Train models
    lgb_model, lgb_val, lgb_auc = train_lightgbm(X_tr, y_tr, X_val, y_val)
    xgb_model, xgb_val, xgb_test, xgb_auc = train_xgboost(
        X_tr, y_tr, X_val, y_val, X_test)
    rf_model, rf_val, rf_auc = train_random_forest(X_tr, y_tr, X_val, y_val)

    lgb_test = lgb_model.predict(X_test)
    rf_test = rf_model.predict_proba(X_test)[:, 1]

    # 4. Ensemble
    val_preds = {"lightgbm": lgb_val, "xgboost": xgb_val, "randomforest": rf_val}
    test_preds = {"lightgbm": lgb_test, "xgboost": xgb_test, "randomforest": rf_test}
    aucs = {"lightgbm": lgb_auc, "xgboost": xgb_auc, "randomforest": rf_auc}

    ensemble_val, ensemble_test, weights, ensemble_auc = build_ensemble(
        val_preds, test_preds, aucs, y_val)
    aucs["ensemble"] = ensemble_auc

    # 5. Visualizations
    models = {"lightgbm": lgb_model, "randomforest": rf_model}
    generate_all_plots(y_val, val_preds, ensemble_val, aucs, models, PLOTS_DIR)

    # 6. Save submissions
    save_submissions(submission, test_preds, ensemble_test, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, auc in aucs.items():
        print(f"  {name:20s} AUC: {auc:.6f}")
    print(f"\nPlots saved to:       {PLOTS_DIR}/")
    print(f"Submissions saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
