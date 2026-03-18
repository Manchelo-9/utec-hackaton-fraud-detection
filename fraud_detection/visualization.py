"""Evaluation plots: ROC curves, confusion matrix, feature importance,
prediction distributions, model comparison, correlation heatmap, and VIF."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             precision_recall_curve, roc_auc_score)
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ── NA visualizations ───────────────────────────────────────────────────


def plot_na_bar(df: pd.DataFrame, title: str = "Missing Values by Column",
                top_n: int = 50, save_path=None):
    """Horizontal bar chart of NA percentage per column (top *top_n*)."""
    na_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    na_pct = na_pct[na_pct > 0].head(top_n)

    if na_pct.empty:
        print(f"  {title}: no missing values found — skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(na_pct) * 0.3)))
    colors = ["#e74c3c" if v > 50 else "#f39c12" if v > 20 else "#2ecc71"
              for v in na_pct.values]
    na_pct.plot.barh(ax=ax, color=colors)
    ax.axvline(x=50, color="red", linestyle="--", alpha=0.6, label="50 % threshold")
    ax.set_xlabel("Missing (%)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_na_heatmap(df: pd.DataFrame, title: str = "NA Heatmap",
                    sample_n: int = 500, save_path=None):
    """Binary heatmap of missingness (rows × columns).

    Samples *sample_n* rows for readability.  Colored = NA, white = present.
    """
    sample = df.sample(n=min(sample_n, len(df)), random_state=42)
    na_matrix = sample.isnull()

    # Sort columns so high-NA ones cluster to the right
    col_order = na_matrix.sum().sort_values().index
    na_matrix = na_matrix[col_order]

    fig, ax = plt.subplots(figsize=(max(14, len(df.columns) * 0.06), 6))
    ax.imshow(na_matrix.values, aspect="auto", cmap="Reds", interpolation="nearest")
    ax.set_title(f"{title}  ({len(sample)} sampled rows × {len(df.columns)} columns)")
    ax.set_ylabel("Rows")
    ax.set_xlabel("Columns (sorted by NA count →)")

    # Show a few column labels
    n_cols = len(col_order)
    tick_step = max(1, n_cols // 20)
    ax.set_xticks(range(0, n_cols, tick_step))
    ax.set_xticklabels([col_order[i] for i in range(0, n_cols, tick_step)],
                       rotation=90, fontsize=6)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_na_treatment_summary(df_before: pd.DataFrame, df_after: pd.DataFrame,
                              df_after_imputed: pd.DataFrame,
                              save_path=None):
    """Three-panel comparison showing NA counts at each pipeline stage.

    Panels:
      1. Raw data (after load)
      2. After cleaning + feature engineering (NaN preserved for LGB/XGB)
      3. After median imputation (for RandomForest)
    """
    stages = {
        "1. Raw (after load)": df_before,
        "2. After cleaning + FE\n(LGB / XGB input)": df_after,
        "3. After imputation\n(RF input)": df_after_imputed,
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (label, df) in zip(axes, stages.items()):
        na_total = int(df.isnull().sum().sum())
        na_cols = int((df.isnull().sum() > 0).sum())
        na_pct = df.isnull().sum().sum() / df.size * 100

        # Top-10 NA columns as a mini bar
        top = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False).head(10)
        top = top[top > 0]
        if top.empty:
            ax.text(0.5, 0.5, "No NAs", ha="center", va="center",
                    fontsize=16, fontweight="bold", color="#2ecc71",
                    transform=ax.transAxes)
        else:
            top.plot.barh(ax=ax, color="#e74c3c", alpha=0.8)
            ax.invert_yaxis()
            ax.set_xlabel("Missing (%)")

        ax.set_title(f"{label}\n"
                     f"{df.shape[0]:,} rows × {df.shape[1]} cols\n"
                     f"NA cells: {na_total:,} ({na_pct:.2f}%) — {na_cols} cols w/ NA",
                     fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("NA Treatment Pipeline — Before & After", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ── Evaluation plots ────────────────────────────────────────────────────


def plot_roc_curves(y_val, val_preds: dict, ensemble_val, save_path=None):
    """Plot ROC curves for each model and the ensemble on a single figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, preds in val_preds.items():
        RocCurveDisplay.from_predictions(y_val, preds, name=name, ax=ax)
    RocCurveDisplay.from_predictions(y_val, ensemble_val, name="Ensemble",
                                     ax=ax, linestyle="--", linewidth=2)
    ax.set_title("ROC Curves — Model Comparison")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_precision_recall(y_val, val_preds: dict, ensemble_val, save_path=None):
    """Plot Precision-Recall curves for each model and the ensemble."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, preds in val_preds.items():
        precision, recall, _ = precision_recall_curve(y_val, preds)
        ax.plot(recall, precision, label=name)
    precision, recall, _ = precision_recall_curve(y_val, ensemble_val)
    ax.plot(recall, precision, label="Ensemble", linestyle="--", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_confusion_matrix(y_val, val_preds: dict, threshold=0.5, save_path=None):
    """Plot confusion matrices for each model side by side."""
    names = list(val_preds.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        y_pred_binary = (val_preds[name] >= threshold).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            y_val, y_pred_binary, ax=ax, cmap="Blues",
            display_labels=["Legit", "Fraud"],
        )
        auc = roc_auc_score(y_val, val_preds[name])
        ax.set_title(f"{name}\n(AUC={auc:.4f})")
    fig.suptitle("Confusion Matrices (threshold=0.5)", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_prediction_distribution(y_val, val_preds: dict, save_path=None):
    """Plot predicted probability distributions for fraud vs legit transactions."""
    names = list(val_preds.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        preds = val_preds[name]
        ax.hist(preds[y_val == 0], bins=50, alpha=0.6, label="Legit", density=True)
        ax.hist(preds[y_val == 1], bins=50, alpha=0.6, label="Fraud", density=True)
        ax.set_title(name)
        ax.set_xlabel("Predicted probability")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Density")
    fig.suptitle("Prediction Distributions — Fraud vs Legit", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_feature_importance(model, model_name: str, top_n=20, save_path=None):
    """Plot top-N feature importances for a tree-based model.

    Supports LightGBM Booster and sklearn estimators with feature_importances_.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if hasattr(model, "feature_importance"):
        # LightGBM Booster
        importance = model.feature_importance(importance_type="gain")
        features = model.feature_name()
    elif hasattr(model, "feature_importances_"):
        # sklearn (RandomForest)
        importance = model.feature_importances_
        features = (model.feature_names_in_
                     if hasattr(model, "feature_names_in_")
                     else [f"f{i}" for i in range(len(importance))])
    else:
        plt.close(fig)
        return

    idx = np.argsort(importance)[-top_n:]
    ax.barh([features[i] for i in idx], importance[idx])
    ax.set_title(f"Top {top_n} Features — {model_name}")
    ax.set_xlabel("Importance (gain)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_model_comparison(aucs: dict, save_path=None):
    """Bar chart comparing validation AUC across models."""
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(aucs.keys())
    values = list(aucs.values())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(names, values, color=colors[:len(names)])
    ax.set_ylabel("Validation AUC")
    ax.set_title("Model Comparison — Validation AUC")
    ax.set_ylim(min(values) - 0.01, max(values) + 0.005)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_correlation_matrix(X: pd.DataFrame, top_n: int = 40, save_path=None):
    """Heatmap of the Pearson correlation matrix for the *top_n* columns
    (selected by highest variance to keep it readable)."""
    # Pick top-N by variance so the plot isn't 200×200
    variances = X.var().sort_values(ascending=False)
    cols = variances.head(top_n).index.tolist()
    corr = X[cols].corr()

    fig, ax = plt.subplots(figsize=(max(12, top_n * 0.4), max(10, top_n * 0.35)))
    cax = ax.matshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    ax.set_title(f"Correlation Matrix — Top {top_n} Features (by variance)", pad=20)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_vif(X: pd.DataFrame, top_n: int = 30, sample_n: int = 10_000,
             save_path=None):
    """Compute and plot Variance Inflation Factors for the *top_n* features.

    Uses a sample of *sample_n* rows and drops NaN before computation
    to keep runtime reasonable on large datasets.
    """
    # Sample and drop NaN
    Xs = X.sample(n=min(sample_n, len(X)), random_state=42).dropna(axis=1)

    # Pick top_n by variance (VIF on 200+ cols is very slow)
    variances = Xs.var().sort_values(ascending=False)
    cols = variances.head(top_n).index.tolist()
    Xs = Xs[cols].copy()

    # Drop any constant columns (VIF undefined)
    Xs = Xs.loc[:, Xs.nunique() > 1]

    print(f"  Computing VIF for {Xs.shape[1]} features on {len(Xs)} sampled rows...")
    vif_data = pd.DataFrame({
        "feature": Xs.columns,
        "VIF": [variance_inflation_factor(Xs.values, i) for i in range(Xs.shape[1])],
    }).sort_values("VIF", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(6, len(vif_data) * 0.3)))
    colors = ["#e74c3c" if v > 10 else "#f39c12" if v > 5 else "#2ecc71"
              for v in vif_data["VIF"]]
    ax.barh(vif_data["feature"], vif_data["VIF"], color=colors)
    ax.axvline(x=5, color="orange", linestyle="--", alpha=0.7, label="VIF = 5")
    ax.axvline(x=10, color="red", linestyle="--", alpha=0.7, label="VIF = 10")
    ax.set_xlabel("Variance Inflation Factor")
    ax.set_title(f"VIF — Top {len(vif_data)} Features")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return vif_data


def generate_all_plots(y_val, val_preds, ensemble_val, aucs, models, plots_dir,
                       X_train=None):
    """Generate and save all evaluation plots.

    If *X_train* is provided, also generates the correlation heatmap and VIF chart.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating evaluation plots...")

    plot_roc_curves(y_val, val_preds, ensemble_val,
                    save_path=plots_dir / "roc_curves.png")

    plot_precision_recall(y_val, val_preds, ensemble_val,
                          save_path=plots_dir / "precision_recall.png")

    plot_confusion_matrix(y_val, val_preds,
                          save_path=plots_dir / "confusion_matrices.png")

    plot_prediction_distribution(y_val, val_preds,
                                 save_path=plots_dir / "prediction_distributions.png")

    plot_model_comparison(aucs, save_path=plots_dir / "model_comparison.png")

    for name, model in models.items():
        plot_feature_importance(model, name,
                                save_path=plots_dir / f"feature_importance_{name}.png")

    if X_train is not None:
        plot_correlation_matrix(X_train,
                                save_path=plots_dir / "correlation_matrix.png")
        plot_vif(X_train, save_path=plots_dir / "vif.png")

    print("All plots saved.\n")
