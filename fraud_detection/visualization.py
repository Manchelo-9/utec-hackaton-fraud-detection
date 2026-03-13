"""Evaluation plots: ROC curves, confusion matrix, feature importance,
prediction distributions, and model comparison."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             precision_recall_curve, roc_auc_score)


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


def generate_all_plots(y_val, val_preds, ensemble_val, aucs, models, plots_dir):
    """Generate and save all evaluation plots."""
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

    print("All plots saved.\n")
