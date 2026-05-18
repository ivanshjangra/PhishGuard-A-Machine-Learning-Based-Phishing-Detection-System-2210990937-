"""
src/evaluation/evaluate.py

PhishGuard Evaluation Suite
----------------------------
Loads the saved model and test split, then produces:
  - Classification report (precision / recall / F1 per class)
  - Confusion matrix plot
  - ROC-AUC curve plot
  - Precision-Recall curve plot
  - Feature importance bar chart

Run with:
    python src/evaluation/evaluate.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend -- safe on macOS
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

sys.path.insert(0, ".")
import config


# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE = {
    "phishing":   "#E84855",
    "legitimate": "#3A86FF",
    "neutral":    "#8D99AE",
    "background": "#F8F9FA",
    "grid":       "#DEE2E6",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["background"],
    "axes.facecolor":    PALETTE["background"],
    "axes.edgecolor":    PALETTE["neutral"],
    "axes.grid":         True,
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_artifacts():
    model        = joblib.load(config.MODEL_PATH)
    scaler       = joblib.load(config.SCALER_PATH)
    feature_cols = joblib.load(config.FEATURE_NAMES_PATH)
    X_test       = np.load(config.MODELS_DIR / "X_test.npy")
    y_test       = np.load(config.MODELS_DIR / "y_test.npy")
    return model, scaler, feature_cols, X_test, y_test


def savefig(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── 1. Classification Report ──────────────────────────────────────────────────

def print_classification_report(y_true, y_pred, y_proba):
    print()
    print("=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(
        y_true, y_pred,
        target_names=["Legitimate", "Phishing"],
        digits=4,
    ))
    roc = roc_auc_score(y_true, y_proba)
    ap  = average_precision_score(y_true, y_proba)
    print(f"  ROC-AUC score          : {roc:.4f}")
    print(f"  Average Precision (AP) : {ap:.4f}")


# ── 2. Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        linewidths=0.5,
        linecolor=PALETTE["grid"],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label", labelpad=10)
    ax.set_title("Confusion Matrix")

    # Annotation: FPR and FNR
    total = len(y_true)
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0
    ax.text(
        0.5, -0.18,
        f"False Positive Rate: {fpr_val:.3f}   |   "
        f"False Negative Rate: {fnr_val:.3f}   |   "
        f"N={total}",
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=9, color=PALETTE["neutral"],
    )

    fig.tight_layout(pad=2)
    return savefig(fig, "confusion_matrix.png")


# ── 3. ROC Curve ──────────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, color=PALETTE["phishing"], lw=2,
            label=f"ROC Curve  (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"],
            lw=1.2, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["phishing"])

    # Mark the operating point closest to threshold 0.5
    idx = np.argmin(np.abs(thresholds - config.DECISION_THRESHOLD))
    ax.scatter(fpr[idx], tpr[idx], s=80, zorder=5,
               color=PALETTE["phishing"],
               label=f"Threshold={config.DECISION_THRESHOLD}  "
                     f"(TPR={tpr[idx]:.3f}, FPR={fpr[idx]:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout(pad=2)
    return savefig(fig, "roc_curve.png")


# ── 4. Precision-Recall Curve ─────────────────────────────────────────────────

def plot_precision_recall_curve(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recall, precision, color=PALETTE["legitimate"], lw=2,
            label=f"PR Curve  (AP = {ap:.4f})")
    ax.axhline(baseline, color=PALETTE["neutral"], lw=1.2,
               linestyle="--", label=f"Baseline (prevalence={baseline:.2f})")
    ax.fill_between(recall, precision, baseline,
                    where=(precision >= baseline),
                    alpha=0.08, color=PALETTE["legitimate"])

    # Mark operating point at threshold 0.5
    idx = np.argmin(np.abs(thresholds - config.DECISION_THRESHOLD))
    ax.scatter(recall[idx], precision[idx], s=80, zorder=5,
               color=PALETTE["legitimate"],
               label=f"Threshold={config.DECISION_THRESHOLD}  "
                     f"(P={precision[idx]:.3f}, R={recall[idx]:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout(pad=2)
    return savefig(fig, "precision_recall_curve.png")


# ── 5. Feature Importance ─────────────────────────────────────────────────────

def plot_feature_importance(model, feature_cols, top_n=20):
    """
    Works for RandomForest and XGBoost (both expose feature_importances_).
    Falls back gracefully for VotingClassifier by averaging sub-estimator importances.
    """
    from sklearn.ensemble import VotingClassifier

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif isinstance(model, VotingClassifier):
        # Average importances across estimators that support it
        imps = []
        for name, est in model.estimators_:
            inner = est
            # Unwrap Pipeline if needed
            if hasattr(est, "named_steps"):
                for step_name, step in est.named_steps.items():
                    if hasattr(step, "feature_importances_"):
                        inner = step
                        break
            if hasattr(inner, "feature_importances_"):
                imps.append(inner.feature_importances_)
        if not imps:
            print("  [!] VotingClassifier: no sub-estimator has feature_importances_. "
                  "Skipping importance plot.")
            return None
        importances = np.mean(imps, axis=0)
    else:
        print("  [!] Model does not expose feature_importances_. Skipping.")
        return None

    # Sort and select top N
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_cols[i] for i in indices]
    top_imp      = importances[indices]

    # Colour bars by importance tier
    colours = [
        PALETTE["phishing"] if v >= 0.05
        else PALETTE["legitimate"] if v >= 0.02
        else PALETTE["neutral"]
        for v in top_imp
    ]

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.38)))
    bars = ax.barh(
        range(top_n), top_imp[::-1],
        color=colours[::-1], edgecolor="white", linewidth=0.5,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Value labels
    for bar, val in zip(bars, top_imp[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8,
                color=PALETTE["neutral"])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["phishing"],   label="High (>= 5%)"),
        Patch(facecolor=PALETTE["legitimate"], label="Medium (2-5%)"),
        Patch(facecolor=PALETTE["neutral"],    label="Low (< 2%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout(pad=2)
    return savefig(fig, "feature_importance.png")


# ── 6. Summary Table ──────────────────────────────────────────────────────────

def print_summary_table(y_true, y_pred, y_proba):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, average_precision_score,
        matthews_corrcoef,
    )
    cm  = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Accuracy":           accuracy_score(y_true, y_pred),
        "Precision":          precision_score(y_true, y_pred),
        "Recall (TPR)":       recall_score(y_true, y_pred),
        "F1-Score":           f1_score(y_true, y_pred),
        "Specificity (TNR)":  tn / (tn + fp) if (tn + fp) > 0 else 0,
        "False Positive Rate":fp / (fp + tn) if (fp + tn) > 0 else 0,
        "False Negative Rate":fn / (fn + tp) if (fn + tp) > 0 else 0,
        "ROC-AUC":            roc_auc_score(y_true, y_proba),
        "Avg Precision (AP)": average_precision_score(y_true, y_proba),
        "MCC":                matthews_corrcoef(y_true, y_pred),
    }

    print()
    print("=" * 60)
    print("Summary Metrics")
    print("=" * 60)
    for name, val in metrics.items():
        bar_len = int(val * 30) if 0 <= val <= 1 else 0
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {name:<25}  {val:.4f}  |{bar}|")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate():
    print()
    print("=" * 60)
    print("PhishGuard -- Full Evaluation Suite")
    print("=" * 60)

    model, scaler, feature_cols, X_test, y_test = load_artifacts()
    print(f"  Model         : {type(model).__name__}")
    print(f"  Test samples  : {len(y_test)}")
    print(f"  Features      : {len(feature_cols)}")

    # Predictions
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Reports
    print_classification_report(y_test, y_pred, y_proba)
    print_summary_table(y_test, y_pred, y_proba)

    # Plots
    print("Generating plots ...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall_curve(y_test, y_proba)
    plot_feature_importance(model, feature_cols, top_n=20)

    print()
    print("All evaluation artefacts saved to:", config.FIGURES_DIR)
    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    evaluate()
