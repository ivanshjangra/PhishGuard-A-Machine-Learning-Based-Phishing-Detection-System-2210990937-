import sys, numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, ".")
import config
from src.features.extractor import feature_names as get_feature_names

def load_data():
    df = pd.read_csv(config.PROCESSED_DATASET_PATH)
    feature_cols = get_feature_names()
    X = df[feature_cols].values.astype(np.float32)
    y = df[config.LABEL_COLUMN].values.astype(np.int32)
    return X, y, feature_cols

def compute_permutation_importance(X_val, y_val, model, feature_cols, n_repeats=15):
    print("  Computing permutation importance...")
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        scoring="f1",
    )
    df_imp = pd.DataFrame({
        "feature":    feature_cols,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df_imp

def select_top_features(df_imp, top_n=14):
    return df_imp.head(top_n)["feature"].tolist()

def compare_full_vs_reduced(X, y, feature_cols, selected_features):
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    full_idx    = list(range(len(feature_cols)))
    reduced_idx = [feature_cols.index(f) for f in selected_features]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                class_weight="balanced",
                                random_state=config.RANDOM_STATE, n_jobs=-1)
    print()
    print("  {:<22}  {:>10}  {:>10}  {:>10}  {:>10}".format(
        "Model", "Accuracy", "F1", "Precision", "Recall"))
    print("  " + "-" * 58)
    for name, idx in [("Full (" + str(len(feature_cols)) + ")", full_idx),
                      ("Reduced (" + str(len(selected_features)) + ")", reduced_idx)]:
        scores = {}
        for metric in ["accuracy", "f1", "precision", "recall"]:
            s = cross_val_score(rf, X_sc[:, idx], y, cv=cv, scoring=metric, n_jobs=-1)
            scores[metric] = s.mean()
        print("  {:<22}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}".format(
            name, scores["accuracy"], scores["f1"],
            scores["precision"], scores["recall"]))

def plot_permutation_importance(df_imp, selected_features):
    fig, ax = plt.subplots(figsize=(9, max(6, len(df_imp) * 0.4)))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")
    colours = ["#3A86FF" if r["feature"] in selected_features else "#DEE2E6"
               for _, r in df_imp.iterrows()][::-1]
    imp_r  = df_imp["importance"].values[::-1]
    std_r  = df_imp["std"].values[::-1]
    feat_r = df_imp["feature"].values[::-1]
    ax.barh(range(len(feat_r)), imp_r, xerr=std_r, color=colours,
            edgecolor="white", linewidth=0.4, height=0.65,
            error_kw={"elinewidth": 1, "ecolor": "#8D99AE"})
    ax.set_yticks(range(len(feat_r)))
    ax.set_yticklabels(feat_r, fontsize=9)
    ax.set_xlabel("Permutation Importance (mean decrease in F1)")
    ax.set_title("Permutation Feature Importance", pad=12)
    sel_patch = mpatches.Patch(color="#3A86FF", label="Selected")
    drp_patch = mpatches.Patch(color="#DEE2E6", label="Dropped")
    ax.legend(handles=[sel_patch, drp_patch], fontsize=9, loc="lower right")
    fig.tight_layout(pad=2)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "permutation_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: " + str(out))

def save_selected_features(selected_features):
    path = config.MODELS_DIR / "selected_features.joblib"
    joblib.dump(selected_features, path)
    print("  Saved selected features -> " + str(path))

def run_feature_selection(top_n=14):
    print()
    print("=" * 60)
    print("PhishGuard -- Permutation Feature Selection")
    print("=" * 60)
    X, y, feature_cols = load_data()
    print("  Loaded " + str(len(X)) + " samples, " + str(len(feature_cols)) + " features")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    rf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                class_weight="balanced",
                                random_state=config.RANDOM_STATE, n_jobs=-1)
    print("  Training RF for importance computation...")
    rf.fit(X_train_sc, y_train)
    df_imp   = compute_permutation_importance(X_val_sc, y_val, rf, feature_cols)
    selected = select_top_features(df_imp, top_n=top_n)
    print()
    print("  Rank  Feature                    Importance")
    print("  " + "-" * 48)
    for i, row in df_imp.iterrows():
        marker = "  <-- SELECT" if row["feature"] in selected else ""
        print("  {:<5} {:<25}  {:>10.4f}{}".format(
            i+1, row["feature"], row["importance"], marker))
    print()
    print("  Selected " + str(len(selected)) + " features:")
    for f in selected:
        print("    - " + f)
    compare_full_vs_reduced(X, y, feature_cols, selected)
    plot_permutation_importance(df_imp, selected)
    save_selected_features(selected)
    print()
    print("Feature selection complete.")
    return selected

if __name__ == "__main__":
    run_feature_selection(top_n=config.TOP_N_FEATURES)
