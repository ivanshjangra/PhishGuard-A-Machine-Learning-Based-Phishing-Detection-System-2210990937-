import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap

sys.path.insert(0, ".")
import config
from src.features.extractor import extract, extract_batch
from src.training.predict import load_model

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PALETTE = {
    "phishing":   "#E84855",
    "legitimate": "#3A86FF",
    "neutral":    "#8D99AE",
    "background": "#F8F9FA",
    "grid":       "#DEE2E6",
    "positive":   "#E84855",
    "negative":   "#3A86FF",
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
})

_EXPLAINER = None


def _get_explainer(model, X_background=None):
    global _EXPLAINER
    if _EXPLAINER is not None:
        return _EXPLAINER
    from sklearn.ensemble import VotingClassifier
    if isinstance(model, VotingClassifier):
        if X_background is None:
            raise ValueError("KernelExplainer requires X_background for VotingClassifier.")
        bg = shap.sample(X_background, 100)
        _EXPLAINER = shap.KernelExplainer(model.predict_proba, bg)
    else:
        _EXPLAINER = shap.TreeExplainer(model)
    return _EXPLAINER


def _feature_vector(url, feature_cols):
    _, scaler, _ = load_model()
    feats = extract(url)
    X_raw = np.array([[feats[c] for c in feature_cols]], dtype=np.float32)
    return scaler.transform(X_raw)


def _shap_values_for_url(url):
    model, scaler, feature_cols = load_model()
    X_scaled  = _feature_vector(url, feature_cols)
    explainer = _get_explainer(model)
    raw = explainer.shap_values(X_scaled)
    if isinstance(raw, list):
        shap_vals = raw[1][0]
        base_val  = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[1])
        else:
            base_val = float(base_val)
    else:
        shap_vals = raw[0, :, 1]
        base_val  = float(explainer.expected_value[1])
    return shap_vals, X_scaled, feature_cols, base_val


FEATURE_DESCRIPTIONS = {
    "has_ip_address":      "URL uses a raw IP address instead of a domain name",
    "has_suspicious_tld":  "Domain uses a high-risk TLD (.xyz, .tk, .ml, etc.)",
    "brand_in_subdomain":  "A known brand name appears in the subdomain (impersonation)",
    "n_phish_keywords":    "URL contains phishing-associated words (verify, secure, login...)",
    "has_https":           "Connection uses HTTPS (encrypted / trusted)",
    "has_port":            "URL specifies a non-standard network port",
    "url_length":          "URL is unusually long",
    "n_hyphens":           "URL contains many hyphens (common in fake brand domains)",
    "n_subdomains":        "URL has many subdomain layers (obfuscation)",
    "url_entropy":         "URL string has high randomness (suggests auto-generated domain)",
    "domain_length":       "Domain name is unusually long",
    "n_dots":              "URL contains many dots (excessive subdomains)",
    "query_length":        "Query string is unusually long",
    "n_query_params":      "URL has many query parameters",
    "n_at_symbols":        "URL contains @ symbol (redirect trick)",
    "has_double_slash":    "URL contains // after the protocol (redirect trick)",
    "ratio_digits_url":    "High proportion of digits in URL",
    "domain_hyphens":      "Domain name itself contains hyphens",
    "n_percent_encoding":  "URL uses percent-encoded characters (obfuscation)",
    "path_length":         "URL path is unusually long",
    "n_path_segments":     "URL path has many directory segments",
    "n_digits_in_url":     "URL contains many digit characters",
    "ratio_digits_domain": "Domain name has a high proportion of digits",
    "n_slashes":           "URL contains many slashes",
    "n_question_marks":    "URL contains multiple question marks",
    "n_equal_signs":       "URL contains many equal signs (long query string)",
    "n_ampersands":        "URL contains many ampersands (many parameters)",
}


def _build_explanation_text(prediction, top_drivers, base_value):
    verdict = prediction["verdict"]
    prob    = prediction["phish_prob"]
    risk    = prediction["risk_level"]
    url     = prediction["url"][:70]

    lines = [
        "VERDICT: " + verdict + "  |  Risk: " + risk + "  |  Phishing probability: " + str(round(prob * 100, 1)) + "%",
        "",
        "URL (truncated): " + url,
        "",
        "Top signals driving this decision:",
    ]

    for driver in top_drivers[:5]:
        feat      = driver["feature"]
        direction = driver["direction"]
        shap_v    = driver["shap"]
        val       = driver["value"]
        desc      = FEATURE_DESCRIPTIONS.get(feat, feat.replace("_", " "))
        arrow     = "-> PHISHING" if direction == "phishing" else "-> LEGITIMATE"
        impact    = "HIGH" if abs(shap_v) >= 0.05 else "MEDIUM" if abs(shap_v) >= 0.02 else "LOW"
        line = "  [" + impact + "] " + desc
        line += "\n            (feature=" + feat + ", value=" + str(val) + ", SHAP=" + "{:+.4f}".format(shap_v) + " " + arrow + ")"
        lines.append(line)

    lines += [
        "",
        "Base rate (expected output without features): " + "{:.4f}".format(base_value),
        "",
        "NOTE: This tool is for defensive triage only. Always confirm before blocking.",
    ]
    return "\n".join(lines)


def explain(url, top_n=10):
    from src.training.predict import predict
    prediction = predict(url)
    shap_vals, X_scaled, feature_cols, base_val = _shap_values_for_url(url)
    feat_vals = extract(url)
    shap_dict = {col: float(shap_vals[i]) for i, col in enumerate(feature_cols)}
    sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_drivers = []
    for feat, shap_v in sorted_items[:top_n]:
        top_drivers.append({
            "feature":   feat,
            "value":     feat_vals[feat],
            "shap":      round(shap_v, 5),
            "direction": "phishing" if shap_v > 0 else "legitimate",
        })
    explanation_text = _build_explanation_text(prediction, top_drivers, base_val)
    return {
        **prediction,
        "shap_values":      shap_dict,
        "top_drivers":      top_drivers,
        "base_value":       round(base_val, 4),
        "explanation_text": explanation_text,
    }


def plot_shap_waterfall(url, top_n=15):
    shap_vals, X_scaled, feature_cols, base_val = _shap_values_for_url(url)
    feat_vals = extract(url)
    order   = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    top_idx = order[::-1]
    feats   = [feature_cols[i] for i in top_idx]
    svals   = [shap_vals[i]    for i in top_idx]
    fvals   = [feat_vals[feature_cols[i]] for i in top_idx]
    colours = [PALETTE["positive"] if s > 0 else PALETTE["negative"] for s in svals]

    fig, ax = plt.subplots(figsize=(9, max(5, len(feats) * 0.45)))
    bars = ax.barh(range(len(feats)), svals, color=colours,
                   edgecolor="white", linewidth=0.4, height=0.65)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([f + "  (=" + str(v) + ")" for f, v in zip(feats, fvals)], fontsize=9)
    ax.axvline(0, color=PALETTE["neutral"], linewidth=1.0, linestyle="-")
    ax.set_xlabel("SHAP Value  (positive = pushes toward Phishing)")
    title = "SHAP Waterfall  |  " + url[:60] + ("..." if len(url) > 60 else "")
    ax.set_title(title, pad=12)
    for bar, val in zip(bars, svals):
        x_pos = val + (0.002 if val >= 0 else -0.002)
        ha    = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                "{:+.4f}".format(val), va="center", ha=ha, fontsize=8,
                color=PALETTE["neutral"])
    pos_patch = mpatches.Patch(color=PALETTE["positive"], label="Pushes toward Phishing")
    neg_patch = mpatches.Patch(color=PALETTE["negative"], label="Pushes toward Legitimate")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)
    fig.tight_layout(pad=2)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "shap_waterfall.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: " + str(out))
    return str(out)


def plot_shap_summary(n_samples=300):
    model, scaler, feature_cols = load_model()
    X_test = np.load(config.MODELS_DIR / "X_test.npy")
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)
    X_sample  = X_test[idx]
    explainer = _get_explainer(model, X_test)
    raw = explainer.shap_values(X_sample)
    shap_matrix = raw[1] if isinstance(raw, list) else raw[:, :, 1]
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.sca(ax)
    shap.summary_plot(shap_matrix, X_sample, feature_names=feature_cols,
                      show=False, plot_size=None, color_bar=True)
    ax.set_title("SHAP Summary  (n=" + str(len(idx)) + " samples)", pad=12)
    fig.tight_layout(pad=2)
    out = config.FIGURES_DIR / "shap_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: " + str(out))
    return str(out)


def plot_shap_bar(n_samples=300):
    model, scaler, feature_cols = load_model()
    X_test = np.load(config.MODELS_DIR / "X_test.npy")
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)
    X_sample  = X_test[idx]
    explainer = _get_explainer(model, X_test)
    raw = explainer.shap_values(X_sample)
    shap_matrix = raw[1] if isinstance(raw, list) else raw[:, :, 1]
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    order    = np.argsort(mean_abs)
    feats    = [feature_cols[i] for i in order]
    vals     = mean_abs[order]
    colours  = [
        PALETTE["phishing"]   if v >= 0.05
        else PALETTE["legitimate"] if v >= 0.02
        else PALETTE["neutral"]
        for v in vals
    ]
    fig, ax = plt.subplots(figsize=(8, max(6, len(feats) * 0.38)))
    ax.barh(range(len(feats)), vals, color=colours, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|  (average impact on model output)")
    ax.set_title("Global Feature Importance via SHAP  (n=" + str(len(idx)) + " samples)", pad=12)
    for i, v in enumerate(vals):
        ax.text(v + 0.0005, i, "{:.4f}".format(v), va="center", fontsize=8,
                color=PALETTE["neutral"])
    pos_patch = mpatches.Patch(color=PALETTE["phishing"],   label="High impact (>= 5%)")
    med_patch = mpatches.Patch(color=PALETTE["legitimate"], label="Medium impact (2-5%)")
    low_patch = mpatches.Patch(color=PALETTE["neutral"],    label="Low impact (< 2%)")
    ax.legend(handles=[pos_patch, med_patch, low_patch], loc="lower right", fontsize=9)
    fig.tight_layout(pad=2)
    out = config.FIGURES_DIR / "shap_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: " + str(out))
    return str(out)


if __name__ == "__main__":
    demo_urls = [
        "https://www.google.com/search?q=weather",
        "http://paypal-secure.login.kqrt8ml.xyz/update/confirm?id=1&tok=abc",
        "http://192.168.1.1:8080/verify/login.html?session=xyz",
    ]

    print()
    print("=" * 70)
    print("PhishGuard -- SHAP Explainability Demo")
    print("=" * 70)

    for url in demo_urls:
        print()
        print("-" * 70)
        result = explain(url, top_n=5)
        print(result["explanation_text"])

    print()
    print("Generating waterfall plot ...")
    plot_shap_waterfall(demo_urls[1], top_n=15)
    print("Generating SHAP summary plot ...")
    plot_shap_summary(n_samples=300)
    print("Generating SHAP bar chart ...")
    plot_shap_bar(n_samples=300)
    print()
    print("All SHAP plots saved to: " + str(config.FIGURES_DIR))
    print()
    print("SHAP demo complete.")
