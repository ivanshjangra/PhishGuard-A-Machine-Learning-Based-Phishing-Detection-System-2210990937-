import sys
import io
import base64
import pathlib

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, ".")
import config

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    .verdict-phishing {
        background: linear-gradient(135deg, #E84855 0%, #c0392b 100%);
        color: white; border-radius: 12px; padding: 1.5rem 2rem;
        text-align: center; margin-bottom: 1rem;
    }
    .verdict-legitimate {
        background: linear-gradient(135deg, #3A86FF 0%, #1a5fd1 100%);
        color: white; border-radius: 12px; padding: 1.5rem 2rem;
        text-align: center; margin-bottom: 1rem;
    }
    .verdict-title  { font-size: 2rem; font-weight: 800; margin: 0; }
    .verdict-prob   { font-size: 1.1rem; margin: 0.3rem 0 0 0; opacity: 0.92; }

    .risk-HIGH   { background:#E84855; color:white; border-radius:6px;
                   padding:3px 12px; font-weight:700; font-size:0.9rem; }
    .risk-MEDIUM { background:#FF9F1C; color:white; border-radius:6px;
                   padding:3px 12px; font-weight:700; font-size:0.9rem; }
    .risk-LOW    { background:#8D99AE; color:white; border-radius:6px;
                   padding:3px 12px; font-weight:700; font-size:0.9rem; }
    .risk-SAFE   { background:#3A86FF; color:white; border-radius:6px;
                   padding:3px 12px; font-weight:700; font-size:0.9rem; }

    .metric-card {
        background: white; border-radius: 10px; padding: 1rem 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center;
    }
    .metric-label { font-size: 0.78rem; color: #8D99AE;
                    text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.6rem; font-weight: 800; color: #2B2D42; }

    .driver-card {
        background: white; border-radius: 8px; padding: 0.75rem 1rem;
        margin-bottom: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border-left: 4px solid #ccc;
    }
    .driver-phishing   { border-left-color: #E84855 !important; }
    .driver-legitimate { border-left-color: #3A86FF !important; }
    .driver-feature    { font-weight: 700; font-size: 0.95rem; color: #2B2D42; }
    .driver-desc       { font-size: 0.82rem; color: #6c757d; margin-top: 2px; }
    .driver-shap       { font-size: 0.82rem; font-weight: 600; }
    .shap-pos { color: #E84855; }
    .shap-neg { color: #3A86FF; }

    .explain-box {
        background: #fff; border-radius: 10px; padding: 1.2rem 1.5rem;
        border: 1px solid #DEE2E6; font-family: monospace; font-size: 0.82rem;
        color: #2B2D42; white-space: pre-wrap; line-height: 1.6;
    }

    .history-item {
        background: white; border-radius: 8px; padding: 0.6rem 1rem;
        margin-bottom: 0.4rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        display: flex; justify-content: space-between; align-items: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3A86FF 0%, #1a5fd1 100%);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600; font-size: 1rem;
        width: 100%; cursor: pointer;
    }
    .stButton>button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ── Cached model loading ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading PhishGuard model...")
def get_model():
    from src.training.predict import load_model
    return load_model()


@st.cache_resource(show_spinner="Initialising SHAP explainer...")
def get_explainer():
    from src.explainability.shap_explain import _get_explainer
    model, _, _ = get_model()
    return _get_explainer(model)


# ── Feature descriptions ──────────────────────────────────────────────────────

FEATURE_DESCRIPTIONS = {
    "has_ip_address":      "URL uses a raw IP address instead of a domain name",
    "has_suspicious_tld":  "Domain uses a high-risk TLD (.xyz .tk .ml etc.)",
    "brand_in_subdomain":  "Known brand name in subdomain (impersonation)",
    "n_phish_keywords":    "Phishing keywords present (verify secure login...)",
    "has_https":           "HTTPS present (encrypted / trusted connection)",
    "has_port":            "Non-standard network port specified",
    "url_length":          "URL is unusually long",
    "n_hyphens":           "Many hyphens (common in fake brand domains)",
    "n_subdomains":        "Many subdomain layers (obfuscation)",
    "url_entropy":         "High randomness in URL (auto-generated domain)",
    "domain_length":       "Domain name is unusually long",
    "n_dots":              "Many dots — excessive subdomains",
    "query_length":        "Query string is unusually long",
    "n_query_params":      "Many query parameters",
    "n_at_symbols":        "@ symbol present (redirect trick)",
    "has_double_slash":    "// after protocol (redirect trick)",
    "ratio_digits_url":    "High digit ratio in URL",
    "domain_hyphens":      "Hyphens in domain name",
    "n_percent_encoding":  "Percent-encoded characters (obfuscation)",
    "path_length":         "URL path is unusually long",
    "n_path_segments":     "Many directory segments in path",
    "n_digits_in_url":     "Many digit characters in URL",
    "ratio_digits_domain": "High digit ratio in domain name",
    "n_slashes":           "Many slashes in URL",
    "n_question_marks":    "Multiple question marks",
    "n_equal_signs":       "Many equal signs (long query string)",
    "n_ampersands":        "Many ampersands (many parameters)",
}


# ── Helper: run full prediction + SHAP ───────────────────────────────────────

def run_analysis(url: str) -> dict:
    from src.explainability.shap_explain import explain
    return explain(url, top_n=10)


# ── Helper: waterfall figure → base64 PNG ────────────────────────────────────

def waterfall_figure(result: dict, top_n: int = 12) -> str:
    shap_dict    = result["shap_values"]
    feat_vals    = result["features"]
    feature_cols = list(shap_dict.keys())
    shap_arr     = np.array([shap_dict[c] for c in feature_cols])

    order   = np.argsort(np.abs(shap_arr))[::-1][:top_n]
    top_idx = order[::-1]
    feats   = [feature_cols[i] for i in top_idx]
    svals   = [shap_arr[i]     for i in top_idx]
    fvals   = [feat_vals[feature_cols[i]] for i in top_idx]

    colours = ["#E84855" if s > 0 else "#3A86FF" for s in svals]

    fig, ax = plt.subplots(figsize=(8, max(4, len(feats) * 0.42)))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    bars = ax.barh(range(len(feats)), svals, color=colours,
                   edgecolor="white", linewidth=0.4, height=0.6)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(
        [f + "  (=" + str(round(v, 3)) + ")" for f, v in zip(feats, fvals)],
        fontsize=9,
    )
    ax.axvline(0, color="#8D99AE", linewidth=1.0)
    ax.set_xlabel("SHAP Value  (red = phishing signal, blue = legitimate signal)",
                  fontsize=9)
    ax.set_title("Feature Contributions to This Prediction", fontsize=11,
                 fontweight="bold", pad=10)
    ax.grid(axis="x", color="#DEE2E6", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, svals):
        x_pos = val + (0.001 if val >= 0 else -0.001)
        ha    = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                "{:+.4f}".format(val), va="center", ha=ha,
                fontsize=7.5, color="#6c757d")

    pos_patch = mpatches.Patch(color="#E84855", label="Phishing signal")
    neg_patch = mpatches.Patch(color="#3A86FF", label="Legitimate signal")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8,
              loc="lower right", framealpha=0.9)

    fig.tight_layout(pad=1.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Helper: probability gauge ─────────────────────────────────────────────────

def gauge_figure(prob: float) -> str:
    fig, ax = plt.subplots(figsize=(3.5, 2.2),
                           subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    theta_max = np.pi
    theta     = theta_max * (1 - prob)

    ax.barh(1, theta_max, left=0,        height=0.5,
            color="#3A86FF", alpha=0.25)
    ax.barh(1, theta_max - theta, left=0, height=0.5,
            color=("#E84855" if prob >= 0.5 else "#3A86FF"))

    ax.set_xlim(0, theta_max)
    ax.set_ylim(0, 2)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_thetagrids([])
    ax.set_rgrids([])
    ax.spines["polar"].set_visible(False)

    ax.text(
        np.pi / 2, 0.2,
        "{:.1f}%".format(prob * 100),
        ha="center", va="center",
        fontsize=18, fontweight="bold",
        color=("#E84855" if prob >= 0.5 else "#3A86FF"),
        transform=ax.transData,
    )

    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#F8F9FA")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/shield.png",
            width=64,
        )
        st.title("PhishGuard")
        st.caption("ML-powered phishing detection with SHAP explainability")
        st.divider()

        st.subheader("About")
        st.markdown(
            "PhishGuard analyses URLs using 27 hand-crafted features "
            "across three categories:"
        )
        st.markdown("- **Lexical** — URL string patterns")
        st.markdown("- **Structural** — domain and TLD signals")
        st.markdown("- **Semantic** — brand impersonation & keywords")
        st.divider()

        st.subheader("Model")
        try:
            model, _, feature_cols = get_model()
            st.success("Model loaded")
            st.caption("Type: " + type(model).__name__)
            st.caption("Features: " + str(len(feature_cols)))
        except Exception as e:
            st.error("Model not found. Run train.py first.")
            st.caption(str(e))

        st.divider()
        st.subheader("Quick test URLs")
        examples = {
            "Google (legit)":   "https://www.google.com/search?q=weather",
            "GitHub (legit)":   "https://github.com/scikit-learn/scikit-learn",
            "PayPal phish":     "http://paypal-secure.login.kqrt8ml.xyz/update/confirm",
            "IP-based attack":  "http://192.168.1.1:8080/verify/login.html?tok=abc",
            "Brand subdomain":  "http://apple-id.verify-account.tk/signin",
        }
        for label, url in examples.items():
            if st.button(label, key="ex_" + label):
                st.session_state["url_input"] = url
                st.session_state["run_analysis"] = True

        st.divider()
        st.caption("For defensive use only.")


# ── Main panel ────────────────────────────────────────────────────────────────

def render_main():
    st.markdown("## 🛡️ PhishGuard — URL Analyser")
    st.markdown(
        "Paste any URL below and press **Analyse** to get an instant "
        "phishing verdict with feature-level SHAP explanations."
    )

    # ── URL input ─────────────────────────────────────────────────────────────
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        url_val = st.session_state.get("url_input", "")
        url = st.text_input(
            "URL to analyse",
            value=url_val,
            placeholder="https://example.com/path?query=value",
            label_visibility="collapsed",
            key="url_field",
        )
    with col_btn:
        analyse = st.button("Analyse", use_container_width=True)

    trigger = analyse or st.session_state.pop("run_analysis", False)

    if not trigger:
        render_placeholder()
        return

    if not url.strip():
        st.warning("Please enter a URL before clicking Analyse.")
        return

    # ── Run analysis ──────────────────────────────────────────────────────────
    with st.spinner("Extracting features and computing SHAP values..."):
        try:
            result = run_analysis(url.strip())
        except Exception as e:
            st.error("Analysis failed: " + str(e))
            st.exception(e)
            return

    # ── Store in session history ──────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].insert(0, {
        "url":     url.strip()[:80],
        "verdict": result["verdict"],
        "prob":    result["phish_prob"],
        "risk":    result["risk_level"],
    })
    st.session_state["history"] = st.session_state["history"][:10]

    render_result(result)


def render_placeholder():
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "Enter a URL above and press **Analyse**, or pick a quick-test "
        "example from the sidebar.",
        icon="🔍",
    )
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
<div class='metric-card'>
<div class='metric-label'>Features analysed</div>
<div class='metric-value'>27</div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class='metric-card'>
<div class='metric-label'>Model</div>
<div class='metric-value'>RF</div>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div class='metric-card'>
<div class='metric-label'>Explainability</div>
<div class='metric-value'>SHAP</div>
</div>""", unsafe_allow_html=True)


def render_result(result: dict):
    verdict    = result["verdict"]
    prob       = result["phish_prob"]
    risk       = result["risk_level"]
    is_phish   = result["label"] == 1
    css_class  = "verdict-phishing" if is_phish else "verdict-legitimate"
    icon       = "🚨" if is_phish else "✅"

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Verdict banner + gauge ─────────────────────────────────────────
    col_verdict, col_gauge = st.columns([3, 1])

    with col_verdict:
        st.markdown(
            "<div class='" + css_class + "'>"
            "<p class='verdict-title'>" + icon + "  " + verdict + "</p>"
            "<p class='verdict-prob'>Phishing probability: "
            + "{:.1f}%".format(prob * 100) + "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col_gauge:
        gauge_b64 = gauge_figure(prob)
        st.markdown(
            "<div style='text-align:center;padding-top:0.3rem;'>"
            "<img src='data:image/png;base64," + gauge_b64 + "' "
            "style='width:100%;max-width:220px;border-radius:10px;'>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Row 2: Metric cards ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    risk_colour = {
        "HIGH": "#E84855", "MEDIUM": "#FF9F1C",
        "LOW":  "#8D99AE", "SAFE":   "#3A86FF",
    }.get(risk, "#8D99AE")

    def metric_card(label, value, colour="#2B2D42"):
        return (
            "<div class='metric-card'>"
            "<div class='metric-label'>" + label + "</div>"
            "<div class='metric-value' style='color:" + colour + ";'>"
            + str(value) + "</div></div>"
        )

    with m1:
        st.markdown(metric_card("Risk Level", risk, risk_colour),
                    unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Phish Prob",
                                "{:.1f}%".format(prob * 100),
                                "#E84855" if is_phish else "#3A86FF"),
                    unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card("Legit Prob",
                                "{:.1f}%".format(result["legit_prob"] * 100),
                                "#3A86FF" if not is_phish else "#8D99AE"),
                    unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card("Features Used", "27", "#2B2D42"),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: SHAP waterfall + top drivers ───────────────────────────────────
    col_wf, col_drivers = st.columns([3, 2])

    with col_wf:
        st.markdown("#### SHAP Feature Contributions")
        wf_b64 = waterfall_figure(result, top_n=12)
        st.markdown(
            "<img src='data:image/png;base64," + wf_b64 + "' "
            "style='width:100%;border-radius:10px;'>",
            unsafe_allow_html=True,
        )

    with col_drivers:
        st.markdown("#### Top Driving Features")
        for driver in result["top_drivers"][:8]:
            feat      = driver["feature"]
            shap_v    = driver["shap"]
            val       = driver["value"]
            direction = driver["direction"]
            desc      = FEATURE_DESCRIPTIONS.get(feat, feat.replace("_", " "))
            shap_cls  = "shap-pos" if shap_v > 0 else "shap-neg"
            card_cls  = "driver-" + direction

            st.markdown(
                "<div class='driver-card " + card_cls + "'>"
                "<div class='driver-feature'>" + feat.replace("_", " ") + "</div>"
                "<div class='driver-desc'>" + desc + "</div>"
                "<div class='driver-shap " + shap_cls + "'>"
                "value=" + str(val) + "  &nbsp;|&nbsp;  "
                "SHAP <span class='" + shap_cls + "'>"
                + "{:+.4f}".format(shap_v) + "</span></div>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 4: SOC analyst explanation ───────────────────────────────────────
    st.markdown("#### SOC Analyst Explanation")
    st.markdown(
        "<div class='explain-box'>" + result["explanation_text"] + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 5: Raw feature values (expandable) ────────────────────────────────
    with st.expander("Raw feature values (all 27 features)", expanded=False):
        import pandas as pd
        feats_df = pd.DataFrame([
            {
                "Feature":     k,
                "Value":       round(v, 4) if isinstance(v, float) else v,
                "SHAP":        round(result["shap_values"].get(k, 0), 5),
                "Direction":   "Phishing" if result["shap_values"].get(k, 0) > 0
                               else "Legitimate",
                "Description": FEATURE_DESCRIPTIONS.get(k, ""),
            }
            for k, v in result["features"].items()
        ]).sort_values("SHAP", key=abs, ascending=False)

        st.dataframe(
            feats_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "SHAP": st.column_config.NumberColumn(format="%.5f"),
            },
        )

    # ── Row 6: Analysis history ───────────────────────────────────────────────
    if len(st.session_state.get("history", [])) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Session History")
        for item in st.session_state["history"]:
            risk_col = {
                "HIGH": "#E84855", "MEDIUM": "#FF9F1C",
                "LOW":  "#8D99AE", "SAFE":   "#3A86FF",
            }.get(item["risk"], "#8D99AE")
            icon_h = "🚨" if item["verdict"] == "Phishing" else "✅"
            st.markdown(
                "<div class='history-item'>"
                "<span style='font-size:0.85rem;color:#2B2D42;'>"
                + icon_h + "  " + item["url"] + "</span>"
                "<span style='font-size:0.8rem;font-weight:700;"
                "color:" + risk_col + ";white-space:nowrap;margin-left:1rem;'>"
                + item["verdict"] + "  " + "{:.0f}%".format(item["prob"] * 100)
                + "</span></div>",
                unsafe_allow_html=True,
            )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
