"""
src/training/predict.py
"""

import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, ".")
import config
from src.features.extractor import extract, extract_batch

_MODEL        = None
_SCALER       = None
_FEATURE_COLS = None


def load_model():
    global _MODEL, _SCALER, _FEATURE_COLS
    if _MODEL is None:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {config.MODEL_PATH}. "
                "Run: python src/training/train.py"
            )
        _MODEL        = joblib.load(config.MODEL_PATH)
        _SCALER       = joblib.load(config.SCALER_PATH)
        _FEATURE_COLS = joblib.load(config.FEATURE_NAMES_PATH)
    return _MODEL, _SCALER, _FEATURE_COLS


def _risk_level(prob):
    if prob >= 0.85:
        return "HIGH"
    elif prob >= 0.60:
        return "MEDIUM"
    elif prob >= 0.40:
        return "LOW"
    else:
        return "SAFE"


def predict(url):
    model, scaler, feature_cols = load_model()
    feats  = extract(url)
    X_raw  = np.array([[feats[c] for c in feature_cols]], dtype=np.float32)
    X_sc   = scaler.transform(X_raw)
    proba  = model.predict_proba(X_sc)[0]
    label  = int(proba[1] >= config.DECISION_THRESHOLD)
    return {
        "url":        url,
        "label":      label,
        "verdict":    "Phishing" if label == 1 else "Legitimate",
        "phish_prob": round(float(proba[1]), 4),
        "legit_prob": round(float(proba[0]), 4),
        "risk_level": _risk_level(float(proba[1])),
        "features":   feats,
    }


def predict_batch(urls):
    model, scaler, feature_cols = load_model()
    df_feats = extract_batch(urls, show_progress=True)
    X_raw    = df_feats[feature_cols].values.astype(np.float32)
    X_sc     = scaler.transform(X_raw)
    probas   = model.predict_proba(X_sc)
    labels   = (probas[:, 1] >= config.DECISION_THRESHOLD).astype(int)
    return pd.DataFrame({
        "url":        urls,
        "label":      labels,
        "verdict":    ["Phishing" if l == 1 else "Legitimate" for l in labels],
        "phish_prob": probas[:, 1].round(4),
        "legit_prob": probas[:, 0].round(4),
        "risk_level": [_risk_level(p) for p in probas[:, 1]],
    })


if __name__ == "__main__":
    test_cases = [
        ("LEGIT",    "https://www.google.com/search?q=weather"),
        ("LEGIT",    "https://github.com/scikit-learn/scikit-learn"),
        ("LEGIT",    "https://www.linkedin.com/in/someprofile"),
        ("PHISHING", "http://192.168.1.1:8080/verify/login.html?tok=abc"),
        ("PHISHING", "http://paypal-secure.login.kqrt8ml.xyz/update/confirm"),
        ("PHISHING", "http://apple-id@evil-domain.tk/account/suspended"),
        ("PHISHING", "http://microsoft-account-verify.suspicious.ml/signin"),
    ]

    print()
    print("=" * 70)
    print("PhishGuard -- Prediction Quick Test")
    print("=" * 70)

    model, scaler, _ = load_model()
    print(f"Model loaded: {type(model).__name__}")
    print()

    all_correct = True
    for expected, url in test_cases:
        result = predict(url)
        correct = (expected == "PHISHING") == (result["label"] == 1)
        status  = "OK" if correct else "FAIL"
        if not correct:
            all_correct = False
        print(f"[{status}] [{expected:<8}] verdict={result['verdict']:<11} "
              f"prob={result['phish_prob']:.3f}  risk={result['risk_level']:<6}  "
              f"{url[:55]}")

    print()
    if all_correct:
        print("All predictions correct.")
    else:
        print("WARNING: Some predictions wrong -- check features or retrain.")
