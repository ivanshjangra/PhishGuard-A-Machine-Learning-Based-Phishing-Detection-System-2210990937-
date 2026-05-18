"""
src/training/train.py

PhishGuard Model Trainer
------------------------
Loads:   data/processed/features.csv
Trains:  Random Forest classifier (default)
Saves:   models/phishguard_rf.joblib
         models/scaler.joblib
         models/feature_names.joblib

Run with:
    python src/training/train.py
    python src/training/train.py --model rf
    python src/training/train.py --model xgb
    python src/training/train.py --model ensemble
"""

import sys
import argparse
import pathlib
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, ".")
import config
from src.features.extractor import feature_names as get_feature_names


def load_features(path):
    print(f"Loading features: {path}")
    df = pd.read_csv(path)
    feature_cols = get_feature_names()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if config.LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column not found.")
    X = df[feature_cols].values.astype(np.float32)
    y = df[config.LABEL_COLUMN].values.astype(np.int32)
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Class 0 (legitimate) : {(y == 0).sum()}")
    print(f"  Class 1 (phishing)   : {(y == 1).sum()}")
    return X, y, feature_cols


def build_model(model_type):
    if model_type == "rf":
        print("Building: Random Forest")
        return RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            min_samples_leaf=config.MIN_SAMPLES_LEAF,
            max_features="sqrt",
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_type == "xgb":
        from xgboost import XGBClassifier
        print("Building: XGBoost")
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_type == "ensemble":
        print("Building: Voting Ensemble (RF + XGB + SVM)")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        svm = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=config.RANDOM_STATE,
            )),
        ])
        return VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb), ("svm", svm)],
            voting="soft",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def cross_validate(model, X, y):
    print()
    print("-- 5-Fold Stratified Cross-Validation ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        print(f"  {metric:<12}  {scores.mean():.4f} +/- {scores.std():.4f}"
              f"  [{scores.min():.4f} - {scores.max():.4f}]")


def save_artifacts(model, scaler, feature_cols):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,        config.MODEL_PATH)
    joblib.dump(scaler,       config.SCALER_PATH)
    joblib.dump(feature_cols, config.FEATURE_NAMES_PATH)
    print()
    print("-- Saved artifacts --")
    for p in [config.MODEL_PATH, config.SCALER_PATH, config.FEATURE_NAMES_PATH]:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<30}  {size_kb:.1f} KB")


def train(model_type="rf"):
    print()
    print("=" * 60)
    print(f"PhishGuard Trainer  --  model={model_type.upper()}")
    print("=" * 60)
    print()

    X, y, feature_cols = load_features(config.PROCESSED_DATASET_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    print(f"Train size : {len(X_train)}  |  Test size : {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = build_model(model_type)
    cross_validate(model, X_train_scaled, y_train)

    print()
    print("-- Training final model --")
    t0 = time.time()
    model.fit(X_train_scaled, y_train)
    elapsed = time.time() - t0
    print(f"  Training time  : {elapsed:.2f}s")

    train_acc = model.score(X_train_scaled, y_train)
    test_acc  = model.score(X_test_scaled,  y_test)
    print(f"  Train accuracy : {train_acc:.4f}")
    print(f"  Test  accuracy : {test_acc:.4f}")

    save_artifacts(model, scaler, feature_cols)

    np.save(config.MODELS_DIR / "X_test.npy", X_test_scaled)
    np.save(config.MODELS_DIR / "y_test.npy", y_test)
    print("  X_test.npy, y_test.npy saved for evaluation.")

    print()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PhishGuard classifier")
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "xgb", "ensemble"],
        help="Model type to train (default: rf)",
    )
    args = parser.parse_args()
    train(model_type=args.model)
