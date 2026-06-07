"""Offline model training script.

Usage:
    python src/ml/train.py

Reads data/train.csv, trains a GradientBoosting classifier,
saves the full pipeline (preprocessor + model) to artifacts/model.pkl.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.ml.preprocess import make_preprocessor

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
RANDOM_SEED = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load training data and return features and target."""
    df = pd.read_csv(Path(__file__).resolve().parent.parent.parent / "data" / "train.csv")
    X = df.drop(columns=["id", "subscribe"])
    y = (df["subscribe"] == "yes").astype(int)
    return X, y


def train() -> tuple[Pipeline, dict]:
    """Train the full pipeline and return (pipeline, metrics)."""
    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = make_preprocessor()

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_SEED,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor.named_steps["preprocessor"]),
            ("classifier", model),
        ]
    )

    sample_weight = compute_sample_weight("balanced", y_train)
    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)

    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]

    cv_proba = cross_val_predict(pipeline, X_train, y_train, cv=3, method="predict_proba")[:, 1]

    # Compute optimal threshold (maximize F1 on validation set)
    thresholds = np.arange(0.1, 0.6, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    for t in thresholds:
        pred_t = (y_val_proba >= t).astype(int)
        f1_t = f1_score(y_val, pred_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = t

    y_val_pred_opt = (y_val_proba >= best_threshold).astype(int)

    metrics = {
        "val_accuracy": round(accuracy_score(y_val, y_val_pred_opt), 4),
        "val_f1": round(f1_score(y_val, y_val_pred_opt, zero_division=0), 4),
        "val_auc": round(roc_auc_score(y_val, y_val_proba), 4),
        "cv_auc": round(roc_auc_score(y_train, cv_proba), 4),
        "threshold": round(float(best_threshold), 2),
    }

    return pipeline, metrics


def main() -> None:
    """Entry point: train, evaluate, save."""
    print("Loading data...")
    pipeline, metrics = train()

    print("\n=== Model Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    meta_path = ARTIFACTS_DIR / "metrics.json"
    with open(meta_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {model_path}")
    print(f"Metrics saved to {meta_path}")

    if metrics["val_auc"] < 0.85:
        print(f"\nWARNING: val_auc {metrics['val_auc']} < 0.85 threshold")
    else:
        print(f"\nOK: val_auc {metrics['val_auc']} >= 0.85 threshold")


if __name__ == "__main__":
    main()
