"""Model training and evaluation for Bank Marketing prediction."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.preprocess import load_and_preprocess

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

CLASSIFIERS = {
    "logistic_regression": LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=42
    ),
    "random_forest": RandomForestClassifier(
        class_weight="balanced", n_estimators=100, random_state=42
    ),
}


def train_and_evaluate(
    train_path: str = "data/train.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train multiple classifiers, evaluate on a validation split, and save the best.

    Args:
        train_path: Path to training CSV.
        test_size: Fraction of training data to hold out for validation.
        random_state: Random seed for reproducibility.

    Returns:
        dict with keys:
            - best_model_name: name of the best classifier
            - best_accuracy: accuracy of the best model on validation set
            - results: dict of per-model metrics
            - best_model_path: path to saved model file
    """
    # Load and preprocess
    X, y, preprocessor, _ = load_and_preprocess(train_path)

    # Split into train/validation (test.csv has no labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results = {}
    best_name = None
    best_accuracy = -1.0

    for name, clf in CLASSIFIERS.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        acc = float(accuracy_score(y_val, y_pred))
        prec = float(precision_score(y_val, y_pred, zero_division=0))
        rec = float(recall_score(y_val, y_pred, zero_division=0))
        f1 = float(f1_score(y_val, y_pred, zero_division=0))
        cm = confusion_matrix(y_val, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
        }

        if acc > best_accuracy:
            best_accuracy = acc
            best_name = name

    # Save the best model
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_clf = CLASSIFIERS[best_name]
    best_clf.fit(X, y)  # Re-fit on full training data
    joblib.dump(
        {"model": best_clf, "preprocessor": preprocessor},
        BEST_MODEL_PATH,
    )

    # Save metrics
    metrics = {
        "best_model": best_name,
        "best_accuracy": round(best_accuracy, 4),
        "results": results,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return {
        "best_model_name": best_name,
        "best_accuracy": round(best_accuracy, 4),
        "results": results,
        "best_model_path": str(BEST_MODEL_PATH),
    }


def load_model(model_path: str | None = None):
    """Load a saved model and its preprocessor.

    Args:
        model_path: Path to model file. Defaults to BEST_MODEL_PATH.

    Returns:
        dict with keys 'model' and 'preprocessor'.
    """
    path = model_path or str(BEST_MODEL_PATH)
    return joblib.load(path)


def predict(model_data: dict, X: np.ndarray) -> np.ndarray:
    """Predict class labels for the given features.

    Args:
        model_data: Dict from load_model() with 'model' key.
        X: Feature matrix.

    Returns:
        Array of predicted class labels (0 or 1).
    """
    return model_data["model"].predict(X)


def predict_proba(model_data: dict, X: np.ndarray) -> np.ndarray:
    """Predict class probabilities for the given features.

    Args:
        model_data: Dict from load_model() with 'model' key.
        X: Feature matrix.

    Returns:
        Array of shape (n_samples, 2) with [P(class=0), P(class=1)].
    """
    return model_data["model"].predict_proba(X)


def print_report(metrics: dict) -> None:
    """Print a human-readable classification report."""
    print("=" * 60)
    print("Model Training Report")
    print("=" * 60)
    print(f"Best model: {metrics['best_model_name']}")
    print(f"Best validation accuracy: {metrics['best_accuracy']:.4f}")
    print()

    for name, m in metrics["results"].items():
        marker = " ★ BEST" if name == metrics["best_model_name"] else ""
        print(f"--- {name}{marker} ---")
        print(f"  Accuracy :  {m['accuracy']:.4f}")
        print(f"  Precision:  {m['precision']:.4f}")
        print(f"  Recall   :  {m['recall']:.4f}")
        print(f"  F1 Score :  {m['f1_score']:.4f}")
        print("  Confusion Matrix:")
        print(
            f"    TN={m['confusion_matrix'][0][0]:5d}  FP={m['confusion_matrix'][0][1]:5d}"
        )
        print(
            f"    FN={m['confusion_matrix'][1][0]:5d}  TP={m['confusion_matrix'][1][1]:5d}"
        )
        print()


# ---- CLI entry point ----
if __name__ == "__main__":
    metrics = train_and_evaluate()
    print_report(metrics)
