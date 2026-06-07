"""Prediction module for the bank marketing model.

Loads the trained pipeline from artifacts/model.pkl and provides
single-row and batch prediction with probabilities and feature importance.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"

CATEGORICAL_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

NUMERIC_COLS = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_index",
    "cons_conf_index",
    "lending_rate3m",
    "nr_employed",
]

FEATURE_NAMES = NUMERIC_COLS + CATEGORICAL_COLS


def load_model() -> object:
    """Load the trained pipeline from artifacts/model.pkl."""
    model_path = ARTIFACTS_DIR / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Please run: python src/ml/train.py"
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_metrics() -> dict:
    """Load training metrics from artifacts/metrics.json."""
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def predict_single(features: dict) -> dict:
    """Predict for a single customer.

    Args:
        features: Dict mapping feature name to value.

    Returns:
        Dict with keys: prediction ("yes"/"no"), probability, top_features.
    """
    model = load_model()
    metrics = load_metrics()
    threshold = metrics.get("threshold", 0.5)

    df = pd.DataFrame([features])
    proba = model.predict_proba(df)[0]
    prob_yes = float(proba[1])
    prediction = "yes" if prob_yes >= threshold else "no"
    probability = round(prob_yes, 4)

    top_features = _get_top_features(model, features)

    return {
        "prediction": prediction,
        "probability": probability,
        "top_features": top_features,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predict for multiple customers.

    Args:
        df: DataFrame with feature columns.

    Returns:
        DataFrame with added 'prediction' and 'probability' columns.
    """
    model = load_model()
    metrics = load_metrics()
    threshold = metrics.get("threshold", 0.5)

    probas = model.predict_proba(df)[:, 1]
    df = df.copy()
    df["probability"] = np.round(probas, 4)
    df["prediction"] = df["probability"].apply(lambda p: "yes" if p >= threshold else "no")
    return df


def _get_top_features(model, features: dict, n: int = 3) -> list[dict]:
    """Get top N features influencing the prediction."""
    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_

    preprocessor = model.named_steps["preprocessor"]
    cat_ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]

    feature_names = []
    values = []
    for col in NUMERIC_COLS:
        feature_names.append(col)
        values.append(features.get(col, 0))
    for col in CATEGORICAL_COLS:
        cats = cat_ohe.categories_[CATEGORICAL_COLS.index(col)]
        val = features.get(col, "")
        for cat in cats:
            feature_names.append(f"{col}_{cat}")
            values.append(1.0 if str(val) == cat else 0.0)

    paired = list(zip(feature_names, importances, values))
    paired.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {"feature": name, "importance": round(float(imp), 4), "value": val}
        for name, imp, val in paired[:n]
    ]
