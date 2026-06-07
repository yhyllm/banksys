"""Tests for src/train.py — AC1 through AC5."""

import json
from unittest.mock import patch

import numpy as np

from src.train import (
    CLASSIFIERS,
    load_model,
    predict,
    predict_proba,
    print_report,
    train_and_evaluate,
)


# ---- Helpers ---------------------------------------------------------------


def _make_dummy_data(n_samples=100, n_features=64, random_state=42):
    """Generate synthetic feature matrix and binary labels."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    y = (rng.rand(n_samples) > 0.7).astype(int)  # ~30% positive
    return X, y


# ---- AC1: Model saved to models/ -------------------------------------------


def test_best_model_file_created(tmp_path):
    """AC1: train_and_evaluate must save a model file to models/."""
    # Redirect output to a temp directory to avoid overwriting real model.
    model_file = tmp_path / "best_model.joblib"
    metrics_file = tmp_path / "metrics.json"

    with (
        patch("src.train.BEST_MODEL_PATH", model_file),
        patch("src.train.METRICS_PATH", metrics_file),
        patch("src.train.MODEL_DIR", tmp_path),
    ):
        metrics = train_and_evaluate()

        assert model_file.exists(), "Model file was not created"
        assert model_file.stat().st_size > 0, "Model file is empty"

        # Model path returned correctly
        assert metrics["best_model_path"] == str(model_file)


def test_metrics_file_created(tmp_path):
    """AC1: Metrics JSON must be saved."""
    metrics_file = tmp_path / "metrics.json"
    model_file = tmp_path / "best_model.joblib"

    with (
        patch("src.train.BEST_MODEL_PATH", model_file),
        patch("src.train.METRICS_PATH", metrics_file),
        patch("src.train.MODEL_DIR", tmp_path),
    ):
        train_and_evaluate()

        assert metrics_file.exists(), "metrics.json was not created"
        data = json.loads(metrics_file.read_text())
        assert "best_model" in data
        assert "best_accuracy" in data
        assert "results" in data


# ---- AC2: Accuracy >= 0.85 -------------------------------------------------


def test_best_accuracy_above_threshold():
    """AC2: Best model accuracy on real data must be >= 0.85."""
    metrics = train_and_evaluate()
    assert metrics["best_accuracy"] >= 0.85, (
        f"Accuracy {metrics['best_accuracy']:.4f} below 0.85 threshold"
    )


# ---- AC3: Classification report --------------------------------------------


def test_results_contain_all_metrics():
    """AC3: Results must include accuracy, precision, recall, F1, confusion matrix."""
    metrics = train_and_evaluate()

    for name, m in metrics["results"].items():
        assert "accuracy" in m, f"{name}: missing accuracy"
        assert "precision" in m, f"{name}: missing precision"
        assert "recall" in m, f"{name}: missing recall"
        assert "f1_score" in m, f"{name}: missing f1_score"
        assert "confusion_matrix" in m, f"{name}: missing confusion_matrix"

        # Confusion matrix must be 2x2
        cm = m["confusion_matrix"]
        assert len(cm) == 2, f"{name}: confusion matrix not 2 rows"
        assert len(cm[0]) == 2, f"{name}: confusion matrix not 2 columns"

        # All metrics between 0 and 1
        for key in ["accuracy", "precision", "recall", "f1_score"]:
            assert 0.0 <= m[key] <= 1.0, f"{name}: {key} out of range: {m[key]}"


def test_print_report_runs(capsys):
    """AC3: print_report must produce human-readable output without error."""
    metrics = train_and_evaluate()
    print_report(metrics)
    captured = capsys.readouterr()
    assert "Best model" in captured.out
    assert "Accuracy" in captured.out
    assert "Confusion Matrix" in captured.out


# ---- AC4: At least two classifiers -----------------------------------------


def test_at_least_two_classifiers_trained():
    """AC4: Must train at least two classifiers and pick the best."""
    assert len(CLASSIFIERS) >= 2, "Need at least 2 classifiers"

    metrics = train_and_evaluate()
    trained = set(metrics["results"].keys())
    expected = set(CLASSIFIERS.keys())
    assert trained == expected, f"Trained {trained}, expected {expected}"

    # Best model must be one of the trained ones
    assert metrics["best_model_name"] in trained


# ---- AC5: Model load / predict / output format -----------------------------


def test_load_model_works():
    """AC5: load_model must return a dict with 'model' and 'preprocessor'."""
    # Real model was saved by test_best_accuracy_above_threshold
    model_data = load_model()
    assert "model" in model_data, "Loaded model data missing 'model' key"
    assert "preprocessor" in model_data, "Loaded model data missing 'preprocessor' key"


def test_predict_returns_binary():
    """AC5: predict must return 0/1 labels."""
    model_data = load_model()
    # Use real preprocessed data for shape-compatible input
    from src.preprocess import load_and_preprocess

    X, _, _, _ = load_and_preprocess("data/train.csv")
    X_small = X[:10]

    preds = predict(model_data, X_small)
    assert preds.shape == (10,), f"Expected shape (10,), got {preds.shape}"
    assert set(np.unique(preds)) <= {0, 1}, f"Unexpected labels: {set(preds)}"


def test_predict_proba_shape():
    """AC5: predict_proba must return (n_samples, 2) with valid probabilities."""
    model_data = load_model()
    from src.preprocess import load_and_preprocess

    X, _, _, _ = load_and_preprocess("data/train.csv")
    X_small = X[:5]

    proba = predict_proba(model_data, X_small)
    assert proba.shape == (5, 2), f"Expected shape (5, 2), got {proba.shape}"
    assert np.all(proba >= 0), "Probabilities must be >= 0"
    assert np.all(proba <= 1), "Probabilities must be <= 1"
    # Each row sums to 1
    assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities must sum to 1 per row"


def test_predict_consistency():
    """AC5: predict class should match argmax of predict_proba."""
    model_data = load_model()
    from src.preprocess import load_and_preprocess

    X, _, _, _ = load_and_preprocess("data/train.csv")
    X_small = X[:20]

    preds = predict(model_data, X_small)
    proba = predict_proba(model_data, X_small)
    expected = np.argmax(proba, axis=1)

    assert np.array_equal(preds, expected), "predict != argmax(predict_proba)"


def test_model_is_reproducible():
    """AC5: Same random_state must produce same accuracy."""
    m1 = train_and_evaluate(random_state=42)
    m2 = train_and_evaluate(random_state=42)
    assert m1["best_accuracy"] == m2["best_accuracy"], (
        "Same seed produced different accuracy"
    )
