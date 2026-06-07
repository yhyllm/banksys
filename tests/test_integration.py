"""Integration tests for train -> predict pipeline."""

import pickle
from pathlib import Path

import pytest

from src.ml.predict import predict_single
from src.ml.train import train

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


@pytest.fixture(scope="module")
def trained_pipeline():
    """Train a real pipeline for integration tests."""
    pipeline, metrics = train()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline, metrics


def _sample_features():
    return {
        "age": 35,
        "job": "admin.",
        "marital": "single",
        "education": "high.school",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 200,
        "campaign": 1,
        "pdays": 100,
        "previous": 0,
        "poutcome": "success",
        "emp_var_rate": 1.4,
        "cons_price_index": 93.0,
        "cons_conf_index": -40.0,
        "lending_rate3m": 3.5,
        "nr_employed": 5000.0,
    }


class TestTrainPredictIntegration:
    """End-to-end tests: train -> save -> load -> predict."""

    def test_train_produces_metrics(self, trained_pipeline):
        """Given training data, When training, Then metrics dict is returned."""
        _, metrics = trained_pipeline
        assert isinstance(metrics, dict)
        assert "val_auc" in metrics

    def test_train_auc_meets_threshold(self, trained_pipeline):
        """Given training data, When training, Then val_auc >= 0.85 on real data."""
        _, metrics = trained_pipeline
        # Only check AUC threshold when using real data (>= 1000 rows).
        # CI generates small random samples with meaningless AUC.
        from src.ml.train import load_data

        X, _ = load_data()
        if len(X) >= 1000:
            assert metrics["val_auc"] >= 0.85

    def test_predict_after_train_returns_valid(self, trained_pipeline):
        """Given a freshly trained model, When predicting, Then result is valid."""
        result = predict_single(_sample_features())
        assert result["prediction"] in ("yes", "no")
        assert 0.0 <= result["probability"] <= 1.0

    def test_predict_deterministic(self, trained_pipeline):
        """Given same model and input, When predicting twice, Then results are identical."""
        r1 = predict_single(_sample_features())
        r2 = predict_single(_sample_features())
        assert r1["prediction"] == r2["prediction"]
        assert r1["probability"] == r2["probability"]

    def test_model_file_exists_after_train(self, trained_pipeline):
        """Given training completed, Then model.pkl exists on disk."""
        model_path = ARTIFACTS_DIR / "model.pkl"
        assert model_path.exists()


class TestTrainMain:
    """Test the train.py main() entry point."""

    def test_train_main_runs(self, tmp_path, monkeypatch):
        """Given valid data, When running main(), Then it completes without error."""
        from src.ml.train import main

        # Redirect artifacts to tmp_path
        monkeypatch.setattr("src.ml.train.ARTIFACTS_DIR", tmp_path)
        main()  # should not raise
        assert (tmp_path / "model.pkl").exists()
        assert (tmp_path / "metrics.json").exists()
