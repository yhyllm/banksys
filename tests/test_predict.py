"""Tests for prediction module."""

import pickle
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.ml.preprocess import make_preprocessor

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def _ensure_model():
    """Ensure a model.pkl exists for prediction tests by creating a dummy one if needed."""
    model_path = ARTIFACTS_DIR / "model.pkl"
    if not model_path.exists():
        preprocessor = make_preprocessor()
        dummy_model = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
        dummy = pd.DataFrame(
            {
                "age": [35] * 100 + [50] * 100,
                "job": ["admin."] * 100 + ["blue-collar"] * 100,
                "marital": ["single"] * 100 + ["married"] * 100,
                "education": ["high.school"] * 100 + ["basic.9y"] * 100,
                "default": ["no"] * 200,
                "housing": ["yes"] * 100 + ["no"] * 100,
                "loan": ["no"] * 200,
                "contact": ["cellular"] * 200,
                "month": ["may"] * 200,
                "day_of_week": ["mon"] * 200,
                "duration": [200] * 100 + [400] * 100,
                "campaign": [1] * 200,
                "pdays": [100] * 200,
                "previous": [0] * 200,
                "poutcome": ["success"] * 100 + ["nonexistent"] * 100,
                "emp_var_rate": [1.0] * 200,
                "cons_price_index": [93.0] * 200,
                "cons_conf_index": [-40.0] * 200,
                "lending_rate3m": [3.5] * 200,
                "nr_employed": [5000.0] * 200,
            }
        )
        y = [0] * 100 + [1] * 100
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor.named_steps["preprocessor"]),
                ("classifier", dummy_model),
            ]
        )
        pipe.fit(dummy, y)
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)
    yield model_path


def _sample_features():
    """Return a dict of features matching expected input schema."""
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


def test_predict_single_returns_dict(_ensure_model):
    """Given trained model, When calling predict_single, Then returns a dict."""
    from src.ml.predict import predict_single

    result = predict_single(_sample_features())
    assert isinstance(result, dict)


def test_predict_single_has_required_keys(_ensure_model):
    """Given model, When predicting, Then result has prediction, probability, top_features."""
    from src.ml.predict import predict_single

    result = predict_single(_sample_features())
    assert "prediction" in result
    assert "probability" in result
    assert "top_features" in result


def test_predict_single_prediction_is_yes_or_no(_ensure_model):
    """Given trained model, When calling predict_single, Then prediction is 'yes' or 'no'."""
    from src.ml.predict import predict_single

    result = predict_single(_sample_features())
    assert result["prediction"] in ("yes", "no")


def test_predict_single_probability_in_range(_ensure_model):
    """Given trained model, When calling predict_single, Then probability is in [0, 1]."""
    from src.ml.predict import predict_single

    result = predict_single(_sample_features())
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_single_top_features_list(_ensure_model):
    """Given trained model, When calling predict_single, Then top_features is a non-empty list."""
    from src.ml.predict import predict_single

    result = predict_single(_sample_features())
    assert isinstance(result["top_features"], list)
    assert len(result["top_features"]) > 0


def test_predict_batch_returns_dataframe(_ensure_model):
    """Given model and batch, When predicting, Then returns DataFrame with extra columns."""
    from src.ml.predict import predict_batch

    df = pd.DataFrame([_sample_features(), _sample_features()])
    result = predict_batch(df)
    assert isinstance(result, pd.DataFrame)
    assert "prediction" in result.columns
    assert "probability" in result.columns
    assert len(result) == 2


def test_predict_batch_row_count_matches(_ensure_model):
    """Given batch of N rows, When predicting, Then output has N rows."""
    from src.ml.predict import predict_batch

    df = pd.DataFrame([_sample_features()] * 5)
    result = predict_batch(df)
    assert len(result) == 5


def test_load_model_file_not_found(tmp_path, monkeypatch):
    """Given no model file, When calling load_model, Then raises FileNotFoundError."""
    from src.ml import predict as pred_module

    monkeypatch.setattr(pred_module, "ARTIFACTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        pred_module.load_model()


def test_load_metrics_returns_empty_when_missing(tmp_path, monkeypatch):
    """Given no metrics file, When calling load_metrics, Then returns empty dict."""
    from src.ml import predict as pred_module

    monkeypatch.setattr(pred_module, "ARTIFACTS_DIR", tmp_path)
    result = pred_module.load_metrics()
    assert result == {}
