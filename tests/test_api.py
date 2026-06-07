"""Tests for src/api.py — AC1 through AC5."""

import pytest
from fastapi.testclient import TestClient

from src.api import app


# ---- Helpers ---------------------------------------------------------------


@pytest.fixture
def client():
    """Create a TestClient with lifespan (model loaded on startup)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_customer():
    """Return a valid customer payload."""
    return {
        "age": 50,
        "job": "services",
        "marital": "married",
        "education": "high.school",
        "default": "unknown",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 4715,
        "campaign": 1,
        "pdays": 412,
        "previous": 2,
        "poutcome": "nonexistent",
        "emp_var_rate": -1.8,
        "cons_price_index": 96.33,
        "cons_conf_index": -40.58,
        "lending_rate3m": 4.05,
        "nr_employed": 4974.79,
    }


# ---- AC3: Health check -----------------------------------------------------


class TestHealth:
    """AC3: GET /health returns status ok with HTTP 200."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_status_ok(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data == {"status": "ok"}


# ---- AC1: Single prediction ------------------------------------------------


class TestPredictSingle:
    """AC1: POST /predict returns {subscribe, probability}."""

    def test_predict_returns_subscribe_and_probability(self, client, valid_customer):
        resp = client.post("/predict", json=valid_customer)
        assert resp.status_code == 200
        data = resp.json()
        assert "subscribe" in data
        assert "probability" in data

    def test_subscribe_is_binary(self, client, valid_customer):
        resp = client.post("/predict", json=valid_customer)
        data = resp.json()
        assert data["subscribe"] in (0, 1), f"subscribe={data['subscribe']}"

    def test_probability_between_0_and_1(self, client, valid_customer):
        resp = client.post("/predict", json=valid_customer)
        data = resp.json()
        assert 0.0 <= data["probability"] <= 1.0, (
            f"probability={data['probability']} out of range"
        )

    def test_different_inputs_may_give_different_results(self, client):
        """Ensure prediction varies with input (not a constant)."""
        young = {
            "age": 25,
            "job": "student",
            "marital": "single",
            "education": "basic.9y",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "month": "aug",
            "day_of_week": "fri",
            "duration": 60,
            "campaign": 1,
            "pdays": 999,
            "previous": 0,
            "poutcome": "nonexistent",
            "emp_var_rate": 1.4,
            "cons_price_index": 94.0,
            "cons_conf_index": -36.0,
            "lending_rate3m": 2.0,
            "nr_employed": 5000.0,
        }
        old = {
            "age": 60,
            "job": "retired",
            "marital": "divorced",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "telephone",
            "month": "dec",
            "day_of_week": "thu",
            "duration": 900,
            "campaign": 3,
            "pdays": 30,
            "previous": 5,
            "poutcome": "success",
            "emp_var_rate": -2.5,
            "cons_price_index": 92.0,
            "cons_conf_index": -45.0,
            "lending_rate3m": 1.5,
            "nr_employed": 5100.0,
        }
        r1 = client.post("/predict", json=young).json()
        r2 = client.post("/predict", json=old).json()
        # At least one of subscribe/probability should differ
        assert (r1["subscribe"] != r2["subscribe"]) or (
            abs(r1["probability"] - r2["probability"]) > 0.001
        ), "All results identical — model may not be distinguishing inputs"


# ---- AC2: Batch prediction -------------------------------------------------


class TestPredictBatch:
    """AC2: POST /predict/batch returns a list of predictions."""

    def test_batch_returns_correct_count(self, client, valid_customer):
        resp = client.post(
            "/predict/batch",
            json={"customers": [valid_customer, valid_customer, valid_customer]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 3

    def test_batch_single_record_works(self, client, valid_customer):
        resp = client.post("/predict/batch", json={"customers": [valid_customer]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 1
        assert "subscribe" in data["predictions"][0]

    def test_batch_empty_list_returns_422(self, client):
        resp = client.post("/predict/batch", json={"customers": []})
        assert resp.status_code == 422


# ---- AC4: Input validation -------------------------------------------------


class TestInputValidation:
    """AC4: Missing fields / wrong types → 422 with clear error."""

    def test_missing_required_field(self, client):
        incomplete = {"age": 30}  # Only one field
        resp = client.post("/predict", json=incomplete)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert len(detail) > 0, "Expected validation errors"

    def test_wrong_type_string_instead_of_int(self, client, valid_customer):
        valid_customer["age"] = "fifty"
        resp = client.post("/predict", json=valid_customer)
        assert resp.status_code == 422

    def test_negative_age(self, client, valid_customer):
        valid_customer["age"] = -5
        resp = client.post("/predict", json=valid_customer)
        assert resp.status_code == 422

    def test_age_out_of_range(self, client, valid_customer):
        valid_customer["age"] = 150
        resp = client.post("/predict", json=valid_customer)
        assert resp.status_code == 422

    def test_negative_duration(self, client, valid_customer):
        valid_customer["duration"] = -1
        resp = client.post("/predict", json=valid_customer)
        assert resp.status_code == 422

    def test_empty_body(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_batch_too_large(self, client, valid_customer):
        """Batch size must be <= 1000."""
        resp = client.post(
            "/predict/batch",
            json={"customers": [valid_customer] * 1001},
        )
        assert resp.status_code == 422


# ---- AC5: OpenAPI docs -----------------------------------------------------


class TestOpenAPI:
    """AC5: FastAPI auto-generates OpenAPI docs."""

    def test_openapi_schema_accessible(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "Bank Marketing Prediction API"

    def test_docs_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
