"""Tests for preprocessing pipeline."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.ml.preprocess import make_preprocessor


@pytest.fixture
def sample_data():
    """Create a small sample DataFrame matching expected columns."""
    return pd.DataFrame(
        {
            "age": [35, 26, 44, 50],
            "job": ["admin.", "blue-collar", "services", "management"],
            "marital": ["single", "married", "divorced", "married"],
            "education": [
                "high.school",
                "basic.9y",
                "professional.course",
                "university.degree",
            ],
            "default": ["no", "unknown", "no", "yes"],
            "housing": ["yes", "no", "unknown", "yes"],
            "loan": ["no", "no", "unknown", "yes"],
            "contact": ["cellular", "telephone", "cellular", "cellular"],
            "month": ["aug", "may", "nov", "jul"],
            "day_of_week": ["mon", "thu", "fri", "tue"],
            "duration": [300, 200, 150, 400],
            "campaign": [1, 3, 2, 1],
            "pdays": [100, 200, 300, 400],
            "previous": [0, 1, 2, 0],
            "poutcome": ["success", "nonexistent", "failure", "success"],
            "emp_var_rate": [1.4, -1.8, 1.1, -0.1],
            "cons_price_index": [95.0, 91.0, 89.0, 97.0],
            "cons_conf_index": [-33.0, -44.0, -36.0, -37.0],
            "lending_rate3m": [3.6, 3.1, 5.0, 3.2],
            "nr_employed": [5200.0, 4900.0, 4950.0, 5100.0],
        }
    )


def test_make_preprocessor_returns_pipeline(sample_data):
    """Given valid config, When calling make_preprocessor, Then returns a Pipeline."""
    pipe = make_preprocessor()
    assert isinstance(pipe, Pipeline)


def test_preprocessor_transforms_without_error(sample_data):
    """Given valid data, When fitting and transforming, Then no exception."""
    pipe = make_preprocessor()
    X = pipe.fit_transform(sample_data)
    assert X is not None
    assert X.shape[0] == 4
    assert X.shape[1] > 0


def test_preprocessor_output_shape_matches(sample_data):
    """Given 4 rows, When transforming, Then output has 4 rows."""
    pipe = make_preprocessor()
    X = pipe.fit_transform(sample_data)
    assert X.shape[0] == len(sample_data)


def test_preprocessor_handles_unknown_categories():
    """Given unseen categorical, When transforming, Then no crash (handle_unknown=ignore)."""
    pipe = make_preprocessor()
    train = pd.DataFrame(
        {
            "age": [30, 40],
            "job": ["admin.", "blue-collar"],
            "marital": ["single", "married"],
            "education": ["high.school", "basic.9y"],
            "default": ["no", "yes"],
            "housing": ["yes", "no"],
            "loan": ["no", "yes"],
            "contact": ["cellular", "telephone"],
            "month": ["may", "jun"],
            "day_of_week": ["mon", "tue"],
            "duration": [200, 300],
            "campaign": [1, 2],
            "pdays": [100, 200],
            "previous": [0, 1],
            "poutcome": ["success", "nonexistent"],
            "emp_var_rate": [1.0, -1.0],
            "cons_price_index": [93.0, 94.0],
            "cons_conf_index": [-40.0, -35.0],
            "lending_rate3m": [3.0, 4.0],
            "nr_employed": [5000.0, 5100.0],
        }
    )
    test = train.copy()
    test.loc[0, "job"] = "unknown_job"  # unseen category
    test.loc[1, "education"] = "phd"  # unseen category

    pipe.fit(train)
    X = pipe.transform(test)
    assert X.shape[0] == 2


def test_get_feature_names_returns_list(sample_data):
    """Given fitted preprocessor, When calling get_feature_names, Then returns non-empty list."""
    from src.ml.preprocess import get_feature_names

    pipe = make_preprocessor()
    pipe.fit(sample_data)
    names = get_feature_names(pipe)
    assert isinstance(names, list)
    assert len(names) > 0
