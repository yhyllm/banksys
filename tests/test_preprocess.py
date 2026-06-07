"""Tests for src/preprocess.py — AC1 through AC5."""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    NUMERIC_COLS,
    TARGET_COL,
    _add_features,
    build_preprocessor,
    load_and_preprocess,
    load_data,
    preprocess_test,
    preprocess_train,
)


# ---- Helpers ---------------------------------------------------------------


@pytest.fixture
def sample_train_df():
    """A minimal train DataFrame for unit tests (not the real CSV)."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age": [30, 45, 60, 35],
            "job": ["admin.", "blue-collar", "admin.", "unknown"],
            "marital": ["married", "single", "married", "divorced"],
            "education": ["high.school", "university.degree", "basic.9y", "unknown"],
            "default": ["no", "no", "unknown", "no"],
            "housing": ["yes", "yes", "no", "no"],
            "loan": ["no", "no", "no", "yes"],
            "contact": ["cellular", "telephone", "cellular", "cellular"],
            "month": ["may", "jun", "jul", "aug"],
            "day_of_week": ["mon", "tue", "wed", "thu"],
            "duration": [120, 300, 45, 200],
            "campaign": [1, 3, 2, 1],
            "pdays": [999, 5, 100, 999],
            "previous": [0, 2, 1, 0],
            "poutcome": ["nonexistent", "success", "failure", "nonexistent"],
            "emp_var_rate": [-1.8, 1.4, -0.9, 0.5],
            "cons_price_index": [93.0, 94.5, 93.2, 94.1],
            "cons_conf_index": [-40.0, -42.5, -36.1, -41.0],
            "lending_rate3m": [1.5, 2.1, 1.8, 2.0],
            "nr_employed": [5000.0, 5100.0, 5050.0, 5075.0],
            "subscribe": ["no", "yes", "no", "yes"],
        }
    )


# ---- AC1: No missing values, correct shapes --------------------------------


def test_preprocess_train_returns_no_nan(sample_train_df):
    """AC1: X_train must have no missing values."""
    X, y, _ = preprocess_train(sample_train_df)
    assert not np.isnan(X).any(), "X_train contains NaN"
    assert X.shape[0] == len(sample_train_df)


def test_preprocess_train_target_is_binary(sample_train_df):
    """AC1: y_train must be 0/1 integers."""
    _, y, _ = preprocess_train(sample_train_df)
    assert set(np.unique(y)) <= {0, 1}, f"y contains unexpected values: {set(y)}"
    assert y.dtype in (np.int64, np.int32), f"y dtype is {y.dtype}, expected int"


def test_preprocess_id_is_dropped(sample_train_df):
    """ID column must not appear in output features."""
    X, _, preproc = preprocess_train(sample_train_df)
    feature_names = preproc.get_feature_names_out()
    for name in feature_names:
        assert "id" not in name.lower(), f"id still present in: {name}"


# ---- AC2: Categorical encoding ---------------------------------------------


def test_categorical_columns_are_onehot_encoded(sample_train_df):
    """AC2: Each categorical column produces multiple one-hot features."""
    _, _, preproc = preprocess_train(sample_train_df)
    feature_names = list(preproc.get_feature_names_out())

    # Each original category column should appear as cat__<col>_<value>
    cat_features = [f for f in feature_names if f.startswith("cat__")]
    assert len(cat_features) > 0, "No one-hot encoded features found"

    # Known categories should be present
    assert "cat__job_admin." in feature_names
    assert "cat__job_unknown" in feature_names  # unknown is preserved


def test_all_categorical_values_binary_after_encoding(sample_train_df):
    """AC2: One-hot encoded values must be 0 or 1 only."""
    X, _, preproc = preprocess_train(sample_train_df)
    feature_names = preproc.get_feature_names_out()
    cat_indices = [
        i for i, name in enumerate(feature_names) if name.startswith("cat__")
    ]
    cat_values = X[:, cat_indices]
    assert np.all((cat_values == 0) | (cat_values == 1)), (
        "One-hot columns contain non-binary values"
    )


# ---- AC3: Numeric standardization ------------------------------------------


def test_numeric_features_are_scaled(sample_train_df):
    """AC3: Numeric columns should be standardized (mean ~0, std ~1)."""
    X, _, preproc = preprocess_train(sample_train_df)
    feature_names = preproc.get_feature_names_out()
    num_indices = [
        i for i, name in enumerate(feature_names) if name.startswith("num__")
    ]
    assert len(num_indices) > 0, "No numeric features found"

    num_values = X[:, num_indices]
    means = num_values.mean(axis=0)
    stds = num_values.std(axis=0)

    # With StandardScaler, means should be close to 0 and stds close to 1
    assert np.allclose(means, 0, atol=1e-6), f"Numeric means not near 0: {means}"
    assert np.allclose(stds, 1, atol=1e-6), f"Numeric stds not near 1: {stds}"


def test_previously_contacted_feature_present(sample_train_df):
    """AC3: Engineered feature 'previously_contacted' must exist."""
    df = _add_features(sample_train_df)
    assert "previously_contacted" in df.columns
    # pdays == 999 should map to 0
    assert df.loc[0, "previously_contacted"] == 0  # pdays=999
    assert df.loc[1, "previously_contacted"] == 1  # pdays=5


def test_all_numeric_columns_present_in_preprocessor(sample_train_df):
    """Verify all NUMERIC_COLS + previously_contacted appear in output."""
    df = _add_features(sample_train_df.drop(columns=["subscribe", "id"]))
    preproc = build_preprocessor()
    preproc.fit(df)  # Must fit before calling get_feature_names_out
    feature_names = list(preproc.get_feature_names_out())
    num_names = [f for f in feature_names if f.startswith("num__")]
    expected = len(NUMERIC_COLS) + 1  # +1 for previously_contacted
    assert len(num_names) == expected, (
        f"Expected {expected} numeric features, got {len(num_names)}: {num_names}"
    )


# ---- AC4: Train/test consistency -------------------------------------------


def test_train_test_same_feature_count(sample_train_df):
    """AC4: Train and test must produce the same number of features."""
    X_train, _, preproc = preprocess_train(sample_train_df)

    # Build a test df with the same columns minus target
    test_df = sample_train_df.drop(columns=["subscribe"])
    X_test = preprocess_test(test_df, preproc)

    assert X_train.shape[1] == X_test.shape[1], (
        f"Train features {X_train.shape[1]} != test features {X_test.shape[1]}"
    )


def test_test_transform_does_not_refit(sample_train_df):
    """AC4: Calling preprocess_test must not alter the fitted preprocessor."""
    X_train1, _, preproc = preprocess_train(sample_train_df)

    test_df = sample_train_df.drop(columns=["subscribe"])
    X_test1 = preprocess_test(test_df, preproc)
    X_test2 = preprocess_test(test_df, preproc)

    # Same preprocessor, same input → identical output
    assert np.array_equal(X_test1, X_test2), "preprocess_test is not deterministic"


def test_test_handles_unknown_category():
    """AC4: Test data with unseen category should not crash (handle_unknown='ignore')."""
    train_df = pd.DataFrame(
        {
            "id": [1, 2],
            "age": [30, 45],
            "job": ["admin.", "blue-collar"],
            "marital": ["married", "single"],
            "education": ["high.school", "university.degree"],
            "default": ["no", "no"],
            "housing": ["yes", "yes"],
            "loan": ["no", "no"],
            "contact": ["cellular", "cellular"],
            "month": ["may", "jun"],
            "day_of_week": ["mon", "tue"],
            "duration": [120, 300],
            "campaign": [1, 3],
            "pdays": [999, 5],
            "previous": [0, 2],
            "poutcome": ["nonexistent", "success"],
            "emp_var_rate": [-1.8, 1.4],
            "cons_price_index": [93.0, 94.5],
            "cons_conf_index": [-40.0, -42.5],
            "lending_rate3m": [1.5, 2.1],
            "nr_employed": [5000.0, 5100.0],
            "subscribe": ["no", "yes"],
        }
    )

    X_train, _, preproc = preprocess_train(train_df)

    # Test with an unseen job category
    test_df = pd.DataFrame(
        {
            "id": [3],
            "age": [50],
            "job": ["astronaut"],  # Never seen in train
            "marital": ["married"],
            "education": ["high.school"],
            "default": ["no"],
            "housing": ["yes"],
            "loan": ["no"],
            "contact": ["cellular"],
            "month": ["may"],
            "day_of_week": ["mon"],
            "duration": [200],
            "campaign": [2],
            "pdays": [999],
            "previous": [0],
            "poutcome": ["nonexistent"],
            "emp_var_rate": [0.0],
            "cons_price_index": [93.5],
            "cons_conf_index": [-40.0],
            "lending_rate3m": [1.8],
            "nr_employed": [5050.0],
        }
    )

    X_test = preprocess_test(test_df, preproc)
    assert X_test.shape[0] == 1
    assert X_test.shape[1] == X_train.shape[1]
    assert not np.isnan(X_test).any()


# ---- AC5: Integration with real data ---------------------------------------


def test_full_pipeline_on_real_data():
    """AC5: End-to-end test with real CSV files."""
    X_train, y_train, preproc, X_test = load_and_preprocess(
        "data/train.csv", "data/test.csv"
    )

    # Shapes
    assert X_train.shape[0] == 22500
    assert X_test.shape[0] == 7500
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) == 22500

    # No NaN
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()

    # Target is binary
    assert set(np.unique(y_train)) == {0, 1}

    # Feature count is reasonable
    assert X_train.shape[1] >= 50, f"Too few features: {X_train.shape[1]}"
    assert X_train.shape[1] <= 100, f"Too many features: {X_train.shape[1]}"


def test_load_data():
    """load_data should return a DataFrame with expected columns."""
    df = load_data("data/train.csv")
    assert isinstance(df, pd.DataFrame)
    assert TARGET_COL in df.columns
    assert "id" in df.columns
    assert len(df) > 0


def test_previously_contacted_on_real_data():
    """pdays=999 rows must have previously_contacted=0."""
    train = load_data("data/train.csv")
    df = _add_features(train)
    never_contacted = df[df["pdays"] == 999]
    assert (never_contacted["previously_contacted"] == 0).all()
    contacted = df[df["pdays"] != 999]
    assert (contacted["previously_contacted"] == 1).all()
