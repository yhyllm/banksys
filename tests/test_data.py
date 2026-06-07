"""Tests for data loading utilities."""

import pandas as pd

from src.utils.data import load_test_data, load_train_data


def test_load_train_data_returns_dataframe():
    """Given train.csv exists, When calling load_train_data, Then returns a DataFrame."""
    df = load_train_data()
    assert isinstance(df, pd.DataFrame)


def test_load_train_data_shape():
    """Given train.csv exists, When loading, Then has expected number of rows."""
    df = load_train_data()
    # Real data: 22500, CI sample: 500; check it has at least 100 rows
    assert len(df) >= 100


def test_load_train_data_has_subscribe():
    """Given train.csv, When loading, Then subscribe column exists."""
    df = load_train_data()
    assert "subscribe" in df.columns
    assert set(df["subscribe"].unique()).issubset({"yes", "no"})


def test_load_train_data_column_count():
    """Given train.csv, When loading, Then has 22 columns (id + 20 features + subscribe)."""
    df = load_train_data()
    assert len(df.columns) == 22


def test_load_test_data_returns_dataframe():
    """Given test.csv exists, When calling load_test_data, Then returns a DataFrame."""
    df = load_test_data()
    assert isinstance(df, pd.DataFrame)


def test_load_test_data_shape():
    """Given test.csv exists, When loading, Then has expected number of rows."""
    df = load_test_data()
    # Real data: 7500, CI sample: 100; check it has at least 50 rows
    assert len(df) >= 50


def test_load_test_data_no_subscribe():
    """Given test.csv has no labels, When loading, Then subscribe column does not exist."""
    df = load_test_data()
    assert "subscribe" not in df.columns


def test_load_test_data_column_count():
    """Given test.csv, When loading, Then has 21 columns (id + 20 features)."""
    df = load_test_data()
    assert len(df.columns) == 21
