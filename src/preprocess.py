"""Data preprocessing for the Bank Marketing dataset."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Column definitions (based on data/字段说明.md and data exploration)
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

TARGET_COL = "subscribe"
ID_COL = "id"


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features before encoding/scaling.

    - previously_contacted: 1 if pdays != 999 (was contacted before), 0 otherwise.
    """
    df = df.copy()
    df["previously_contacted"] = (df["pdays"] != 999).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer for the Bank Marketing dataset.

    Returns a preprocessor that:
    - One-hot encodes categorical features (treats 'unknown' as a category).
    - Standard-scales numeric features (including engineered ones).
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, drop=None
    )

    numeric_transformer = StandardScaler()

    # Numeric columns after feature engineering include the new flag
    all_numeric = NUMERIC_COLS + ["previously_contacted"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_COLS),
            ("num", numeric_transformer, all_numeric),
        ],
        remainder="drop",  # Drop any column not listed here
    )

    return preprocessor


def preprocess_train(
    train_df: pd.DataFrame,
) -> tuple:
    """Preprocess the training data.

    Args:
        train_df: Raw training DataFrame (must contain TARGET_COL).

    Returns:
        X_train: Feature matrix (numpy array).
        y_train: Target labels (numpy array, 0/1).
        preprocessor: Fitted ColumnTransformer (for reuse on test data).
    """
    df = train_df.copy()

    # Drop id column
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # Extract target and encode yes/no -> 1/0
    y = df[TARGET_COL].map({"yes": 1, "no": 0}).values
    df = df.drop(columns=[TARGET_COL])

    # Add engineered features
    df = _add_features(df)

    # Fit and transform
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df)

    return X, y, preprocessor


def preprocess_test(
    test_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> "np.ndarray":
    """Preprocess test data using a fitted preprocessor.

    Args:
        test_df: Raw test DataFrame (may or may not contain TARGET_COL).
        preprocessor: Fitted ColumnTransformer from preprocess_train.

    Returns:
        X_test: Feature matrix (numpy array).
    """
    df = test_df.copy()

    # Drop id column
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # Drop target if present (test data may not have it)
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # Add engineered features
    df = _add_features(df)

    # Transform only (do not fit)
    X = preprocessor.transform(df)

    return X


def load_and_preprocess(
    train_path: str,
    test_path: str | None = None,
) -> tuple:
    """Convenience function: load CSVs and return preprocessed data.

    Args:
        train_path: Path to train CSV.
        test_path: Path to test CSV (optional).

    Returns:
        X_train: Training feature matrix.
        y_train: Training labels.
        preprocessor: Fitted ColumnTransformer.
        X_test: Test feature matrix (if test_path provided, else None).
    """
    train_df = load_data(train_path)
    X_train, y_train, preprocessor = preprocess_train(train_df)

    if test_path:
        test_df = load_data(test_path)
        X_test = preprocess_test(test_df, preprocessor)
        return X_train, y_train, preprocessor, X_test

    return X_train, y_train, preprocessor, None
