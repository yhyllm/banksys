"""Preprocessing pipeline for bank marketing data."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Categorical columns that need one-hot encoding
CAT_COLS = [
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

# Numeric columns that need scaling
NUM_COLS = [
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


def make_preprocessor() -> Pipeline:
    """Build the preprocessing Pipeline.

    Returns a fitted-ready Pipeline that handles:
    - Missing values (most_frequent for cat, median for numeric)
    - One-hot encoding for categorical features
    - Standard scaling for numeric features
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ],
        remainder="drop",
    )

    return Pipeline([("preprocessor", preprocessor)])


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    """Extract feature names from a fitted preprocessor."""
    ct = preprocessor.named_steps["preprocessor"]
    names = []
    for name, trans, cols in ct.transformers_:
        if trans == "drop":
            continue
        if hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
            ohe = trans.named_steps["onehot"]
            names.extend(ohe.get_feature_names_out(cols).tolist())
        elif hasattr(trans, "named_steps") and "scaler" in trans.named_steps:
            names.extend(cols)
    return names
