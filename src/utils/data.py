"""Data loading utilities for the bank marketing dataset."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load_train_data() -> pd.DataFrame:
    """Load the training dataset with subscribe labels."""
    path = DATA_DIR / "train.csv"
    df = pd.read_csv(path)
    return df


def load_test_data() -> pd.DataFrame:
    """Load the test dataset without subscribe labels."""
    path = DATA_DIR / "test.csv"
    df = pd.read_csv(path)
    return df
