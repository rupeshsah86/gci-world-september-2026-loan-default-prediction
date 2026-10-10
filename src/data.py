"""
Data loading and validation module for GCI World 2026 Loan Default Prediction.
Loads raw datasets from Competition/input/ with strict validation checks.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPETITION_INPUT_DIR = PROJECT_ROOT / "Competition" / "input"

TRAIN_PATH = COMPETITION_INPUT_DIR / "train.csv"
TEST_PATH = COMPETITION_INPUT_DIR / "test.csv"
SAMPLE_SUB_PATH = COMPETITION_INPUT_DIR / "sample_submission.csv"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"


def get_data_paths() -> Dict[str, Path]:
    """Returns a dictionary of raw data file paths."""
    return {
        "train": TRAIN_PATH,
        "test": TEST_PATH,
        "sample_submission": SAMPLE_SUB_PATH,
    }


def load_raw_data(verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train.csv, test.csv, and sample_submission.csv with integrity verification.

    Returns:
        train_df: pd.DataFrame of training samples (171,202 rows)
        test_df: pd.DataFrame of testing samples (61,500 rows)
        sample_sub: pd.DataFrame of sample submission template (61,500 rows)
    """
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found at: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test data not found at: {TEST_PATH}")
    if not SAMPLE_SUB_PATH.exists():
        raise FileNotFoundError(f"Sample submission not found at: {SAMPLE_SUB_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    # Integrity assertions
    assert ID_COL in train_df.columns, f"Missing ID column {ID_COL} in train"
    assert ID_COL in test_df.columns, f"Missing ID column {ID_COL} in test"
    assert ID_COL in sample_sub.columns, f"Missing ID column {ID_COL} in sample submission"
    assert TARGET_COL in train_df.columns, f"Missing target column {TARGET_COL} in train"
    assert TARGET_COL not in test_df.columns, f"Target column {TARGET_COL} must not be in test data"

    # ID overlap check
    train_ids = set(train_df[ID_COL])
    test_ids = set(test_df[ID_COL])
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"Critical error: {len(overlap)} overlapping IDs between train and test!"

    # Row order consistency check
    assert (sample_sub[ID_COL].values == test_df[ID_COL].values).all(), \
        "Sample submission IDs do not match test IDs in exact order"

    if verbose:
        print(f"[data.py] Successfully loaded raw datasets:")
        print(f"  - Train shape: {train_df.shape} (Default rate: {train_df[TARGET_COL].mean():.4%})")
        print(f"  - Test shape:  {test_df.shape}")
        print(f"  - Sample sub:  {sample_sub.shape}")

    return train_df, test_df, sample_sub


if __name__ == "__main__":
    train, test, sub = load_raw_data()
    print("Self-test passed successfully.")
