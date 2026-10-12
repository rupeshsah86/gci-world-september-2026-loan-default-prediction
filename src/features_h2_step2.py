"""
Stage H2 Step 2: Advanced Feature Engineering Pipeline.
Implements:
1. Leak-Free Out-of-Fold Smoothed Target Encoding for high-cardinality categoricals
   (ORGANIZATION_TYPE, OCCUPATION_TYPE) using inner K-fold splitting.
2. High-leverage non-linear cross-interactions combining top gain drivers
   (PAYMENT_RATE, CREDIT_TO_INCOME_RATIO, EXT_SOURCES_MEAN).
3. Frequency / Prevalence Encodings fitted strictly on training partitions.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin

from src.features import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    ID_COL,
    TARGET_COL,
    BaselineFeaturePipeline,
)


def compute_inner_oof_target_encoding(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    cat_cols: List[str],
    n_inner_splits: int = 5,
    smoothing_weight: float = 20.0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes strictly leak-free Out-of-Fold Smoothed Target Encoding.
    
    Training fold encodings are generated via an internal 5-fold cross-split:
    each row's target is excluded from its own encoding.
    Validation fold encodings are derived using the full training fold statistics.
    
    Formula:
        S = (n * mean_cat + m * mean_global) / (n + m)
    where:
        n = category count in inner-training fold
        m = smoothing_weight (Bayesian prior strength)
    """
    X_tr_out = X_train.copy()
    X_va_out = X_val.copy()

    global_mean = float(np.mean(y_train))
    inner_kf = KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)

    for col in cat_cols:
        te_col_name = f"TE_{col}"
        X_tr_out[te_col_name] = np.nan
        X_va_out[te_col_name] = np.nan

        # 1. Inner OOF encoding for training rows
        for in_train_idx, in_val_idx in inner_kf.split(X_train):
            in_X_tr_col = X_train.iloc[in_train_idx][col].values
            in_y_tr = y_train[in_train_idx]
            in_X_val_col = X_train.iloc[in_val_idx][col].values

            in_global_mean = float(np.mean(in_y_tr))

            # Fast vectorized aggregation
            tmp_df = pd.DataFrame({"cat": in_X_tr_col, "target": in_y_tr})
            stats = tmp_df.groupby("cat")["target"].agg(["count", "sum"])

            # Smoothed encoding map: (sum + m * prior) / (count + m)
            counts = stats["count"]
            sums = stats["sum"]
            smoothed = (sums + smoothing_weight * in_global_mean) / (counts + smoothing_weight)
            encoding_map = smoothed.to_dict()

            # Map inner validation fold; unseen categories get in_global_mean
            vals = pd.Series(in_X_val_col).map(encoding_map).fillna(in_global_mean).values
            X_tr_out.iloc[in_val_idx, X_tr_out.columns.get_loc(te_col_name)] = vals

        # 2. Outer validation encoding using full training fold statistics
        full_tmp = pd.DataFrame({"cat": X_train[col].values, "target": y_train})
        full_stats = full_tmp.groupby("cat")["target"].agg(["count", "sum"])
        full_smoothed = (full_stats["sum"] + smoothing_weight * global_mean) / (
            full_stats["count"] + smoothing_weight
        )
        full_encoding_map = full_smoothed.to_dict()

        X_va_out[te_col_name] = X_val[col].map(full_encoding_map).fillna(global_mean).values

    return X_tr_out, X_va_out


class FeaturePipelineH2(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering pipeline supporting:
    - Base 46 domain features
    - Non-linear cross-interactions (Top gain interactions)
    - Frequency / prevalence encodings
    - Clean category typing for LightGBM or CatBoost
    """

    def __init__(
        self,
        include_cross_interactions: bool = False,
        include_frequency_encodings: bool = False,
        encode_categories: bool = True,
    ):
        self.include_cross_interactions = include_cross_interactions
        self.include_frequency_encodings = include_frequency_encodings
        self.encode_categories = encode_categories

        self.base_pipe = BaselineFeaturePipeline(
            add_domain_ratios=True,
            encode_categories=encode_categories,
        )
        self.freq_maps: Dict[str, Dict[str, float]] = {}
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits base transformers and frequency maps strictly on training data."""
        self.base_pipe.fit(X, y)
        X_base = self.base_pipe.transform(X)

        # Learn frequency encodings strictly from training partition
        if self.include_frequency_encodings:
            self.freq_maps = {}
            for col in ["ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE"]:
                if col in X.columns:
                    freqs = X[col].value_counts(normalize=True).to_dict()
                    self.freq_maps[col] = freqs

        X_eng = self._add_engineered_features(X_base, X)
        self.feature_names = [c for c in X_eng.columns if c not in [ID_COL, TARGET_COL]]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame using fitted parameters."""
        X_base = self.base_pipe.transform(X)
        X_eng = self._add_engineered_features(X_base, X)
        return X_eng[self.feature_names].copy()

    def _add_engineered_features(self, df: pd.DataFrame, X_raw: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Non-linear interactions between top gain drivers
        if self.include_cross_interactions:
            # Multiplicative debt-burden x external score
            if "PAYMENT_RATE" in df.columns and "EXT_SOURCES_MEAN" in df.columns:
                df["PAYMENT_RATE_X_EXT_MEAN"] = df["PAYMENT_RATE"] * df["EXT_SOURCES_MEAN"].fillna(0.5)

            if "CREDIT_TO_INCOME_RATIO" in df.columns and "EXT_SOURCES_MEAN" in df.columns:
                df["CREDIT_INCOME_TO_EXT_MEAN"] = df["CREDIT_TO_INCOME_RATIO"] / (
                    df["EXT_SOURCES_MEAN"].fillna(0.5) + 1e-4
                )

            if "ANNUITY_TO_INCOME_RATIO" in df.columns and "EXT_SOURCES_MEAN" in df.columns:
                df["ANNUITY_INCOME_TO_EXT_MEAN"] = df["ANNUITY_TO_INCOME_RATIO"] / (
                    df["EXT_SOURCES_MEAN"].fillna(0.5) + 1e-4
                )

            # Ratio of min external source to max external source (discrepancy term)
            if "EXT_SOURCES_MIN" in df.columns and "EXT_SOURCES_MAX" in df.columns:
                df["EXT_MIN_TO_MAX_RATIO"] = df["EXT_SOURCES_MIN"] / (df["EXT_SOURCES_MAX"] + 1e-5)

            # Age calibrated loan burden
            if "DAYS_BIRTH" in df.columns and "PAYMENT_RATE" in df.columns:
                # -DAYS_BIRTH is positive age in days
                age_years = -df["DAYS_BIRTH"] / 365.25
                df["PAYMENT_RATE_X_AGE"] = df["PAYMENT_RATE"] * age_years

        # 2. Frequency encodings
        if self.include_frequency_encodings:
            for col, f_map in self.freq_maps.items():
                if col in X_raw.columns:
                    df[f"FREQ_{col}"] = X_raw[col].map(f_map).fillna(0.0).values

        return df


if __name__ == "__main__":
    from src.data import load_raw_data

    train, test, _ = load_raw_data(verbose=False)
    p = FeaturePipelineH2(
        include_cross_interactions=True,
        include_frequency_encodings=True,
        encode_categories=True,
    )
    p.fit(train)
    out = p.transform(train)
    print(f"Self-test FeaturePipelineH2 passed: {len(p.feature_names)} features extracted (Base: 46)")
    print(f"New features: {[c for c in p.feature_names if c not in p.base_pipe.feature_names]}")
