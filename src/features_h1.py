"""
Extended Feature Engineering Module for Stage H1.
Provides incremental, well-motivated domain feature groups:
- Group 1: Extended Financial Ratios & Cash Flow (4 features)
- Group 2: Temporal, Life Stage & Bureau Urgency (7 features)
- Group 3: External Score Interactions & Missing Indicators (5 features)
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.features import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    ID_COL,
    TARGET_COL,
    BaselineFeaturePipeline,
)


class ExtendedFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Incremental feature engineering pipeline building on top of BaselineFeaturePipeline.
    
    Parameters:
        include_group1_financial: Add extended financial & cash flow ratios (4 features)
        include_group2_temporal: Add temporal life-stage & bureau urgency features (7 features)
        include_group3_ext_interact: Add external score polynomials & indicators (5 features)
        encode_categories: If True, integer encodes categories (LGBM); if False, outputs strings (CatBoost)
    """

    def __init__(
        self,
        include_group1_financial: bool = False,
        include_group2_temporal: bool = False,
        include_group3_ext_interact: bool = False,
        encode_categories: bool = False,
    ):
        self.include_group1_financial = include_group1_financial
        self.include_group2_temporal = include_group2_temporal
        self.include_group3_ext_interact = include_group3_ext_interact
        self.encode_categories = encode_categories

        # Underlying 46-feature baseline transformer
        self.base_pipe = BaselineFeaturePipeline(
            add_domain_ratios=True,
            encode_categories=encode_categories,
        )
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits underlying transformers strictly on training fold."""
        self.base_pipe.fit(X, y)
        X_base = self.base_pipe.transform(X)
        X_extended = self._engineer_extended_features(X_base)
        self.feature_names = [c for c in X_extended.columns if c not in [ID_COL, TARGET_COL]]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms DataFrame applying base features and active candidate groups."""
        X_base = self.base_pipe.transform(X)
        X_extended = self._engineer_extended_features(X_base)
        return X_extended[self.feature_names].copy()

    def _engineer_extended_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # =====================================================================
        # GROUP 1: Extended Financial Ratios & Discretionary Cash Flow (4 features)
        # =====================================================================
        if self.include_group1_financial:
            # 1. Markup / Cash-out margin: difference between credit amount and goods price
            if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
                df["CREDIT_MINUS_GOODS"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
                df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1e-5)

            # 2. Net disposable income after debt annuity service
            if "AMT_INCOME_TOTAL" in df.columns and "AMT_ANNUITY" in df.columns:
                df["INCOME_MINUS_ANNUITY"] = df["AMT_INCOME_TOTAL"] - df["AMT_ANNUITY"]

            # 3. Approximate repayment term in payment periods
            if "AMT_CREDIT" in df.columns and "AMT_ANNUITY" in df.columns:
                df["CREDIT_TERM_APPROX"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1e-5)

        # =====================================================================
        # GROUP 2: Temporal, Life Stage & Bureau Urgency (7 features)
        # =====================================================================
        if self.include_group2_temporal:
            # 1. Borrower age in positive years
            if "DAYS_BIRTH" in df.columns:
                df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

            # 2. Employment duration in positive years
            if "DAYS_EMPLOYED" in df.columns:
                df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25

            # 3. Identity and registration stability ratios
            if "DAYS_ID_PUBLISH" in df.columns and "DAYS_BIRTH" in df.columns:
                df["ID_PUBLISH_TO_AGE_RATIO"] = df["DAYS_ID_PUBLISH"] / (df["DAYS_BIRTH"] + 1e-5)

            if "DAYS_REGISTRATION" in df.columns and "DAYS_BIRTH" in df.columns:
                df["REGISTRATION_TO_AGE_RATIO"] = df["DAYS_REGISTRATION"] / (df["DAYS_BIRTH"] + 1e-5)

            if "DAYS_LAST_PHONE_CHANGE" in df.columns and "DAYS_BIRTH" in df.columns:
                df["PHONE_TO_AGE_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"] / (df["DAYS_BIRTH"] + 1e-5)

            # 4. Total Bureau inquiries and short-term urgency ratio
            bureau_cols = [
                "AMT_REQ_CREDIT_BUREAU_HOUR",
                "AMT_REQ_CREDIT_BUREAU_MON",
                "AMT_REQ_CREDIT_BUREAU_QRT",
                "AMT_REQ_CREDIT_BUREAU_YEAR",
            ]
            if all(c in df.columns for c in bureau_cols):
                df["BUREAU_ENQUIRIES_TOTAL"] = df[bureau_cols].sum(axis=1)
                recent_enquiries = df["AMT_REQ_CREDIT_BUREAU_HOUR"] + df["AMT_REQ_CREDIT_BUREAU_MON"]
                df["BUREAU_URGENT_SHARE"] = recent_enquiries / (df["BUREAU_ENQUIRIES_TOTAL"] + 1e-5)

        # =====================================================================
        # GROUP 3: External Score Polynomials & Indicators (5 features)
        # =====================================================================
        if self.include_group3_ext_interact:
            # 1. Multiplicative interaction between available external sources
            if all(c in df.columns for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
                df["EXT_SOURCES_PROD"] = (
                    df["EXT_SOURCE_1"].fillna(1.0)
                    * df["EXT_SOURCE_2"].fillna(1.0)
                    * df["EXT_SOURCE_3"].fillna(1.0)
                )

                # 2. External source missing indicators (highly non-random missingness)
                df["EXT_SOURCE_1_ISNA"] = df["EXT_SOURCE_1"].isna().astype(int)
                df["EXT_SOURCE_3_ISNA"] = df["EXT_SOURCE_3"].isna().astype(int)

                # 3. Empirical weighted credit score composite
                df["EXT_WEIGHTED_SCORE"] = (
                    df["EXT_SOURCE_1"].fillna(0.5) * 1.0
                    + df["EXT_SOURCE_2"].fillna(0.5) * 3.0
                    + df["EXT_SOURCE_3"].fillna(0.5) * 5.0
                ) / 9.0

                # 4. Discrepancy between two primary credit bureau scores
                df["EXT_SOURCE_2_3_DIFF"] = df["EXT_SOURCE_2"] - df["EXT_SOURCE_3"]

        return df


if __name__ == "__main__":
    from src.data import load_raw_data

    train, test, _ = load_raw_data(verbose=False)
    p = ExtendedFeaturePipeline(
        include_group1_financial=True,
        include_group2_temporal=True,
        include_group3_ext_interact=True,
        encode_categories=False,
    )
    p.fit(train)
    res = p.transform(train)
    print(f"Extended Feature Pipeline verified: {len(p.feature_names)} features (Base: 46)")
    print(f"First 10 features: {p.feature_names[:10]}")
    print(f"New features sample: {p.feature_names[-16:]}")
