"""
Feature engineering and preprocessing pipeline for GCI World 2026 Loan Default Prediction.
Includes safe handling of domain anomalies, missing values, and categorical encodings
with strict safeguards against data leakage.
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

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]

NUMERICAL_COLS = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_LAST_PHONE_CHANGE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]


class BaselineFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Leakage-free baseline preprocessing pipeline.
    
    Operations:
    1. Removes ID and TARGET from features.
    2. Handles DAYS_EMPLOYED == 365243 anomaly:
       - Adds DAYS_EMPLOYED_ANOM binary indicator (1 if sentinel, 0 otherwise).
       - Replaces 365243 with np.nan so tree splits are not distorted.
    3. Handles OWN_CAR_AGE outliers (flags age >= 60) and adds OWN_CAR_FLAG.
    4. Encodes categorical variables into integer category codes fitted only on training data.
    5. Preserves all numerical features with native NaN representation for LightGBM.
    """

    def __init__(self, add_domain_ratios: bool = False, encode_categories: bool = True):
        self.add_domain_ratios = add_domain_ratios
        self.encode_categories = encode_categories
        self.cat_mappings: Dict[str, Dict[str, int]] = {}
        self.feature_names: List[str] = []

    def _clean_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # DAYS_EMPLOYED anomaly
        if "DAYS_EMPLOYED" in df.columns:
            df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
            df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

        # OWN_CAR_AGE handling
        if "OWN_CAR_AGE" in df.columns:
            df["FLAG_OWN_CAR"] = (~df["OWN_CAR_AGE"].isna()).astype(int)
            df["OWN_CAR_AGE_OUTLIER"] = (df["OWN_CAR_AGE"] >= 60).astype(int)

        # CODE_GENDER anomaly: replace rare 'XNA' with NaN
        if "CODE_GENDER" in df.columns:
            df.loc[df["CODE_GENDER"] == "XNA", "CODE_GENDER"] = np.nan

        return df

    def _engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Financial credit burden ratios
        if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df["CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-5)
        
        if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df["ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1e-5)

        if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
            df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-5)

        if "AMT_GOODS_PRICE" in df.columns and "AMT_CREDIT" in df.columns:
            df["GOODS_PRICE_TO_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1e-5)

        # Demographics & life stage ratios
        if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
            df["DAYS_EMPLOYED_TO_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1e-5)

        if "AMT_INCOME_TOTAL" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
            df["INCOME_PER_FAMILY_MEMBER"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1e-5)

        # External source aggregations (strongest predictors in Home Credit)
        ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
        if ext_cols:
            df["EXT_SOURCES_MEAN"] = df[ext_cols].mean(axis=1)
            df["EXT_SOURCES_STD"] = df[ext_cols].std(axis=1)
            df["EXT_SOURCES_MIN"] = df[ext_cols].min(axis=1)
            df["EXT_SOURCES_MAX"] = df[ext_cols].max(axis=1)
            df["EXT_SOURCES_NAN_COUNT"] = df[ext_cols].isna().sum(axis=1)

        return df

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits categorical label encodings strictly on training fold."""
        X_clean = self._clean_anomalies(X)
        if self.add_domain_ratios:
            X_clean = self._engineer_domain_features(X_clean)

        self.cat_mappings = {}
        if self.encode_categories:
            for col in CATEGORICAL_COLS:
                if col in X_clean.columns:
                    # Find unique non-null categories in train
                    cats = X_clean[col].dropna().unique().tolist()
                    cats.sort()
                    # 0-indexed integer mapping; unseen categories will map to -1
                    self.cat_mappings[col] = {cat: idx for idx, cat in enumerate(cats)}

        # Determine feature names (exclude ID and TARGET)
        cols_to_drop = [c for c in [ID_COL, TARGET_COL] if c in X_clean.columns]
        self.feature_names = [c for c in X_clean.columns if c not in cols_to_drop]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame using fitted mappings or native categories."""
        X_out = self._clean_anomalies(X)
        if self.add_domain_ratios:
            X_out = self._engineer_domain_features(X_out)

        if self.encode_categories:
            # Apply integer label encodings for LightGBM
            for col, mapping in self.cat_mappings.items():
                if col in X_out.columns:
                    X_out[col] = X_out[col].map(mapping).fillna(-1).astype(int)
        else:
            # Keep string categories for CatBoost native categorical handling
            for col in CATEGORICAL_COLS:
                if col in X_out.columns:
                    X_out[col] = X_out[col].fillna("Missing").astype(str)

        # Keep only registered features
        X_out = X_out[self.feature_names].copy()
        return X_out


if __name__ == "__main__":
    from src.data import load_raw_data

    train, test, _ = load_raw_data(verbose=False)
    pipe = BaselineFeaturePipeline(add_domain_ratios=False)
    pipe.fit(train)
    X_tr = pipe.transform(train)
    X_te = pipe.transform(test)

    print("Baseline feature pipeline self-test passed:")
    print(f"  - Extracted features count: {len(pipe.feature_names)}")
    print(f"  - Train shape: {X_tr.shape}")
    print(f"  - Test shape:  {X_te.shape}")
    print(f"  - Features: {pipe.feature_names[:10]}...")
