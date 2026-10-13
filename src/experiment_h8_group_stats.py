"""
Stage H8: Experiment C - Target-Free Demographic Group Aggregations Screening.
Evaluates group mean deviations strictly fitted on training slice against Fold 1 baselines.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS

def engineer_group_features(X_tr_in, X_va_in):
    X_tr = X_tr_in.copy()
    X_va = X_va_in.copy()

    group_specs = [
        ("EXT_SOURCE_2", "NAME_EDUCATION_TYPE"),
        ("EXT_SOURCE_3", "OCCUPATION_TYPE"),
        ("PAYMENT_RATE", "NAME_INCOME_TYPE"),
        ("CREDIT_TO_INCOME_RATIO", "NAME_EDUCATION_TYPE"),
        ("DAYS_BIRTH", "NAME_INCOME_TYPE"),
    ]

    for num_col, cat_col in group_specs:
        if num_col in X_tr.columns and cat_col in X_tr.columns:
            # Fit group means strictly on training fold
            group_means = X_tr.groupby(cat_col)[num_col].mean().to_dict()
            global_mean = X_tr[num_col].mean()

            feature_name = f"{num_col}_DEV_BY_{cat_col}"
            X_tr[feature_name] = X_tr[num_col] - X_tr[cat_col].map(group_means).fillna(global_mean)
            X_va[feature_name] = X_va[num_col] - X_va[cat_col].map(group_means).fillna(global_mean)

    return X_tr, X_va

def run_group_stats_screening():
    print("=" * 80)
    print("STAGE H8: DEMOGRAPHIC GROUP AGGREGATION SCREENING (FOLD 1)")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(skf.split(X_raw, y_true))
    y_tr, y_va = y_true[tr_idx], y_true[va_idx]

    # Preprocess with BaselineFeaturePipeline (46 features)
    pipe = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=True)
    pipe.fit(X_raw.iloc[tr_idx])
    X_tr_base = pipe.transform(X_raw.iloc[tr_idx])
    X_va_base = pipe.transform(X_raw.iloc[va_idx])

    # Engineer group features
    pipe_raw = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
    pipe_raw.fit(X_raw.iloc[tr_idx])
    X_tr_raw = pipe_raw.transform(X_raw.iloc[tr_idx])
    X_va_raw = pipe_raw.transform(X_raw.iloc[va_idx])
    X_tr_grp, X_va_grp = engineer_group_features(X_tr_raw, X_va_raw)

    # Encode categories for LightGBM
    for col, mapping in pipe.cat_mappings.items():
        if col in X_tr_grp.columns:
            X_tr_grp[col] = X_tr_grp[col].map(mapping).fillna(-1).astype(int)
            X_va_grp[col] = X_va_grp[col].map(mapping).fillna(-1).astype(int)

    cat_cols = [c for c in CATEGORICAL_COLS if c in X_tr_base.columns]

    # Test on LightGBM C2
    params_lgb = {
        "learning_rate": 0.025,
        "num_leaves": 20,
        "max_depth": 6,
        "min_child_samples": 120,
        "feature_fraction": 0.65,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "reg_alpha": 2.0,
        "reg_lambda": 10.0,
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 43,
    }

    # Baseline LightGBM C2 (46 feats)
    dtrain_base = lgb.Dataset(X_tr_base, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    dval_base = lgb.Dataset(X_va_base, label=y_va, reference=dtrain_base, categorical_feature=cat_cols, free_raw_data=False)
    m_base = lgb.train(params_lgb, dtrain_base, num_boost_round=1500, valid_sets=[dtrain_base, dval_base],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
    auc_base = roc_auc_score(y_va, m_base.predict(X_va_base, num_iteration=m_base.best_iteration))
    print(f"  LightGBM C2 Baseline (46 feats): Fold 1 AUC = {auc_base:.5f} (Iter: {m_base.best_iteration})")

    # LightGBM C2 with Group Features (51 feats)
    dtrain_grp = lgb.Dataset(X_tr_grp, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    dval_grp = lgb.Dataset(X_va_grp, label=y_va, reference=dtrain_grp, categorical_feature=cat_cols, free_raw_data=False)
    m_grp = lgb.train(params_lgb, dtrain_grp, num_boost_round=1500, valid_sets=[dtrain_grp, dval_grp],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    auc_grp = roc_auc_score(y_va, m_grp.predict(X_va_grp, num_iteration=m_grp.best_iteration))
    delta = auc_grp - auc_base
    status = "GAIN" if delta > 0.0001 else ("FLAT" if abs(delta) <= 0.0001 else "DEGRADED")
    print(f"  LightGBM C2 + Group Stats (51 feats): Fold 1 AUC = {auc_grp:.5f} ({delta:+.5f}) -> {status}")

if __name__ == "__main__":
    run_group_stats_screening()
