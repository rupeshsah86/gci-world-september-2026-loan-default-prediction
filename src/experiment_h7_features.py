"""
Stage H7: Experiment C - Feature Engineering Screening on LightGBM C2.
Evaluate leakage-safe feature candidate pilots against verified LightGBM C2 Fold 1 baseline.
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS
from src.features_h1 import ExtendedFeaturePipeline

def get_pilot_features(X_raw, pilot_name):
    if pilot_name == "Baseline_46":
        pipe = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=True)
        return pipe, False
    elif pilot_name == "Pilot_1_Ext_Interactions_51":
        pipe = ExtendedFeaturePipeline(
            include_group1_financial=False,
            include_group2_temporal=False,
            include_group3_ext_interact=True,
            encode_categories=True,
        )
        return pipe, False
    elif pilot_name == "Pilot_2_Financial_Ratios_50":
        pipe = ExtendedFeaturePipeline(
            include_group1_financial=True,
            include_group2_temporal=False,
            include_group3_ext_interact=False,
            encode_categories=True,
        )
        return pipe, False
    elif pilot_name == "Pilot_3_All_Extended_62":
        pipe = ExtendedFeaturePipeline(
            include_group1_financial=True,
            include_group2_temporal=True,
            include_group3_ext_interact=True,
            encode_categories=True,
        )
        return pipe, False
    elif pilot_name == "Pilot_4_Novel_Interactions_50":
        pipe = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=True)
        return pipe, True
    else:
        raise ValueError(f"Unknown pilot: {pilot_name}")

def add_novel_features(df):
    df = df.copy()
    if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_2" in df.columns:
        df["EXT_1_TO_2_RATIO"] = df["EXT_SOURCE_1"] / (df["EXT_SOURCE_2"] + 1e-4)
    if "EXT_SOURCE_2" in df.columns and "EXT_SOURCE_3" in df.columns:
        df["EXT_2_TO_3_RATIO"] = df["EXT_SOURCE_2"] / (df["EXT_SOURCE_3"] + 1e-4)
    if "AMT_ANNUITY" in df.columns and "DAYS_BIRTH" in df.columns:
        df["ANNUITY_PER_AGE_DAY"] = df["AMT_ANNUITY"] / (-df["DAYS_BIRTH"] + 1e-4)
    if "DAYS_EMPLOYED" in df.columns and "DAYS_REGISTRATION" in df.columns:
        df["EMPLOYED_TO_REGISTRATION_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_REGISTRATION"] - 1.0)
    return df

def run_feature_screening():
    print("=" * 80)
    print("STAGE H7: FEATURE PILOT SCREENING (LIGHTGBM C2, FOLD 1)")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(skf.split(X_raw, y_true))
    y_tr, y_va = y_true[tr_idx], y_true[va_idx]

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

    pilots = [
        "Baseline_46",
        "Pilot_1_Ext_Interactions_51",
        "Pilot_2_Financial_Ratios_50",
        "Pilot_3_All_Extended_62",
        "Pilot_4_Novel_Interactions_50",
    ]

    results = {}
    for pilot in pilots:
        t0 = time.time()
        pipe, is_novel = get_pilot_features(X_raw, pilot)
        pipe.fit(X_raw.iloc[tr_idx])
        X_tr = pipe.transform(X_raw.iloc[tr_idx])
        X_va = pipe.transform(X_raw.iloc[va_idx])

        if is_novel:
            X_tr = add_novel_features(X_tr)
            X_va = add_novel_features(X_va)

        cat_cols = [c for c in CATEGORICAL_COLS if c in X_tr.columns]
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, categorical_feature=cat_cols, free_raw_data=False)

        m = lgb.train(
            params_lgb,
            dtrain,
            num_boost_round=1500,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        preds = m.predict(X_va, num_iteration=m.best_iteration)
        score = float(roc_auc_score(y_va, preds))
        elapsed = time.time() - t0
        results[pilot] = {
            "n_features": X_tr.shape[1],
            "auc": score,
            "best_iter": int(m.best_iteration),
            "elapsed": elapsed,
        }
        print(f"  {pilot:30s} ({X_tr.shape[1]:2d} feats) -> Fold 1 AUC: {score:.5f} (Best Iter: {m.best_iteration:4d}, {elapsed:.1f}s)")

    baseline_score = results["Baseline_46"]["auc"]
    print("\nComparison vs Baseline_46 (Fold 1):")
    for pilot, res in results.items():
        delta = res["auc"] - baseline_score
        status = "GAIN" if delta > 0.0001 else ("FLAT" if abs(delta) <= 0.0001 else "DEGRADED")
        print(f"  {pilot:30s}: AUC = {res['auc']:.5f} ({delta:+.5f}) -> {status}")

if __name__ == "__main__":
    run_feature_screening()
