"""
Stage H7: Experiment A - XGBoost Hyperparameter Exploration.
Test bounded, regularized XGBoost configurations against verified Config 3 baseline (0.76015).
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS

def get_xgb_features(X_tr_raw, X_va_raw):
    pipe = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
    pipe.fit(X_tr_raw)
    X_tr = pipe.transform(X_tr_raw)
    X_va = pipe.transform(X_va_raw)
    for col in CATEGORICAL_COLS:
        if col in X_tr.columns:
            train_cats = sorted(X_tr[col].dropna().unique().tolist())
            cat_dt = pd.CategoricalDtype(categories=train_cats, ordered=False)
            X_tr[col] = X_tr[col].astype(cat_dt)
            X_va[col] = X_va[col].where(X_va[col].isin(train_cats), np.nan).astype(cat_dt)
    return X_tr, X_va

def run_fold1_pilot():
    print("=" * 80)
    print("STAGE H7: XGBOOST FOLD 1 SCREENING PILOT")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(skf.split(X_raw, y_true))

    print(f"[+] Transforming Fold 1 data (Train: {len(tr_idx):,}, Val: {len(va_idx):,})...")
    X_tr, X_va = get_xgb_features(X_raw.iloc[tr_idx], X_raw.iloc[va_idx])
    y_tr, y_va = y_true[tr_idx], y_true[va_idx]

    configs = {
        "Config_3_Baseline": {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.60,
            "reg_alpha": 2.0,
            "reg_lambda": 10.0,
            "tree_method": "hist",
            "enable_categorical": True,
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        },
        "Config_3A_HighL2_MCW15_LR025": {
            "n_estimators": 1200,
            "learning_rate": 0.025,
            "max_depth": 6,
            "min_child_weight": 15,
            "subsample": 0.85,
            "colsample_bytree": 0.65,
            "reg_alpha": 3.0,
            "reg_lambda": 15.0,
            "tree_method": "hist",
            "enable_categorical": True,
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        },
        "Config_3B_Depth5_MCW15_LR03": {
            "n_estimators": 1200,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 15,
            "subsample": 0.85,
            "colsample_bytree": 0.70,
            "reg_alpha": 2.0,
            "reg_lambda": 12.0,
            "tree_method": "hist",
            "enable_categorical": True,
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        },
        "Config_3C_Col055_Sub08_L2_12": {
            "n_estimators": 1200,
            "learning_rate": 0.025,
            "max_depth": 6,
            "min_child_weight": 10,
            "subsample": 0.80,
            "colsample_bytree": 0.55,
            "reg_alpha": 2.0,
            "reg_lambda": 12.0,
            "tree_method": "hist",
            "enable_categorical": True,
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        },
    }

    results = {}
    for name, params in configs.items():
        t0 = time.time()
        clf = xgb.XGBClassifier(**params, early_stopping_rounds=50)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        preds = clf.predict_proba(X_va)[:, 1]
        score = float(roc_auc_score(y_va, preds))
        best_iter = clf.best_iteration
        elapsed = time.time() - t0
        results[name] = {
            "auc": score,
            "best_iter": int(best_iter),
            "elapsed": elapsed,
        }
        print(f"  {name:30s} -> Fold 1 AUC: {score:.5f} (Best Iter: {best_iter:4d}, Time: {elapsed:.1f}s)")

    baseline_auc = results["Config_3_Baseline"]["auc"]
    for name, res in results.items():
        delta = res["auc"] - baseline_auc
        print(f"  vs Baseline: {name:30s} Delta: {delta:+.5f}")

    return results

if __name__ == "__main__":
    run_fold1_pilot()
