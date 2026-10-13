"""
Stage H7: Experiment B - CatBoost Hyperparameter Exploration.
Test bounded, regularized CatBoost configurations against verified Config 4 baseline (0.76080 OOF).
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features_h1 import ExtendedFeaturePipeline
from src.features import CATEGORICAL_COLS

def get_cb_features(X_tr_raw, X_va_raw):
    pipe = ExtendedFeaturePipeline(
        include_group1_financial=True,
        include_group2_temporal=True,
        include_group3_ext_interact=True,
        encode_categories=False,
    )
    pipe.fit(X_tr_raw)
    X_tr = pipe.transform(X_tr_raw)
    X_va = pipe.transform(X_va_raw)
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_tr.columns]
    return X_tr, X_va, cat_cols

def run_fold1_pilot():
    print("=" * 80)
    print("STAGE H7: CATBOOST FOLD 1 SCREENING PILOT")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(skf.split(X_raw, y_true))

    print(f"[+] Transforming Fold 1 data (Train: {len(tr_idx):,}, Val: {len(va_idx):,})...")
    X_tr, X_va, cat_cols = get_cb_features(X_raw.iloc[tr_idx], X_raw.iloc[va_idx])
    y_tr, y_va = y_true[tr_idx], y_true[va_idx]

    pool_tr = Pool(X_tr, y_tr, cat_features=cat_cols)
    pool_va = Pool(X_va, y_va, cat_features=cat_cols)

    configs = {
        "Config_4_Baseline": {
            "iterations": 1500,
            "learning_rate": 0.04,
            "depth": 6,
            "l2_leaf_reg": 10.0,
            "random_strength": 1.5,
            "eval_metric": "AUC",
            "loss_function": "Logloss",
            "random_seed": 42,
            "verbose": False,
            "thread_count": -1,
        },
        "Config_5_L2_15_LR035": {
            "iterations": 1600,
            "learning_rate": 0.035,
            "depth": 6,
            "l2_leaf_reg": 15.0,
            "random_strength": 1.5,
            "eval_metric": "AUC",
            "loss_function": "Logloss",
            "random_seed": 42,
            "verbose": False,
            "thread_count": -1,
        },
        "Config_6_L2_20_RS2_LR030": {
            "iterations": 1800,
            "learning_rate": 0.030,
            "depth": 6,
            "l2_leaf_reg": 20.0,
            "random_strength": 2.0,
            "eval_metric": "AUC",
            "loss_function": "Logloss",
            "random_seed": 42,
            "verbose": False,
            "thread_count": -1,
        },
        "Config_7_Depth7_L2_20_LR03": {
            "iterations": 1500,
            "learning_rate": 0.030,
            "depth": 7,
            "l2_leaf_reg": 20.0,
            "random_strength": 1.5,
            "eval_metric": "AUC",
            "loss_function": "Logloss",
            "random_seed": 42,
            "verbose": False,
            "thread_count": -1,
        },
    }

    results = {}
    for name, params in configs.items():
        t0 = time.time()
        clf = CatBoostClassifier(**params, early_stopping_rounds=60)
        clf.fit(pool_tr, eval_set=pool_va, verbose=False)
        preds = clf.predict_proba(pool_va)[:, 1]
        score = float(roc_auc_score(y_va, preds))
        best_iter = clf.get_best_iteration()
        elapsed = time.time() - t0
        results[name] = {
            "auc": score,
            "best_iter": int(best_iter),
            "elapsed": elapsed,
        }
        print(f"  {name:30s} -> Fold 1 AUC: {score:.5f} (Best Iter: {best_iter:4d}, Time: {elapsed:.1f}s)")

    baseline_auc = results["Config_4_Baseline"]["auc"]
    for name, res in results.items():
        delta = res["auc"] - baseline_auc
        print(f"  vs Baseline: {name:30s} Delta: {delta:+.5f}")

    return results

if __name__ == "__main__":
    run_fold1_pilot()
