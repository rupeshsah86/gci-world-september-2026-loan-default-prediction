"""
Stage H10: Experiment Set 3 & 4 - Model Tuning and CatBoost on Target-Encoded Features (Fold 1).
Tests:
1. CatBoost on Target-Encoded features:
   - Config TE-1: depth=6, l2=10, lr=0.04
   - Config TE-2: depth=6, l2=20, lr=0.03
   - Config TE-3: depth=5, l2=10, lr=0.035
2. LightGBM C2 Hyperparameter search on Target-Encoded features:
   - num_leaves in [16, 20, 24, 28]
   - max_depth in [5, 6, 7]
   - min_child_samples in [80, 120, 160]
   - feature_fraction in [0.60, 0.65, 0.70]
   - extra_trees in [False, True]
3. XGBoost Config 3A Hyperparameter variations:
   - max_depth in [5, 6]
   - min_child_weight in [12, 15, 20]
   - colsample_bytree in [0.55, 0.65]
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS

def main():
    print("=" * 80)
    print("STAGE H10: MODEL TUNING & CATBOOST ON TE FEATURES (FOLD 1)")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(skf.split(X_raw, y_true))

    y_tr = y_true[tr_idx]
    y_va = y_true[va_idx]

    high_card = ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
    low_card = [c for c in CATEGORICAL_COLS if c not in high_card]

    # Preprocess
    pipe = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
    pipe.fit(X_raw.iloc[tr_idx])
    X_tr = pipe.transform(X_raw.iloc[tr_idx])
    X_va = pipe.transform(X_raw.iloc[va_idx])

    # Target Encoding
    skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    te = TargetEncoder(smooth="auto", cv=skf_inner)
    X_tr_te = te.fit_transform(X_tr[high_card], y_tr)
    X_va_te = te.transform(X_va[high_card])

    for i, c in enumerate(high_card):
        X_tr[f"{c}_TE"] = X_tr_te[:, i]
        X_va[f"{c}_TE"] = X_va_te[:, i]
        X_tr.drop(columns=[c], inplace=True)
        X_va.drop(columns=[c], inplace=True)

    # For LightGBM & CatBoost (integer encoded low-card)
    X_tr_lgb = X_tr.copy()
    X_va_lgb = X_va.copy()
    for col in low_card:
        cats = sorted(X_tr_lgb[col].dropna().unique().tolist())
        cat_map = {c: idx for idx, c in enumerate(cats)}
        X_tr_lgb[col] = X_tr_lgb[col].map(cat_map).fillna(-1).astype(int)
        X_va_lgb[col] = X_va_lgb[col].map(cat_map).fillna(-1).astype(int)

    # For XGBoost (categorical dtype)
    X_tr_xgb = X_tr.copy()
    X_va_xgb = X_va.copy()
    for col in low_card:
        cats = sorted(X_tr_xgb[col].dropna().unique().tolist())
        dt = pd.CategoricalDtype(categories=cats, ordered=False)
        X_tr_xgb[col] = X_tr_xgb[col].astype(dt)
        X_va_xgb[col] = X_va_xgb[col].where(X_va_xgb[col].isin(cats), np.nan).astype(dt)

    dtr_lgb = lgb.Dataset(X_tr_lgb, label=y_tr, categorical_feature=low_card, free_raw_data=False)
    dva_lgb = lgb.Dataset(X_va_lgb, label=y_va, reference=dtr_lgb, categorical_feature=low_card, free_raw_data=False)

    results = {}

    # 1. CatBoost on TE features
    print("\n--- 1. Testing CatBoost on Target-Encoded Features ---")
    cb_configs = [
        ("CB_TE_1", {"depth": 6, "l2_leaf_reg": 10.0, "learning_rate": 0.04, "iterations": 1500}),
        ("CB_TE_2", {"depth": 6, "l2_leaf_reg": 20.0, "learning_rate": 0.03, "iterations": 1800}),
        ("CB_TE_3", {"depth": 5, "l2_leaf_reg": 10.0, "learning_rate": 0.035, "iterations": 1600}),
    ]
    for name, params in cb_configs:
        cb_clf = CatBoostClassifier(
            **params,
            eval_metric="AUC",
            random_seed=42,
            early_stopping_rounds=50,
            verbose=False,
            thread_count=-1
        )
        cb_clf.fit(X_tr_lgb, y_tr, eval_set=(X_va_lgb, y_va), verbose=False)
        p_cb = cb_clf.predict_proba(X_va_lgb)[:, 1]
        auc_cb = float(roc_auc_score(y_va, p_cb))
        print(f"  {name}: Fold 1 AUC = {auc_cb:.5f} (Best Iter: {cb_clf.get_best_iteration()})")
        results[name] = {"auc": auc_cb, "best_iter": cb_clf.get_best_iteration()}

    # 2. LightGBM C2 Variations
    print("\n--- 2. LightGBM C2 Hyperparameter Variations on TE Features ---")
    lgb_base = {
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

    # Baseline score
    m_base = lgb.train(lgb_base, dtr_lgb, num_boost_round=1500, valid_sets=[dtr_lgb, dva_lgb], callbacks=[lgb.early_stopping(50, verbose=False)])
    auc_lgb_base = float(roc_auc_score(y_va, m_base.predict(X_va_lgb, num_iteration=m_base.best_iteration)))
    print(f"  LGB C2 Baseline (leaves=20, depth=6, mcw=120): Fold 1 AUC = {auc_lgb_base:.5f} (Best Iter: {m_base.best_iteration})")
    results["LGB_C2_baseline"] = {"auc": auc_lgb_base, "best_iter": m_base.best_iteration}

    lgb_variations = [
        ("LGB_Leaves24_Depth6", {"num_leaves": 24, "max_depth": 6}),
        ("LGB_Leaves16_Depth5", {"num_leaves": 16, "max_depth": 5}),
        ("LGB_Leaves28_Depth7", {"num_leaves": 28, "max_depth": 7}),
        ("LGB_MCW150", {"min_child_samples": 150}),
        ("LGB_MCW80", {"min_child_samples": 80}),
        ("LGB_Col060", {"feature_fraction": 0.60}),
        ("LGB_Col070", {"feature_fraction": 0.70}),
        ("LGB_RegL15_Alpha3", {"reg_lambda": 15.0, "reg_alpha": 3.0}),
        ("LGB_ExtraTrees", {"extra_trees": True}),
    ]

    for name, overrides in lgb_variations:
        p_ = lgb_base.copy()
        p_.update(overrides)
        m_ = lgb.train(p_, dtr_lgb, num_boost_round=1500, valid_sets=[dtr_lgb, dva_lgb], callbacks=[lgb.early_stopping(50, verbose=False)])
        auc_ = float(roc_auc_score(y_va, m_.predict(X_va_lgb, num_iteration=m_.best_iteration)))
        delta = auc_ - auc_lgb_base
        print(f"  {name:25s}: Fold 1 AUC = {auc_:.5f} (Delta: {delta:+.5f}, Best Iter: {m_.best_iteration})")
        results[name] = {"auc": auc_, "delta": delta, "best_iter": m_.best_iteration}

    # 3. XGBoost Config 3A Variations
    print("\n--- 3. XGBoost Config 3A Hyperparameter Variations on TE Features ---")
    xgb_base = {
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
        "early_stopping_rounds": 50,
    }

    clf_base = xgb.XGBClassifier(**xgb_base)
    clf_base.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    auc_xgb_base = float(roc_auc_score(y_va, clf_base.predict_proba(X_va_xgb)[:, 1]))
    print(f"  XGB C3A Baseline: Fold 1 AUC = {auc_xgb_base:.5f} (Best Iter: {clf_base.best_iteration})")
    results["XGB_C3A_baseline"] = {"auc": auc_xgb_base, "best_iter": clf_base.best_iteration}

    xgb_variations = [
        ("XGB_Depth5_MCW15", {"max_depth": 5, "min_child_weight": 15}),
        ("XGB_Depth6_MCW20", {"max_depth": 6, "min_child_weight": 20}),
        ("XGB_Depth6_MCW10", {"max_depth": 6, "min_child_weight": 10}),
        ("XGB_Col055", {"colsample_bytree": 0.55}),
        ("XGB_Col075", {"colsample_bytree": 0.75}),
        ("XGB_RegL20_Alpha5", {"reg_lambda": 20.0, "reg_alpha": 5.0}),
    ]

    for name, overrides in xgb_variations:
        p_ = xgb_base.copy()
        p_.update(overrides)
        clf_ = xgb.XGBClassifier(**p_)
        clf_.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
        auc_ = float(roc_auc_score(y_va, clf_.predict_proba(X_va_xgb)[:, 1]))
        delta = auc_ - auc_xgb_base
        print(f"  {name:25s}: Fold 1 AUC = {auc_:.5f} (Delta: {delta:+.5f}, Best Iter: {clf_.best_iteration})")
        results[name] = {"auc": auc_, "delta": delta, "best_iter": clf_.best_iteration}

    out_file = PROJECT_ROOT / "experiments" / "stage_h10" / "model_tuning_fold1_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved tuning results to: {out_file}")

if __name__ == "__main__":
    main()

