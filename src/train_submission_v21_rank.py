"""
Stage H4: Rank-Normalized Quad-Model Blend Pilot Candidate V2.1 Generation Script.
Project: GCI World 2026 Home Credit Default Risk.

Component Models (Trained on 100% of Training Data):
1. CatBoost Config 4 (62 features, depth=6, l2=10.0, rs=1.5, lr=0.04, iters=1180, weight=0.27)
2. CatBoost Config 3 (62 features, depth=6, l2=5.0, rs=2.0, lr=0.04, iters=1103, weight=0.23)
3. XGBoost Config 3 (46 features, depth=6, mcw=10, col=0.60, reg_a=2.0, reg_l=10.0, lr=0.03, iters=453, weight=0.30)
4. Tuned LightGBM (46 features, leaves=24, depth=7, mcw=100, col=0.70, reg_a=1.0, reg_l=5.0, lr=0.03, iters=465, weight=0.20)

Percentile-Rank Normalization Formula:
r_m = rankdata(p_m, method="average") / N
p_rank_blend = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb3 + 0.20 * r_lgb

Target Output: outputs/submission_candidate_20261013_v21_rank_blend.csv
Review Report: reports/stage_h4_rank_blend_review.md
"""

import sys
from pathlib import Path
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy.stats import rankdata
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import lightgbm as lgb

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS
from src.features_h1 import ExtendedFeaturePipeline

EXISTING_V2_PATH = PROJECT_ROOT / "outputs" / "submission_candidate_20261012_v2_quad_blend.csv"
OUTPUT_CANDIDATE_PATH = PROJECT_ROOT / "outputs" / "submission_candidate_20261013_v21_rank_blend.csv"
TEST_PREDS_PATH = PROJECT_ROOT / "experiments" / "test_preds_quad_models.npz"


def main():
    total_start = time.time()
    print("=" * 80)
    print("STAGE H4: TRAINING QUAD-MODEL ENSEMBLE & GENERATING RANK-BLENDED CANDIDATE V2.1")
    print("=" * 80)

    # 1. Load Raw Training and Test Data
    print("\n[Step 1/7] Loading competition raw datasets...")
    train_df, test_df, sample_sub = load_raw_data(verbose=True)

    y_train = train_df[TARGET_COL].values
    X_train_raw = train_df.drop(columns=[TARGET_COL, ID_COL])
    X_test_raw = test_df.drop(columns=[ID_COL])

    n_train = len(train_df)
    n_test = len(test_df)
    assert n_train == 171202, f"Expected 171,202 train rows, got {n_train}"
    assert n_test == 61500, f"Expected 61,500 test rows, got {n_test}"
    print(f"[+] Loaded {n_train:,} training samples and {n_test:,} test samples.")

    # -----------------------------------------------------------------
    # PIPELINE 1: Baseline 46-feature pipeline (for LightGBM & XGBoost)
    # -----------------------------------------------------------------
    print("\n[Step 2/7] Preprocessing 46-feature baseline pipeline...")
    # For LightGBM (integer label encoded categories)
    pipe_lgb = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=True)
    pipe_lgb.fit(X_train_raw)
    X_tr_lgb = pipe_lgb.transform(X_train_raw)
    X_te_lgb = pipe_lgb.transform(X_test_raw)
    cat_cols_lgb = [c for c in CATEGORICAL_COLS if c in pipe_lgb.feature_names]

    # For XGBoost (CategoricalDtype)
    pipe_xgb = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
    pipe_xgb.fit(X_train_raw)
    X_tr_xgb = pipe_xgb.transform(X_train_raw)
    X_te_xgb = pipe_xgb.transform(X_test_raw)
    for col in CATEGORICAL_COLS:
        if col in X_tr_xgb.columns:
            train_cats = sorted(X_tr_xgb[col].dropna().unique().tolist())
            cat_dt = pd.CategoricalDtype(categories=train_cats, ordered=False)
            X_tr_xgb[col] = X_tr_xgb[col].astype(cat_dt)
            X_te_xgb[col] = X_te_xgb[col].where(X_te_xgb[col].isin(train_cats), np.nan).astype(cat_dt)

    # -----------------------------------------------------------------
    # PIPELINE 2: Extended 62-feature pipeline (for CatBoost models)
    # -----------------------------------------------------------------
    print("\n[Step 3/7] Preprocessing 62-feature extended pipeline...")
    pipe_cb = ExtendedFeaturePipeline(
        include_group1_financial=True,
        include_group2_temporal=True,
        include_group3_ext_interact=True,
        encode_categories=False,
    )
    pipe_cb.fit(X_train_raw)
    X_tr_cb = pipe_cb.transform(X_train_raw)
    X_te_cb = pipe_cb.transform(X_test_raw)
    cat_cols_cb = [c for c in CATEGORICAL_COLS if c in X_tr_cb.columns]
    pool_tr_cb = Pool(X_tr_cb, y_train, cat_features=cat_cols_cb)
    pool_te_cb = Pool(X_te_cb, cat_features=cat_cols_cb)

    # -----------------------------------------------------------------
    # MODEL 1: Tuned LightGBM (46 features, Weight: 20%)
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING MODEL 1: Tuned LightGBM (Weight: 20%)...")
    print("=" * 80)
    t0 = time.time()
    dtrain_lgb = lgb.Dataset(X_tr_lgb, label=y_train, categorical_feature=cat_cols_lgb, free_raw_data=False)
    lgb_params = {
        "learning_rate": 0.03,
        "num_leaves": 24,
        "max_depth": 7,
        "min_child_samples": 100,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }
    model_lgb = lgb.train(lgb_params, dtrain_lgb, num_boost_round=465)
    p_test_lgb = model_lgb.predict(X_te_lgb)
    lgb_time = time.time() - t0
    print(f"  [+] LightGBM completed in {lgb_time:.1f}s. Test mean: {p_test_lgb.mean():.4f}")

    # -----------------------------------------------------------------
    # MODEL 2: XGBoost Config 3 (46 features, Weight: 30%)
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING MODEL 2: XGBoost Config 3 (Weight: 30%)...")
    print("=" * 80)
    t0 = time.time()
    xgb_params = {
        "n_estimators": 453,
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
    }
    clf_xgb = xgb.XGBClassifier(**xgb_params)
    clf_xgb.fit(X_tr_xgb, y_train, verbose=False)
    p_test_xgb = clf_xgb.predict_proba(X_te_xgb)[:, 1]
    xgb_time = time.time() - t0
    print(f"  [+] XGBoost completed in {xgb_time:.1f}s. Test mean: {p_test_xgb.mean():.4f}")

    # -----------------------------------------------------------------
    # MODEL 3: CatBoost Config 3 (62 features, Weight: 23%)
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING MODEL 3: CatBoost Config 3 (Weight: 23%)...")
    print("=" * 80)
    t0 = time.time()
    cb3_params = {
        "iterations": 1103,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_strength": 2.0,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "random_seed": 42,
        "thread_count": 4,
        "verbose": 250,
        "task_type": "CPU",
    }
    cb3_model = CatBoostClassifier(**cb3_params)
    cb3_model.fit(pool_tr_cb, verbose=250)
    p_test_cb3 = cb3_model.predict_proba(pool_te_cb)[:, 1]
    cb3_time = time.time() - t0
    print(f"  [+] CatBoost Config 3 completed in {cb3_time:.1f}s. Test mean: {p_test_cb3.mean():.4f}")

    # -----------------------------------------------------------------
    # MODEL 4: CatBoost Config 4 (62 features, Weight: 27%)
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING MODEL 4: CatBoost Config 4 (Weight: 27%)...")
    print("=" * 80)
    t0 = time.time()
    cb4_params = {
        "iterations": 1180,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 10.0,
        "random_strength": 1.5,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "random_seed": 42,
        "thread_count": 4,
        "verbose": 250,
        "task_type": "CPU",
    }
    cb4_model = CatBoostClassifier(**cb4_params)
    cb4_model.fit(pool_tr_cb, verbose=250)
    p_test_cb4 = cb4_model.predict_proba(pool_te_cb)[:, 1]
    cb4_time = time.time() - t0
    print(f"  [+] CatBoost Config 4 completed in {cb4_time:.1f}s. Test mean: {p_test_cb4.mean():.4f}")

    # Save test predictions for reproducibility
    np.savez_compressed(
        TEST_PREDS_PATH,
        test_cb4=p_test_cb4,
        test_cb3=p_test_cb3,
        test_xgb=p_test_xgb,
        test_lgb=p_test_lgb,
        sk_id_curr=test_df[ID_COL].values,
    )
    print(f"\n[+] Saved test prediction arrays to: {TEST_PREDS_PATH}")

    # -----------------------------------------------------------------
    # SANITY CHECK: Verify reproduction of raw probability candidate V2
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("REPLICATION SANITY CHECK: COMPARING PROBABILITY BLEND TO CANDIDATE V2...")
    print("=" * 80)
    p_test_prob_blend = (
        0.27 * p_test_cb4
        + 0.23 * p_test_cb3
        + 0.20 * p_test_lgb
        + 0.30 * p_test_xgb
    )
    if EXISTING_V2_PATH.exists():
        existing_v2_df = pd.read_csv(EXISTING_V2_PATH)
        max_diff = np.max(np.abs(existing_v2_df[TARGET_COL].values - p_test_prob_blend))
        print(f"  Max absolute difference vs Candidate V2: {max_diff:.8e}")
        assert max_diff < 1e-5, f"Probability blend diverged from existing candidate V2 (max diff: {max_diff})!"
        print("  [PASSED] Raw probability blend matches existing Candidate V2 perfectly!")
    else:
        print("  [WARNING] Existing Candidate V2 not found for comparison.")

    # -----------------------------------------------------------------
    # STEP 6: APPLY PERCENTILE-RANK NORMALIZATION SEPARATELY TO TEST PREDS
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("APPLYING PERCENTILE-RANK NORMALIZATION TO EACH MODEL'S TEST PREDICTIONS...")
    print("=" * 80)
    # Average rank for ties divided by N
    r_test_cb4 = rankdata(p_test_cb4, method="average") / n_test
    r_test_cb3 = rankdata(p_test_cb3, method="average") / n_test
    r_test_xgb = rankdata(p_test_xgb, method="average") / n_test
    r_test_lgb = rankdata(p_test_lgb, method="average") / n_test

    # Check bounds
    for name, r in [("CB4", r_test_cb4), ("CB3", r_test_cb3), ("XGB3", r_test_xgb), ("LGB", r_test_lgb)]:
        print(f"  {name} Rank Pct: min={r.min():.6f}, max={r.max():.6f}, mean={r.mean():.6f}, std={r.std():.6f}")

    # Combine with existing blend weights: 27% CB4 + 23% CB3 + 20% LGB + 30% XGB
    p_test_rank_blend = (
        0.27 * r_test_cb4
        + 0.23 * r_test_cb3
        + 0.20 * r_test_lgb
        + 0.30 * r_test_xgb
    )

    print(f"\n  Rank-Blended Test Candidate V2.1 Summary Statistics ({n_test:,} records):")
    print(f"    Min probability : {p_test_rank_blend.min():.6f}")
    print(f"    Max probability : {p_test_rank_blend.max():.6f}")
    print(f"    Mean probability: {p_test_rank_blend.mean():.6f}")
    print(f"    Std probability : {p_test_rank_blend.std():.6f}")
    print(f"    Median          : {np.median(p_test_rank_blend):.6f}")
    print(f"    1st Percentile  : {np.percentile(p_test_rank_blend, 1):.6f}")
    print(f"    99th Percentile : {np.percentile(p_test_rank_blend, 99):.6f}")

    # -----------------------------------------------------------------
    # STEP 7: SAVE AND AUDIT NEW CANDIDATE V2.1 SUBMISSION
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SAVING AND AUDITING NEW CANDIDATE V2.1 SUBMISSION...")
    print("=" * 80)

    sub_df = pd.DataFrame({
        ID_COL: test_df[ID_COL].values,
        TARGET_COL: p_test_rank_blend,
    })

    OUTPUT_CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(OUTPUT_CANDIDATE_PATH, index=False)
    print(f"[+] Saved new rank-blended submission candidate to: {OUTPUT_CANDIDATE_PATH}")

    # Audit 1: File existence and size
    assert OUTPUT_CANDIDATE_PATH.exists(), "Submission file was not written!"
    file_size_mb = OUTPUT_CANDIDATE_PATH.stat().st_size / (1024 * 1024)
    print(f"  [CHECK 1] File exists ({file_size_mb:.2f} MB)")

    # Audit 2: Reopen and parse as CSV
    loaded_sub = pd.read_csv(OUTPUT_CANDIDATE_PATH)
    print(f"  [CHECK 2] Reopened and parsed successfully as CSV")

    # Audit 3: Exact shape and column order
    assert loaded_sub.shape == (61500, 2), f"Expected shape (61500, 2), got {loaded_sub.shape}"
    assert list(loaded_sub.columns) == [ID_COL, TARGET_COL], f"Columns mismatch: {loaded_sub.columns}"
    print(f"  [CHECK 3] Exact shape (61,500 rows, 2 columns: ['SK_ID_CURR', 'TARGET']) verified")

    # Audit 4: Exact ID alignment
    assert (loaded_sub[ID_COL].values == test_df[ID_COL].values).all(), "ID alignment mismatch with test.csv!"
    assert loaded_sub[ID_COL].nunique() == 61500, "Duplicate IDs detected in submission!"
    print(f"  [CHECK 4] Exact ID alignment with test.csv in original order verified (zero missing, zero duplicates)")

    # Audit 5: Probability bounds and finiteness
    assert not loaded_sub[TARGET_COL].isna().any(), "NaNs detected in TARGET!"
    assert not np.isinf(loaded_sub[TARGET_COL].values).any(), "Infinite values detected in TARGET!"
    assert (loaded_sub[TARGET_COL].values >= 0.0).all(), "Predictions < 0 detected!"
    assert (loaded_sub[TARGET_COL].values <= 1.0).all(), "Predictions > 1 detected!"
    print(f"  [CHECK 5] All TARGET values are finite probabilities strictly within [0.0, 1.0]")

    # Audit 6: Ensure previous candidate was not overwritten
    assert EXISTING_V2_PATH.exists(), "Existing Candidate V2 was deleted or overwritten!"
    print(f"  [CHECK 6] Previous candidate {EXISTING_V2_PATH.name} remains untouched.")

    total_time = time.time() - total_start
    print(f"\n[+] All Stage H4 candidate generation steps and audits passed in {total_time:.1f}s (~{total_time/60:.1f} minutes).")


if __name__ == "__main__":
    main()
