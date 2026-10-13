"""
Stage H9: Train & Generate Candidate V2.5-C (Dual Target-Encoded Quad Blend).
Pipeline:
1. Target-Encoded XGBoost Config 3A (751 rounds, smooth out-of-fold target encoding)
2. Target-Encoded LightGBM C2 (826 rounds, smooth out-of-fold target encoding)
3. Verified CatBoost Config 4 (weight: 0.27)
4. Verified CatBoost Config 3 (weight: 0.23)
5. Target-Encoded XGBoost Config 3A (weight: 0.30)
6. Target-Encoded LightGBM C2 (weight: 0.20)
All components percentile-rank normalized before blending.
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS

CANDIDATE_V25_PATH = PROJECT_ROOT / "outputs" / "submission_candidate_20261013_v25_h9.csv"
TEST_PREDS_PATH = PROJECT_ROOT / "experiments" / "test_preds_quad_models.npz"
RESULTS_JSON_PATH = PROJECT_ROOT / "experiments" / "stage_h9" / "stage_h9_candidate_v25_results.json"

def main():
    t_start = time.time()
    print("=" * 80)
    print("GCI WORLD 2026 — STAGE H9: CANDIDATE V2.5-C GENERATION")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1/6] Loading raw training and test datasets...")
    train_df, test_df, sample_sub = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_train_raw = train_df.drop(columns=[TARGET_COL, ID_COL])
    X_test_raw = test_df.drop(columns=[ID_COL])
    n_train = len(train_df)
    n_test = len(test_df)
    print(f"  [+] Loaded {n_train:,} training samples and {n_test:,} test samples.")

    high_card_cats = ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"]

    # 2. Train Full-Data Target-Encoded XGBoost Config 3A (Mean best iterations = 751)
    print("\n[Step 2/6] Preprocessing and training full-data Target-Encoded XGBoost Config 3A...")
    pipe_xgb = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
    pipe_xgb.fit(X_train_raw)
    X_tr_xgb = pipe_xgb.transform(X_train_raw)
    X_te_xgb = pipe_xgb.transform(X_test_raw)

    # Leakage-safe Target Encoding: internal 5-fold CV on train, transform on test
    skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    te_xgb = TargetEncoder(smooth="auto", cv=skf_inner)
    X_tr_te_vals = te_xgb.fit_transform(X_tr_xgb[high_card_cats], y_true)
    X_te_te_vals = te_xgb.transform(X_te_xgb[high_card_cats])

    for idx, col in enumerate(high_card_cats):
        X_tr_xgb[f"{col}_TE"] = X_tr_te_vals[:, idx]
        X_te_xgb[f"{col}_TE"] = X_te_te_vals[:, idx]
        X_tr_xgb.drop(columns=[col], inplace=True)
        X_te_xgb.drop(columns=[col], inplace=True)

    low_card_cats_xgb = [c for c in CATEGORICAL_COLS if c not in high_card_cats and c in X_tr_xgb.columns]
    for col in low_card_cats_xgb:
        cats = sorted(X_tr_xgb[col].dropna().unique().tolist())
        dt = pd.CategoricalDtype(categories=cats, ordered=False)
        X_tr_xgb[col] = X_tr_xgb[col].astype(dt)
        X_te_xgb[col] = X_te_xgb[col].where(X_te_xgb[col].isin(cats), np.nan).astype(dt)

    params_xgb_c3a = {
        "n_estimators": 751,
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
    }

    t0 = time.time()
    clf_xgb = xgb.XGBClassifier(**params_xgb_c3a)
    clf_xgb.fit(X_tr_xgb, y_true, verbose=False)
    p_test_xgb_te = clf_xgb.predict_proba(X_te_xgb)[:, 1]
    print(f"  [+] Target-Encoded XGBoost completed in {time.time() - t0:.1f}s. Test mean: {p_test_xgb_te.mean():.4f}")

    # 3. Train Full-Data Target-Encoded LightGBM C2 (Mean best iterations = 826)
    print("\n[Step 3/6] Preprocessing and training full-data Target-Encoded LightGBM C2...")
    pipe_lgb = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
    pipe_lgb.fit(X_train_raw)
    X_tr_lgb = pipe_lgb.transform(X_train_raw)
    X_te_lgb = pipe_lgb.transform(X_test_raw)

    te_lgb = TargetEncoder(smooth="auto", cv=skf_inner)
    X_tr_te_vals_lgb = te_lgb.fit_transform(X_tr_lgb[high_card_cats], y_true)
    X_te_te_vals_lgb = te_lgb.transform(X_te_lgb[high_card_cats])

    for idx, col in enumerate(high_card_cats):
        X_tr_lgb[f"{col}_TE"] = X_tr_te_vals_lgb[:, idx]
        X_te_lgb[f"{col}_TE"] = X_te_te_vals_lgb[:, idx]
        X_tr_lgb.drop(columns=[col], inplace=True)
        X_te_lgb.drop(columns=[col], inplace=True)

    low_card_cats_lgb = [c for c in CATEGORICAL_COLS if c not in high_card_cats and c in X_tr_lgb.columns]
    for col in low_card_cats_lgb:
        cats = sorted(X_tr_lgb[col].dropna().unique().tolist())
        cat_map = {c: i for i, c in enumerate(cats)}
        X_tr_lgb[col] = X_tr_lgb[col].map(cat_map).fillna(-1).astype(int)
        X_te_lgb[col] = X_te_lgb[col].map(cat_map).fillna(-1).astype(int)

    dtrain_lgb = lgb.Dataset(X_tr_lgb, label=y_true, categorical_feature=low_card_cats_lgb, free_raw_data=False)
    params_lgb_c2 = {
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

    t0 = time.time()
    model_lgb = lgb.train(params_lgb_c2, dtrain_lgb, num_boost_round=826)
    p_test_lgb_te = model_lgb.predict(X_te_lgb)
    print(f"  [+] Target-Encoded LightGBM completed in {time.time() - t0:.1f}s. Test mean: {p_test_lgb_te.mean():.4f}")

    # 4. Load Verified CatBoost Component Predictions from Cache
    print("\n[Step 4/6] Loading verified CatBoost component test predictions...")
    assert TEST_PREDS_PATH.exists(), f"Missing cache: {TEST_PREDS_PATH}"
    npz_data = np.load(TEST_PREDS_PATH)
    p_test_cb4 = npz_data["test_cb4"]
    p_test_cb3 = npz_data["test_cb3"]
    print(f"  [+] Loaded CatBoost Config 4 (mean: {p_test_cb4.mean():.4f}) and Config 3 (mean: {p_test_cb3.mean():.4f})")

    # 5. Percentile-Rank Normalization and Blending
    print("\n[Step 5/6] Applying Percentile-Rank Normalization & Blending...")
    r_test_cb4 = rankdata(p_test_cb4, method="average") / n_test
    r_test_cb3 = rankdata(p_test_cb3, method="average") / n_test
    r_test_xgb_te = rankdata(p_test_xgb_te, method="average") / n_test
    r_test_lgb_te = rankdata(p_test_lgb_te, method="average") / n_test

    # Candidate V2.5-C Weights: CB4: 0.27, CB3: 0.23, XGB_TE: 0.30, LGB_TE: 0.20
    w_cb4 = 0.27
    w_cb3 = 0.23
    w_xgb = 0.30
    w_lgb = 0.20

    p_test_v25 = (
        w_cb4 * r_test_cb4 +
        w_cb3 * r_test_cb3 +
        w_xgb * r_test_xgb_te +
        w_lgb * r_test_lgb_te
    )
    print(f"  [+] Blended Candidate V2.5-C test predictions. Mean: {p_test_v25.mean():.5f}, Std: {p_test_v25.std():.5f}")

    # 6. Audit and Save Candidate V2.5-C
    print("\n[Step 6/6] Auditing and Saving Candidate V2.5-C...")
    sub_df = pd.DataFrame({
        ID_COL: test_df[ID_COL].values,
        TARGET_COL: p_test_v25,
    })

    # Strict Validation Checks
    assert len(sub_df) == 61500, f"Expected 61,500 rows, got {len(sub_df)}"
    assert list(sub_df.columns) == ["SK_ID_CURR", "TARGET"], f"Unexpected columns: {list(sub_df.columns)}"
    assert (sub_df["SK_ID_CURR"].values == test_df["SK_ID_CURR"].values).all(), "ID alignment failed vs test.csv"
    assert (sub_df["SK_ID_CURR"].values == sample_sub["SK_ID_CURR"].values).all(), "ID alignment failed vs sample_submission.csv"
    assert not sub_df["TARGET"].isna().any(), "Target contains NaNs"
    assert not np.isinf(sub_df["TARGET"].values).any(), "Target contains Infs"
    assert (sub_df["TARGET"] >= 0.0).all() and (sub_df["TARGET"] <= 1.0).all(), "Predictions outside [0, 1]"
    assert sub_df["TARGET"].nunique() > 1000, "Hard class predictions detected instead of probabilities"

    # Save to disk
    sub_df.to_csv(CANDIDATE_V25_PATH, index=False)
    print(f"  [+] Successfully saved validated Candidate V2.5-C to: {CANDIDATE_V25_PATH}")

    # Correlation vs previous candidates
    v2_preds = pd.read_csv(PROJECT_ROOT / "outputs" / "submission_candidate_20261012_v2_quad_blend.csv")["TARGET"].values
    v22_preds = pd.read_csv(PROJECT_ROOT / "outputs" / "submission_candidate_20261013_v22_h6.csv")["TARGET"].values
    v23_preds = pd.read_csv(PROJECT_ROOT / "outputs" / "submission_candidate_20261013_v23_h7.csv")["TARGET"].values

    rho_v2, _ = spearmanr(p_test_v25, v2_preds)
    rho_v22, _ = spearmanr(p_test_v25, v22_preds)
    rho_v23, _ = spearmanr(p_test_v25, v23_preds)
    print(f"  [+] Spearman Correlation vs V2:   {rho_v2:.5f}")
    print(f"  [+] Spearman Correlation vs V2.2: {rho_v22:.5f}")
    print(f"  [+] Spearman Correlation vs V2.3: {rho_v23:.5f}")

    # Structured Results JSON
    results = {
        "candidate": "Candidate_V2.5_C_H9",
        "artifact": str(CANDIDATE_V25_PATH),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "ensemble_weights": {
            "CatBoost_Config_4": w_cb4,
            "CatBoost_Config_3": w_cb3,
            "XGBoost_Config_3A_TargetEncoded": w_xgb,
            "LightGBM_C2_TargetEncoded": w_lgb,
        },
        "component_oof_scores": {
            "CatBoost_Config_4": 0.76080,
            "CatBoost_Config_3": 0.76072,
            "XGBoost_Config_3A_TargetEncoded": 0.76166,
            "LightGBM_C2_TargetEncoded": 0.76183,
        },
        "oof_scores": {
            "v23_benchmark_rank": 0.76265,
            "v23_benchmark_nested": 0.76260,
            "v25c_rank_blend": 0.76302,
            "v25c_nested_cv_exploratory": 0.76300,
            "delta_vs_v23_rank": 0.00037,
            "delta_vs_v23_nested": 0.00040,
        },
        "public_leaderboard_benchmarks": {
            "v1_confirmed": 0.76456,
            "v2_confirmed": 0.76482,
            "v23_confirmed": 0.76535,
        },
        "test_distribution": {
            "min": float(p_test_v25.min()),
            "max": float(p_test_v25.max()),
            "mean": float(p_test_v25.mean()),
            "median": float(np.median(p_test_v25)),
            "std": float(p_test_v25.std()),
        },
        "correlations": {
            "spearman_vs_v2": float(rho_v2),
            "spearman_vs_v22": float(rho_v22),
            "spearman_vs_v23": float(rho_v23),
        },
        "total_runtime_seconds": time.time() - t_start,
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [+] Saved structured metrics to: {RESULTS_JSON_PATH}")
    print(f"\n[+] Stage H9 Candidate Generation Complete in {time.time() - t_start:.1f}s.")

if __name__ == "__main__":
    main()
