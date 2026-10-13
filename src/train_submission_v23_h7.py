"""
Stage H7: Full-Data Training and Submission Candidate V2.3 Generation.
Project: GCI World 2026 September Home Credit Default Risk Prediction.

Candidate V2.3 Ensemble Composition:
- 27% CatBoost Config 4 (62 features, OOF ROC-AUC: 0.76080)
- 23% CatBoost Config 3 (62 features, OOF ROC-AUC: 0.76072)
- 30% Upgraded XGBoost Config 3A (46 features, OOF ROC-AUC: 0.76088)
- 20% Upgraded LightGBM C2 (46 features, OOF ROC-AUC: 0.75970)

Ensemble Metrics:
- 5-Fold OOF Rank-Normalized ROC-AUC: 0.76265 (+0.00030 vs V2.2 0.76235; +0.00035 vs V2.1 0.76230)
- 5-Fold OOF Probability Blend ROC-AUC: 0.76258 (+0.00034 vs V2 benchmark 0.76224)
- Unbiased Nested 5-Fold CV ROC-AUC: 0.76263

Output:
- outputs/submission_candidate_20261013_v23_h7.csv
"""

import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS

STAGE_H7_DIR = PROJECT_ROOT / "experiments" / "stage_h7"
STAGE_H7_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATE_V23_PATH = PROJECT_ROOT / "outputs" / "submission_candidate_20261013_v23_h7.csv"
RESULTS_JSON_PATH = STAGE_H7_DIR / "stage_h7_results.json"
TEST_PREDS_PATH = PROJECT_ROOT / "experiments" / "test_preds_quad_models.npz"

def main():
    t_start = time.time()
    print("=" * 80)
    print("STAGE H7: TRAINING FULL-DATA PIPELINE & GENERATING CANDIDATE V2.3")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1/6] Loading competition raw datasets...")
    train_df, test_df, sample_sub = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_train_raw = train_df.drop(columns=[TARGET_COL, ID_COL])
    X_test_raw = test_df.drop(columns=[ID_COL])
    n_train = len(train_df)
    n_test = len(test_df)
    assert n_train == 171202, f"Expected 171,202 train rows, got {n_train}"
    assert n_test == 61500, f"Expected 61,500 test rows, got {n_test}"
    print(f"  [+] Loaded {n_train:,} training samples and {n_test:,} test samples.")

    # 2. Train Full-Data Upgraded XGBoost Config 3A (Mean best iterations = 700)
    print("\n[Step 2/6] Preprocessing and training full-data XGBoost Config 3A...")
    pipe_xgb = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
    pipe_xgb.fit(X_train_raw)
    X_tr_xgb = pipe_xgb.transform(X_train_raw)
    X_te_xgb = pipe_xgb.transform(X_test_raw)

    for col in CATEGORICAL_COLS:
        if col in X_tr_xgb.columns:
            cats = sorted(X_tr_xgb[col].dropna().unique().tolist())
            dt = pd.CategoricalDtype(categories=cats, ordered=False)
            X_tr_xgb[col] = X_tr_xgb[col].astype(dt)
            X_te_xgb[col] = X_te_xgb[col].where(X_te_xgb[col].isin(cats), np.nan).astype(dt)

    params_xgb_c3a = {
        "n_estimators": 700,
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
    p_test_xgb_3a = clf_xgb.predict_proba(X_te_xgb)[:, 1]
    print(f"  [+] XGBoost Config 3A completed in {time.time() - t0:.1f}s. Test mean: {p_test_xgb_3a.mean():.4f}")

    # 3. Train Full-Data LightGBM C2 (Mean best iterations = 748)
    print("\n[Step 3/6] Preprocessing and training full-data LightGBM C2...")
    pipe_lgb = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=True)
    pipe_lgb.fit(X_train_raw)
    X_tr_lgb = pipe_lgb.transform(X_train_raw)
    X_te_lgb = pipe_lgb.transform(X_test_raw)
    cat_cols_lgb = [c for c in CATEGORICAL_COLS if c in pipe_lgb.feature_names]

    dtrain_lgb = lgb.Dataset(X_tr_lgb, label=y_true, categorical_feature=cat_cols_lgb, free_raw_data=False)
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
        "random_state": 42,
    }

    t0 = time.time()
    model_lgb = lgb.train(params_lgb_c2, dtrain_lgb, num_boost_round=748)
    p_test_lgb_c2 = model_lgb.predict(X_te_lgb)
    print(f"  [+] LightGBM C2 completed in {time.time() - t0:.1f}s. Test mean: {p_test_lgb_c2.mean():.4f}")

    # 4. Load Verified CatBoost Test Predictions from Cache
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
    r_test_xgb3a = rankdata(p_test_xgb_3a, method="average") / n_test
    r_test_lgb_c2 = rankdata(p_test_lgb_c2, method="average") / n_test

    p_test_v23 = (
        0.27 * r_test_cb4
        + 0.23 * r_test_cb3
        + 0.30 * r_test_xgb3a
        + 0.20 * r_test_lgb_c2
    )

    # 6. Audit and Save Candidate V2.3
    print("\n[Step 6/6] Auditing and Saving Candidate V2.3...")
    sub_df = pd.DataFrame({
        ID_COL: test_df[ID_COL].values,
        TARGET_COL: p_test_v23,
    })

    # Verification checks
    assert len(sub_df) == 61500, f"Expected 61,500 rows, got {len(sub_df)}"
    assert list(sub_df.columns) == ["SK_ID_CURR", "TARGET"], f"Unexpected columns: {list(sub_df.columns)}"
    assert (sub_df["SK_ID_CURR"].values == test_df["SK_ID_CURR"].values).all(), "ID alignment failed vs test.csv"
    assert (sub_df["SK_ID_CURR"].values == sample_sub["SK_ID_CURR"].values).all(), "ID alignment failed vs sample_submission.csv"
    assert not sub_df["TARGET"].isna().any(), "Target contains NaNs"
    assert not np.isinf(sub_df["TARGET"].values).any(), "Target contains Infs"
    assert (sub_df["TARGET"] >= 0.0).all() and (sub_df["TARGET"] <= 1.0).all(), "Predictions outside [0, 1]"
    assert sub_df["TARGET"].nunique() > 1000, "Hard class predictions detected instead of probabilities"

    # Save to disk
    sub_df.to_csv(CANDIDATE_V23_PATH, index=False)
    print(f"  [+] Successfully saved validated Candidate V2.3 to: {CANDIDATE_V23_PATH}")

    # Compare correlation with Candidate V2, V2.1, V2.2
    v2_preds = pd.read_csv(PROJECT_ROOT / "outputs" / "submission_candidate_20261012_v2_quad_blend.csv")["TARGET"].values
    v21_preds = pd.read_csv(PROJECT_ROOT / "outputs" / "submission_candidate_20261013_v21_rank_blend.csv")["TARGET"].values
    v22_preds = pd.read_csv(PROJECT_ROOT / "outputs" / "submission_candidate_20261013_v22_h6.csv")["TARGET"].values

    rho_v2, _ = spearmanr(p_test_v23, v2_preds)
    rho_v21, _ = spearmanr(p_test_v23, v21_preds)
    rho_v22, _ = spearmanr(p_test_v23, v22_preds)
    print(f"  [+] Spearman Correlation vs V2:   {rho_v2:.5f}")
    print(f"  [+] Spearman Correlation vs V2.1: {rho_v21:.5f}")
    print(f"  [+] Spearman Correlation vs V2.2: {rho_v22:.5f}")

    # Save Stage H7 Structured Metrics JSON
    results = {
        "candidate": "Candidate_V2.3_H7",
        "artifact": str(CANDIDATE_V23_PATH),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "ensemble_weights": {
            "CatBoost_Config_4": 0.27,
            "CatBoost_Config_3": 0.23,
            "XGBoost_Config_3A": 0.30,
            "LightGBM_C2": 0.20,
        },
        "component_oof_scores": {
            "CatBoost_Config_4": 0.76080,
            "CatBoost_Config_3": 0.76072,
            "XGBoost_Config_3A": 0.76088,
            "LightGBM_C2": 0.75970,
        },
        "oof_scores": {
            "v2_prob_benchmark": 0.76224,
            "v21_rank_blend": 0.76230,
            "v22_rank_blend": 0.76235,
            "v23_prob_blend": 0.76258,
            "v23_rank_blend": 0.76265,
            "v23_nested_cv_unbiased": 0.76263,
            "delta_vs_v2_prob": 0.00034,
            "delta_vs_v22_rank": 0.00030,
        },
        "public_leaderboard_benchmarks": {
            "v1_confirmed": 0.76456,
            "v2_confirmed": 0.76482,
            "latest_displayed": 0.76506,
        },
        "test_distribution": {
            "min": float(p_test_v23.min()),
            "max": float(p_test_v23.max()),
            "mean": float(p_test_v23.mean()),
            "median": float(np.median(p_test_v23)),
            "std": float(p_test_v23.std()),
        },
        "correlations": {
            "spearman_vs_v2": float(rho_v2),
            "spearman_vs_v21": float(rho_v21),
            "spearman_vs_v22": float(rho_v22),
        },
        "total_runtime_seconds": time.time() - t_start,
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [+] Saved structured metrics to: {RESULTS_JSON_PATH}")
    print(f"\n[+] Stage H7 Complete in {time.time() - t_start:.1f}s.")

if __name__ == "__main__":
    main()
