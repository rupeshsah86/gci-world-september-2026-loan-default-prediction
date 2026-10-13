"""
Stage H6: Advanced ROC-AUC Improvement and Model Optimization Pipeline.
Project: GCI World 2026 September Home Credit Default Risk Prediction.

Summary of Stage H6 Improvements:
1. Discovered and validated regularized LightGBM C2 configuration:
   - max_depth: 6, num_leaves: 20, min_child_samples: 120
   - feature_fraction: 0.65, bagging_fraction: 0.75, bagging_freq: 1
   - reg_alpha: 2.0, reg_lambda: 10.0, learning_rate: 0.025
   - Single Model OOF ROC-AUC: 0.75970 (Gain: +0.00070 over baseline 0.75900)
   - Improved on 4 out of 5 folds.
2. Upgraded Quad-Model Ensemble:
   - 27% CatBoost Config 4 (62 feats, OOF: 0.76080)
   - 23% CatBoost Config 3 (62 feats, OOF: 0.76072)
   - 30% XGBoost Config 3 (46 feats, OOF: 0.76015)
   - 20% Upgraded LightGBM C2 (46 feats, OOF: 0.75970)
   - Upgraded Probability Blend OOF ROC-AUC: 0.76228 (+0.00004 vs V2 baseline 0.76224)
   - Upgraded Rank-Normalized Blend OOF ROC-AUC: 0.76235 (+0.00005 vs V2.1 baseline 0.76230)

Outputs:
- OOF predictions: experiments/stage_h6/oof_lightgbm_c2.csv
- Structured metrics: experiments/stage_h6/stage_h6_results.json
- Candidate V2.2: outputs/submission_candidate_20261013_v22_h6.csv
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
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS

STAGE_H6_DIR = PROJECT_ROOT / "experiments" / "stage_h6"
STAGE_H6_DIR.mkdir(parents=True, exist_ok=True)

OOF_LGB_C2_PATH = STAGE_H6_DIR / "oof_lightgbm_c2.csv"
RESULTS_JSON_PATH = STAGE_H6_DIR / "stage_h6_results.json"
CANDIDATE_V22_PATH = PROJECT_ROOT / "outputs" / "submission_candidate_20261013_v22_h6.csv"
TEST_PREDS_PATH = PROJECT_ROOT / "experiments" / "test_preds_quad_models.npz"


def main():
    total_start = time.time()
    print("=" * 80)
    print("STAGE H6: ADVANCED MODEL OPTIMIZATION & CANDIDATE V2.2 GENERATION")
    print("=" * 80)

    # 1. Load Datasets
    print("\n[Step 1/6] Loading competition raw datasets...")
    train_df, test_df, sample_sub = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_train_raw = train_df.drop(columns=[TARGET_COL, ID_COL])
    X_test_raw = test_df.drop(columns=[ID_COL])
    n_train = len(train_df)
    n_test = len(test_df)
    assert n_train == 171202, f"Expected 171,202 train rows, got {n_train}"
    assert n_test == 61500, f"Expected 61,500 test rows, got {n_test}"
    print(f"[+] Loaded {n_train:,} training samples and {n_test:,} test samples.")

    # 2. Load Existing Verified Component OOF Predictions
    print("\n[Step 2/6] Loading verified existing component OOF predictions...")
    p_cb4 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_4_High_L2_Regularization.csv")["oof_pred"].values
    p_cb3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_3_RandomStrength_2_LR04.csv")["oof_pred"].values
    p_xgb3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_XGB_Config_3_Depth6_Colsample06_Reg10.csv")["oof_pred"].values
    p_lgb_orig = pd.read_csv(PROJECT_ROOT / "experiments" / "oof_tuned_lightgbm.csv")["oof_best_tuned_lgb"].values

    v2_base_prob = 0.27 * p_cb4 + 0.23 * p_cb3 + 0.30 * p_xgb3 + 0.20 * p_lgb_orig
    r_cb4 = rankdata(p_cb4, method="average") / n_train
    r_cb3 = rankdata(p_cb3, method="average") / n_train
    r_xgb3 = rankdata(p_xgb3, method="average") / n_train
    r_lgb_orig = rankdata(p_lgb_orig, method="average") / n_train
    v21_base_rank = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb3 + 0.20 * r_lgb_orig

    auc_base_prob = float(roc_auc_score(y_true, v2_base_prob))
    auc_base_rank = float(roc_auc_score(y_true, v21_base_rank))
    print(f"  [+] Confirmed Baseline V2 Prob OOF ROC-AUC:  {auc_base_prob:.5f}")
    print(f"  [+] Confirmed Baseline V2.1 Rank OOF ROC-AUC: {auc_base_rank:.5f}")

    # 3. 5-Fold Cross-Validation of Upgraded LightGBM C2
    print("\n[Step 3/6] Generating 5-Fold OOF Predictions for Upgraded LightGBM C2...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(X_train_raw, y_true))

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
    }

    oof_lgb_c2 = np.zeros(n_train)
    fold_scores_c2 = []
    best_iters_c2 = []
    t_lgb_start = time.time()

    for fold_idx, (tr_idx, val_idx) in enumerate(folds, 1):
        pipe_fold = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=True)
        pipe_fold.fit(X_train_raw.iloc[tr_idx])
        X_tr_f = pipe_fold.transform(X_train_raw.iloc[tr_idx])
        X_va_f = pipe_fold.transform(X_train_raw.iloc[val_idx])
        cat_cols_f = [c for c in CATEGORICAL_COLS if c in pipe_fold.feature_names]

        dtrain_f = lgb.Dataset(X_tr_f, label=y_true[tr_idx], categorical_feature=cat_cols_f, free_raw_data=False)
        dval_f = lgb.Dataset(X_va_f, label=y_true[val_idx], reference=dtrain_f, categorical_feature=cat_cols_f, free_raw_data=False)

        m = lgb.train(
            {**params_lgb_c2, "random_state": 42 + fold_idx},
            dtrain_f,
            num_boost_round=1500,
            valid_sets=[dtrain_f, dval_f],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        preds_va = m.predict(X_va_f, num_iteration=m.best_iteration)
        oof_lgb_c2[val_idx] = preds_va
        score = float(roc_auc_score(y_true[val_idx], preds_va))
        fold_scores_c2.append(score)
        best_iters_c2.append(int(m.best_iteration))
        print(f"  Fold {fold_idx}: AUC = {score:.5f} (Best Iter: {m.best_iteration})")

    auc_lgb_c2 = float(roc_auc_score(y_true, oof_lgb_c2))
    mean_iter_c2 = int(round(np.mean(best_iters_c2)))
    print(f"  [+] Upgraded LightGBM C2 OOF ROC-AUC: {auc_lgb_c2:.5f} (Mean: {np.mean(fold_scores_c2):.5f} +/- {np.std(fold_scores_c2):.5f})")
    print(f"      Delta vs Baseline Tuned LightGBM (0.75900): {auc_lgb_c2 - 0.75900:+.5f}")
    print(f"      Mean Early Stopping Iterations: {mean_iter_c2}")

    # Save OOF predictions to disk
    df_oof_c2 = pd.DataFrame({
        ID_COL: train_df[ID_COL].values,
        TARGET_COL: y_true,
        "oof_pred": oof_lgb_c2,
    })
    df_oof_c2.to_csv(OOF_LGB_C2_PATH, index=False)
    print(f"  [+] Saved LightGBM C2 OOF predictions to: {OOF_LGB_C2_PATH}")

    # 4. Ensemble Evaluation: Candidate V2.2
    print("\n[Step 4/6] Evaluating Upgraded Quad-Model Ensemble (Candidate V2.2)...")
    r_lgb_c2 = rankdata(oof_lgb_c2, method="average") / n_train

    # A. Upgraded Probability Blend
    upgraded_prob_blend = (
        0.27 * p_cb4
        + 0.23 * p_cb3
        + 0.30 * p_xgb3
        + 0.20 * oof_lgb_c2
    )
    auc_upgraded_prob = float(roc_auc_score(y_true, upgraded_prob_blend))

    # B. Upgraded Rank-Normalized Blend
    upgraded_rank_blend = (
        0.27 * r_cb4
        + 0.23 * r_cb3
        + 0.30 * r_xgb3
        + 0.20 * r_lgb_c2
    )
    auc_upgraded_rank = float(roc_auc_score(y_true, upgraded_rank_blend))

    print(f"  --> Upgraded Probability Blend OOF ROC-AUC:      {auc_upgraded_prob:.5f} (Delta vs V2:   {auc_upgraded_prob - auc_base_prob:+.5f})")
    print(f"  --> Upgraded Rank-Normalized Blend OOF ROC-AUC:  {auc_upgraded_rank:.5f} (Delta vs V2.1: {auc_upgraded_rank - auc_base_rank:+.5f})")

    fold_scores_upgraded_rank = []
    fold_scores_base_rank = []
    for _, val_idx in folds:
        fold_scores_upgraded_rank.append(float(roc_auc_score(y_true[val_idx], upgraded_rank_blend[val_idx])))
        fold_scores_base_rank.append(float(roc_auc_score(y_true[val_idx], v21_base_rank[val_idx])))

    print(f"\n  Fold Comparison (Rank Blend):")
    for idx, (b, u) in enumerate(zip(fold_scores_base_rank, fold_scores_upgraded_rank), 1):
        print(f"    Fold {idx}: Baseline {b:.5f} -> Upgraded {u:.5f} (Delta: {u - b:+.5f})")

    # 5. Full-Train Inference for LightGBM C2 & Test Ensemble Combination
    print("\n[Step 5/6] Training LightGBM C2 on 100% Training Data & Generating Test Predictions...")
    pipe_full = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=True)
    pipe_full.fit(X_train_raw)
    X_tr_full = pipe_full.transform(X_train_raw)
    X_te_full = pipe_full.transform(X_test_raw)
    cat_cols_full = [c for c in CATEGORICAL_COLS if c in pipe_full.feature_names]

    dtrain_full = lgb.Dataset(X_tr_full, label=y_true, categorical_feature=cat_cols_full, free_raw_data=False)
    m_full = lgb.train(
        {**params_lgb_c2, "random_state": 42},
        dtrain_full,
        num_boost_round=mean_iter_c2,
    )
    p_test_lgb_c2 = m_full.predict(X_te_full)
    print(f"  [+] LightGBM C2 full test predictions generated. Mean: {p_test_lgb_c2.mean():.4f}")

    # Load component test predictions from experiments/test_preds_quad_models.npz
    assert TEST_PREDS_PATH.exists(), f"Missing component test predictions: {TEST_PREDS_PATH}"
    npz_data = np.load(TEST_PREDS_PATH)
    p_test_cb4 = npz_data["test_cb4"]
    p_test_cb3 = npz_data["test_cb3"]
    p_test_xgb = npz_data["test_xgb"]

    # Percentile-rank normalization on test predictions
    r_test_cb4 = rankdata(p_test_cb4, method="average") / n_test
    r_test_cb3 = rankdata(p_test_cb3, method="average") / n_test
    r_test_xgb = rankdata(p_test_xgb, method="average") / n_test
    r_test_lgb_c2 = rankdata(p_test_lgb_c2, method="average") / n_test

    p_test_candidate_v22 = (
        0.27 * r_test_cb4
        + 0.23 * r_test_cb3
        + 0.30 * r_test_xgb
        + 0.20 * r_test_lgb_c2
    )

    # 6. Save and Audit Candidate V2.2
    print("\n[Step 6/6] Saving and Auditing Candidate V2.2...")
    sub_df = pd.DataFrame({
        ID_COL: test_df[ID_COL].values,
        TARGET_COL: p_test_candidate_v22,
    })
    sub_df.to_csv(CANDIDATE_V22_PATH, index=False)
    print(f"[+] Saved Candidate V2.2 submission to: {CANDIDATE_V22_PATH}")

    # Audits
    assert CANDIDATE_V22_PATH.exists()
    loaded_sub = pd.read_csv(CANDIDATE_V22_PATH)
    assert loaded_sub.shape == (61500, 2), f"Expected shape (61500, 2), got {loaded_sub.shape}"
    assert list(loaded_sub.columns) == [ID_COL, TARGET_COL]
    assert (loaded_sub[ID_COL].values == test_df[ID_COL].values).all()
    assert loaded_sub[ID_COL].nunique() == 61500
    assert not loaded_sub[TARGET_COL].isna().any()
    assert not np.isinf(loaded_sub[TARGET_COL].values).any()
    assert (loaded_sub[TARGET_COL].values >= 0.0).all() and (loaded_sub[TARGET_COL].values <= 1.0).all()

    # Compare with sample submission format
    assert (loaded_sub[ID_COL].values == sample_sub[ID_COL].values).all()
    print("  [AUDIT PASSED] Shape, columns, ID order, finiteness, and bounds strictly verified.")

    # Save structured results JSON
    summary_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "baseline_scores": {
            "v2_prob_blend_oof": round(auc_base_prob, 5),
            "v21_rank_blend_oof": round(auc_base_rank, 5),
            "v2_public_leaderboard": 0.76482,
        },
        "lightgbm_c2_results": {
            "name": "LightGBM_C2_Regularized_Depth6",
            "hyperparameters": params_lgb_c2,
            "mean_iterations": mean_iter_c2,
            "oof_auc": round(auc_lgb_c2, 5),
            "fold_scores": [round(s, 5) for s in fold_scores_c2],
            "mean_fold_auc": round(float(np.mean(fold_scores_c2)), 5),
            "std_fold_auc": round(float(np.std(fold_scores_c2)), 5),
            "delta_vs_baseline_lgb": round(auc_lgb_c2 - 0.75900, 5),
        },
        "candidate_v22_ensemble": {
            "artifact": str(CANDIDATE_V22_PATH),
            "weights": {"cb4": 0.27, "cb3": 0.23, "xgb3": 0.30, "lgb_c2": 0.20},
            "prob_blend_oof_auc": round(auc_upgraded_prob, 5),
            "rank_blend_oof_auc": round(auc_upgraded_rank, 5),
            "delta_prob_vs_v2": round(auc_upgraded_prob - auc_base_prob, 5),
            "delta_rank_vs_v21": round(auc_upgraded_rank - auc_base_rank, 5),
            "rank_fold_scores": [round(s, 5) for s in fold_scores_upgraded_rank],
            "rank_mean_fold": round(float(np.mean(fold_scores_upgraded_rank)), 5),
            "rank_std_fold": round(float(np.std(fold_scores_upgraded_rank)), 5),
        },
        "test_distribution_summary": {
            "min": round(float(p_test_candidate_v22.min()), 6),
            "max": round(float(p_test_candidate_v22.max()), 6),
            "mean": round(float(p_test_candidate_v22.mean()), 6),
            "median": round(float(np.median(p_test_candidate_v22)), 6),
            "std": round(float(p_test_candidate_v22.std()), 6),
        },
        "total_runtime_seconds": round(time.time() - total_start, 1),
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"[+] Saved structured experiment results to: {RESULTS_JSON_PATH}")

    total_time = time.time() - total_start
    print(f"\n[+] Stage H6 complete in {total_time:.1f}s (~{total_time/60:.1f} minutes).")


if __name__ == "__main__":
    main()
