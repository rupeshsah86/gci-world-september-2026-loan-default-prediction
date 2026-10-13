"""
Stage H9: Task 3-6 - Comprehensive Hybrid Ensemble Exploration and Nested CV Evaluation.
Investigates complementary error diversity between single-seed and bagged models,
tests bounded hybrid blend weights, and validates unbiased generalization via 5-Fold Nested CV.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL

STAGE_H9_DIR = PROJECT_ROOT / "experiments" / "stage_h9"
STAGE_H9_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = STAGE_H9_DIR / "stage_h9_hybrid_results.json"

def optimize_weights_scipy(preds_list, y_true, bounds=None, use_rank=True):
    n = len(y_true)
    if use_rank:
        features = np.column_stack([rankdata(p) / n for p in preds_list])
    else:
        features = np.column_stack([
            (p - np.min(p)) / (np.max(p) - np.min(p) + 1e-12) for p in preds_list
        ])

    k = features.shape[1]
    if bounds is None:
        bounds = [(0.02, 0.50) for _ in range(k)]

    def loss(w):
        w_norm = np.array(w) / np.sum(w)
        blend = features @ w_norm
        return -roc_auc_score(y_true, blend)

    init_w = np.ones(k) / k
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(loss, init_w, bounds=bounds, constraints=constraints, method="SLSQP")
    opt_w = res.x / np.sum(res.x)
    opt_auc = -res.fun
    return opt_w, opt_auc

def nested_cv_weights(preds_list, y_true, folds, bounds=None, use_rank=True):
    n = len(y_true)
    outer_oof_blend = np.zeros(n)
    fold_weights = []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        y_tr, y_va = y_true[tr_idx], y_true[va_idx]
        tr_preds = [p[tr_idx] for p in preds_list]
        va_preds = [p[va_idx] for p in preds_list]

        opt_w, _ = optimize_weights_scipy(tr_preds, y_tr, bounds=bounds, use_rank=use_rank)
        fold_weights.append(opt_w.tolist())

        if use_rank:
            n_va = len(va_idx)
            va_mat = np.column_stack([rankdata(p) / n_va for p in va_preds])
        else:
            va_mat = np.column_stack([
                (p - np.min(p)) / (np.max(p) - np.min(p) + 1e-12) for p in va_preds
            ])

        outer_oof_blend[va_idx] = va_mat @ opt_w

    nested_auc = float(roc_auc_score(y_true, outer_oof_blend))
    return nested_auc, fold_weights, outer_oof_blend

def main():
    print("=" * 80)
    print("STAGE H9: TARGETED HYBRID ENSEMBLE INVESTIGATION")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    n = len(y_true)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(train_df, y_true))

    # Load verified OOF predictions
    p_cb4 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_4_High_L2_Regularization.csv")["oof_pred"].values
    p_cb3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_3_RandomStrength_2_LR04.csv")["oof_pred"].values
    p_cb6 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h7" / "oof_catboost_config6.csv")["oof_pred"].values

    p_xgb_s = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h7" / "oof_xgboost_c3a.csv")["oof_pred"].values
    p_xgb_b = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h8" / "oof_xgboost_c3a_bagged_3seed.csv")["oof_pred"].values

    p_lgb_s = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h6" / "oof_lightgbm_c2.csv")["oof_pred"].values
    p_lgb_b = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h8" / "oof_lightgbm_c2_bagged_3seed.csv")["oof_pred"].values

    # Percentile ranks
    r_cb4 = rankdata(p_cb4) / n
    r_cb3 = rankdata(p_cb3) / n
    r_cb6 = rankdata(p_cb6) / n
    r_xgb_s = rankdata(p_xgb_s) / n
    r_xgb_b = rankdata(p_xgb_b) / n
    r_lgb_s = rankdata(p_lgb_s) / n
    r_lgb_b = rankdata(p_lgb_b) / n

    # Min-max normalized probabilities
    def minmax(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-12)

    norm_cb4 = minmax(p_cb4)
    norm_cb3 = minmax(p_cb3)
    norm_xgb_s = minmax(p_xgb_s)
    norm_xgb_b = minmax(p_xgb_b)
    norm_lgb_s = minmax(p_lgb_s)
    norm_lgb_b = minmax(p_lgb_b)

    # 1. Error Residual Correlation Analysis
    print("\n[Step 1] Error Residual Correlation Analysis:")
    # Residual error = abs(rank_pred - y_true)
    res_cb4 = r_cb4 - y_true
    res_cb3 = r_cb3 - y_true
    res_xgb_s = r_xgb_s - y_true
    res_xgb_b = r_xgb_b - y_true
    res_lgb_s = r_lgb_s - y_true
    res_lgb_b = r_lgb_b - y_true

    res_df = pd.DataFrame({
        "CB4": res_cb4,
        "CB3": res_cb3,
        "XGB_Single": res_xgb_s,
        "XGB_Bagged": res_xgb_b,
        "LGB_Single": res_lgb_s,
        "LGB_Bagged": res_lgb_b,
    })
    res_corr = res_df.corr(method="spearman")
    print(res_corr.round(4).to_string())

    # 2. Candidate V2.3 Benchmark
    v23_rank = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb_s + 0.20 * r_lgb_s
    v23_prob = 0.27 * norm_cb4 + 0.23 * norm_cb3 + 0.30 * norm_xgb_s + 0.20 * norm_lgb_s
    v23_rank_auc = roc_auc_score(y_true, v23_rank)
    v23_prob_auc = roc_auc_score(y_true, v23_prob)

    print("\n[Step 2] Baseline Candidate V2.3 Scores:")
    print(f"  V2.3 Rank Blend OOF AUC: {v23_rank_auc:.5f} (LB: 0.76535)")
    print(f"  V2.3 Prob Blend OOF AUC: {v23_prob_auc:.5f}")

    # 3. Systematic Bounded Grid Search on XGBoost Weight Partitioning
    print("\n[Step 3] XGBoost Single vs Bagged Weight Partitioning (Holding CB4=0.27, CB3=0.23, LGB=0.20):")
    xgb_grid_results = []
    for w_bagged in np.linspace(0.0, 0.30, 13):
        w_single = round(0.30 - w_bagged, 4)
        w_bagged = round(w_bagged, 4)

        blend_rank = 0.27 * r_cb4 + 0.23 * r_cb3 + w_single * r_xgb_s + w_bagged * r_xgb_b + 0.20 * r_lgb_s
        blend_prob = 0.27 * norm_cb4 + 0.23 * norm_cb3 + w_single * norm_xgb_s + w_bagged * norm_xgb_b + 0.20 * norm_lgb_s

        score_rank = roc_auc_score(y_true, blend_rank)
        score_prob = roc_auc_score(y_true, blend_prob)
        delta_rank = score_rank - v23_rank_auc

        xgb_grid_results.append({
            "w_xgb_single": w_single,
            "w_xgb_bagged": w_bagged,
            "rank_auc": score_rank,
            "prob_auc": score_prob,
            "delta_vs_v23": delta_rank,
        })
        print(f"  XGB Single: {w_single:5.3f} | Bagged: {w_bagged:5.3f} -> Rank AUC: {score_rank:.5f} ({delta_rank:+.6f}) | Prob AUC: {score_prob:.5f}")

    # 4. LightGBM Single vs Bagged Partitioning
    print("\n[Step 4] LightGBM Single vs Bagged Partitioning (with optimal XGB split):")
    lgb_grid_results = []
    # Test with 50/50 XGB split (0.15 / 0.15)
    for w_lgb_b in [0.0, 0.05, 0.10, 0.15, 0.20]:
        w_lgb_s = round(0.20 - w_lgb_b, 4)
        blend_rank = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.15 * r_xgb_s + 0.15 * r_xgb_b + w_lgb_s * r_lgb_s + w_lgb_b * r_lgb_b
        score_rank = roc_auc_score(y_true, blend_rank)
        delta_rank = score_rank - v23_rank_auc
        lgb_grid_results.append({
            "w_lgb_single": w_lgb_s,
            "w_lgb_bagged": w_lgb_b,
            "rank_auc": score_rank,
            "delta": delta_rank,
        })
        print(f"  LGB Single: {w_lgb_s:5.3f} | Bagged: {w_lgb_b:5.3f} -> Rank AUC: {score_rank:.5f} ({delta_rank:+.6f})")

    # 5. Nested 5-Fold Cross-Validation (Unbiased Generalization Test)
    print("\n[Step 5] Rigorous Nested 5-Fold Cross-Validation on Hybrid Model Pool:")
    pool_preds = [p_cb4, p_cb3, p_xgb_s, p_xgb_b, p_lgb_s, p_lgb_b]
    pool_names = ["CB4", "CB3", "XGB_Single", "XGB_Bagged", "LGB_Single", "LGB_Bagged"]

    bounds_pool = [
        (0.15, 0.35),  # CB4
        (0.10, 0.30),  # CB3
        (0.05, 0.25),  # XGB_Single
        (0.05, 0.25),  # XGB_Bagged
        (0.05, 0.20),  # LGB_Single
        (0.00, 0.15),  # LGB_Bagged
    ]

    opt_w_rank, in_sample_auc_rank = optimize_weights_scipy(pool_preds, y_true, bounds=bounds_pool, use_rank=True)
    opt_w_prob, in_sample_auc_prob = optimize_weights_scipy(pool_preds, y_true, bounds=bounds_pool, use_rank=False)

    print("  In-Sample Fitted Weights (Rank):", {name: round(float(w), 3) for name, w in zip(pool_names, opt_w_rank)})
    print(f"  In-Sample Fitted Rank AUC:       {in_sample_auc_rank:.5f}")

    nested_auc_rank, nested_weights_rank, oof_nested_rank = nested_cv_weights(pool_preds, y_true, folds, bounds=bounds_pool, use_rank=True)
    nested_auc_prob, nested_weights_prob, oof_nested_prob = nested_cv_weights(pool_preds, y_true, folds, bounds=bounds_pool, use_rank=False)

    print(f"  Nested CV Rank AUC (Unbiased):   {nested_auc_rank:.5f}")
    print(f"  Nested CV Prob AUC (Unbiased):   {nested_auc_prob:.5f}")

    # Check fold-level performance of nested CV
    nested_fold_scores = [
        roc_auc_score(y_true[va], oof_nested_rank[va]) for tr, va in folds
    ]
    print(f"  Nested CV Fold Scores: {[round(s, 5) for s in nested_fold_scores]}")
    print(f"  Nested CV Fold Mean +/- Std: {np.mean(nested_fold_scores):.5f} +/- {np.std(nested_fold_scores):.5f}")

    # Save structured results
    summary = {
        "candidate_v23_baseline": {
            "rank_auc": v23_rank_auc,
            "prob_auc": v23_prob_auc,
            "public_leaderboard": 0.76535,
        },
        "residual_correlation_matrix": res_corr.to_dict(),
        "xgb_partition_grid": xgb_grid_results,
        "lgb_partition_grid": lgb_grid_results,
        "nested_cv_results": {
            "nested_auc_rank": nested_auc_rank,
            "nested_auc_prob": nested_auc_prob,
            "in_sample_auc_rank": in_sample_auc_rank,
            "in_sample_auc_prob": in_sample_auc_prob,
            "fold_scores": nested_fold_scores,
            "mean_fold_auc": float(np.mean(nested_fold_scores)),
            "std_fold_auc": float(np.std(nested_fold_scores)),
            "nested_weights": nested_weights_rank,
            "delta_vs_v23_nested": nested_auc_rank - 0.76263,
        },
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Saved Stage H9 hybrid results to: {RESULTS_JSON}")

if __name__ == "__main__":
    main()
