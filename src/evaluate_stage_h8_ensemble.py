"""
Stage H8: Comprehensive Ensemble Optimization and Nested Cross-Validation Engine.
Evaluates 3-seed bagged XGBoost 3A (0.76128) + 3-seed bagged LightGBM C2 (0.75983)
combined with verified CatBoost models under probability, rank, and power transforms.
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

STAGE_H8_DIR = PROJECT_ROOT / "experiments" / "stage_h8"
STAGE_H7_DIR = PROJECT_ROOT / "experiments" / "stage_h7"
RESULTS_JSON = STAGE_H8_DIR / "stage_h8_ensemble_results.json"

def optimize_weights_scipy(preds_list, y_true, use_rank=True):
    n = len(y_true)
    if use_rank:
        features = np.column_stack([rankdata(p) / n for p in preds_list])
    else:
        features = np.column_stack(preds_list)

    k = features.shape[1]
    def loss(w):
        w = np.array(w)
        w = w / np.sum(w)
        blend = features @ w
        return -roc_auc_score(y_true, blend)

    init_w = np.ones(k) / k
    bounds = [(0.05, 0.50) for _ in range(k)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(loss, init_w, bounds=bounds, constraints=constraints, method="SLSQP")
    opt_w = res.x / np.sum(res.x)
    opt_auc = -res.fun
    return opt_w, opt_auc

def nested_cv_weights(preds_list, y_true, folds, use_rank=True):
    n = len(y_true)
    k = len(preds_list)
    outer_oof_blend = np.zeros(n)
    fold_weights = []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        y_tr, y_va = y_true[tr_idx], y_true[va_idx]
        tr_preds = [p[tr_idx] for p in preds_list]
        va_preds = [p[va_idx] for p in preds_list]

        opt_w, _ = optimize_weights_scipy(tr_preds, y_tr, use_rank=use_rank)
        fold_weights.append(opt_w.tolist())

        if use_rank:
            n_va = len(va_idx)
            va_mat = np.column_stack([rankdata(p) / n_va for p in va_preds])
        else:
            va_mat = np.column_stack(va_preds)

        outer_oof_blend[va_idx] = va_mat @ opt_w

    nested_auc = float(roc_auc_score(y_true, outer_oof_blend))
    return nested_auc, fold_weights, outer_oof_blend

def main():
    print("=" * 80)
    print("STAGE H8: ENSEMBLE BLENDING & NESTED CV ANALYSIS")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    n = len(y_true)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(train_df, y_true))

    # Load component predictions
    p_cb4 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_4_High_L2_Regularization.csv")["oof_pred"].values
    p_cb3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_3_RandomStrength_2_LR04.csv")["oof_pred"].values
    p_cb6 = pd.read_csv(STAGE_H7_DIR / "oof_catboost_config6.csv")["oof_pred"].values

    p_xgb3_orig = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_XGB_Config_3_Depth6_Colsample06_Reg10.csv")["oof_pred"].values
    p_xgb3a_single = pd.read_csv(STAGE_H7_DIR / "oof_xgboost_c3a.csv")["oof_pred"].values
    p_xgb3a_bagged = pd.read_csv(STAGE_H8_DIR / "oof_xgboost_c3a_bagged_3seed.csv")["oof_pred"].values

    p_lgb_orig = pd.read_csv(PROJECT_ROOT / "experiments" / "oof_tuned_lightgbm.csv")["oof_best_tuned_lgb"].values
    p_lgb_c2_single = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h6" / "oof_lightgbm_c2.csv")["oof_pred"].values
    p_lgb_c2_bagged = pd.read_csv(STAGE_H8_DIR / "oof_lightgbm_c2_bagged_3seed.csv")["oof_pred"].values

    print("\n[Step 1] Individual Model OOF ROC-AUC Summary:")
    all_models = {
        "CatBoost_Config_4": p_cb4,
        "CatBoost_Config_3": p_cb3,
        "CatBoost_Config_6": p_cb6,
        "XGBoost_Config_3_Orig": p_xgb3_orig,
        "XGBoost_Config_3A_Single": p_xgb3a_single,
        "XGBoost_Config_3A_Bagged_3Seed": p_xgb3a_bagged,
        "LightGBM_C1_Orig": p_lgb_orig,
        "LightGBM_C2_Single": p_lgb_c2_single,
        "LightGBM_C2_Bagged_3Seed": p_lgb_c2_bagged,
    }
    for name, p in all_models.items():
        print(f"  {name:32s}: {roc_auc_score(y_true, p):.5f}")

    # Standard ranks
    r_cb4 = rankdata(p_cb4) / n
    r_cb3 = rankdata(p_cb3) / n
    r_xgb3a_single = rankdata(p_xgb3a_single) / n
    r_xgb3a_bagged = rankdata(p_xgb3a_bagged) / n
    r_lgb_c2_single = rankdata(p_lgb_c2_single) / n
    r_lgb_c2_bagged = rankdata(p_lgb_c2_bagged) / n

    # Benchmark V2.3 Rank (0.76265)
    v23_rank = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb3a_single + 0.20 * r_lgb_c2_single
    v23_prob = 0.27 * p_cb4 + 0.23 * p_cb3 + 0.30 * p_xgb3a_single + 0.20 * p_lgb_c2_single

    # Stage H8 Upgraded Blend (Candidate V2.4) with Bagged Components
    v24_rank = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb3a_bagged + 0.20 * r_lgb_c2_bagged
    v24_prob = 0.27 * p_cb4 + 0.23 * p_cb3 + 0.30 * p_xgb3a_bagged + 0.20 * p_lgb_c2_bagged

    print("\n[Step 2] Candidate Benchmark Comparisons:")
    print(f"  Candidate V2.3 Prob Blend:          {roc_auc_score(y_true, v23_prob):.5f}")
    print(f"  Candidate V2.3 Rank Blend (LB .76535): {roc_auc_score(y_true, v23_rank):.5f}")
    print(f"  Candidate V2.4 Prob Blend (Bagged): {roc_auc_score(y_true, v24_prob):.5f} (+{roc_auc_score(y_true, v24_prob) - roc_auc_score(y_true, v23_prob):.5f})")
    print(f"  Candidate V2.4 Rank Blend (Bagged): {roc_auc_score(y_true, v24_rank):.5f} (+{roc_auc_score(y_true, v24_rank) - roc_auc_score(y_true, v23_rank):.5f})")

    # Fold scores for Candidate V2.4
    fold_scores_v24 = []
    for tr_idx, va_idx in folds:
        score_f = roc_auc_score(y_true[va_idx], v24_rank[va_idx])
        fold_scores_v24.append(score_f)
    print(f"  Candidate V2.4 Fold Scores: {[round(s, 5) for s in fold_scores_v24]}")
    print(f"  Candidate V2.4 Mean +/- Std: {np.mean(fold_scores_v24):.5f} +/- {np.std(fold_scores_v24):.5f}")

    # Optimize Weights via Scipy & Nested CV
    print("\n[Step 3] Weight Optimization & Nested 5-Fold Cross-Validation:")
    active_preds = [p_cb4, p_cb3, p_xgb3a_bagged, p_lgb_c2_bagged]
    active_names = ["CB4", "CB3", "XGB3A_Bagged", "LGB_C2_Bagged"]

    opt_w_prob, opt_auc_prob = optimize_weights_scipy(active_preds, y_true, use_rank=False)
    opt_w_rank, opt_auc_rank = optimize_weights_scipy(active_preds, y_true, use_rank=True)
    print("  In-Sample Optimized Weights (Prob):", {name: round(float(w), 3) for name, w in zip(active_names, opt_w_prob)})
    print(f"  In-Sample Optimized Prob AUC:       {opt_auc_prob:.5f}")
    print("  In-Sample Optimized Weights (Rank):", {name: round(float(w), 3) for name, w in zip(active_names, opt_w_rank)})
    print(f"  In-Sample Optimized Rank AUC:       {opt_auc_rank:.5f}")

    nested_auc_prob, nested_weights_prob, _ = nested_cv_weights(active_preds, y_true, folds, use_rank=False)
    nested_auc_rank, nested_weights_rank, _ = nested_cv_weights(active_preds, y_true, folds, use_rank=True)
    print(f"  Nested CV Prob AUC (Unbiased):     {nested_auc_prob:.5f}")
    print(f"  Nested CV Rank AUC (Unbiased):     {nested_auc_rank:.5f}")

    # Non-linear Power Transform Test
    print("\n[Step 4] Rank Power Transform Exploration (p in [0.8, 1.2]):")
    best_p = 1.0
    best_p_auc = roc_auc_score(y_true, v24_rank)
    for p_val in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]:
        r_p_blend = (
            0.27 * (r_cb4 ** p_val)
            + 0.23 * (r_cb3 ** p_val)
            + 0.30 * (r_xgb3a_bagged ** p_val)
            + 0.20 * (r_lgb_c2_bagged ** p_val)
        )
        score_p = roc_auc_score(y_true, r_p_blend)
        if score_p > best_p_auc:
            best_p_auc = score_p
            best_p = p_val
        print(f"  Power p = {p_val:.2f}: OOF AUC = {score_p:.5f} ({score_p - roc_auc_score(y_true, v24_rank):+.5f})")

    # Save structured results
    summary = {
        "individual_models": {name: float(roc_auc_score(y_true, p)) for name, p in all_models.items()},
        "candidate_v23_baseline": {
            "prob_oof_auc": float(roc_auc_score(y_true, v23_prob)),
            "rank_oof_auc": float(roc_auc_score(y_true, v23_rank)),
            "public_leaderboard": 0.76535,
        },
        "candidate_v24_bagged": {
            "prob_oof_auc": float(roc_auc_score(y_true, v24_prob)),
            "rank_oof_auc": float(roc_auc_score(y_true, v24_rank)),
            "fold_scores": fold_scores_v24,
            "mean_fold_auc": float(np.mean(fold_scores_v24)),
            "std_fold_auc": float(np.std(fold_scores_v24)),
            "delta_vs_v23_rank": float(roc_auc_score(y_true, v24_rank) - roc_auc_score(y_true, v23_rank)),
            "nested_cv_rank_unbiased": float(nested_auc_rank),
        },
        "optimized_weights": {name: float(w) for name, w in zip(active_names, opt_w_rank)},
        "nested_weights": nested_weights_rank,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Saved Stage H8 ensemble results to: {RESULTS_JSON}")

if __name__ == "__main__":
    main()
