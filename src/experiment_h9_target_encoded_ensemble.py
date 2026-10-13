"""
Stage H9: Experiment 4 - Comprehensive Evaluation of Target-Encoded LightGBM C2 Ensemble.
Tests:
1. Spearman and residual error correlation of Target-Encoded LightGBM C2 with CB4, CB3, XGB3A Single, XGB3A Bagged.
2. Grid search over blend weights (both rank and probability blending).
3. Leakage-safe 5-fold Nested Cross-Validation to get an unbiased estimate.
4. Comparison against Candidate V2.3 (0.76265 OOF, 0.76263 Nested CV, 0.76535 Public LB).
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata, spearmanr
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def to_rank(p):
    return (rankdata(p) - 0.5) / len(p)

def run_analysis():
    print("=" * 80)
    print("STAGE H9: COMPREHENSIVE ENSEMBLE ANALYSIS WITH TARGET-ENCODED LIGHTGBM")
    print("=" * 80)

    # Load All Component OOFs
    cb4 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_4_High_L2_Regularization.csv")
    cb3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_3_RandomStrength_2_LR04.csv")
    xgb_single = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h7" / "oof_xgboost_c3a.csv")
    xgb_bagged = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h8" / "oof_xgboost_c3a_bagged_3seed.csv")
    lgb_orig = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h6" / "oof_lightgbm_c2.csv")
    lgb_te = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h9" / "oof_lightgbm_c2_target_encoded.csv")

    y_true = cb4["TARGET"].values
    n = len(y_true)

    p_cb4 = cb4["oof_pred"].values
    p_cb3 = cb3["oof_pred"].values
    p_xgb_s = xgb_single["oof_pred"].values
    p_xgb_b = xgb_bagged["oof_pred"].values
    p_lgb_o = lgb_orig["oof_pred"].values
    p_lgb_te = lgb_te["oof_pred"].values

    r_cb4 = to_rank(p_cb4)
    r_cb3 = to_rank(p_cb3)
    r_xgb_s = to_rank(p_xgb_s)
    r_xgb_b = to_rank(p_xgb_b)
    r_lgb_o = to_rank(p_lgb_o)
    r_lgb_te = to_rank(p_lgb_te)

    # 1. Component Standalone AUCs
    print("\n--- Component Model Standalone 5-Fold OOF AUC ---")
    print(f"  1. CatBoost Config 4:              {roc_auc_score(y_true, p_cb4):.5f}")
    print(f"  2. CatBoost Config 3:              {roc_auc_score(y_true, p_cb3):.5f}")
    print(f"  3. XGBoost Config 3A (Single):     {roc_auc_score(y_true, p_xgb_s):.5f}")
    print(f"  4. XGBoost Config 3A (Bagged):     {roc_auc_score(y_true, p_xgb_b):.5f}")
    print(f"  5. LightGBM C2 (Original):         {roc_auc_score(y_true, p_lgb_o):.5f}")
    print(f"  6. LightGBM C2 (Target-Encoded):   {roc_auc_score(y_true, p_lgb_te):.5f}")

    # 2. Correlation Analysis
    print("\n--- Correlation Analysis: Original LGB vs Target-Encoded LGB ---")
    for name, p_ref in [("CB4", p_cb4), ("CB3", p_cb3), ("XGB_Single", p_xgb_s), ("XGB_Bagged", p_xgb_b)]:
        rho_orig, _ = spearmanr(p_ref, p_lgb_o)
        rho_te, _ = spearmanr(p_ref, p_lgb_te)
        print(f"  vs {name:12s}: Spearman with LGB Orig = {rho_orig:.4f} | with LGB TE = {rho_te:.4f} (Delta: {rho_te - rho_orig:+.4f})")

    # 3. Fixed-Weight V2.3 Variants Comparison
    print("\n--- Fixed-Weight V2.3 Configuration Variants ---")
    # Base V2.3: CB4=0.27, CB3=0.23, XGB_Single=0.30, LGB_Orig=0.20
    b_v23_orig = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb_s + 0.20 * r_lgb_o
    auc_v23_orig = roc_auc_score(y_true, b_v23_orig)
    print(f"  [Candidate V2.3 Benchmark] Rank Blend AUC: {auc_v23_orig:.5f}")

    # V2.3 with LGB TE:
    b_v23_te = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb_s + 0.20 * r_lgb_te
    auc_v23_te = roc_auc_score(y_true, b_v23_te)
    print(f"  [Variant A: V2.3 + LGB TE] Rank Blend AUC: {auc_v23_te:.5f} (Delta: {auc_v23_te - auc_v23_orig:+.5f})")

    # Variant B: V2.3 with Bagged XGBoost + LGB TE
    b_v23_bag_te = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb_b + 0.20 * r_lgb_te
    auc_v23_bag_te = roc_auc_score(y_true, b_v23_bag_te)
    print(f"  [Variant B: V2.4 + LGB TE] Rank Blend AUC: {auc_v23_bag_te:.5f} (Delta: {auc_v23_bag_te - auc_v23_orig:+.5f})")

    # Variant C: Hybrid XGB (0.15 Single + 0.15 Bagged) + LGB TE
    b_v23_hyb_te = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.15 * r_xgb_s + 0.15 * r_xgb_b + 0.20 * r_lgb_te
    auc_v23_hyb_te = roc_auc_score(y_true, b_v23_hyb_te)
    print(f"  [Variant C: Hybrid XGB + LGB TE] Rank Blend AUC: {auc_v23_hyb_te:.5f} (Delta: {auc_v23_hyb_te - auc_v23_orig:+.5f})")

    # Probability Blends of the Same Variants
    p_v23_orig = 0.27 * p_cb4 + 0.23 * p_cb3 + 0.30 * p_xgb_s + 0.20 * p_lgb_o
    p_v23_te = 0.27 * p_cb4 + 0.23 * p_cb3 + 0.30 * p_xgb_s + 0.20 * p_lgb_te
    print(f"  [Prob Blend V2.3 Benchmark]: {roc_auc_score(y_true, p_v23_orig):.5f}")
    print(f"  [Prob Blend V2.3 + LGB TE]:  {roc_auc_score(y_true, p_v23_te):.5f} (Delta: {roc_auc_score(y_true, p_v23_te) - roc_auc_score(y_true, p_v23_orig):+.5f})")

    # 4. Rigorous 5-Fold Nested Cross-Validation (Unbiased Evaluation)
    print("\n--- Rigorous 5-Fold Nested Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Let's test two candidate model sets in Nested CV:
    # Model Set 1: [CB4, CB3, XGB_Single, LGB_TE] (Quad blend with TE)
    # Model Set 2: [CB4, CB3, XGB_Single, XGB_Bagged, LGB_TE] (Penta blend)
    
    def run_nested_cv(model_ranks, name):
        k_models = len(model_ranks)
        nested_oof = np.zeros(n, dtype=float)
        outer_aucs = []
        outer_weights = []

        for outer_fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n), y_true), 1):
            y_inner_tr = y_true[train_idx]
            R_inner_tr = [r[train_idx] for r in model_ranks]

            # Optimize weights on inner folds using SLSQP
            def obj(w):
                w = np.array(w)
                w = w / np.sum(w)
                blend = sum(w[i] * R_inner_tr[i] for i in range(k_models))
                return -roc_auc_score(y_inner_tr, blend)

            init_w = np.ones(k_models) / k_models
            bounds = [(0.05, 0.50) for _ in range(k_models)]
            res = minimize(obj, init_w, bounds=bounds, method="SLSQP")
            best_w = res.x / np.sum(res.x)
            outer_weights.append(best_w)

            # Evaluate on held-out outer fold
            test_blend = sum(best_w[i] * model_ranks[i][test_idx] for i in range(k_models))
            nested_oof[test_idx] = test_blend
            auc_outer = roc_auc_score(y_true[test_idx], test_blend)
            outer_aucs.append(auc_outer)

        overall_nested_auc = roc_auc_score(y_true, nested_oof)
        mean_nested_auc = float(np.mean(outer_aucs))
        std_nested_auc = float(np.std(outer_aucs))
        mean_w = np.mean(outer_weights, axis=0)

        print(f"  {name}:")
        print(f"    Nested Overall OOF AUC: {overall_nested_auc:.5f}")
        print(f"    Nested Mean Fold AUC:   {mean_nested_auc:.5f} (+/- {std_nested_auc:.5f})")
        print(f"    Average Learned Weights: {np.round(mean_w, 4)}")
        return overall_nested_auc, mean_nested_auc, outer_weights

    nested_v23_bench, _, _ = run_nested_cv([r_cb4, r_cb3, r_xgb_s, r_lgb_o], "Candidate V2.3 Benchmark (CB4, CB3, XGB_s, LGB_orig)")
    nested_v23_te, _, _ = run_nested_cv([r_cb4, r_cb3, r_xgb_s, r_lgb_te], "Candidate V2.5-TE Quad Blend (CB4, CB3, XGB_s, LGB_te)")
    nested_penta_te, _, _ = run_nested_cv([r_cb4, r_cb3, r_xgb_s, r_xgb_b, r_lgb_te], "Candidate V2.5-TE Penta Blend (CB4, CB3, XGB_s, XGB_b, LGB_te)")

    print(f"\nNested CV Delta for Quad Blend: {nested_v23_te - nested_v23_bench:+.5f}")
    print(f"Nested CV Delta for Penta Blend: {nested_penta_te - nested_v23_bench:+.5f}")

    # Check fold-by-fold consistency
    print("\n--- Fold-by-Fold Comparison (Quad Blend with LGB TE vs Benchmark V2.3) ---")
    for f_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y_true), 1):
        f_auc_orig = roc_auc_score(y_true[va_idx], b_v23_orig[va_idx])
        f_auc_te = roc_auc_score(y_true[va_idx], b_v23_te[va_idx])
        print(f"  Fold {f_idx}: V2.3 Orig = {f_auc_orig:.5f} | V2.3+TE = {f_auc_te:.5f} (Delta: {f_auc_te - f_auc_orig:+.5f})")

    out_file = PROJECT_ROOT / "experiments" / "stage_h9" / "stage_h9_target_encoded_ensemble_results.json"
    results_summary = {
        "candidate_v23_benchmark_oof": float(auc_v23_orig),
        "candidate_v23_benchmark_nested": float(nested_v23_bench),
        "v23_plus_lgb_te_oof": float(auc_v23_te),
        "v23_plus_lgb_te_nested": float(nested_v23_te),
        "v23_plus_lgb_te_lift_oof": float(auc_v23_te - auc_v23_orig),
        "v23_plus_lgb_te_lift_nested": float(nested_v23_te - nested_v23_bench),
        "penta_blend_te_nested": float(nested_penta_te),
    }
    with open(out_file, "w") as f:
        json.dump(results_summary, f, indent=2)

if __name__ == "__main__":
    run_analysis()
