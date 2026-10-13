"""
Stage H9: Final Comprehensive Ensemble Evaluation.
Evaluates:
- CatBoost Config 4 (0.76080)
- CatBoost Config 3 (0.76072)
- XGBoost Config 3A Original (0.76088)
- XGBoost Config 3A Bagged (0.76128)
- XGBoost Config 3A Target-Encoded (0.76166)
- LightGBM C2 Original (0.75970)
- LightGBM C2 Bagged (0.75983)
- LightGBM C2 Target-Encoded (0.76183)

Tests:
1. Error and prediction correlation matrices.
2. Probability blending vs rank blending.
3. 5-Fold Nested CV with strictly held-out weight optimization.
4. Fold-by-fold delta against Candidate V2.3 benchmark.
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

def run_evaluation():
    print("=" * 80)
    print("STAGE H9: FINAL COMPREHENSIVE ENSEMBLE EVALUATION")
    print("=" * 80)

    # Load All Component OOFs
    cb4 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_4_High_L2_Regularization.csv")
    cb3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_3_RandomStrength_2_LR04.csv")
    xgb_orig = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h7" / "oof_xgboost_c3a.csv")
    xgb_bag = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h8" / "oof_xgboost_c3a_bagged_3seed.csv")
    xgb_te = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h9" / "oof_xgboost_c3a_target_encoded.csv")
    lgb_orig = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h6" / "oof_lightgbm_c2.csv")
    lgb_bag = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h8" / "oof_lightgbm_c2_bagged_3seed.csv")
    lgb_te = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h9" / "oof_lightgbm_c2_target_encoded.csv")

    y_true = cb4["TARGET"].values
    n = len(y_true)

    models_raw = {
        "CB4": cb4["oof_pred"].values,
        "CB3": cb3["oof_pred"].values,
        "XGB_orig": xgb_orig["oof_pred"].values,
        "XGB_bag": xgb_bag["oof_pred"].values,
        "XGB_te": xgb_te["oof_pred"].values,
        "LGB_orig": lgb_orig["oof_pred"].values,
        "LGB_bag": lgb_bag["oof_pred"].values,
        "LGB_te": lgb_te["oof_pred"].values,
    }

    models_rank = {k: to_rank(v) for k, v in models_raw.items()}

    print("\n1. STANDALONE 5-FOLD OOF AUC:")
    for k, v in models_raw.items():
        auc = roc_auc_score(y_true, v)
        print(f"  - {k:12s}: {auc:.5f}")

    # Spearman Correlation Matrix
    keys = list(models_raw.keys())
    corr_matrix = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for k1 in keys:
        for k2 in keys:
            corr_matrix.loc[k1, k2], _ = spearmanr(models_raw[k1], models_raw[k2])
    print("\n2. SPEARMAN CORRELATION MATRIX:")
    print(corr_matrix.round(4).to_string())

    # Residual Error Correlation Matrix (e_i = y - p_i)
    err_corr_matrix = pd.DataFrame(index=keys, columns=keys, dtype=float)
    errors = {k: y_true - models_raw[k] for k in keys}
    for k1 in keys:
        for k2 in keys:
            err_corr_matrix.loc[k1, k2] = float(np.corrcoef(errors[k1], errors[k2])[0, 1])
    print("\n3. RESIDUAL ERROR CORRELATION MATRIX:")
    print(err_corr_matrix.round(4).to_string())

    # 4. FIXED-WEIGHT BLEND COMPARISONS
    print("\n4. FIXED-WEIGHT ENSEMBLE COMPARISONS:")
    
    # Baseline Candidate V2.3
    # Weights: CB4: 0.27, CB3: 0.23, XGB: 0.30, LGB: 0.20
    b_v23_rank = 0.27 * models_rank["CB4"] + 0.23 * models_rank["CB3"] + 0.30 * models_rank["XGB_orig"] + 0.20 * models_rank["LGB_orig"]
    b_v23_prob = 0.27 * models_raw["CB4"] + 0.23 * models_raw["CB3"] + 0.30 * models_raw["XGB_orig"] + 0.20 * models_raw["LGB_orig"]
    auc_v23_rank = roc_auc_score(y_true, b_v23_rank)
    auc_v23_prob = roc_auc_score(y_true, b_v23_prob)
    print(f"  [Benchmark V2.3]               Rank: {auc_v23_rank:.5f} | Prob: {auc_v23_prob:.5f}")

    # V2.4 (Fully Bagged)
    b_v24_rank = 0.27 * models_rank["CB4"] + 0.23 * models_rank["CB3"] + 0.30 * models_rank["XGB_bag"] + 0.20 * models_rank["LGB_bag"]
    auc_v24_rank = roc_auc_score(y_true, b_v24_rank)
    print(f"  [V2.4 Fully Bagged]            Rank: {auc_v24_rank:.5f} (Delta: {auc_v24_rank - auc_v23_rank:+.5f})")

    # V2.5-A (Target-Encoded LGB only)
    b_v25a_rank = 0.27 * models_rank["CB4"] + 0.23 * models_rank["CB3"] + 0.30 * models_rank["XGB_orig"] + 0.20 * models_rank["LGB_te"]
    b_v25a_prob = 0.27 * models_raw["CB4"] + 0.23 * models_raw["CB3"] + 0.30 * models_raw["XGB_orig"] + 0.20 * models_raw["LGB_te"]
    auc_v25a_rank = roc_auc_score(y_true, b_v25a_rank)
    auc_v25a_prob = roc_auc_score(y_true, b_v25a_prob)
    print(f"  [V2.5-A: V2.3 + LGB_TE]        Rank: {auc_v25a_rank:.5f} | Prob: {auc_v25a_prob:.5f} (Delta Rank: {auc_v25a_rank - auc_v23_rank:+.5f})")

    # V2.5-B (Target-Encoded XGB only)
    b_v25b_rank = 0.27 * models_rank["CB4"] + 0.23 * models_rank["CB3"] + 0.30 * models_rank["XGB_te"] + 0.20 * models_rank["LGB_orig"]
    auc_v25b_rank = roc_auc_score(y_true, b_v25b_rank)
    print(f"  [V2.5-B: V2.3 + XGB_TE]        Rank: {auc_v25b_rank:.5f} (Delta Rank: {auc_v25b_rank - auc_v23_rank:+.5f})")

    # V2.5-C (Dual Target-Encoded: Both XGB_TE and LGB_TE)
    b_v25c_rank = 0.27 * models_rank["CB4"] + 0.23 * models_rank["CB3"] + 0.30 * models_rank["XGB_te"] + 0.20 * models_rank["LGB_te"]
    b_v25c_prob = 0.27 * models_raw["CB4"] + 0.23 * models_raw["CB3"] + 0.30 * models_raw["XGB_te"] + 0.20 * models_raw["LGB_te"]
    auc_v25c_rank = roc_auc_score(y_true, b_v25c_rank)
    auc_v25c_prob = roc_auc_score(y_true, b_v25c_prob)
    print(f"  [V2.5-C: Dual TE (XGB+LGB)]    Rank: {auc_v25c_rank:.5f} | Prob: {auc_v25c_prob:.5f} (Delta Rank: {auc_v25c_rank - auc_v23_rank:+.5f})")

    # V2.5-D (Dual TE + Equalized Quad Blend: 0.25 each)
    b_v25d_rank = 0.25 * models_rank["CB4"] + 0.25 * models_rank["CB3"] + 0.25 * models_rank["XGB_te"] + 0.25 * models_rank["LGB_te"]
    auc_v25d_rank = roc_auc_score(y_true, b_v25d_rank)
    print(f"  [V2.5-D: Dual TE (Equal 0.25)] Rank: {auc_v25d_rank:.5f} (Delta Rank: {auc_v25d_rank - auc_v23_rank:+.5f})")

    # 5. STRICT 5-FOLD NESTED CROSS-VALIDATION
    print("\n5. STRICT 5-FOLD NESTED CROSS-VALIDATION (UNBIASED ESTIMATE):")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate_nested_cv(candidate_keys, name):
        k = len(candidate_keys)
        ranks_list = [models_rank[k_] for k_ in candidate_keys]
        nested_oof = np.zeros(n, dtype=float)
        outer_aucs = []
        outer_weights = []

        for outer_fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n), y_true), 1):
            y_inner_tr = y_true[train_idx]
            R_inner_tr = [r[train_idx] for r in ranks_list]

            def obj(w):
                w = np.array(w)
                w = w / np.sum(w)
                blend = sum(w[i] * R_inner_tr[i] for i in range(k))
                return -roc_auc_score(y_inner_tr, blend)

            init_w = np.ones(k) / k
            bounds = [(0.05, 0.50) for _ in range(k)]
            res = minimize(obj, init_w, bounds=bounds, method="SLSQP")
            best_w = res.x / np.sum(res.x)
            outer_weights.append(best_w)

            test_blend = sum(best_w[i] * ranks_list[i][test_idx] for i in range(k))
            nested_oof[test_idx] = test_blend
            auc_outer = roc_auc_score(y_true[test_idx], test_blend)
            outer_aucs.append(auc_outer)

        overall_nested_auc = float(roc_auc_score(y_true, nested_oof))
        mean_nested_auc = float(np.mean(outer_aucs))
        std_nested_auc = float(np.std(outer_aucs))
        mean_w = np.mean(outer_weights, axis=0)

        print(f"  {name}:")
        print(f"    Nested Overall OOF AUC: {overall_nested_auc:.5f}")
        print(f"    Nested Mean Fold AUC:   {mean_nested_auc:.5f} (+/- {std_nested_auc:.5f})")
        print(f"    Mean Weights:           {[round(x, 4) for x in mean_w]}")
        return overall_nested_auc, mean_nested_auc, std_nested_auc, outer_aucs, mean_w

    bench_nested = evaluate_nested_cv(["CB4", "CB3", "XGB_orig", "LGB_orig"], "Candidate V2.3 Benchmark")
    v25a_nested = evaluate_nested_cv(["CB4", "CB3", "XGB_orig", "LGB_te"], "V2.5-A (with LGB_TE)")
    v25c_nested = evaluate_nested_cv(["CB4", "CB3", "XGB_te", "LGB_te"], "V2.5-C (Dual TE: XGB_TE + LGB_TE)")

    # 6. FOLD-BY-FOLD COMPARISON ACROSS ALL 5 FOLDS
    print("\n6. FOLD-BY-FOLD COMPARISON:")
    print(f"  {'Fold':<6s} | {'V2.3 Benchmark':<15s} | {'V2.5-A (LGB_TE)':<15s} | {'V2.5-C (Dual TE)':<15s} | {'Delta C vs V2.3':<15s}")
    print("  " + "-" * 75)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y_true), 1):
        f_v23 = roc_auc_score(y_true[va_idx], b_v23_rank[va_idx])
        f_v25a = roc_auc_score(y_true[va_idx], b_v25a_rank[va_idx])
        f_v25c = roc_auc_score(y_true[va_idx], b_v25c_rank[va_idx])
        print(f"  Fold {fold:<1d} | {f_v23:<15.5f} | {f_v25a:<15.5f} | {f_v25c:<15.5f} | {f_v25c - f_v23:<+15.5f}")

    # Summary JSON output
    summary = {
        "standalone_oof_aucs": {k: float(roc_auc_score(y_true, v)) for k, v in models_raw.items()},
        "fixed_weight_ensembles": {
            "v23_benchmark_rank": float(auc_v23_rank),
            "v23_benchmark_prob": float(auc_v23_prob),
            "v24_bagged_rank": float(auc_v24_rank),
            "v25a_lgb_te_rank": float(auc_v25a_rank),
            "v25b_xgb_te_rank": float(auc_v25b_rank),
            "v25c_dual_te_rank": float(auc_v25c_rank),
            "v25d_dual_te_equal_rank": float(auc_v25d_rank),
        },
        "nested_cv_results": {
            "v23_benchmark": {"overall": bench_nested[0], "mean": bench_nested[1], "std": bench_nested[2]},
            "v25a_lgb_te": {"overall": v25a_nested[0], "mean": v25a_nested[1], "std": v25a_nested[2]},
            "v25c_dual_te": {"overall": v25c_nested[0], "mean": v25c_nested[1], "std": v25c_nested[2]},
        },
        "deltas_vs_v23_benchmark": {
            "v25a_rank_lift": float(auc_v25a_rank - auc_v23_rank),
            "v25c_rank_lift": float(auc_v25c_rank - auc_v23_rank),
            "v25a_nested_lift": float(v25a_nested[0] - bench_nested[0]),
            "v25c_nested_lift": float(v25c_nested[0] - bench_nested[0]),
        }
    }

    out_json = PROJECT_ROOT / "experiments" / "stage_h9" / "stage_h9_final_ensemble_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved final summary to: {out_json}")

if __name__ == "__main__":
    run_evaluation()
