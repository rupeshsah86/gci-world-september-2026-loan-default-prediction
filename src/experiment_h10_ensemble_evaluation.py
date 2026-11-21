"""
Stage H10: Final Comprehensive Ensemble Evaluation & Nested CV.
Evaluates:
- CatBoost Config 4 (0.76080)
- CatBoost Config 3 (0.76072)
- XGBoost Config 3A Target-Encoded (0.76166)
- XGBoost Config 3B Target-Encoded (0.76166)
- LightGBM C2 Target-Encoded (0.76183)
- LightGBM C3 Target-Encoded (0.76194)
- CatBoost C1 Target-Encoded (0.76013)

Tests:
1. Spearman & Residual error correlation matrices.
2. Candidate V2.5-C benchmark reproduction (0.76302).
3. Candidate V2.6-A: Replacing LightGBM C2_TE with LightGBM C3_TE (CB4 27%, CB3 23%, XGB_3A 30%, LGB_C3 20%).
4. Candidate V2.6-B: Replacing XGBoost C3A_TE with XGBoost C3B_TE.
5. Candidate V2.6-C: Dual upgrade (XGB_3B + LGB_C3).
6. Non-linear blending: Logit rank blending (z = logit(r)), power blending.
7. Strict 5-Fold Nested Cross-Validation (unbiased weight learning).
8. Fold-by-fold comparison and monotonicity check vs Candidate V2.5-C.
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
from scipy.special import logit, expit

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def to_rank(p):
    return (rankdata(p) - 0.5) / len(p)

def main():
    print("=" * 80)
    print("STAGE H10: COMPREHENSIVE ENSEMBLE EVALUATION & NESTED CV")
    print("=" * 80)

    # 1. Load All OOF Predictions
    cb4 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_4_High_L2_Regularization.csv")
    cb3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_3_RandomStrength_2_LR04.csv")
    xgb_3a = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h9" / "oof_xgboost_c3a_target_encoded.csv")
    xgb_3b = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h10" / "oof_xgboost_c3b_te.csv")
    lgb_c2 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h9" / "oof_lightgbm_c2_target_encoded.csv")
    lgb_c3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h10" / "oof_lightgbm_c3_te.csv")
    cb_te = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h10" / "oof_catboost_c1_te.csv")

    y_true = cb4["TARGET"].values
    n = len(y_true)

    models_raw = {
        "CB4": cb4["oof_pred"].values,
        "CB3": cb3["oof_pred"].values,
        "XGB_3A_TE": xgb_3a["oof_pred"].values,
        "XGB_3B_TE": xgb_3b["oof_pred"].values,
        "LGB_C2_TE": lgb_c2["oof_pred"].values,
        "LGB_C3_TE": lgb_c3["oof_pred"].values,
        "CB_TE": cb_te["oof_pred"].values,
    }

    models_rank = {k: to_rank(v) for k, v in models_raw.items()}

    print("\n1. STANDALONE 5-FOLD OOF AUC:")
    for k, v in models_raw.items():
        auc = roc_auc_score(y_true, v)
        print(f"  - {k:15s}: {auc:.5f}")

    # Spearman Correlation Matrix
    keys = list(models_raw.keys())
    corr_df = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for k1 in keys:
        for k2 in keys:
            corr_df.loc[k1, k2], _ = spearmanr(models_raw[k1], models_raw[k2])
    print("\n2. SPEARMAN CORRELATION MATRIX:")
    print(corr_df.round(4).to_string())

    # Residual Error Correlation Matrix
    err_corr_df = pd.DataFrame(index=keys, columns=keys, dtype=float)
    errors = {k: y_true - models_raw[k] for k in keys}
    for k1 in keys:
        for k2 in keys:
            err_corr_df.loc[k1, k2] = float(np.corrcoef(errors[k1], errors[k2])[0, 1])
    print("\n3. RESIDUAL ERROR CORRELATION MATRIX:")
    print(err_corr_df.round(4).to_string())

    # 4. Ensemble Comparisons
    print("\n4. ENSEMBLE CONFIGURATION COMPARISONS:")

    # Benchmark: Candidate V2.5-C (Dual TE Quad: CB4 0.27, CB3 0.23, XGB_3A 0.30, LGB_C2 0.20)
    b_v25c_rank = 0.27 * models_rank["CB4"] + 0.23 * models_rank["CB3"] + 0.30 * models_rank["XGB_3A_TE"] + 0.20 * models_rank["LGB_C2_TE"]
    b_v25c_prob = 0.27 * models_raw["CB4"] + 0.23 * models_raw["CB3"] + 0.30 * models_raw["XGB_3A_TE"] + 0.20 * models_raw["LGB_C2_TE"]
    auc_v25c_rank = roc_auc_score(y_true, b_v25c_rank)
    auc_v25c_prob = roc_auc_score(y_true, b_v25c_prob)
    print(f"  [Benchmark V2.5-C]          Rank: {auc_v25c_rank:.5f} | Prob: {auc_v25c_prob:.5f}")

    # Candidate V2.6-A: Upgrade LGB to C3_TE
    b_v26a_rank = 0.27 * models_rank["CB4"] + 0.23 * models_rank["CB3"] + 0.30 * models_rank["XGB_3A_TE"] + 0.20 * models_rank["LGB_C3_TE"]
    b_v26a_prob = 0.27 * models_raw["CB4"] + 0.23 * models_raw["CB3"] + 0.30 * models_raw["XGB_3A_TE"] + 0.20 * models_raw["LGB_C3_TE"]
    auc_v26a_rank = roc_auc_score(y_true, b_v26a_rank)
    auc_v26a_prob = roc_auc_score(y_true, b_v26a_prob)
    print(f"  [Candidate V2.6-A: LGB_C3]  Rank: {auc_v26a_rank:.5f} | Prob: {auc_v26a_prob:.5f} (Delta: {auc_v26a_rank - auc_v25c_rank:+.5f})")

    # Candidate V2.6-B: Upgrade XGB to C3B_TE
    b_v26b_rank = 0.27 * models_rank["CB4"] + 0.23 * models_rank["CB3"] + 0.30 * models_rank["XGB_3B_TE"] + 0.20 * models_rank["LGB_C2_TE"]
    auc_v26b_rank = roc_auc_score(y_true, b_v26b_rank)
    print(f"  [Candidate V2.6-B: XGB_3B]  Rank: {auc_v26b_rank:.5f} (Delta: {auc_v26b_rank - auc_v25c_rank:+.5f})")

    # Candidate V2.6-C: Dual Upgrade (XGB_3B + LGB_C3)
    b_v26c_rank = 0.27 * models_rank["CB4"] + 0.23 * models_rank["CB3"] + 0.30 * models_rank["XGB_3B_TE"] + 0.20 * models_rank["LGB_C3_TE"]
    b_v26c_prob = 0.27 * models_raw["CB4"] + 0.23 * models_raw["CB3"] + 0.30 * models_raw["XGB_3B_TE"] + 0.20 * models_raw["LGB_C3_TE"]
    auc_v26c_rank = roc_auc_score(y_true, b_v26c_rank)
    auc_v26c_prob = roc_auc_score(y_true, b_v26c_prob)
    print(f"  [Candidate V2.6-C: Dual Up] Rank: {auc_v26c_rank:.5f} | Prob: {auc_v26c_prob:.5f} (Delta: {auc_v26c_rank - auc_v25c_rank:+.5f})")

    # Candidate V2.6-D: Equal Weights Quad (0.25 each: CB4, CB3, XGB_3A, LGB_C3)
    b_v26d_rank = 0.25 * models_rank["CB4"] + 0.25 * models_rank["CB3"] + 0.25 * models_rank["XGB_3A_TE"] + 0.25 * models_rank["LGB_C3_TE"]
    auc_v26d_rank = roc_auc_score(y_true, b_v26d_rank)
    print(f"  [Candidate V2.6-D: Equal W] Rank: {auc_v26d_rank:.5f} (Delta: {auc_v26d_rank - auc_v25c_rank:+.5f})")

    # Candidate V2.6-E: 5-Model Ensemble (adding CB_TE with 5% weight)
    b_v26e_rank = 0.25 * models_rank["CB4"] + 0.20 * models_rank["CB3"] + 0.30 * models_rank["XGB_3A_TE"] + 0.20 * models_rank["LGB_C3_TE"] + 0.05 * models_rank["CB_TE"]
    auc_v26e_rank = roc_auc_score(y_true, b_v26e_rank)
    print(f"  [Candidate V2.6-E: 5-Model] Rank: {auc_v26e_rank:.5f} (Delta: {auc_v26e_rank - auc_v25c_rank:+.5f})")

    # 5. Non-linear Rank Blending: Logit Rank Blending
    print("\n5. NON-LINEAR TRANSFORMATIONS (LOGIT RANK BLENDING):")
    # Clip ranks to avoid inf: [eps, 1-eps]
    eps = 1e-5
    r_cb4_clip = np.clip(models_rank["CB4"], eps, 1 - eps)
    r_cb3_clip = np.clip(models_rank["CB3"], eps, 1 - eps)
    r_xgb_clip = np.clip(models_rank["XGB_3A_TE"], eps, 1 - eps)
    r_lgb_clip = np.clip(models_rank["LGB_C3_TE"], eps, 1 - eps)

    z_blend = 0.27 * logit(r_cb4_clip) + 0.23 * logit(r_cb3_clip) + 0.30 * logit(r_xgb_clip) + 0.20 * logit(r_lgb_clip)
    auc_logit = roc_auc_score(y_true, z_blend)
    print(f"  Logit Rank Blend AUC: {auc_logit:.5f} (Delta: {auc_logit - auc_v25c_rank:+.5f})")

    # Power Rank Blend: r^1.2
    p_blend = 0.27 * (models_rank["CB4"]**1.2) + 0.23 * (models_rank["CB3"]**1.2) + 0.30 * (models_rank["XGB_3A_TE"]**1.2) + 0.20 * (models_rank["LGB_C3_TE"]**1.2)
    auc_power = roc_auc_score(y_true, p_blend)
    print(f"  Power (1.2) Rank Blend AUC: {auc_power:.5f} (Delta: {auc_power - auc_v25c_rank:+.5f})")

    # 6. Strict 5-Fold Nested Cross-Validation (Unbiased Evaluation)
    print("\n6. STRICT 5-FOLD NESTED CROSS-VALIDATION:")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate_nested(candidate_keys, name):
        k = len(candidate_keys)
        ranks = [models_rank[k_] for k_ in candidate_keys]
        nested_oof = np.zeros(n, dtype=float)
        outer_aucs = []
        outer_weights = []

        for out_fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y_true), 1):
            y_in_tr = y_true[tr_idx]
            R_in_tr = [r[tr_idx] for r in ranks]

            def obj(w):
                w = np.array(w)
                w = w / np.sum(w)
                blend = sum(w[i] * R_in_tr[i] for i in range(k))
                return -roc_auc_score(y_in_tr, blend)

            init_w = np.ones(k) / k
            bounds = [(0.05, 0.50) for _ in range(k)]
            res = minimize(obj, init_w, bounds=bounds, method="SLSQP")
            best_w = res.x / np.sum(res.x)
            outer_weights.append(best_w)

            test_blend = sum(best_w[i] * ranks[i][va_idx] for i in range(k))
            nested_oof[va_idx] = test_blend
            auc_out = roc_auc_score(y_true[va_idx], test_blend)
            outer_aucs.append(auc_out)

        ov_auc = float(roc_auc_score(y_true, nested_oof))
        mean_auc = float(np.mean(outer_aucs))
        std_auc = float(np.std(outer_aucs))
        mean_w = np.mean(outer_weights, axis=0)

        print(f"  {name}:")
        print(f"    Nested Overall OOF AUC: {ov_auc:.5f}")
        print(f"    Nested Mean Fold AUC:   {mean_auc:.5f} (+/- {std_auc:.5f})")
        print(f"    Mean Learned Weights:   {[round(x, 4) for x in mean_w]}")
        return ov_auc, mean_auc, std_auc, outer_aucs, mean_w

    bench_nested = evaluate_nested(["CB4", "CB3", "XGB_3A_TE", "LGB_C2_TE"], "Benchmark Candidate V2.5-C")
    v26a_nested = evaluate_nested(["CB4", "CB3", "XGB_3A_TE", "LGB_C3_TE"], "Candidate V2.6-A (with LGB_C3)")
    v26c_nested = evaluate_nested(["CB4", "CB3", "XGB_3B_TE", "LGB_C3_TE"], "Candidate V2.6-C (Dual Upgrade)")

    # 7. Fold-by-Fold Comparison & Monotonicity
    print("\n7. FOLD-BY-FOLD COMPARISON ACROSS ALL 5 FOLDS:")
    print(f"  {'Fold':<6s} | {'V2.5-C (Benchmark)':<18s} | {'V2.6-A (LGB_C3)':<18s} | {'Delta A vs V2.5-C':<18s} | {'Winner':<10s}")
    print("  " + "-" * 75)
    fold_wins = 0
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y_true), 1):
        f_v25 = roc_auc_score(y_true[va_idx], b_v25c_rank[va_idx])
        f_v26a = roc_auc_score(y_true[va_idx], b_v26a_rank[va_idx])
        diff = f_v26a - f_v25
        winner = "V2.6-A" if diff > 0 else ("TIE" if diff == 0 else "V2.5-C")
        if diff >= 0:
            fold_wins += 1
        print(f"  Fold {fold:<1d} | {f_v25:<18.5f} | {f_v26a:<18.5f} | {diff:<+18.5f} | {winner:<10s}")

    print(f"\nMonotonicity Score: {fold_wins}/5 folds improved or equal.")

    # Save summary JSON
    results = {
        "timestamp": "Stage_H10_Evaluation",
        "standalone_oof_aucs": {k: float(roc_auc_score(y_true, v)) for k, v in models_raw.items()},
        "fixed_weight_ensembles": {
            "v25c_benchmark_rank": float(auc_v25c_rank),
            "v25c_benchmark_prob": float(auc_v25c_prob),
            "v26a_lgb_c3_rank": float(auc_v26a_rank),
            "v26a_lgb_c3_prob": float(auc_v26a_prob),
            "v26b_xgb_c3b_rank": float(auc_v26b_rank),
            "v26c_dual_upgrade_rank": float(auc_v26c_rank),
            "v26d_equal_weight_rank": float(auc_v26d_rank),
            "v26e_5model_rank": float(auc_v26e_rank),
            "logit_rank_blend": float(auc_logit),
            "power_rank_blend": float(auc_power),
        },
        "nested_cv_results": {
            "v25c_benchmark": {"overall": bench_nested[0], "mean": bench_nested[1], "std": bench_nested[2]},
            "v26a_lgb_c3": {"overall": v26a_nested[0], "mean": v26a_nested[1], "std": v26a_nested[2]},
            "v26c_dual_upgrade": {"overall": v26c_nested[0], "mean": v26c_nested[1], "std": v26c_nested[2]},
        },
        "deltas_vs_v25c_benchmark": {
            "v26a_rank_lift": float(auc_v26a_rank - auc_v25c_rank),
            "v26a_nested_lift": float(v26a_nested[0] - bench_nested[0]),
            "v26c_rank_lift": float(auc_v26c_rank - auc_v25c_rank),
            "v26c_nested_lift": float(v26c_nested[0] - bench_nested[0]),
        }
    }

    out_json = PROJECT_ROOT / "experiments" / "stage_h10" / "stage_h10_ensemble_summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved final summary to: {out_json}")

if __name__ == "__main__":
    main()

