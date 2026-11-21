"""
Stage H11: Phase 5 - Comprehensive Ensemble Exploration & Nested CV.
Evaluates:
1. Multi-model combinations across:
   - CatBoost Config 4 (0.76080)
   - CatBoost Config 3 (0.76072)
   - XGBoost Config 3A (0.76166)
   - XGBoost Config 3B (0.76166)
   - LightGBM C2 (0.76183)
   - LightGBM C3 (0.76194)
2. Regularized Out-of-Fold Stacking (Logistic Regression C=0.01, 0.1, 1.0)
3. Non-linear power rank blending (p in 0.8 to 1.5)
4. Strict 5-Fold Nested Cross-Validation (unbiased weight evaluation)
5. Monotonicity and fold-by-fold comparison against Candidate V2.5-C (0.76302).
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata, spearmanr
from scipy.optimize import minimize
from scipy.special import logit, expit

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def to_rank(p):
    return (rankdata(p) - 0.5) / len(p)

def main():
    print("=" * 80)
    print("STAGE H11: PHASE 5 - COMPREHENSIVE ENSEMBLE EXPLORATION")
    print("=" * 80)

    # Load All Component OOFs
    cb4 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_4_High_L2_Regularization.csv")
    cb3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_3_RandomStrength_2_LR04.csv")
    xgb_3a = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h9" / "oof_xgboost_c3a_target_encoded.csv")
    xgb_3b = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h10" / "oof_xgboost_c3b_te.csv")
    lgb_c2 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h9" / "oof_lightgbm_c2_target_encoded.csv")
    lgb_c3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h10" / "oof_lightgbm_c3_te.csv")

    y_true = cb4["TARGET"].values
    n = len(y_true)

    models_raw = {
        "CB4": cb4["oof_pred"].values,
        "CB3": cb3["oof_pred"].values,
        "XGB_3A": xgb_3a["oof_pred"].values,
        "XGB_3B": xgb_3b["oof_pred"].values,
        "LGB_C2": lgb_c2["oof_pred"].values,
        "LGB_C3": lgb_c3["oof_pred"].values,
    }

    models_rank = {k: to_rank(v) for k, v in models_raw.items()}

    # 1. Baseline Benchmark: Candidate V2.5-C
    b_v25c = 0.27 * models_rank["CB4"] + 0.23 * models_rank["CB3"] + 0.30 * models_rank["XGB_3A"] + 0.20 * models_rank["LGB_C2"]
    auc_v25c = float(roc_auc_score(y_true, b_v25c))
    print(f"\n[Benchmark V2.5-C] OOF ROC-AUC = {auc_v25c:.5f}")

    results = {"benchmark_v25c": auc_v25c}

    # 2. Exploration: Blends of Top GBDTs
    candidates = {}

    # Blend 1: Candidate V2.6-A (LGB C3 upgrade)
    b_26a = 0.27 * models_rank["CB4"] + 0.23 * models_rank["CB3"] + 0.30 * models_rank["XGB_3A"] + 0.20 * models_rank["LGB_C3"]
    candidates["V2.6-A (LGB_C3)"] = b_26a

    # Blend 2: Candidate V2.6-C (Dual Upgrade: XGB 3B + LGB C3)
    b_26c = 0.27 * models_rank["CB4"] + 0.23 * models_rank["CB3"] + 0.30 * models_rank["XGB_3B"] + 0.20 * models_rank["LGB_C3"]
    candidates["V2.6-C (Dual Up)"] = b_26c

    # Blend 3: Hexa-Model Blend (Equally blending both variants of XGB and LGB)
    # CB4: 0.27, CB3: 0.23, XGB3A: 0.15, XGB3B: 0.15, LGBC2: 0.10, LGBC3: 0.10
    b_hexa = 0.27 * models_rank["CB4"] + 0.23 * models_rank["CB3"] + 0.15 * models_rank["XGB_3A"] + 0.15 * models_rank["XGB_3B"] + 0.10 * models_rank["LGB_C2"] + 0.10 * models_rank["LGB_C3"]
    candidates["Hexa-Blend"] = b_hexa

    # Blend 4: Pure 3-Family Blend (CB4 0.35, XGB3A 0.35, LGBC3 0.30)
    b_tri = 0.35 * models_rank["CB4"] + 0.35 * models_rank["XGB_3A"] + 0.30 * models_rank["LGB_C3"]
    candidates["Tri-Model (CB4/XGB3A/LGBC3)"] = b_tri

    print("\n--- Fixed-Weight Candidate Ensembles ---")
    for name, pred in candidates.items():
        auc_ = float(roc_auc_score(y_true, pred))
        delta = auc_ - auc_v25c
        print(f"  {name:28s}: OOF AUC = {auc_:.5f} (Delta vs V2.5-C: {delta:+.5f})")
        results[name] = {"oof_auc": auc_, "delta": delta}

    # 3. Power Rank Blending Search
    print("\n--- Power Rank Blending (p in 0.8 to 1.5) on Candidate V2.6-A ---")
    power_results = {}
    for p_val in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        b_p = 0.27 * (models_rank["CB4"]**p_val) + 0.23 * (models_rank["CB3"]**p_val) + 0.30 * (models_rank["XGB_3A"]**p_val) + 0.20 * (models_rank["LGB_C3"]**p_val)
        auc_p = float(roc_auc_score(y_true, b_p))
        delta_p = auc_p - auc_v25c
        power_results[str(p_val)] = auc_p
        print(f"  Power p={p_val:.1f}: OOF AUC = {auc_p:.5f} (Delta: {delta_p:+.5f})")
    results["power_rank_blends"] = power_results

    # 4. Out-of-Fold Stacking with Logistic Regression
    print("\n--- Out-of-Fold Stacking (Logistic Regression) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Stack inputs (raw probabilities of 6 models)
    X_stack = np.column_stack([models_raw[k] for k in ["CB4", "CB3", "XGB_3A", "XGB_3B", "LGB_C2", "LGB_C3"]])

    stack_results = {}
    for C_val in [0.001, 0.01, 0.1, 1.0]:
        stack_oof = np.zeros(n, dtype=float)
        for tr_idx, va_idx in skf.split(X_stack, y_true):
            meta = LogisticRegression(C=C_val, solver="lbfgs", max_iter=200, random_state=42)
            meta.fit(X_stack[tr_idx], y_true[tr_idx])
            stack_oof[va_idx] = meta.predict_proba(X_stack[va_idx])[:, 1]
        auc_stack = float(roc_auc_score(y_true, stack_oof))
        delta_stack = auc_stack - auc_v25c
        stack_results[f"C={C_val}"] = auc_stack
        print(f"  Stacking (LogReg C={C_val:5f}): OOF AUC = {auc_stack:.5f} (Delta: {delta_stack:+.5f})")
    results["stacking_logreg"] = stack_results

    # 5. Strict 5-Fold Nested Cross-Validation (Unbiased Evaluation)
    print("\n--- Strict 5-Fold Nested Cross-Validation ---")
    def run_nested(rank_keys, name):
        k = len(rank_keys)
        ranks = [models_rank[k_] for k_ in rank_keys]
        nested_oof = np.zeros(n, dtype=float)
        outer_aucs = []
        outer_weights = []

        for out_fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y_true), 1):
            y_in = y_true[tr_idx]
            R_in = [r[tr_idx] for r in ranks]

            def loss(w):
                w = np.array(w)
                w = w / np.sum(w)
                blend = sum(w[i] * R_in[i] for i in range(k))
                return -roc_auc_score(y_in, blend)

            init_w = np.ones(k) / k
            bnds = [(0.05, 0.50) for _ in range(k)]
            res = minimize(loss, init_w, bounds=bnds, method="SLSQP")
            w_opt = res.x / np.sum(res.x)
            outer_weights.append(w_opt)

            va_blend = sum(w_opt[i] * ranks[i][va_idx] for i in range(k))
            nested_oof[va_idx] = va_blend
            outer_aucs.append(float(roc_auc_score(y_true[va_idx], va_blend)))

        ov_auc = float(roc_auc_score(y_true, nested_oof))
        mean_auc = float(np.mean(outer_aucs))
        std_auc = float(np.std(outer_aucs))
        print(f"  {name}:")
        print(f"    Nested Overall OOF AUC: {ov_auc:.5f}")
        print(f"    Nested Mean Fold AUC:   {mean_auc:.5f} (+/- {std_auc:.5f})")
        print(f"    Mean Learned Weights:   {[round(x, 4) for x in np.mean(outer_weights, axis=0)]}")
        return ov_auc, mean_auc, std_auc, outer_aucs

    nested_v25c = run_nested(["CB4", "CB3", "XGB_3A", "LGB_C2"], "Candidate V2.5-C Benchmark")
    nested_v26a = run_nested(["CB4", "CB3", "XGB_3A", "LGB_C3"], "Candidate V2.6-A")
    nested_hexa = run_nested(["CB4", "CB3", "XGB_3A", "XGB_3B", "LGB_C2", "LGB_C3"], "Hexa-Model Blend")

    # 6. Fold-by-Fold Monotonicity Audit: Hexa-Blend and V2.6-A vs V2.5-C
    print("\n--- Fold-by-Fold Stability & Monotonicity Audit ---")
    print(f"  {'Fold':<6s} | {'V2.5-C (Benchmark)':<18s} | {'V2.6-A (LGB_C3)':<18s} | {'Hexa-Blend':<18s} | {'Delta Hexa vs V2.5-C':<20s}")
    print("  " + "-" * 85)
    hexa_wins = 0
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y_true), 1):
        f_v25c = roc_auc_score(y_true[va_idx], b_v25c[va_idx])
        f_v26a = roc_auc_score(y_true[va_idx], b_26a[va_idx])
        f_hexa = roc_auc_score(y_true[va_idx], b_hexa[va_idx])
        diff_hexa = f_hexa - f_v25c
        if diff_hexa >= 0:
            hexa_wins += 1
        print(f"  Fold {fold:<1d} | {f_v25c:<18.5f} | {f_v26a:<18.5f} | {f_hexa:<18.5f} | {diff_hexa:<+20.5f}")

    print(f"\nHexa-Blend Monotonicity vs V2.5-C: {hexa_wins}/5 folds.")

    out_file = PROJECT_ROOT / "experiments" / "stage_h11" / "stage_h11_ensemble_exploration_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved ensemble exploration results to: {out_file}")

if __name__ == "__main__":
    main()

