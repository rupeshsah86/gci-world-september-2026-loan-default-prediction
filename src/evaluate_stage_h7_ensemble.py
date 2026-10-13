"""
Stage H7: Comprehensive Ensemble Blending & Nested Cross-Validation Analysis.
Evaluates upgraded component models (XGBoost Config 3A, CatBoost Config 6, LightGBM C2)
under both Probability and Percentile-Rank normalization.
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

STAGE_H7_DIR = PROJECT_ROOT / "experiments" / "stage_h7"
RESULTS_JSON = STAGE_H7_DIR / "stage_h7_ensemble_results.json"

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
    bounds = [(0.02, 0.60) for _ in range(k)]
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
    print("STAGE H7: ENSEMBLE BLENDING & NESTED CV ANALYSIS")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    n = len(y_true)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(train_df, y_true))

    # Load component OOF predictions
    p_cb4 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_4_High_L2_Regularization.csv")["oof_pred"].values
    p_cb3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_3_RandomStrength_2_LR04.csv")["oof_pred"].values
    p_xgb3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_XGB_Config_3_Depth6_Colsample06_Reg10.csv")["oof_pred"].values
    p_lgb_orig = pd.read_csv(PROJECT_ROOT / "experiments" / "oof_tuned_lightgbm.csv")["oof_best_tuned_lgb"].values
    p_lgb_c2 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h6" / "oof_lightgbm_c2.csv")["oof_pred"].values
    p_xgb3a = pd.read_csv(STAGE_H7_DIR / "oof_xgboost_c3a.csv")["oof_pred"].values

    # Optional CatBoost Config 6 if available
    cb6_path = STAGE_H7_DIR / "oof_catboost_config6.csv"
    has_cb6 = cb6_path.exists()
    p_cb6 = pd.read_csv(cb6_path)["oof_pred"].values if has_cb6 else None

    # Individual Model Scores
    print("\n[Step 1] Individual Model OOF ROC-AUC Summary:")
    models = {
        "CatBoost_Config_4": p_cb4,
        "CatBoost_Config_3": p_cb3,
        "XGBoost_Config_3_Orig": p_xgb3,
        "XGBoost_Config_3A_Upgraded": p_xgb3a,
        "LightGBM_C1_Orig": p_lgb_orig,
        "LightGBM_C2_Upgraded": p_lgb_c2,
    }
    if has_cb6:
        models["CatBoost_Config_6_Upgraded"] = p_cb6

    for name, p in models.items():
        auc = roc_auc_score(y_true, p)
        print(f"  {name:30s}: {auc:.5f}")

    # Spearman Correlations
    print("\n[Step 2] Prediction Spearman Rank Correlation Matrix:")
    corr_df = pd.DataFrame({name: p for name, p in models.items()}).corr(method="spearman")
    print(corr_df.round(4).to_string())

    # Baseline Comparisons
    r_cb4 = rankdata(p_cb4) / n
    r_cb3 = rankdata(p_cb3) / n
    r_xgb3 = rankdata(p_xgb3) / n
    r_xgb3a = rankdata(p_xgb3a) / n
    r_lgb_orig = rankdata(p_lgb_orig) / n
    r_lgb_c2 = rankdata(p_lgb_c2) / n

    v2_prob = 0.27 * p_cb4 + 0.23 * p_cb3 + 0.30 * p_xgb3 + 0.20 * p_lgb_orig
    v21_rank = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb3 + 0.20 * r_lgb_orig
    v22_prob = 0.27 * p_cb4 + 0.23 * p_cb3 + 0.30 * p_xgb3 + 0.20 * p_lgb_c2
    v22_rank = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb3 + 0.20 * r_lgb_c2

    # Upgraded Blends with XGBoost 3A
    h7_base_prob = 0.27 * p_cb4 + 0.23 * p_cb3 + 0.30 * p_xgb3a + 0.20 * p_lgb_c2
    h7_base_rank = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb3a + 0.20 * r_lgb_c2

    print("\n[Step 3] Standard Quad-Model Blends (Weights: 0.27 CB4, 0.23 CB3, 0.30 XGB, 0.20 LGB):")
    print(f"  V2 Prob Blend (LB: 0.76482):       {roc_auc_score(y_true, v2_prob):.5f}")
    print(f"  V2.1 Rank Blend:                   {roc_auc_score(y_true, v21_rank):.5f}")
    print(f"  V2.2 Prob Blend (with LGB C2):     {roc_auc_score(y_true, v22_prob):.5f}")
    print(f"  V2.2 Rank Blend (with LGB C2):     {roc_auc_score(y_true, v22_rank):.5f}")
    print(f"  H7 Prob Blend (with XGB 3A):       {roc_auc_score(y_true, h7_base_prob):.5f} (+{roc_auc_score(y_true, h7_base_prob) - roc_auc_score(y_true, v2_prob):.5f} vs V2)")
    print(f"  H7 Rank Blend (with XGB 3A):       {roc_auc_score(y_true, h7_base_rank):.5f} (+{roc_auc_score(y_true, h7_base_rank) - roc_auc_score(y_true, v22_rank):.5f} vs V2.2)")

    # Test with CB6 if available
    if has_cb6:
        r_cb6 = rankdata(p_cb6) / n
        h7_cb6_prob = 0.27 * p_cb6 + 0.23 * p_cb3 + 0.30 * p_xgb3a + 0.20 * p_lgb_c2
        h7_cb6_rank = 0.27 * r_cb6 + 0.23 * r_cb3 + 0.30 * r_xgb3a + 0.20 * r_lgb_c2
        print(f"  H7 Prob Blend (with CB6 + XGB 3A): {roc_auc_score(y_true, h7_cb6_prob):.5f}")
        print(f"  H7 Rank Blend (with CB6 + XGB 3A): {roc_auc_score(y_true, h7_cb6_rank):.5f}")

    # Optimize Weights via Scipy & Nested CV
    print("\n[Step 4] Weight Optimization & Nested 5-Fold Cross-Validation:")
    active_preds = [p_cb4, p_cb3, p_xgb3a, p_lgb_c2]
    active_names = ["CB4", "CB3", "XGB3A", "LGB_C2"]
    if has_cb6:
        active_preds = [p_cb6, p_cb4, p_cb3, p_xgb3a, p_lgb_c2]
        active_names = ["CB6", "CB4", "CB3", "XGB3A", "LGB_C2"]

    opt_w_prob, opt_auc_prob = optimize_weights_scipy(active_preds, y_true, use_rank=False)
    opt_w_rank, opt_auc_rank = optimize_weights_scipy(active_preds, y_true, use_rank=True)
    print("  In-Sample Optimized Weights (Prob):", {name: round(float(w), 3) for name, w in zip(active_names, opt_w_prob)})
    print(f"  In-Sample Optimized Prob AUC:       {opt_auc_prob:.5f}")
    print("  In-Sample Optimized Weights (Rank):", {name: round(float(w), 3) for name, w in zip(active_names, opt_w_rank)})
    print(f"  In-Sample Optimized Rank AUC:       {opt_auc_rank:.5f}")

    nested_auc_prob, nested_weights_prob, _ = nested_cv_weights(active_preds, y_true, folds, use_rank=False)
    nested_auc_rank, nested_weights_rank, oof_nested_rank = nested_cv_weights(active_preds, y_true, folds, use_rank=True)
    print(f"  Nested CV Prob AUC (Unbiased):     {nested_auc_prob:.5f}")
    print(f"  Nested CV Rank AUC (Unbiased):     {nested_auc_rank:.5f}")

    # Save results to json
    results_payload = {
        "individual_models": {name: float(roc_auc_score(y_true, p)) for name, p in models.items()},
        "baseline_ensembles": {
            "v2_prob": float(roc_auc_score(y_true, v2_prob)),
            "v21_rank": float(roc_auc_score(y_true, v21_rank)),
            "v22_prob": float(roc_auc_score(y_true, v22_prob)),
            "v22_rank": float(roc_auc_score(y_true, v22_rank)),
        },
        "h7_upgraded_ensembles": {
            "h7_base_prob": float(roc_auc_score(y_true, h7_base_prob)),
            "h7_base_rank": float(roc_auc_score(y_true, h7_base_rank)),
            "in_sample_opt_prob": float(opt_auc_prob),
            "in_sample_opt_rank": float(opt_auc_rank),
            "nested_cv_prob": float(nested_auc_prob),
            "nested_cv_rank": float(nested_auc_rank),
        },
        "opt_weights_rank": {name: float(w) for name, w in zip(active_names, opt_w_rank)},
        "nested_weights_rank": nested_weights_rank,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[+] Saved ensemble results to: {RESULTS_JSON}")

if __name__ == "__main__":
    main()
