"""
Stage H4: Rank-Normalized Quad-Model Blend Evaluation and Audit.
Computes OOF metrics, fold-level consistency, distribution statistics,
correlation analysis, and produces data for reports/stage_h4_rank_blend_review.md.
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data import load_raw_data, TARGET_COL, ID_COL

CB4_PATH = PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_4_High_L2_Regularization.csv"
CB3_PATH = PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_3_RandomStrength_2_LR04.csv"
XGB3_PATH = PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_XGB_Config_3_Depth6_Colsample06_Reg10.csv"
LGB_PATH = PROJECT_ROOT / "experiments" / "oof_tuned_lightgbm.csv"


def run_evaluation():
    print("=" * 80)
    print("STAGE H4: FULL OOF & FOLD-LEVEL AUDIT")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    train_ids = train_df[ID_COL].values
    n_train = len(train_df)

    cb4_df = pd.read_csv(CB4_PATH)
    cb3_df = pd.read_csv(CB3_PATH)
    xgb3_df = pd.read_csv(XGB3_PATH)
    lgb_df = pd.read_csv(LGB_PATH)

    p_cb4 = cb4_df["oof_pred"].values
    p_cb3 = cb3_df["oof_pred"].values
    p_xgb3 = xgb3_df["oof_pred"].values
    p_lgb = lgb_df["oof_best_tuned_lgb"].values

    # Weights
    w_cb4, w_cb3, w_xgb, w_lgb = 0.27, 0.23, 0.30, 0.20

    # 1. Probability blend
    prob_blend = w_cb4 * p_cb4 + w_cb3 * p_cb3 + w_xgb * p_xgb3 + w_lgb * p_lgb

    # 2. Percentile-rank normalization
    r_cb4 = rankdata(p_cb4, method="average") / n_train
    r_cb3 = rankdata(p_cb3, method="average") / n_train
    r_xgb3 = rankdata(p_xgb3, method="average") / n_train
    r_lgb = rankdata(p_lgb, method="average") / n_train

    rank_blend = w_cb4 * r_cb4 + w_cb3 * r_cb3 + w_xgb * r_xgb3 + w_lgb * r_lgb

    # Overall OOF AUC
    auc_cb4 = float(roc_auc_score(y_true, p_cb4))
    auc_cb3 = float(roc_auc_score(y_true, p_cb3))
    auc_xgb3 = float(roc_auc_score(y_true, p_xgb3))
    auc_lgb = float(roc_auc_score(y_true, p_lgb))
    auc_prob = float(roc_auc_score(y_true, prob_blend))
    auc_rank = float(roc_auc_score(y_true, rank_blend))

    # Fold-by-fold evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_indices = list(skf.split(train_df, y_true))

    models_dict = {
        "CatBoost_Config_4": p_cb4,
        "CatBoost_Config_3": p_cb3,
        "XGBoost_Config_3": p_xgb3,
        "Tuned_LightGBM": p_lgb,
        "Probability_Blend_V2": prob_blend,
        "Rank_Normalized_Blend_V21": rank_blend,
    }

    fold_results = {}
    for name, preds in models_dict.items():
        f_scores = [float(roc_auc_score(y_true[val_idx], preds[val_idx])) for _, val_idx in fold_indices]
        fold_results[name] = {
            "fold_scores": [round(s, 5) for s in f_scores],
            "mean_auc": round(float(np.mean(f_scores)), 5),
            "std_auc": round(float(np.std(f_scores)), 5),
            "overall_oof_auc": round(float(roc_auc_score(y_true, preds)), 5),
        }

    # Spearman and Pearson correlation
    corr_df = pd.DataFrame({
        "CB4": p_cb4,
        "CB3": p_cb3,
        "XGB3": p_xgb3,
        "LGB": p_lgb,
        "Prob_Blend": prob_blend,
        "Rank_Blend": rank_blend,
    })

    summary = {
        "n_samples": n_train,
        "weights": {"CB4": w_cb4, "CB3": w_cb3, "XGB3": w_xgb, "LGB": w_lgb},
        "overall_scores": {
            "CB4": round(auc_cb4, 5),
            "CB3": round(auc_cb3, 5),
            "XGB3": round(auc_xgb3, 5),
            "LGB": round(auc_lgb, 5),
            "prob_blend": round(auc_prob, 5),
            "rank_blend": round(auc_rank, 5),
            "delta_rank_vs_prob": round(auc_rank - auc_prob, 5),
        },
        "fold_results": fold_results,
        "pearson_correlation": corr_df.corr(method="pearson").round(5).to_dict(),
        "spearman_correlation": corr_df.corr(method="spearman").round(5).to_dict(),
    }

    output_json = PROJECT_ROOT / "experiments" / "stage_h4_rank_blend_results.json"
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved evaluation results to {output_json}")

    return summary


if __name__ == "__main__":
    run_evaluation()
