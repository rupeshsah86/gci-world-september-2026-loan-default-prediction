"""
Stage H11: Phase 1 & Phase 2 - Audit, Benchmark Reproduction & Deep Error Analysis.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL

def to_rank(p):
    return (rankdata(p) - 0.5) / len(p)

def main():
    print("=" * 80)
    print("STAGE H11: PHASE 1 (AUDIT & REPRODUCTION) & PHASE 2 (ERROR ANALYSIS)")
    print("=" * 80)

    # 1. Load Data
    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    sk_ids = train_df[ID_COL].values
    n = len(train_df)

    # 2. Phase 1: Load Predictions & Reproduce Benchmarks
    print("\n--- PHASE 1: BENCHMARK REPRODUCTION & INTEGRITY AUDIT ---")
    cb4 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_4_High_L2_Regularization.csv")
    cb3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_3_RandomStrength_2_LR04.csv")
    xgb_3a = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h9" / "oof_xgboost_c3a_target_encoded.csv")
    lgb_c2 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h9" / "oof_lightgbm_c2_target_encoded.csv")
    lgb_c3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h10" / "oof_lightgbm_c3_te.csv")

    for name, df_ in [("CB4", cb4), ("CB3", cb3), ("XGB_3A", xgb_3a), ("LGB_C2", lgb_c2), ("LGB_C3", lgb_c3)]:
        assert len(df_) == n, f"Row count mismatch in {name}"
        assert (df_[ID_COL].values == sk_ids).all(), f"ID alignment failure in {name}"
        assert (df_[TARGET_COL].values == y_true).all(), f"Target mismatch in {name}"
        assert not df_["oof_pred"].isna().any(), f"NaNs in {name}"

    r_cb4 = to_rank(cb4["oof_pred"].values)
    r_cb3 = to_rank(cb3["oof_pred"].values)
    r_xgb_3a = to_rank(xgb_3a["oof_pred"].values)
    r_lgb_c2 = to_rank(lgb_c2["oof_pred"].values)
    r_lgb_c3 = to_rank(lgb_c3["oof_pred"].values)

    # Candidate V2.5-C
    b_v25c_rank = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb_3a + 0.20 * r_lgb_c2
    auc_v25c = float(roc_auc_score(y_true, b_v25c_rank))

    # Candidate V2.6-A
    b_v26a_rank = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb_3a + 0.20 * r_lgb_c3
    auc_v26a = float(roc_auc_score(y_true, b_v26a_rank))

    print(f"  [+] Reproduced Candidate V2.5-C OOF AUC: {auc_v25c:.5f} (Target: 0.76302)")
    print(f"  [+] Reproduced Candidate V2.6-A OOF AUC: {auc_v26a:.5f} (Target: 0.76306)")
    assert abs(auc_v25c - 0.76302) < 1e-4, "Failed to reproduce V2.5-C!"
    assert abs(auc_v26a - 0.76306) < 1e-4, "Failed to reproduce V2.6-A!"
    print("  [+] Phase 1 Audit Passed with 100% precision.")

    # 3. Phase 2: Error Analysis
    print("\n--- PHASE 2: ERROR ANALYSIS & HARD EXAMPLE SEGMENTATION ---")
    p_blend = b_v25c_rank # Rank-normalized probability surrogate
    residuals = y_true - p_blend

    analysis_df = train_df.copy()
    analysis_df["rank_pred"] = p_blend
    analysis_df["residual"] = residuals
    analysis_df["abs_residual"] = np.abs(residuals)

    # A. Difficult Examples: Worst False Negatives and Worst False Positives
    # Worst FN: Defaulted (y=1) but predicted lowest rank
    worst_fn = analysis_df[analysis_df[TARGET_COL] == 1].sort_values("rank_pred", ascending=True).head(1000)
    # Worst FP: Non-defaulted (y=0) but predicted highest rank
    worst_fp = analysis_df[analysis_df[TARGET_COL] == 0].sort_values("rank_pred", ascending=False).head(1000)

    print(f"  Top 1,000 Worst False Negatives (y=1): Mean Rank = {worst_fn['rank_pred'].mean():.4f}")
    print(f"  Top 1,000 Worst False Positives (y=0): Mean Rank = {worst_fp['rank_pred'].mean():.4f}")

    # Inspect key driver features in Worst FN vs Typical Defaults
    typical_defaults = analysis_df[analysis_df[TARGET_COL] == 1]
    print("\n  Comparison: Worst False Negatives vs Typical Defaults:")
    for feat in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "AMT_CREDIT", "AMT_INCOME_TOTAL"]:
        if feat in analysis_df.columns:
            fn_mean = worst_fn[feat].mean()
            typ_mean = typical_defaults[feat].mean()
            print(f"    {feat:20s}: Worst FN Mean = {fn_mean:.4f} | All Defaults Mean = {typ_mean:.4f}")

    # B. Performance by Segments:
    # 1. External Score Availability Pattern
    print("\n  ROC-AUC by External Source Availability Pattern:")
    has_ext1 = (~analysis_df["EXT_SOURCE_1"].isna()).astype(int)
    has_ext2 = (~analysis_df["EXT_SOURCE_2"].isna()).astype(int)
    has_ext3 = (~analysis_df["EXT_SOURCE_3"].isna()).astype(int)
    analysis_df["ext_pattern"] = has_ext1.astype(str) + has_ext2.astype(str) + has_ext3.astype(str)
    
    pattern_metrics = {}
    for pat, grp in analysis_df.groupby("ext_pattern"):
        if grp[TARGET_COL].nunique() > 1 and len(grp) > 500:
            pat_auc = float(roc_auc_score(grp[TARGET_COL], grp["rank_pred"]))
            pat_count = len(grp)
            def_rate = grp[TARGET_COL].mean()
            pattern_metrics[pat] = {"auc": pat_auc, "count": pat_count, "default_rate": def_rate}
            print(f"    Pattern (EXT1/2/3: {pat}): Count = {pat_count:6d} | Default Rate = {def_rate:.3%} | AUC = {pat_auc:.5f}")

    # 2. Age Groups
    analysis_df["age_years"] = -analysis_df["DAYS_BIRTH"] / 365.25
    analysis_df["age_group"] = pd.cut(analysis_df["age_years"], bins=[0, 30, 40, 50, 60, 100], labels=["<30", "30-40", "40-50", "50-60", "60+"])
    print("\n  ROC-AUC by Age Group:")
    age_metrics = {}
    for ag, grp in analysis_df.groupby("age_group", observed=False):
        ag_auc = float(roc_auc_score(grp[TARGET_COL], grp["rank_pred"]))
        ag_cnt = len(grp)
        ag_def = grp[TARGET_COL].mean()
        age_metrics[str(ag)] = {"auc": ag_auc, "count": ag_cnt, "default_rate": ag_def}
        print(f"    Age Group {str(ag):6s}: Count = {ag_cnt:6d} | Default Rate = {ag_def:.3%} | AUC = {ag_auc:.5f}")

    # 3. Income Deciles
    analysis_df["income_decile"] = pd.qcut(analysis_df["AMT_INCOME_TOTAL"], q=10, labels=False, duplicates="drop")
    print("\n  ROC-AUC by Income Decile (0=Lowest, 9=Highest):")
    income_metrics = {}
    for inc, grp in analysis_df.groupby("income_decile"):
        inc_auc = float(roc_auc_score(grp[TARGET_COL], grp["rank_pred"]))
        income_metrics[int(inc)] = {"auc": inc_auc, "count": len(grp), "default_rate": grp[TARGET_COL].mean()}
        print(f"    Decile {inc}: Count = {len(grp):6d} | Default Rate = {grp[TARGET_COL].mean():.3%} | AUC = {inc_auc:.5f}")

    # 4. Model Disagreement Analysis: Where do CB4, XGB3A, and LGB_C3 differ most?
    print("\n  Model Disagreement Diagnostics:")
    analysis_df["diff_cb_xgb"] = np.abs(r_cb4 - r_xgb_3a)
    analysis_df["diff_cb_lgb"] = np.abs(r_cb4 - r_lgb_c3)
    analysis_df["diff_xgb_lgb"] = np.abs(r_xgb_3a - r_lgb_c3)
    analysis_df["total_disagreement"] = analysis_df["diff_cb_xgb"] + analysis_df["diff_cb_lgb"] + analysis_df["diff_xgb_lgb"]

    high_disagree = analysis_df.sort_values("total_disagreement", ascending=False).head(5000)
    low_disagree = analysis_df.sort_values("total_disagreement", ascending=True).head(5000)

    print(f"    High Disagreement (Top 5,000) Default Rate: {high_disagree[TARGET_COL].mean():.3%} | V2.5-C AUC: {roc_auc_score(high_disagree[TARGET_COL], high_disagree['rank_pred']):.5f}")
    print(f"    Low Disagreement (Top 5,000) Default Rate:  {low_disagree[TARGET_COL].mean():.3%} | V2.5-C AUC: {roc_auc_score(low_disagree[TARGET_COL], low_disagree['rank_pred']):.5f}")

    # Check which features correlate with model disagreement
    print("\n  Top Feature Correlations with Model Disagreement:")
    num_cols = analysis_df.select_dtypes(include=[np.number]).columns
    disagree_corrs = {}
    for col in num_cols:
        if col not in ["total_disagreement", "diff_cb_xgb", "diff_cb_lgb", "diff_xgb_lgb", "rank_pred", "residual", "abs_residual", "TARGET", "SK_ID_CURR"]:
            mask = ~analysis_df[col].isna()
            if mask.sum() > 100:
                c = float(analysis_df.loc[mask, col].corr(analysis_df.loc[mask, "total_disagreement"]))
                disagree_corrs[col] = abs(c)

    sorted_disagree = sorted(disagree_corrs.items(), key=lambda x: x[1], reverse=True)[:8]
    for col, c in sorted_disagree:
        print(f"    {col:30s}: |Correlation| = {c:.4f}")

    # Save summary
    results = {
        "verified_v25c_oof": auc_v25c,
        "verified_v26a_oof": auc_v26a,
        "pattern_metrics": pattern_metrics,
        "age_metrics": age_metrics,
        "income_metrics": income_metrics,
        "top_disagreement_features": dict(sorted_disagree)
    }

    out_file = PROJECT_ROOT / "experiments" / "stage_h11" / "phase1_phase2_audit_error_summary.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved Phase 1 & 2 audit results to: {out_file}")

if __name__ == "__main__":
    main()

