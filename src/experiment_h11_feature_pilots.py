"""
Stage H11: Phase 3 - Systematic Feature Engineering Screening Pilots (Fold 1).
Tests hypothesis-driven features derived from Phase 2 error analysis:
1. Candidate Set A: External Source Interactions & Robust Aggregations
   - EXT_SOURCE_PROD_2_3 = EXT_SOURCE_2 * EXT_SOURCE_3
   - EXT_SOURCE_DIFF_2_3 = EXT_SOURCE_2 - EXT_SOURCE_3
   - EXT_SOURCE_WEIGHTED_MEAN = (0.32*EXT1 + 0.31*EXT2 + 0.37*EXT3) / weights_sum
   - EXT_SOURCE_MIN_MAX_RATIO = EXT_SOURCES_MIN / (EXT_SOURCES_MAX + 1e-5)
2. Candidate Set B: Debt Burden, Loan Term & Financing Differences
   - GOODS_CREDIT_DIFF = AMT_CREDIT - AMT_GOODS_PRICE (Financed excess / down payment)
   - CREDIT_TERM_YEARS = AMT_CREDIT / (AMT_ANNUITY + 1e-5)
   - DISPOSABLE_INCOME = AMT_INCOME_TOTAL - AMT_ANNUITY
   - ANNUITY_PER_PERSON = AMT_ANNUITY / (CNT_FAM_MEMBERS + 1e-5)
3. Candidate Set C: Youth & Employment Stability Features
   - ADULT_EMPLOYED_RATIO = DAYS_EMPLOYED / (DAYS_BIRTH + 1e-5)
   - INCOME_TO_AGE = AMT_INCOME_TOTAL / (-DAYS_BIRTH / 365.25 + 1e-5)
   - REGISTRATION_TO_BIRTH = DAYS_REGISTRATION / (DAYS_BIRTH + 1e-5)
   - ID_PUBLISH_TO_BIRTH = DAYS_ID_PUBLISH / (DAYS_BIRTH + 1e-5)
4. Candidate Set D: Leakage-Safe Out-of-Fold Peer Group Aggregations
   - Median income by (NAME_EDUCATION_TYPE, OCCUPATION_TYPE) fitted strictly on training fold
   - RELATIVE_INCOME_TO_PEER = AMT_INCOME_TOTAL / median_peer_income
5. Candidate Set E: Combined Best (All non-redundant candidates)

All evaluated on Fold 1 against the verified LightGBM C3_TE baseline (Fold 1 AUC = 0.75940).
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS

def main():
    print("=" * 80)
    print("STAGE H11: FEATURE ENGINEERING SCREENING PILOTS (FOLD 1)")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(skf.split(X_raw, y_true))

    y_tr = y_true[tr_idx]
    y_va = y_true[va_idx]

    params_lgb = {
        "learning_rate": 0.025,
        "num_leaves": 20,
        "max_depth": 6,
        "min_child_samples": 120,
        "feature_fraction": 0.65,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "reg_alpha": 3.0,
        "reg_lambda": 15.0,
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 43,
    }

    # Step 1: Baseline Pipeline
    pipe = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
    pipe.fit(X_raw.iloc[tr_idx])
    X_tr_base = pipe.transform(X_raw.iloc[tr_idx])
    X_va_base = pipe.transform(X_raw.iloc[va_idx])

    high_card = ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
    low_card = [c for c in CATEGORICAL_COLS if c not in high_card and c in X_tr_base.columns]

    skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    te = TargetEncoder(smooth="auto", cv=skf_inner)
    X_tr_te = te.fit_transform(X_tr_base[high_card], y_tr)
    X_va_te = te.transform(X_va_base[high_card])

    for i, c in enumerate(high_card):
        X_tr_base[f"{c}_TE"] = X_tr_te[:, i]
        X_va_base[f"{c}_TE"] = X_va_te[:, i]
        X_tr_base.drop(columns=[c], inplace=True)
        X_va_base.drop(columns=[c], inplace=True)

    for col in low_card:
        cats = sorted(X_tr_base[col].dropna().unique().tolist())
        cat_map = {c: idx for idx, c in enumerate(cats)}
        X_tr_base[col] = X_tr_base[col].map(cat_map).fillna(-1).astype(int)
        X_va_base[col] = X_va_base[col].map(cat_map).fillna(-1).astype(int)

    def evaluate_features(X_tr, X_va, name):
        dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=low_card, free_raw_data=False)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=low_card, free_raw_data=False)
        model = lgb.train(
            params_lgb,
            dtr,
            num_boost_round=1500,
            valid_sets=[dtr, dva],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        p_va = model.predict(X_va, num_iteration=model.best_iteration)
        auc = float(roc_auc_score(y_va, p_va))
        return auc, model.best_iteration

    auc_base, iter_base = evaluate_features(X_tr_base, X_va_base, "Baseline")
    print(f"\n[Baseline] LightGBM C3_TE (46 features): Fold 1 AUC = {auc_base:.5f} (Best Iter: {iter_base})")
    results = {"baseline": {"auc": auc_base, "best_iter": iter_base}}

    # -------------------------------------------------------------
    # Candidate Set A: External Source Interactions & Combinations
    # -------------------------------------------------------------
    print("\n--- Testing Candidate Set A: External Source Interactions ---")
    X_tr_A = X_tr_base.copy()
    X_va_A = X_va_base.copy()

    # EXT2 * EXT3
    X_tr_A["EXT_PROD_2_3"] = X_tr_A["EXT_SOURCE_2"] * X_tr_A["EXT_SOURCE_3"]
    X_va_A["EXT_PROD_2_3"] = X_va_A["EXT_SOURCE_2"] * X_va_A["EXT_SOURCE_3"]

    # EXT2 - EXT3
    X_tr_A["EXT_DIFF_2_3"] = X_tr_A["EXT_SOURCE_2"] - X_tr_A["EXT_SOURCE_3"]
    X_va_A["EXT_DIFF_2_3"] = X_va_A["EXT_SOURCE_2"] - X_va_A["EXT_SOURCE_3"]

    # Weighted mean: 0.32 EXT1, 0.31 EXT2, 0.37 EXT3
    w1, w2, w3 = 0.32, 0.31, 0.37
    def calc_weighted_ext(df):
        s1 = df["EXT_SOURCE_1"].fillna(0) * w1
        s2 = df["EXT_SOURCE_2"].fillna(0) * w2
        s3 = df["EXT_SOURCE_3"].fillna(0) * w3
        denom = (~df["EXT_SOURCE_1"].isna())*w1 + (~df["EXT_SOURCE_2"].isna())*w2 + (~df["EXT_SOURCE_3"].isna())*w3
        return (s1 + s2 + s3) / (denom + 1e-5)

    X_tr_A["EXT_WEIGHTED_MEAN"] = calc_weighted_ext(X_tr_A)
    X_va_A["EXT_WEIGHTED_MEAN"] = calc_weighted_ext(X_va_A)

    auc_A, iter_A = evaluate_features(X_tr_A, X_va_A, "Set A")
    delta_A = auc_A - auc_base
    print(f"  Candidate Set A: Fold 1 AUC = {auc_A:.5f} (Delta: {delta_A:+.5f}, Best Iter: {iter_A})")
    results["candidate_set_A"] = {"auc": auc_A, "delta": delta_A, "best_iter": iter_A}

    # -------------------------------------------------------------
    # Candidate Set B: Debt Burden, Term & Financing Differences
    # -------------------------------------------------------------
    print("\n--- Testing Candidate Set B: Debt Burden & Financing Differences ---")
    X_tr_B = X_tr_base.copy()
    X_va_B = X_va_base.copy()

    # Credit minus Goods price (financing excess)
    X_tr_B["GOODS_CREDIT_DIFF"] = X_tr_B["AMT_CREDIT"] - X_tr_B["AMT_GOODS_PRICE"]
    X_va_B["GOODS_CREDIT_DIFF"] = X_va_B["AMT_CREDIT"] - X_va_B["AMT_GOODS_PRICE"]

    # Implied loan duration in years
    X_tr_B["CREDIT_TERM_YEARS"] = X_tr_B["AMT_CREDIT"] / (X_tr_B["AMT_ANNUITY"] + 1e-5)
    X_va_B["CREDIT_TERM_YEARS"] = X_va_B["AMT_CREDIT"] / (X_va_B["AMT_ANNUITY"] + 1e-5)

    # Disposable income
    X_tr_B["DISPOSABLE_INCOME"] = X_tr_B["AMT_INCOME_TOTAL"] - X_tr_B["AMT_ANNUITY"]
    X_va_B["DISPOSABLE_INCOME"] = X_va_B["AMT_INCOME_TOTAL"] - X_va_B["AMT_ANNUITY"]

    # Annuity per family member
    X_tr_B["ANNUITY_PER_PERSON"] = X_tr_B["AMT_ANNUITY"] / (X_tr_B["CNT_FAM_MEMBERS"] + 1e-5)
    X_va_B["ANNUITY_PER_PERSON"] = X_va_B["AMT_ANNUITY"] / (X_va_B["CNT_FAM_MEMBERS"] + 1e-5)

    auc_B, iter_B = evaluate_features(X_tr_B, X_va_B, "Set B")
    delta_B = auc_B - auc_base
    print(f"  Candidate Set B: Fold 1 AUC = {auc_B:.5f} (Delta: {delta_B:+.5f}, Best Iter: {iter_B})")
    results["candidate_set_B"] = {"auc": auc_B, "delta": delta_B, "best_iter": iter_B}

    # -------------------------------------------------------------
    # Candidate Set C: Youth & Employment Stability Features
    # -------------------------------------------------------------
    print("\n--- Testing Candidate Set C: Youth & Employment Stability ---")
    X_tr_C = X_tr_base.copy()
    X_va_C = X_va_base.copy()

    # Fraction of adult life employed
    X_tr_C["ADULT_EMPLOYED_RATIO"] = X_tr_C["DAYS_EMPLOYED"] / (X_tr_C["DAYS_BIRTH"] + 1e-5)
    X_va_C["ADULT_EMPLOYED_RATIO"] = X_va_C["DAYS_EMPLOYED"] / (X_va_C["DAYS_BIRTH"] + 1e-5)

    # Income to age in years
    X_tr_C["INCOME_TO_AGE"] = X_tr_C["AMT_INCOME_TOTAL"] / (-X_tr_C["DAYS_BIRTH"] / 365.25 + 1e-5)
    X_va_C["INCOME_TO_AGE"] = X_va_C["AMT_INCOME_TOTAL"] / (-X_va_C["DAYS_BIRTH"] / 365.25 + 1e-5)

    # Days registration / Days birth
    X_tr_C["REGISTRATION_TO_BIRTH"] = X_tr_C["DAYS_REGISTRATION"] / (X_tr_C["DAYS_BIRTH"] + 1e-5)
    X_va_C["REGISTRATION_TO_BIRTH"] = X_va_C["DAYS_REGISTRATION"] / (X_va_C["DAYS_BIRTH"] + 1e-5)

    auc_C, iter_C = evaluate_features(X_tr_C, X_va_C, "Set C")
    delta_C = auc_C - auc_base
    print(f"  Candidate Set C: Fold 1 AUC = {auc_C:.5f} (Delta: {delta_C:+.5f}, Best Iter: {iter_C})")
    results["candidate_set_C"] = {"auc": auc_C, "delta": delta_C, "best_iter": iter_C}

    # -------------------------------------------------------------
    # Candidate Set D: Leakage-Safe Out-of-Fold Peer Group Aggregations
    # -------------------------------------------------------------
    print("\n--- Testing Candidate Set D: Peer Group Aggregations ---")
    X_tr_D = X_tr_base.copy()
    X_va_D = X_va_base.copy()

    # Group by (EDUCATION_TYPE, OCCUPATION_TYPE) to get median income, fitted ONLY on training fold
    train_slice = X_raw.iloc[tr_idx].copy()
    val_slice = X_raw.iloc[va_idx].copy()
    group_cols = ["NAME_EDUCATION_TYPE", "OCCUPATION_TYPE"]

    group_medians = train_slice.groupby(group_cols)["AMT_INCOME_TOTAL"].median().reset_index()
    group_medians.rename(columns={"AMT_INCOME_TOTAL": "PEER_MEDIAN_INCOME"}, inplace=True)
    global_median_income = train_slice["AMT_INCOME_TOTAL"].median()

    train_merged = train_slice[group_cols].merge(group_medians, on=group_cols, how="left")
    val_merged = val_slice[group_cols].merge(group_medians, on=group_cols, how="left")

    X_tr_D["RELATIVE_INCOME_TO_PEER"] = X_tr_D["AMT_INCOME_TOTAL"] / (train_merged["PEER_MEDIAN_INCOME"].fillna(global_median_income).values + 1e-5)
    X_va_D["RELATIVE_INCOME_TO_PEER"] = X_va_D["AMT_INCOME_TOTAL"] / (val_merged["PEER_MEDIAN_INCOME"].fillna(global_median_income).values + 1e-5)

    auc_D, iter_D = evaluate_features(X_tr_D, X_va_D, "Set D")
    delta_D = auc_D - auc_base
    print(f"  Candidate Set D: Fold 1 AUC = {auc_D:.5f} (Delta: {delta_D:+.5f}, Best Iter: {iter_D})")
    results["candidate_set_D"] = {"auc": auc_D, "delta": delta_D, "best_iter": iter_D}

    # -------------------------------------------------------------
    # Candidate Set E: Synergistic Combinations (Best Individual Features)
    # -------------------------------------------------------------
    print("\n--- Testing Candidate Set E: Individual Feature Screening ---")
    individual_tests = [
        ("EXT_PROD_2_3", X_tr_A[["EXT_PROD_2_3"]], X_va_A[["EXT_PROD_2_3"]]),
        ("EXT_DIFF_2_3", X_tr_A[["EXT_DIFF_2_3"]], X_va_A[["EXT_DIFF_2_3"]]),
        ("EXT_WEIGHTED_MEAN", X_tr_A[["EXT_WEIGHTED_MEAN"]], X_va_A[["EXT_WEIGHTED_MEAN"]]),
        ("GOODS_CREDIT_DIFF", X_tr_B[["GOODS_CREDIT_DIFF"]], X_va_B[["GOODS_CREDIT_DIFF"]]),
        ("CREDIT_TERM_YEARS", X_tr_B[["CREDIT_TERM_YEARS"]], X_va_B[["CREDIT_TERM_YEARS"]]),
        ("DISPOSABLE_INCOME", X_tr_B[["DISPOSABLE_INCOME"]], X_va_B[["DISPOSABLE_INCOME"]]),
        ("ANNUITY_PER_PERSON", X_tr_B[["ANNUITY_PER_PERSON"]], X_va_B[["ANNUITY_PER_PERSON"]]),
        ("ADULT_EMPLOYED_RATIO", X_tr_C[["ADULT_EMPLOYED_RATIO"]], X_va_C[["ADULT_EMPLOYED_RATIO"]]),
        ("INCOME_TO_AGE", X_tr_C[["INCOME_TO_AGE"]], X_va_C[["INCOME_TO_AGE"]]),
        ("RELATIVE_INCOME_TO_PEER", X_tr_D[["RELATIVE_INCOME_TO_PEER"]], X_va_D[["RELATIVE_INCOME_TO_PEER"]]),
    ]

    for fname, tr_col, va_col in individual_tests:
        X_tr_ind = X_tr_base.copy()
        X_va_ind = X_va_base.copy()
        X_tr_ind[fname] = tr_col.values
        X_va_ind[fname] = va_col.values
        auc_ind, iter_ind = evaluate_features(X_tr_ind, X_va_ind, fname)
        delta_ind = auc_ind - auc_base
        marker = "==> POSITIVE" if delta_ind > 0 else ""
        print(f"    {fname:25s}: Fold 1 AUC = {auc_ind:.5f} (Delta: {delta_ind:+.5f}) {marker}")
        results[f"ind_{fname}"] = {"auc": auc_ind, "delta": delta_ind, "best_iter": iter_ind}

    out_file = PROJECT_ROOT / "experiments" / "stage_h11" / "feature_engineering_fold1_screening.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved screening results to: {out_file}")

if __name__ == "__main__":
    main()

