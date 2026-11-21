"""
Stage H10: Experiment Set 1 & 2 Pilot Screening on Fold 1.
Tests:
1. Baseline: LightGBM C2 with standard Target-Encoded (OCCUPATION_TYPE, ORGANIZATION_TYPE).
2. Variant 1A: Interaction Target Encoding (OCC_ORG = OCCUPATION_TYPE + "__" + ORGANIZATION_TYPE).
3. Variant 1B: Expanded Target Encoding (OCC, ORG, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE).
4. Variant 1C: Target Encoding smoothing sensitivity (smooth=5.0, 10.0, 20.0 vs "auto").
5. Variant 2A: Missingness indicators (MISSING_EXT1, MISSING_EXT3, MISSING_BUREAU).
6. Variant 2B: Interaction Target Encoding + Missingness indicators.
All fitted strictly on Training Fold 1; evaluated strictly on Validation Fold 1.
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
    print("STAGE H10: PILOT SCREENING (FOLD 1 STRICT ISOLATION)")
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
        "reg_alpha": 2.0,
        "reg_lambda": 10.0,
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 43,
    }

    # Pilot 0: Stage H9 Baseline LightGBM C2 (Fold 1 score = 0.75871)
    print("\n--- Pilot 0: Stage H9 Baseline (OCC & ORG Target-Encoded) ---")
    pipe0 = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
    pipe0.fit(X_raw.iloc[tr_idx])
    X_tr0 = pipe0.transform(X_raw.iloc[tr_idx])
    X_va0 = pipe0.transform(X_raw.iloc[va_idx])

    high_card = ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
    low_card = [c for c in CATEGORICAL_COLS if c not in high_card and c in X_tr0.columns]

    te0 = TargetEncoder(smooth="auto", cv=5, random_state=42)
    X_tr0_te = te0.fit_transform(X_tr0[high_card], y_tr)
    X_va0_te = te0.transform(X_va0[high_card])

    for i, c in enumerate(high_card):
        X_tr0[f"{c}_TE"] = X_tr0_te[:, i]
        X_va0[f"{c}_TE"] = X_va0_te[:, i]
        X_tr0.drop(columns=[c], inplace=True)
        X_va0.drop(columns=[c], inplace=True)

    for col in low_card:
        cats = sorted(X_tr0[col].dropna().unique().tolist())
        cat_map = {c: idx for idx, c in enumerate(cats)}
        X_tr0[col] = X_tr0[col].map(cat_map).fillna(-1).astype(int)
        X_va0[col] = X_va0[col].map(cat_map).fillna(-1).astype(int)

    dtr0 = lgb.Dataset(X_tr0, label=y_tr, categorical_feature=low_card, free_raw_data=False)
    dva0 = lgb.Dataset(X_va0, label=y_va, reference=dtr0, categorical_feature=low_card, free_raw_data=False)
    m0 = lgb.train(params_lgb, dtr0, num_boost_round=1500, valid_sets=[dtr0, dva0], callbacks=[lgb.early_stopping(50, verbose=False)])
    auc0 = roc_auc_score(y_va, m0.predict(X_va0, num_iteration=m0.best_iteration))
    print(f"  Pilot 0 (Baseline Fold 1): AUC = {auc0:.5f} (Best Iter: {m0.best_iteration})")

    results = {"pilot_0_baseline": float(auc0)}

    # Pilot 1A: Interaction Target Encoding (OCC_ORG)
    print("\n--- Pilot 1A: Interaction Target Encoding (OCC_ORG) ---")
    X_tr1a = X_tr0.copy()
    X_va1a = X_va0.copy()

    # Create interaction
    occ_org_tr = X_raw.iloc[tr_idx]["OCCUPATION_TYPE"].astype(str) + "__" + X_raw.iloc[tr_idx]["ORGANIZATION_TYPE"].astype(str)
    occ_org_va = X_raw.iloc[va_idx]["OCCUPATION_TYPE"].astype(str) + "__" + X_raw.iloc[va_idx]["ORGANIZATION_TYPE"].astype(str)

    df_occ_org_tr = pd.DataFrame({"OCC_ORG": occ_org_tr})
    df_occ_org_va = pd.DataFrame({"OCC_ORG": occ_org_va})

    te1a = TargetEncoder(smooth="auto", cv=5, random_state=42)
    X_tr1a["OCC_ORG_TE"] = te1a.fit_transform(df_occ_org_tr, y_tr).ravel()
    X_va1a["OCC_ORG_TE"] = te1a.transform(df_occ_org_va).ravel()

    dtr1a = lgb.Dataset(X_tr1a, label=y_tr, categorical_feature=low_card, free_raw_data=False)
    dva1a = lgb.Dataset(X_va1a, label=y_va, reference=dtr1a, categorical_feature=low_card, free_raw_data=False)
    m1a = lgb.train(params_lgb, dtr1a, num_boost_round=1500, valid_sets=[dtr1a, dva1a], callbacks=[lgb.early_stopping(50, verbose=False)])
    auc1a = roc_auc_score(y_va, m1a.predict(X_va1a, num_iteration=m1a.best_iteration))
    print(f"  Pilot 1A (OCC_ORG Interaction): AUC = {auc1a:.5f} (Delta: {auc1a - auc0:+.5f})")
    results["pilot_1a_occ_org"] = float(auc1a)

    # Pilot 1B: Expanded Target Encoding (also encode INCOME_TYPE, EDUCATION_TYPE)
    print("\n--- Pilot 1B: Expanded Target Encoding (OCC, ORG, INCOME, EDUCATION) ---")
    pipe1b = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
    pipe1b.fit(X_raw.iloc[tr_idx])
    X_tr1b = pipe1b.transform(X_raw.iloc[tr_idx])
    X_va1b = pipe1b.transform(X_raw.iloc[va_idx])

    expanded_cats = ["OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE"]
    remaining_cats = [c for c in CATEGORICAL_COLS if c not in expanded_cats and c in X_tr1b.columns]

    te1b = TargetEncoder(smooth="auto", cv=5, random_state=42)
    X_tr1b_te = te1b.fit_transform(X_tr1b[expanded_cats], y_tr)
    X_va1b_te = te1b.transform(X_va1b[expanded_cats])

    for i, c in enumerate(expanded_cats):
        X_tr1b[f"{c}_TE"] = X_tr1b_te[:, i]
        X_va1b[f"{c}_TE"] = X_va1b_te[:, i]
        X_tr1b.drop(columns=[c], inplace=True)
        X_va1b.drop(columns=[c], inplace=True)

    for col in remaining_cats:
        cats = sorted(X_tr1b[col].dropna().unique().tolist())
        cat_map = {c: idx for idx, c in enumerate(cats)}
        X_tr1b[col] = X_tr1b[col].map(cat_map).fillna(-1).astype(int)
        X_va1b[col] = X_va1b[col].map(cat_map).fillna(-1).astype(int)

    dtr1b = lgb.Dataset(X_tr1b, label=y_tr, categorical_feature=remaining_cats, free_raw_data=False)
    dva1b = lgb.Dataset(X_va1b, label=y_va, reference=dtr1b, categorical_feature=remaining_cats, free_raw_data=False)
    m1b = lgb.train(params_lgb, dtr1b, num_boost_round=1500, valid_sets=[dtr1b, dva1b], callbacks=[lgb.early_stopping(50, verbose=False)])
    auc1b = roc_auc_score(y_va, m1b.predict(X_va1b, num_iteration=m1b.best_iteration))
    print(f"  Pilot 1B (Expanded TE): AUC = {auc1b:.5f} (Delta: {auc1b - auc0:+.5f})")
    results["pilot_1b_expanded_te"] = float(auc1b)

    # Pilot 1C: Target Encoding with smooth=10.0 and smooth=20.0
    print("\n--- Pilot 1C: Target Encoding Smoothing Tuning ---")
    for s_val in [5.0, 10.0, 20.0, 50.0]:
        pipe1c = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
        pipe1c.fit(X_raw.iloc[tr_idx])
        X_tr1c = pipe1c.transform(X_raw.iloc[tr_idx])
        X_va1c = pipe1c.transform(X_raw.iloc[va_idx])

        te1c = TargetEncoder(smooth=s_val, cv=5, random_state=42)
        X_tr1c_te = te1c.fit_transform(X_tr1c[high_card], y_tr)
        X_va1c_te = te1c.transform(X_va1c[high_card])

        for i, c in enumerate(high_card):
            X_tr1c[f"{c}_TE"] = X_tr1c_te[:, i]
            X_va1c[f"{c}_TE"] = X_va1c_te[:, i]
            X_tr1c.drop(columns=[c], inplace=True)
            X_va1c.drop(columns=[c], inplace=True)

        for col in low_card:
            cats = sorted(X_tr1c[col].dropna().unique().tolist())
            cat_map = {c: idx for idx, c in enumerate(cats)}
            X_tr1c[col] = X_tr1c[col].map(cat_map).fillna(-1).astype(int)
            X_va1c[col] = X_va1c[col].map(cat_map).fillna(-1).astype(int)

        dtr1c = lgb.Dataset(X_tr1c, label=y_tr, categorical_feature=low_card, free_raw_data=False)
        dva1c = lgb.Dataset(X_va1c, label=y_va, reference=dtr1c, categorical_feature=low_card, free_raw_data=False)
        m1c = lgb.train(params_lgb, dtr1c, num_boost_round=1500, valid_sets=[dtr1c, dva1c], callbacks=[lgb.early_stopping(50, verbose=False)])
        auc1c = roc_auc_score(y_va, m1c.predict(X_va1c, num_iteration=m1c.best_iteration))
        print(f"  Pilot 1C (smooth={s_val}): AUC = {auc1c:.5f} (Delta: {auc1c - auc0:+.5f})")
        results[f"pilot_1c_smooth_{s_val}"] = float(auc1c)

    # Pilot 2A: Missingness Indicators
    print("\n--- Pilot 2A: Explicit Missingness Indicators ---")
    X_tr2a = X_tr0.copy()
    X_va2a = X_va0.copy()

    X_tr2a["MISSING_EXT1"] = X_raw.iloc[tr_idx]["EXT_SOURCE_1"].isna().astype(int).values
    X_va2a["MISSING_EXT1"] = X_raw.iloc[va_idx]["EXT_SOURCE_1"].isna().astype(int).values
    X_tr2a["MISSING_EXT3"] = X_raw.iloc[tr_idx]["EXT_SOURCE_3"].isna().astype(int).values
    X_va2a["MISSING_EXT3"] = X_raw.iloc[va_idx]["EXT_SOURCE_3"].isna().astype(int).values
    X_tr2a["MISSING_BUREAU"] = X_raw.iloc[tr_idx]["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().astype(int).values
    X_va2a["MISSING_BUREAU"] = X_raw.iloc[va_idx]["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().astype(int).values
    X_tr2a["EXT_1_3_BOTH_MISSING"] = (X_tr2a["MISSING_EXT1"] & X_tr2a["MISSING_EXT3"]).astype(int).values
    X_va2a["EXT_1_3_BOTH_MISSING"] = (X_va2a["MISSING_EXT1"] & X_va2a["MISSING_EXT3"]).astype(int).values

    dtr2a = lgb.Dataset(X_tr2a, label=y_tr, categorical_feature=low_card, free_raw_data=False)
    dva2a = lgb.Dataset(X_va2a, label=y_va, reference=dtr2a, categorical_feature=low_card, free_raw_data=False)
    m2a = lgb.train(params_lgb, dtr2a, num_boost_round=1500, valid_sets=[dtr2a, dva2a], callbacks=[lgb.early_stopping(50, verbose=False)])
    auc2a = roc_auc_score(y_va, m2a.predict(X_va2a, num_iteration=m2a.best_iteration))
    print(f"  Pilot 2A (Missingness Indicators): AUC = {auc2a:.5f} (Delta: {auc2a - auc0:+.5f})")
    results["pilot_2a_missingness"] = float(auc2a)

    out_file = PROJECT_ROOT / "experiments" / "stage_h10" / "pilot_screening_fold1_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved screening results to: {out_file}")

if __name__ == "__main__":
    main()

