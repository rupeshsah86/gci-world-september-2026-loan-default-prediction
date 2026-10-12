"""
Stage H2 Step 2: Feature Pilot Experiments.
Evaluates controlled feature engineering hypotheses across 5 stratified folds:
- Pilot 1: Out-of-Fold Smoothed Target Encoding (F1)
- Pilot 2: Non-Linear Cross-Interactions between Top Gain Drivers (F2/F3)
- Pilot 3: Frequency / Prevalence Encodings (F4)
- Pilot 4: Combined Winning Feature Candidate
- Verification against the verified baselines:
    LightGBM Baseline: Mean 0.75905, OOF 0.75900
    CatBoost Baseline: Mean 0.76060, OOF 0.76060
    Ensemble Baseline: OOF 0.76203
"""

import sys
from pathlib import Path
import json
import time
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import CATEGORICAL_COLS
from src.features_h2_step2 import (
    FeaturePipelineH2,
    compute_inner_oof_target_encoding,
)

# Tuned LightGBM hyperparameters
LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 24,
    "max_depth": 7,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.75,
    "bagging_freq": 1,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "verbose": -1,
    "n_jobs": -1,
}


def evaluate_lightgbm_pipeline(
    config_name: str,
    X_raw: pd.DataFrame,
    y: np.ndarray,
    use_target_encoding: bool = False,
    include_cross_interactions: bool = False,
    include_frequency_encodings: bool = False,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluates a feature configuration on LightGBM across 5 stratified folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(y))
    fold_scores = []
    fold_times = []
    best_iters = []
    feature_count = 0

    print(f"\n" + "-" * 65)
    print(f"EVALUATING LIGHTGBM: {config_name}")
    print("-" * 65)

    start_time = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_raw, y), 1):
        f_start = time.time()
        X_tr_raw = X_raw.iloc[train_idx].copy()
        y_tr = y[train_idx]
        X_va_raw = X_raw.iloc[val_idx].copy()
        y_va = y[val_idx]

        # 1. Feature transformations
        pipe = FeaturePipelineH2(
            include_cross_interactions=include_cross_interactions,
            include_frequency_encodings=include_frequency_encodings,
            encode_categories=True,
        )
        pipe.fit(X_tr_raw)
        X_tr_enc = pipe.transform(X_tr_raw)
        X_va_enc = pipe.transform(X_va_raw)

        # 2. Out-of-fold target encoding if requested
        if use_target_encoding:
            te_cats = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]
            X_tr_enc, X_va_enc = compute_inner_oof_target_encoding(
                X_tr_enc,
                y_tr,
                X_va_enc,
                cat_cols=te_cats,
                n_inner_splits=5,
                smoothing_weight=20.0,
                random_state=random_state + fold_idx,
            )

        feature_count = len(X_tr_enc.columns)
        cat_cols = [c for c in CATEGORICAL_COLS if c in X_tr_enc.columns]

        lgb_tr = lgb.Dataset(X_tr_enc, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
        lgb_va = lgb.Dataset(X_va_enc, label=y_va, reference=lgb_tr, categorical_feature=cat_cols, free_raw_data=False)

        params = {**LGB_PARAMS, "random_state": random_state + fold_idx}

        model = lgb.train(
            params,
            lgb_tr,
            num_boost_round=1500,
            valid_sets=[lgb_tr, lgb_va],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        preds = model.predict(X_va_enc, num_iteration=model.best_iteration)
        oof_preds[val_idx] = preds

        auc = float(roc_auc_score(y_va, preds))
        fold_scores.append(auc)
        best_iters.append(int(model.best_iteration))
        fold_times.append(float(time.time() - f_start))

        print(f"  Fold {fold_idx}: AUC = {auc:.5f} (Best Iter: {model.best_iteration}, Time: {fold_times[-1]:.1f}s)")

    total_time = time.time() - start_time
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    oof_auc = float(roc_auc_score(y, oof_preds))

    res = {
        "config_name": config_name,
        "feature_count": feature_count,
        "fold_scores": [round(s, 5) for s in fold_scores],
        "mean_auc": round(mean_auc, 5),
        "std_auc": round(std_auc, 5),
        "oof_auc": round(oof_auc, 5),
        "mean_best_iteration": round(float(np.mean(best_iters)), 1),
        "total_runtime_seconds": round(total_time, 2),
    }

    print(f"--> {config_name}: Mean AUC = {mean_auc:.5f} +/- {std_auc:.5f} | OOF = {oof_auc:.5f} | Time: {total_time:.1f}s")
    return res, oof_preds


def main():
    total_start = time.time()
    print("=" * 70)
    print("STAGE H2 STEP 2: FEATURE PILOT EXPERIMENTS")
    print("=" * 70)

    train_df, _, _ = load_raw_data(verbose=False)
    y = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])

    ref_lgb_mean = 0.75905
    ref_lgb_oof = 0.75900
    ref_ensemble_oof = 0.76203

    pilot_configs = [
        {
            "name": "Pilot_1_Target_Encoding",
            "use_target_encoding": True,
            "include_cross_interactions": False,
            "include_frequency_encodings": False,
            "desc": "Smoothed OOF target encoding for ORGANIZATION_TYPE and OCCUPATION_TYPE (m=20)",
        },
        {
            "name": "Pilot_2_Cross_Interactions",
            "use_target_encoding": False,
            "include_cross_interactions": True,
            "include_frequency_encodings": False,
            "desc": "5 non-linear interactions combining top gain drivers (PAYMENT_RATE, CREDIT_INCOME, EXT_SOURCES)",
        },
        {
            "name": "Pilot_3_Frequency_Encodings",
            "use_target_encoding": False,
            "include_cross_interactions": False,
            "include_frequency_encodings": True,
            "desc": "Frequency / prevalence encodings for high-cardinality categoricals",
        },
        {
            "name": "Pilot_4_Combined_Best_Features",
            "use_target_encoding": True,
            "include_cross_interactions": True,
            "include_frequency_encodings": True,
            "desc": "Full combination of OOF Target Encoding, Cross-Interactions, and Frequency Encodings",
        },
    ]

    all_pilot_results = []
    oof_predictions_dict = {}

    for cfg in pilot_configs:
        res, oof = evaluate_lightgbm_pipeline(
            config_name=cfg["name"],
            X_raw=X_raw,
            y=y,
            use_target_encoding=cfg["use_target_encoding"],
            include_cross_interactions=cfg["include_cross_interactions"],
            include_frequency_encodings=cfg["include_frequency_encodings"],
        )
        res["description"] = cfg["desc"]
        res["delta_vs_lgb_ref"] = round(res["mean_auc"] - ref_lgb_mean, 5)
        all_pilot_results.append(res)
        oof_predictions_dict[cfg["name"]] = oof

    # Identify best LightGBM pilot
    best_pilot = max(all_pilot_results, key=lambda r: r["mean_auc"])
    print("\n" + "=" * 70)
    print(f"PILOT SUMMARY: Best Performing Feature Candidate: {best_pilot['config_name']}")
    print(f"  Mean ROC-AUC: {best_pilot['mean_auc']:.5f} (Delta vs Ref: {best_pilot['delta_vs_lgb_ref']:+.5f})")
    print(f"  OOF ROC-AUC:  {best_pilot['oof_auc']:.5f}")
    print("=" * 70)

    # Save OOF of best pilot
    best_pilot_oof_path = PROJECT_ROOT / "experiments" / f"oof_lightgbm_{best_pilot['config_name']}.csv"
    pd.DataFrame({
        ID_COL: train_df[ID_COL],
        TARGET_COL: y,
        "oof_pred": oof_predictions_dict[best_pilot["config_name"]],
    }).to_csv(best_pilot_oof_path, index=False)
    print(f"[+] Saved best pilot OOF predictions to: {best_pilot_oof_path}")

    # Save experiment results JSON
    json_path = PROJECT_ROOT / "experiments" / "stage_h2_step2_results.json"
    full_json = {
        "reference_baseline_lgb": {"mean_auc": ref_lgb_mean, "oof_auc": ref_lgb_oof},
        "reference_baseline_ensemble": {"oof_auc": ref_ensemble_oof},
        "pilot_experiments": all_pilot_results,
        "best_pilot": best_pilot,
        "total_runtime_seconds": round(time.time() - total_start, 2),
    }
    with open(json_path, "w") as f:
        json.dump(full_json, f, indent=2)
    print(f"[+] Saved Stage H2 Step 2 experiment results to: {json_path}")

    # Generate Markdown Report
    report_path = PROJECT_ROOT / "reports" / "stage_h2_step2_feature_pilots.md"
    report_lines = [
        "# Stage H2 Step 2: Feature Pilot Experiments Report",
        "",
        "## 1. Executive Summary",
        f"- **Reference LightGBM Baseline**: Mean = **0.75905**, OOF = **0.75900** (46 features)",
        f"- **Reference Ensemble Baseline**: OOF = **0.76203**",
        "- **Evaluation Setup**: Identical 5-Fold Stratified Cross-Validation (`random_state=42`), zero data leakage.",
        f"- **Total Runtime**: {time.time() - total_start:.1f} seconds (~{(time.time() - total_start)/60:.1f} minutes) on Apple Silicon M2.",
        "",
        "## 2. Pilot Experiment Configurations",
        "1. **Pilot 1: Out-of-Fold Smoothed Target Encoding (F1)**:",
        "   - Category targets encoded using an internal 5-fold cross-split inside each training fold (Bayesian smoothing $m=20$).",
        "   - Evaluated on high-cardinality features: `ORGANIZATION_TYPE` and `OCCUPATION_TYPE`.",
        "2. **Pilot 2: Non-Linear Cross-Interactions (F2/F3)**:",
        "   - Non-linear cross-ratios combining the top gain drivers: `PAYMENT_RATE_X_EXT_MEAN`, `CREDIT_INCOME_TO_EXT_MEAN`, `ANNUITY_INCOME_TO_EXT_MEAN`, `EXT_MIN_TO_MAX_RATIO`, `PAYMENT_RATE_X_AGE`.",
        "3. **Pilot 3: Frequency / Prevalence Encodings (F4)**:",
        "   - Normalized category frequencies learned strictly from training folds for `ORGANIZATION_TYPE`, `OCCUPATION_TYPE`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`.",
        "4. **Pilot 4: Combined Feature Set**:",
        "   - Full unified pipeline incorporating Target Encoding, Cross-Interactions, and Frequency Encodings.",
        "",
        "## 3. Cross-Validation Results Comparison Table",
        "| Experiment Name | Feature Count | Folds ROC-AUC (1 to 5) | Mean ROC-AUC (+/- Std) | OOF ROC-AUC | Delta vs Reference (0.75905) | Runtime |",
        "| :--- | :---: | :--- | :--- | :---: | :---: | :---: |",
        f"| **Reference LightGBM (46 feats)** | 46 | `[0.75530, 0.76093, 0.76386, 0.75571, 0.75944]` | **0.75905 +/- 0.00323** | **0.75900** | Baseline | 34.7s |",
    ]

    for r in all_pilot_results:
        folds_str = str(r["fold_scores"])
        report_lines.append(
            f"| **{r['config_name']}** | {r['feature_count']} | `{folds_str}` | **{r['mean_auc']:.5f} +/- {r['std_auc']:.5f}** | **{r['oof_auc']:.5f}** | **{r['delta_vs_lgb_ref']:+.5f}** | {r['total_runtime_seconds']:.1f}s |"
        )

    report_lines.extend([
        "",
        "## 4. Key Findings: What Helped vs. What Hurt",
        f"- **Best Feature Set**: **`{best_pilot['config_name']}`** ({best_pilot['feature_count']} features) with Mean ROC-AUC of **{best_pilot['mean_auc']:.5f}** ({best_pilot['delta_vs_lgb_ref']:+.5f}) and OOF of **{best_pilot['oof_auc']:.5f}**.",
        "",
        "### Detailed Breakdown by Feature Group:",
    ])

    for r in all_pilot_results:
        report_lines.append(f"- **{r['config_name']}**: Mean = **{r['mean_auc']:.5f}** ({r['delta_vs_lgb_ref']:+.5f}). {r['description']}.")

    report_lines.extend([
        "",
        "## 5. Summary & Next Steps",
        "- All original competition files remain untouched.",
        "- No final submission or upload performed.",
        "- Results presented for user review before proceeding to CatBoost evaluation or Stage H2 Step 3.",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"[+] Saved Stage H2 Step 2 report to: {report_path}")


if __name__ == "__main__":
    main()
