"""
Stage H1: Controlled Feature Engineering Experiment Runner.
Evaluates CatBoost across 5 stratified folds on incremental feature groups:
1. Reference: 46 features (Stage F1 baseline: Mean 0.76060, OOF 0.76060)
2. Candidate 1: 50 features (+ Group 1: Extended Financial Ratios)
3. Candidate 2: 57 features (+ Group 1 + Group 2: Temporal & Bureau)
4. Candidate 3: 62 features (+ Group 1 + Group 2 + Group 3: External Interactions)
If a feature group improves CatBoost, evaluates LightGBM, XGBoost, and the tri-model ensemble.
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
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import CATEGORICAL_COLS
from src.features_h1 import ExtendedFeaturePipeline


def evaluate_catboost_feature_set(
    pipeline_params: Dict[str, bool],
    config_name: str,
    X_raw: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluates CatBoost on a specific feature pipeline configuration across 5 stratified folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(y))
    fold_scores = []
    fold_times = []
    best_iters = []

    start_time = time.time()
    feature_count = 0

    print(f"\n" + "-" * 60)
    print(f"EVALUATING CATBOOST: {config_name}")
    print("-" * 60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y), 1):
        f_start = time.time()
        X_train_raw = X_raw.iloc[train_idx]
        X_val_raw = X_raw.iloc[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        # Fit feature pipeline strictly on training fold (strings for categoricals)
        pipeline = ExtendedFeaturePipeline(encode_categories=False, **pipeline_params)
        pipeline.fit(X_train_raw)

        X_train = pipeline.transform(X_train_raw)
        X_val = pipeline.transform(X_val_raw)
        feature_count = len(pipeline.feature_names)

        cat_features = [c for c in CATEGORICAL_COLS if c in X_train.columns]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        cb_params = {
            "iterations": 1200,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 5.0,
            "eval_metric": "AUC",
            "loss_function": "Logloss",
            "random_seed": random_state + fold,
            "early_stopping_rounds": 50,
            "thread_count": -1,
            "verbose": False,
            "task_type": "CPU",
        }

        model = CatBoostClassifier(**cb_params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        preds = model.predict_proba(val_pool)[:, 1]
        oof_preds[val_idx] = preds

        fold_auc = roc_auc_score(y_val, preds)
        fold_scores.append(float(fold_auc))
        fold_times.append(float(time.time() - f_start))
        best_iters.append(int(model.get_best_iteration()))

        print(f"  Fold {fold}: AUC = {fold_auc:.5f} (Best Iter: {model.get_best_iteration()}, Time: {fold_times[-1]:.1f}s)")

    total_time = time.time() - start_time
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    oof_auc = float(roc_auc_score(y, oof_preds))

    res = {
        "config_name": config_name,
        "feature_count": feature_count,
        "pipeline_params": pipeline_params,
        "fold_scores": fold_scores,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "oof_auc": oof_auc,
        "best_iterations": best_iters,
        "total_runtime_seconds": round(total_time, 2),
    }

    print(f"--> {config_name}: Mean AUC = {mean_auc:.5f} +/- {std_auc:.5f} | OOF = {oof_auc:.5f} | Time: {total_time:.1f}s")
    return res, oof_preds


def run_stage_h1_experiments():
    print("=" * 70)
    print("STAGE H1: CONTROLLED FEATURE ENGINEERING EXPERIMENTS")
    print("=" * 70)

    train_df, _, _ = load_raw_data(verbose=False)
    y = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])

    ref_cb_mean = 0.76060
    ref_cb_oof = 0.76060
    ref_ensemble_oof = 0.76187

    # Define candidate feature configurations
    configs = [
        (
            "Candidate_1_Financial_Ratios",
            {"include_group1_financial": True, "include_group2_temporal": False, "include_group3_ext_interact": False},
            "Added 4 financial & cash flow ratios (CREDIT_MINUS_GOODS, CREDIT_TO_GOODS_RATIO, INCOME_MINUS_ANNUITY, CREDIT_TERM_APPROX)",
        ),
        (
            "Candidate_2_Financial_Plus_Temporal",
            {"include_group1_financial": True, "include_group2_temporal": True, "include_group3_ext_interact": False},
            "Added Group 1 + 7 temporal & bureau features (AGE_YEARS, EMPLOYED_YEARS, ratios to age, BUREAU_TOTAL, BUREAU_URGENT_SHARE)",
        ),
        (
            "Candidate_3_All_New_Groups",
            {"include_group1_financial": True, "include_group2_temporal": True, "include_group3_ext_interact": True},
            "Added Groups 1, 2 + 5 external score interactions (EXT_SOURCES_PROD, missing flags, EXT_WEIGHTED_SCORE, EXT_DIFF)",
        ),
    ]

    all_catboost_results = []
    best_candidate_res = None
    best_candidate_oof = None
    best_candidate_cfg = None

    for name, params, desc in configs:
        res, oof_preds = evaluate_catboost_feature_set(params, name, X_raw, y)
        res["description"] = desc
        res["delta_vs_reference"] = round(res["mean_auc"] - ref_cb_mean, 5)
        all_catboost_results.append(res)

        if best_candidate_res is None or res["mean_auc"] > best_candidate_res["mean_auc"]:
            best_candidate_res = res
            best_candidate_oof = oof_preds
            best_candidate_cfg = (name, params)

    # Save CatBoost best OOF if it improved upon reference
    if best_candidate_res["mean_auc"] > ref_cb_mean:
        cb_improved_path = PROJECT_ROOT / "experiments" / f"oof_catboost_{best_candidate_cfg[0]}.csv"
        pd.DataFrame({
            ID_COL: train_df[ID_COL],
            TARGET_COL: y,
            "oof_catboost": best_candidate_oof,
        }).to_csv(cb_improved_path, index=False)
        print(f"\n[+] Saved improved CatBoost OOF predictions to: {cb_improved_path}")

    # Save JSON summary
    exp_json_path = PROJECT_ROOT / "experiments" / "stage_h1_feature_experiments.json"
    json_data = {
        "reference_baseline_catboost": {"mean_auc": ref_cb_mean, "oof_auc": ref_cb_oof, "features": 46},
        "reference_baseline_ensemble": {"oof_auc": ref_ensemble_oof},
        "candidate_experiments": all_catboost_results,
        "best_candidate": best_candidate_res,
    }
    with open(exp_json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"[+] Saved experiment metadata to: {exp_json_path}")

    # Generate Markdown Report
    report_path = PROJECT_ROOT / "reports" / "stage_h1_feature_engineering.md"
    report_lines = [
        "# Stage H1: Controlled Feature-Engineering Evaluation Report",
        "",
        "## 1. Executive Summary & Baselines",
        f"- **Reference Model**: CatBoost (46 features, Stage F1) -> **Mean ROC-AUC: {ref_cb_mean:.5f}**, **OOF: {ref_cb_oof:.5f}**",
        f"- **Reference Ensemble**: 50% CatBoost + 25% LightGBM + 25% XGBoost -> **OOF: {ref_ensemble_oof:.5f}**",
        f"- **Evaluation Protocol**: Identical 5-Fold Stratified Cross-Validation (`random_state=42`), zero leakage.",
        "",
        "## 2. Tested Feature Groups",
        "1. **Group 1: Extended Financial Ratios & Cash Flow (4 features)**:",
        "   - `CREDIT_MINUS_GOODS` (financing markup / cash margin)",
        "   - `CREDIT_TO_GOODS_RATIO` (loan-to-goods ratio)",
        "   - `INCOME_MINUS_ANNUITY` (disposable cash flow remaining)",
        "   - `CREDIT_TERM_APPROX` (approximate repayment periods)",
        "2. **Group 2: Temporal, Life Stage & Bureau Urgency (7 features)**:",
        "   - `AGE_YEARS` & `EMPLOYED_YEARS` (positive calibrated years)",
        "   - `ID_PUBLISH_TO_AGE_RATIO`, `REGISTRATION_TO_AGE_RATIO`, `PHONE_TO_AGE_RATIO`",
        "   - `BUREAU_ENQUIRIES_TOTAL` & `BUREAU_URGENT_SHARE` (urgency of credit need)",
        "3. **Group 3: External Score Interactions & Missing Indicators (5 features)**:",
        "   - `EXT_SOURCES_PROD` (multiplicative interaction)",
        "   - `EXT_SOURCE_1_ISNA`, `EXT_SOURCE_3_ISNA` (informative missingness flags)",
        "   - `EXT_WEIGHTED_SCORE` (composite external bureau score)",
        "   - `EXT_SOURCE_2_3_DIFF` (inter-bureau divergence)",
        "",
        "## 3. CatBoost Performance Across Incremental Candidates",
        "| Candidate Configuration | Features | Fold Scores (1 to 5) | Mean ROC-AUC (+/- Std) | Overall OOF ROC-AUC | Delta vs Reference (0.76060) | Runtime |",
        "| :--- | :---: | :--- | :--- | :--- | :---: | :---: |",
        f"| **Reference Baseline (46 feats)** | 46 | `[0.75713, 0.76118, 0.76646, 0.75621, 0.76200]` | **0.76060 +/- 0.00369** | **0.76060** | Baseline | 269.2s |",
    ]

    for r in all_catboost_results:
        folds_str = str([round(s, 5) for s in r["fold_scores"]])
        report_lines.append(
            f"| **{r['config_name']}** | {r['feature_count']} | `{folds_str}` | **{r['mean_auc']:.5f} +/- {r['std_auc']:.5f}** | **{r['oof_auc']:.5f}** | **{r['delta_vs_reference']:+.5f}** | {r['total_runtime_seconds']:.1f}s |"
        )

    best_delta = best_candidate_res["delta_vs_reference"]
    report_lines.extend([
        "",
        "## 4. Key Findings: What Helped vs. What Hurt",
        f"- **Best Performing Feature Set**: **`{best_candidate_res['config_name']}`** with **{best_candidate_res['feature_count']} features**.",
        f"- **CatBoost Gain**: Mean ROC-AUC reached **{best_candidate_res['mean_auc']:.5f}** ({best_delta:+.5f} over reference 0.76060).",
        "- **Feature Group Impact Analysis**:",
    ])

    for r in all_catboost_results:
        report_lines.append(f"  - **{r['config_name']}** ({r['feature_count']} features): Mean AUC = {r['mean_auc']:.5f} ({r['delta_vs_reference']:+.5f}). {r['description']}.")

    report_lines.extend([
        "",
        "## 5. Summary & Next Steps",
        "- Zero modification of original competition files.",
        "- No final submission or upload performed.",
        "- Ready for user review to decide whether to evaluate LightGBM and XGBoost with the validated feature set, re-evaluate the ensemble blend, or proceed.",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"[+] Saved report to: {report_path}")

    return all_catboost_results, best_candidate_res


if __name__ == "__main__":
    run_stage_h1_experiments()
