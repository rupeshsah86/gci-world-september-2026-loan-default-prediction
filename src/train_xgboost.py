"""
XGBoost 5-Fold Stratified Cross-Validation pipeline for GCI World 2026 Loan Default Prediction.
Evaluates XGBoost with native partition-based categorical handling (enable_categorical=True)
on the identical 5 stratified folds as LightGBM and CatBoost.
Saves out-of-fold predictions, fold metrics, and benchmark comparisons.
"""

import sys
from pathlib import Path
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS


def run_xgboost_cv(
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 1200,
    learning_rate: float = 0.03,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.7,
    reg_alpha: float = 1.0,
    reg_lambda: float = 5.0,
):
    print("=" * 70)
    print("STAGE F2: EVALUATING XGBOOST WITH NATIVE CATEGORICAL SUPPORT")
    print("=" * 70)

    train_df, _, _ = load_raw_data(verbose=False)
    y = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(train_df))
    fold_scores = []
    fold_times = []
    best_iterations = []

    total_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y), 1):
        f_start = time.time()
        print(f"\n--- Fold {fold}/{n_splits} ---")

        X_train_raw = X_raw.iloc[train_idx]
        X_val_raw = X_raw.iloc[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        # Pipeline fitted strictly per fold (string categorical outputs)
        pipeline = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
        pipeline.fit(X_train_raw)

        X_train = pipeline.transform(X_train_raw)
        X_val = pipeline.transform(X_val_raw)

        # Convert categorical columns to pandas CategoricalDtype fitted strictly on training fold
        for col in CATEGORICAL_COLS:
            if col in X_train.columns:
                train_categories = X_train[col].dropna().unique().tolist()
                train_categories.sort()
                cat_dtype = pd.CategoricalDtype(categories=train_categories, ordered=False)

                X_train[col] = X_train[col].astype(cat_dtype)
                # For validation fold, map unseen categories to NaN cleanly
                X_val[col] = X_val[col].where(X_val[col].isin(train_categories), np.nan).astype(cat_dtype)

        xgb_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "tree_method": "hist",
            "enable_categorical": True,
            "eval_metric": "auc",
            "early_stopping_rounds": 50,
            "random_state": random_state + fold,
            "n_jobs": -1,
        }

        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100,
        )

        val_preds = clf.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds

        fold_auc = roc_auc_score(y_val, val_preds)
        f_time = time.time() - f_start
        fold_scores.append(float(fold_auc))
        fold_times.append(float(f_time))
        best_iterations.append(int(clf.best_iteration))

        print(f"Fold {fold} ROC-AUC: {fold_auc:.5f} (Best Iter: {clf.best_iteration}, Time: {f_time:.1f}s)")

    total_time = time.time() - total_start
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    oof_auc = float(roc_auc_score(y, oof_preds))

    # Verification: every training row received exactly one valid prediction
    assert len(oof_preds) == len(train_df), "OOF predictions count mismatch"
    assert not np.isnan(oof_preds).any(), "Found NaN in OOF predictions"
    assert not np.isinf(oof_preds).any(), "Found Inf in OOF predictions"
    assert (oof_preds >= 0.0).all() and (oof_preds <= 1.0).all(), "Predictions outside [0, 1] interval"

    # Previous benchmarks
    lgb_tuned_mean = 0.75905
    lgb_tuned_oof = 0.75900
    cb_mean = 0.76060
    cb_oof = 0.76060

    print("\n" + "=" * 70)
    print("XGBOOST 5-FOLD CV SUMMARY & BENCHMARK COMPARISON:")
    print("=" * 70)
    print(f"XGBoost Folds ROC-AUC:      {[round(s, 5) for s in fold_scores]}")
    print(f"XGBoost Mean ROC-AUC:       {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f"XGBoost OOF ROC-AUC:        {oof_auc:.5f}")
    print(f"Tuned LightGBM Benchmark:   Mean {lgb_tuned_mean:.5f} | OOF {lgb_tuned_oof:.5f}")
    print(f"CatBoost Benchmark:         Mean {cb_mean:.5f} | OOF {cb_oof:.5f}")
    print(f"Difference (XGB - LightGBM): Mean {mean_auc - lgb_tuned_mean:+.5f} | OOF {oof_auc - lgb_tuned_oof:+.5f}")
    print(f"Difference (XGB - CatBoost): Mean {mean_auc - cb_mean:+.5f} | OOF {oof_auc - cb_oof:+.5f}")
    print(f"Total CV Runtime:           {total_time:.1f} seconds")
    print("=" * 70)

    # 1. Save OOF predictions
    oof_df = pd.DataFrame({
        "SK_ID_CURR": train_df[ID_COL],
        "TARGET": y,
        "oof_xgboost": oof_preds,
    })
    oof_path = PROJECT_ROOT / "experiments" / "oof_xgboost.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"[+] Saved XGBoost OOF predictions to: {oof_path}")

    # 2. Save experiment results JSON
    xgb_results = {
        "model": "XGBClassifier",
        "features_count": len(X_train.columns),
        "n_splits": n_splits,
        "effective_parameters": {k: v for k, v in xgb_params.items()},
        "fold_scores": fold_scores,
        "best_iterations": best_iterations,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "oof_auc": oof_auc,
        "fold_times": fold_times,
        "total_runtime_seconds": round(total_time, 2),
        "benchmark_comparison": {
            "lgb_tuned_mean": lgb_tuned_mean,
            "lgb_tuned_oof": lgb_tuned_oof,
            "catboost_mean": cb_mean,
            "catboost_oof": cb_oof,
            "delta_vs_lgb_mean": round(mean_auc - lgb_tuned_mean, 5),
            "delta_vs_lgb_oof": round(oof_auc - lgb_tuned_oof, 5),
            "delta_vs_cb_mean": round(mean_auc - cb_mean, 5),
            "delta_vs_cb_oof": round(oof_auc - cb_oof, 5),
        },
    }
    exp_path = PROJECT_ROOT / "experiments" / "xgboost_experiment_results.json"
    with open(exp_path, "w") as f:
        json.dump(xgb_results, f, indent=2)
    print(f"[+] Saved XGBoost experiment results to: {exp_path}")

    # 3. Save Markdown Report
    report_path = PROJECT_ROOT / "reports" / "xgboost_validation_report.md"
    report_content = f"""# XGBoost 5-Fold Cross-Validation Report (Stage F2)

## Overview
- **Model**: `XGBClassifier` with Native Categorical Splits (`enable_categorical=True`, `tree_method="hist"`)
- **Dataset**: `Competition/input/train.csv` (171,202 rows, 8.07% positive default rate)
- **Features Used**: Identical 46 domain-engineered features
- **Categorical Columns (8)**: {CATEGORICAL_COLS} formatted as pandas `CategoricalDtype` fitted strictly per fold
- **Validation**: 5-Fold Stratified K-Fold (`n_splits=5`, `shuffle=True`, `random_state=42`)
- **Leakage Controls**: Preprocessing fitted strictly on fold training data; `SK_ID_CURR` and `TARGET` excluded.

## Performance Benchmark Comparison
| Metric | Tuned LightGBM | CatBoost | XGBoost (Stage F2) | Delta vs LGB | Delta vs CatBoost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fold 1 ROC-AUC** | 0.75530 | 0.75713 | **{fold_scores[0]:.5f}** | {fold_scores[0] - 0.75530:+.5f} | {fold_scores[0] - 0.75713:+.5f} |
| **Fold 2 ROC-AUC** | 0.76093 | 0.76118 | **{fold_scores[1]:.5f}** | {fold_scores[1] - 0.76093:+.5f} | {fold_scores[1] - 0.76118:+.5f} |
| **Fold 3 ROC-AUC** | 0.76386 | 0.76646 | **{fold_scores[2]:.5f}** | {fold_scores[2] - 0.76386:+.5f} | {fold_scores[2] - 0.76646:+.5f} |
| **Fold 4 ROC-AUC** | 0.75571 | 0.75621 | **{fold_scores[3]:.5f}** | {fold_scores[3] - 0.75571:+.5f} | {fold_scores[3] - 0.75621:+.5f} |
| **Fold 5 ROC-AUC** | 0.75944 | 0.76200 | **{fold_scores[4]:.5f}** | {fold_scores[4] - 0.75944:+.5f} | {fold_scores[4] - 0.76200:+.5f} |
| **Mean Fold ROC-AUC** | **0.75905** | **0.76060** | **{mean_auc:.5f} +/- {std_auc:.5f}** | **{mean_auc - lgb_tuned_mean:+.5f}** | **{mean_auc - cb_mean:+.5f}** |
| **Overall OOF ROC-AUC** | **0.75900** | **0.76060** | **{oof_auc:.5f}** | **{oof_auc - lgb_tuned_oof:+.5f}** | **{oof_auc - cb_oof:+.5f}** |
| **Total Runtime** | ~35s | ~269s | {total_time:.1f}s | — | — |

## Feature Handling & Implementation Details
1. **Native Categorical Partitioning**: Handled via XGBoost's `tree_method="hist"` with `enable_categorical=True`. Categories are represented without arbitrary ordinal assumptions.
2. **Missing Values**: Missing values in numerical columns are natively routed by XGBoost to the default split direction maximizing gain.
3. **Data Verification**:
   - Every training row received exactly one out-of-fold prediction.
   - Out-of-fold predictions align row-for-row with `SK_ID_CURR`.
   - Predictions strictly bounded in continuous probability range `[0.0, 1.0]`.
"""
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"[+] Saved XGBoost report to: {report_path}")

    return xgb_results


if __name__ == "__main__":
    run_xgboost_cv()
