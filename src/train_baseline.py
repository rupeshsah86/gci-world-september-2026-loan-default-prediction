"""
Baseline training and holdout validation script for GCI World 2026 Loan Default Prediction.
Compares models on a stratified 70/30 holdout split using all permitted features.
Saves experiment logs and metrics to experiments/ and reports/.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS


def evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Calculates comprehensive classification metrics at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "roc_auc": float(auc),
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def run_baseline_experiment(random_state: int = 42):
    print("=" * 70)
    print("STAGE C: EVALUATING FIRST IMPROVED BASELINE (STRATIFIED HOLDOUT)")
    print("=" * 70)

    # 1. Load data
    train_df, _, _ = load_raw_data()
    y = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])

    # 2. Stratified train/validation split (70% train, 30% valid)
    X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
        X_raw, y, test_size=0.30, stratify=y, random_state=random_state
    )

    print(f"Training split:   {X_train_raw.shape[0]} samples (Default rate: {y_train.mean():.4%})")
    print(f"Validation split: {X_valid_raw.shape[0]} samples (Default rate: {y_valid.mean():.4%})")

    # 3. Fit feature pipeline strictly on training split
    pipeline = BaselineFeaturePipeline(add_domain_ratios=False)
    pipeline.fit(X_train_raw)

    X_train = pipeline.transform(X_train_raw)
    X_valid = pipeline.transform(X_valid_raw)
    feature_names = pipeline.feature_names
    cat_indices = [col for col in CATEGORICAL_COLS if col in feature_names]

    print(f"Total features extracted: {len(feature_names)}")

    results = {}

    # 4. Model 1: LightGBM Baseline (all features, anomaly flags, native handling)
    print("\n--- Training Model 1: Baseline LightGBM ---")
    start_time = time.time()
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices, free_raw_data=False)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train, categorical_feature=cat_indices, free_raw_data=False)

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "random_state": random_state,
        "n_jobs": -1,
    }

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    lgb_time = time.time() - start_time

    lgb_train_preds = lgb_model.predict(X_train, num_iteration=lgb_model.best_iteration)
    lgb_valid_preds = lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration)

    lgb_train_metrics = evaluate_predictions(y_train, lgb_train_preds)
    lgb_valid_metrics = evaluate_predictions(y_valid, lgb_valid_preds)

    print(f"LightGBM Best Iteration: {lgb_model.best_iteration}")
    print(f"LightGBM Training ROC-AUC:   {lgb_train_metrics['roc_auc']:.5f}")
    print(f"LightGBM Validation ROC-AUC: {lgb_valid_metrics['roc_auc']:.5f} (Elapsed: {lgb_time:.2f}s)")

    results["LightGBM_Baseline"] = {
        "model": "LightGBM",
        "features_count": len(feature_names),
        "train_roc_auc": lgb_train_metrics["roc_auc"],
        "valid_roc_auc": lgb_valid_metrics["roc_auc"],
        "metrics_at_0.5": lgb_valid_metrics,
        "best_iteration": int(lgb_model.best_iteration),
        "runtime_seconds": round(lgb_time, 2),
        "hyperparameters": lgb_params,
    }

    # 5. Model 2: Random Forest (imputing NaNs with median for comparison)
    print("\n--- Training Model 2: Random Forest (All features, max_depth=12) ---")
    start_time = time.time()
    # Fill NaN for scikit-learn RF
    rf_train = X_train.fillna(X_train.median())
    rf_valid = X_valid.fillna(X_train.median())

    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1)
    rf.fit(rf_train, y_train)
    rf_time = time.time() - start_time

    rf_train_preds = rf.predict_proba(rf_train)[:, 1]
    rf_valid_preds = rf.predict_proba(rf_valid)[:, 1]

    rf_train_metrics = evaluate_predictions(y_train, rf_train_preds)
    rf_valid_metrics = evaluate_predictions(y_valid, rf_valid_preds)

    print(f"Random Forest Training ROC-AUC:   {rf_train_metrics['roc_auc']:.5f}")
    print(f"Random Forest Validation ROC-AUC: {rf_valid_metrics['roc_auc']:.5f} (Elapsed: {rf_time:.2f}s)")

    results["RandomForest_AllFeatures"] = {
        "model": "RandomForestClassifier",
        "features_count": len(feature_names),
        "train_roc_auc": rf_train_metrics["roc_auc"],
        "valid_roc_auc": rf_valid_metrics["roc_auc"],
        "metrics_at_0.5": rf_valid_metrics,
        "runtime_seconds": round(rf_time, 2),
    }

    # Benchmarks comparison
    tutorial_lr_auc = 0.66307
    tutorial_rf_auc = 0.66879
    lb_reference = 0.79362

    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON SUMMARY:")
    print("=" * 70)
    print(f"1. Tutorial Logistic Regression (5 features):  {tutorial_lr_auc:.5f}")
    print(f"2. Tutorial Random Forest (5 features):        {tutorial_rf_auc:.5f}")
    print(f"3. Improved Random Forest (All 35 features):    {rf_valid_metrics['roc_auc']:.5f} (+{rf_valid_metrics['roc_auc'] - tutorial_rf_auc:+.5f})")
    print(f"4. Improved LightGBM Baseline (All 35 features): {lgb_valid_metrics['roc_auc']:.5f} (+{lgb_valid_metrics['roc_auc'] - tutorial_rf_auc:+.5f})")
    print(f"5. Target Observed Leaderboard Score:          ~{lb_reference:.5f}")
    print("=" * 70)

    # 6. Save experiment results
    exp_file = PROJECT_ROOT / "experiments" / "baseline_holdout_results.json"
    with open(exp_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved experiment results to: {exp_file}")

    # 7. Write concise markdown report
    report_file = PROJECT_ROOT / "reports" / "baseline_holdout_report.md"
    report_content = f"""# Baseline Holdout Validation Report

## Overview
- **Evaluation Split**: 70% Train ({X_train_raw.shape[0]} rows) / 30% Valid ({X_valid_raw.shape[0]} rows), Stratified, Seed = {random_state}
- **Features Used**: All 32 permitted raw features + 3 domain anomaly indicators ({len(feature_names)} features total)
- **Leakage Guards**: Preprocessing fitted solely on the training fold; target and `SK_ID_CURR` excluded.

## Performance Comparison
| Model | Features | Train ROC-AUC | Valid ROC-AUC | Improvement vs Tutorial | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tutorial Logistic Regression** | 5 | — | 0.66307 | Baseline | — |
| **Tutorial Random Forest** | 5 | — | 0.66879 | +0.00572 | — |
| **Random Forest (All Features)** | {len(feature_names)} | {rf_train_metrics['roc_auc']:.5f} | **{rf_valid_metrics['roc_auc']:.5f}** | **+{rf_valid_metrics['roc_auc'] - tutorial_rf_auc:.5f}** | {rf_time:.1f}s |
| **LightGBM Baseline** | {len(feature_names)} | {lgb_train_metrics['roc_auc']:.5f} | **{lgb_valid_metrics['roc_auc']:.5f}** | **+{lgb_valid_metrics['roc_auc'] - tutorial_rf_auc:.5f}** | {lgb_time:.1f}s |

## Classification Metrics at Threshold 0.5 (LightGBM)
- **ROC-AUC**: {lgb_valid_metrics['roc_auc']:.5f}
- **Accuracy**: {lgb_valid_metrics['accuracy']:.4%}
- **Precision**: {lgb_valid_metrics['precision']:.4%}
- **Recall**: {lgb_valid_metrics['recall']:.4%}
- **F1 Score**: {lgb_valid_metrics['f1']:.4%}
- **Confusion Matrix**: `TN={lgb_valid_metrics['confusion_matrix'][0][0]}`, `FP={lgb_valid_metrics['confusion_matrix'][0][1]}`, `FN={lgb_valid_metrics['confusion_matrix'][1][0]}`, `TP={lgb_valid_metrics['confusion_matrix'][1][1]}`

## Observations
1. Moving from 5 features to all available features dramatically boosts ROC-AUC from **~0.668** to **{lgb_valid_metrics['roc_auc']:.5f}**.
2. LightGBM handles high cardinality categorical variables and native missing values cleanly and swiftly ({lgb_time:.1f}s training time).
3. The next step is validating this performance across 5-fold Stratified Cross-Validation and testing engineered financial domain ratios.
"""
    with open(report_file, "w") as f:
        f.write(report_content)
    print(f"[+] Saved report to: {report_file}")

    return results


if __name__ == "__main__":
    run_baseline_experiment()
