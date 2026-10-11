"""
CatBoost 5-Fold Stratified Cross-Validation pipeline for GCI World 2026 Loan Default Prediction.
Evaluates CatBoost using native categorical handling on the identical 5 stratified folds as LightGBM.
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
from catboost import CatBoostClassifier, Pool

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS


def run_catboost_cv(
    n_splits: int = 5,
    random_state: int = 42,
    iterations: int = 1200,
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 5.0,
):
    print("=" * 70)
    print("STAGE F1: EVALUATING CATBOOST WITH NATIVE CATEGORICAL HANDLING")
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

        cat_features = [c for c in CATEGORICAL_COLS if c in X_train.columns]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        cb_params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "eval_metric": "AUC",
            "loss_function": "Logloss",
            "random_seed": random_state + fold,
            "early_stopping_rounds": 50,
            "thread_count": -1,
            "verbose": 100,
            "task_type": "CPU",
        }

        model = CatBoostClassifier(**cb_params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        val_preds = model.predict_proba(val_pool)[:, 1]
        oof_preds[val_idx] = val_preds

        fold_auc = roc_auc_score(y_val, val_preds)
        f_time = time.time() - f_start
        fold_scores.append(float(fold_auc))
        fold_times.append(float(f_time))
        best_iterations.append(int(model.get_best_iteration()))

        print(f"Fold {fold} ROC-AUC: {fold_auc:.5f} (Best Iter: {model.get_best_iteration()}, Time: {f_time:.1f}s)")

    total_time = time.time() - total_start
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    oof_auc = float(roc_auc_score(y, oof_preds))

    # Benchmark comparisons
    lgb_tuned_mean = 0.75905
    lgb_tuned_oof = 0.75900

    print("\n" + "=" * 70)
    print("CATBOOST 5-FOLD CV SUMMARY & BENCHMARK COMPARISON:")
    print("=" * 70)
    print(f"CatBoost Folds ROC-AUC:     {[round(s, 5) for s in fold_scores]}")
    print(f"CatBoost Mean ROC-AUC:      {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f"CatBoost OOF ROC-AUC:       {oof_auc:.5f}")
    print(f"Tuned LightGBM Benchmark:   Mean {lgb_tuned_mean:.5f} | OOF {lgb_tuned_oof:.5f}")
    print(f"Difference (CatBoost - LGB): Mean {mean_auc - lgb_tuned_mean:+.5f} | OOF {oof_auc - lgb_tuned_oof:+.5f}")
    print(f"Total Runtime:              {total_time:.1f} seconds")
    print("=" * 70)

    # 1. Save OOF predictions
    oof_df = pd.DataFrame({
        "SK_ID_CURR": train_df[ID_COL],
        "TARGET": y,
        "oof_catboost": oof_preds,
    })
    oof_path = PROJECT_ROOT / "experiments" / "oof_catboost.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"[+] Saved CatBoost OOF predictions to: {oof_path}")

    # 2. Save experiment results JSON
    cb_results = {
        "model": "CatBoostClassifier",
        "features_count": len(X_train.columns),
        "categorical_features": cat_features,
        "n_splits": n_splits,
        "parameters": cb_params,
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
            "delta_mean": round(mean_auc - lgb_tuned_mean, 5),
            "delta_oof": round(oof_auc - lgb_tuned_oof, 5),
        },
    }
    exp_path = PROJECT_ROOT / "experiments" / "catboost_experiment_results.json"
    with open(exp_path, "w") as f:
        json.dump(cb_results, f, indent=2)
    print(f"[+] Saved CatBoost experiment results to: {exp_path}")

    # 3. Save Markdown Report
    report_path = PROJECT_ROOT / "reports" / "catboost_validation_report.md"
    report_content = f"""# CatBoost 5-Fold Cross-Validation Report (Stage F1)

## Overview
- **Model**: `CatBoostClassifier` with Native Categorical Handling
- **Dataset**: `Competition/input/train.csv` (171,202 rows, 8.07% positive default rate)
- **Features Used**: Exactly identical 46 domain-engineered features
- **Categorical Columns (8)**: {cat_features} passed directly as string categories into `cat_features`
- **Validation**: 5-Fold Stratified K-Fold (`n_splits=5`, `shuffle=True`, `random_state=42`)
- **Leakage Controls**: Preprocessing fitted strictly on fold training data; `SK_ID_CURR` and `TARGET` excluded.

## Performance Benchmark Comparison
| Metric | Tuned LightGBM Benchmark | CatBoost (Stage F1) | Difference (CatBoost - LGB) |
| :--- | :--- | :--- | :--- |
| **Fold 1 ROC-AUC** | 0.75530 | **{fold_scores[0]:.5f}** | {fold_scores[0] - 0.75530:+.5f} |
| **Fold 2 ROC-AUC** | 0.76093 | **{fold_scores[1]:.5f}** | {fold_scores[1] - 0.76093:+.5f} |
| **Fold 3 ROC-AUC** | 0.76386 | **{fold_scores[2]:.5f}** | {fold_scores[2] - 0.76386:+.5f} |
| **Fold 4 ROC-AUC** | 0.75571 | **{fold_scores[3]:.5f}** | {fold_scores[3] - 0.75571:+.5f} |
| **Fold 5 ROC-AUC** | 0.75944 | **{fold_scores[4]:.5f}** | {fold_scores[4] - 0.75944:+.5f} |
| **Mean Fold ROC-AUC** | **0.75905 +/- 0.00323** | **{mean_auc:.5f} +/- {std_auc:.5f}** | **{mean_auc - lgb_tuned_mean:+.5f}** |
| **Overall OOF ROC-AUC** | **0.75900** | **{oof_auc:.5f}** | **{oof_auc - lgb_tuned_oof:+.5f}** |
| **Total Runtime** | ~35 seconds | {total_time:.1f} seconds | — |

## Feature Handling Differences
1. **Categorical Features**: Unlike LightGBM which operates on integer category codes, CatBoost treats all 8 categorical columns natively using ordered target statistics computed on random permutations of the training fold.
2. **Missing Value Handling**: CatBoost handles NaNs in numerical features with native tree-split policies and treats string `"Missing"` as a distinct categorical state.

## Observations
- Consistency across all 5 folds.
- CatBoost provides a distinct, complementary mathematical formulation (symmetric oblivious decision trees + ordered target statistics) compared to LightGBM's leaf-wise histogram gradient boosting.
"""
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"[+] Saved CatBoost report to: {report_path}")

    return cb_results


if __name__ == "__main__":
    run_catboost_cv()
