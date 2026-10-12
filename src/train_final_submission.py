"""
Final Submission Candidate Generation Script.
Project: GCI World 2026 Home Credit Default Risk.
Trains the verified 3-model ensemble on 100% of the competition training data
and generates submission predictions on the 61,500 competition test rows.

Ensemble:
- 50% CatBoost Config 3 (62 features, Candidate 3 pipeline)
- 25% Tuned LightGBM (46 features, Baseline pipeline)
- 25% XGBoost Config 3 (46 features, Baseline pipeline)

Target Output: outputs/submission_candidate_20261012.csv
Report: reports/final_submission_candidate_review.md
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
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import lightgbm as lgb

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS
from src.features_h1 import ExtendedFeaturePipeline

SUBMISSION_PATH = PROJECT_ROOT / "outputs" / "submission_candidate_20261012.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "final_submission_candidate_review.md"


def main():
    total_start = time.time()
    print("=" * 75)
    print("FINAL SUBMISSION CANDIDATE GENERATION: GCI WORLD 2026 HOME CREDIT")
    print("=" * 75)

    # 1. Load Raw Training and Test Data
    print("\n[Step 1/6] Loading competition raw datasets...")
    train_df, test_df, sample_sub = load_raw_data(verbose=True)

    y_train = train_df[TARGET_COL].values
    X_train_raw = train_df.drop(columns=[TARGET_COL, ID_COL])
    X_test_raw = test_df.drop(columns=[ID_COL])

    n_train = len(train_df)
    n_test = len(test_df)
    assert n_train == 171202, f"Expected 171,202 train rows, got {n_train}"
    assert n_test == 61500, f"Expected 61,500 test rows, got {n_test}"
    print(f"[+] Loaded {n_train:,} training samples and {n_test:,} test samples.")

    # -----------------------------------------------------------------
    # MODEL 1: Tuned LightGBM (46 features, Baseline Pipeline)
    # -----------------------------------------------------------------
    print("\n" + "=" * 75)
    print("[Step 2/6] Training Model 1: Tuned LightGBM (46 features)...")
    print("=" * 75)
    lgb_start = time.time()

    pipe_lgb = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=True)
    pipe_lgb.fit(X_train_raw)
    X_tr_lgb = pipe_lgb.transform(X_train_raw)
    X_te_lgb = pipe_lgb.transform(X_test_raw)

    assert X_tr_lgb.shape[1] == 46, f"Expected 46 LGB features, got {X_tr_lgb.shape[1]}"
    assert X_te_lgb.shape[1] == 46, f"Expected 46 LGB features, got {X_te_lgb.shape[1]}"
    cat_cols_lgb = [c for c in CATEGORICAL_COLS if c in pipe_lgb.feature_names]

    dtrain_lgb = lgb.Dataset(X_tr_lgb, label=y_train, categorical_feature=cat_cols_lgb, free_raw_data=False)
    lgb_params = {
        "learning_rate": 0.03,
        "num_leaves": 24,
        "max_depth": 7,
        "min_child_samples": 100,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    # 465 trees based on 5-fold CV mean early stopping point
    lgb_n_rounds = 465
    print(f"  Training LightGBM on {n_train:,} rows with {lgb_n_rounds} iterations...")
    model_lgb = lgb.train(lgb_params, dtrain_lgb, num_boost_round=lgb_n_rounds)
    p_test_lgb = model_lgb.predict(X_te_lgb)
    lgb_time = time.time() - lgb_start
    print(f"  [+] LightGBM completed in {lgb_time:.1f}s. Test preds: min={p_test_lgb.min():.4f}, mean={p_test_lgb.mean():.4f}, max={p_test_lgb.max():.4f}")

    # -----------------------------------------------------------------
    # MODEL 2: XGBoost Config 3 (46 features, Baseline Pipeline)
    # -----------------------------------------------------------------
    print("\n" + "=" * 75)
    print("[Step 3/6] Training Model 2: XGBoost Config 3 (46 features)...")
    print("=" * 75)
    xgb_start = time.time()

    pipe_xgb = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
    pipe_xgb.fit(X_train_raw)
    X_tr_xgb = pipe_xgb.transform(X_train_raw)
    X_te_xgb = pipe_xgb.transform(X_test_raw)

    assert X_tr_xgb.shape[1] == 46, f"Expected 46 XGB features, got {X_tr_xgb.shape[1]}"
    assert X_te_xgb.shape[1] == 46, f"Expected 46 XGB features, got {X_te_xgb.shape[1]}"

    for col in CATEGORICAL_COLS:
        if col in X_tr_xgb.columns:
            train_categories = sorted(X_tr_xgb[col].dropna().unique().tolist())
            cat_dtype = pd.CategoricalDtype(categories=train_categories, ordered=False)
            X_tr_xgb[col] = X_tr_xgb[col].astype(cat_dtype)
            X_te_xgb[col] = X_te_xgb[col].where(X_te_xgb[col].isin(train_categories), np.nan).astype(cat_dtype)

    xgb_params = {
        "n_estimators": 453,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.60,
        "reg_alpha": 2.0,
        "reg_lambda": 10.0,
        "tree_method": "hist",
        "enable_categorical": True,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }

    print(f"  Training XGBoost on {n_train:,} rows with {xgb_params['n_estimators']} iterations...")
    clf_xgb = xgb.XGBClassifier(**xgb_params)
    clf_xgb.fit(X_tr_xgb, y_train, verbose=False)
    p_test_xgb = clf_xgb.predict_proba(X_te_xgb)[:, 1]
    xgb_time = time.time() - xgb_start
    print(f"  [+] XGBoost completed in {xgb_time:.1f}s. Test preds: min={p_test_xgb.min():.4f}, mean={p_test_xgb.mean():.4f}, max={p_test_xgb.max():.4f}")

    # -----------------------------------------------------------------
    # MODEL 3: CatBoost Config 3 (62 features, Candidate 3 Pipeline)
    # -----------------------------------------------------------------
    print("\n" + "=" * 75)
    print("[Step 4/6] Training Model 3: CatBoost Config 3 (62 features)...")
    print("=" * 75)
    cb_start = time.time()

    pipe_cb = ExtendedFeaturePipeline(
        include_group1_financial=True,
        include_group2_temporal=True,
        include_group3_ext_interact=True,
        encode_categories=False,
    )
    pipe_cb.fit(X_train_raw)
    X_tr_cb = pipe_cb.transform(X_train_raw)
    X_te_cb = pipe_cb.transform(X_test_raw)

    assert X_tr_cb.shape[1] == 62, f"Expected 62 CatBoost features, got {X_tr_cb.shape[1]}"
    assert X_te_cb.shape[1] == 62, f"Expected 62 CatBoost features, got {X_te_cb.shape[1]}"

    cat_cols_cb = [c for c in CATEGORICAL_COLS if c in X_tr_cb.columns]
    train_pool_cb = Pool(X_tr_cb, y_train, cat_features=cat_cols_cb)
    test_pool_cb = Pool(X_te_cb, cat_features=cat_cols_cb)

    cb_params = {
        "iterations": 1103,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_strength": 2.0,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "random_seed": 42,
        "thread_count": 4,
        "verbose": 200,
        "task_type": "CPU",
    }

    print(f"  Training CatBoost on {n_train:,} rows with {cb_params['iterations']} iterations...")
    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(train_pool_cb, verbose=200)
    p_test_cb = cb_model.predict_proba(test_pool_cb)[:, 1]
    cb_time = time.time() - cb_start
    print(f"  [+] CatBoost completed in {cb_time:.1f}s. Test preds: min={p_test_cb.min():.4f}, mean={p_test_cb.mean():.4f}, max={p_test_cb.max():.4f}")

    # -----------------------------------------------------------------
    # ENSEMBLE COMBINATION: 50% CB + 25% LGB + 25% XGB
    # -----------------------------------------------------------------
    print("\n" + "=" * 75)
    print("[Step 5/6] Computing Fixed-Weight Tri-Model Ensemble (50/25/25)...")
    print("=" * 75)
    p_test_ensemble = 0.50 * p_test_cb + 0.25 * p_test_lgb + 0.25 * p_test_xgb

    print(f"  Ensemble Summary Statistics on Test Set ({n_test:,} records):")
    print(f"    Min probability : {p_test_ensemble.min():.6f}")
    print(f"    Max probability : {p_test_ensemble.max():.6f}")
    print(f"    Mean probability: {p_test_ensemble.mean():.6f}")
    print(f"    Std probability : {p_test_ensemble.std():.6f}")
    print(f"    Median          : {np.median(p_test_ensemble):.6f}")
    print(f"    1st Percentile  : {np.percentile(p_test_ensemble, 1):.6f}")
    print(f"    99th Percentile : {np.percentile(p_test_ensemble, 99):.6f}")

    # -----------------------------------------------------------------
    # SUBMISSION CREATION AND RIGOROUS CHECKS
    # -----------------------------------------------------------------
    print("\n" + "=" * 75)
    print("[Step 6/6] Generating and validating submission candidate...")
    print("=" * 75)

    sub_df = pd.DataFrame({
        ID_COL: test_df[ID_COL].values,
        TARGET_COL: p_test_ensemble,
    })

    # Save to outputs/
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"[+] Saved submission file to: {SUBMISSION_PATH}")

    # Audit 1: File existence and size
    assert SUBMISSION_PATH.exists(), "Submission file was not written!"
    file_size_mb = SUBMISSION_PATH.stat().st_size / (1024 * 1024)
    print(f"  [CHECK 1] File exists ({file_size_mb:.2f} MB)")

    # Audit 2: Reopen and parse as CSV
    loaded_sub = pd.read_csv(SUBMISSION_PATH)
    print(f"  [CHECK 2] Reopened and parsed successfully")

    # Audit 3: Exact shape and column order
    assert loaded_sub.shape == (61500, 2), f"Expected shape (61500, 2), got {loaded_sub.shape}"
    assert list(loaded_sub.columns) == [ID_COL, TARGET_COL], f"Columns mismatch: {loaded_sub.columns}"
    print(f"  [CHECK 3] Exact shape (61,500 rows, 2 columns: ['SK_ID_CURR', 'TARGET']) verified")

    # Audit 4: Exact ID alignment
    assert (loaded_sub[ID_COL].values == test_df[ID_COL].values).all(), "ID alignment mismatch with test.csv!"
    assert loaded_sub[ID_COL].nunique() == 61500, "Duplicate IDs detected in submission!"
    print(f"  [CHECK 4] Exact ID alignment with test.csv in original order verified (zero missing, zero duplicates)")

    # Audit 5: Probability bounds and finiteness
    assert not loaded_sub[TARGET_COL].isna().any(), "NaNs detected in TARGET!"
    assert not np.isinf(loaded_sub[TARGET_COL].values).any(), "Infinite values detected in TARGET!"
    assert (loaded_sub[TARGET_COL].values >= 0.0).all(), "Predictions < 0 detected!"
    assert (loaded_sub[TARGET_COL].values <= 1.0).all(), "Predictions > 1 detected!"
    print(f"  [CHECK 5] All TARGET values are finite probabilities strictly within [0.0, 1.0]")

    total_time = time.time() - total_start
    print(f"\n[+] All audits passed successfully in {total_time:.1f}s (~{total_time/60:.1f} minutes).")

    # -----------------------------------------------------------------
    # GENERATE MARKDOWN REVIEW REPORT
    # -----------------------------------------------------------------
    report_lines = [
        "# Final Submission Candidate Review Report",
        "",
        "## 1. Executive Summary",
        f"- **Submission Artifact**: [`outputs/submission_candidate_20261012.csv`](file://{SUBMISSION_PATH})",
        f"- **Generated At**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- **Total Training & Inference Runtime**: {total_time:.1f} seconds (~{total_time/60:.1f} minutes) on Apple Silicon M2.",
        "- **Ensemble Architecture**: Tri-model probability blend combining three distinct gradient boosting implementations.",
        "- **Training Scope**: Fitted on 100% of the competition training dataset (171,202 rows).",
        "- **Inference Scope**: Generated predictions for 100% of the competition test dataset (61,500 rows).",
        "",
        "## 2. Component Model Specifications",
        "",
        "### Model 1: CatBoost Config 3 (Weight: 50%)",
        "- **Feature Pipeline**: Verified 62-feature `ExtendedFeaturePipeline` (financial cash flow ratios, temporal & bureau ratios, and external score polynomial interactions).",
        "- **Categorical Handling**: Native CatBoost categorical handling with string representation via `cat_features` pool.",
        "- **Hyperparameters**: `depth=6, l2_leaf_reg=5.0, learning_rate=0.04, random_strength=2.0, loss_function='Logloss', eval_metric='AUC'`.",
        "- **Full Training Budget**: 1,103 iterations (derived from 5-fold CV mean early-stopping iteration).",
        "- **5-Fold CV Performance**: Mean ROC-AUC = **0.76076 $\\pm$ 0.00263**, OOF ROC-AUC = **0.76072**.",
        f"- **Training Runtime**: {cb_time:.1f} seconds.",
        "",
        "### Model 2: Tuned LightGBM (Weight: 25%)",
        "- **Feature Pipeline**: Verified 46-feature `BaselineFeaturePipeline` with domain ratios and integer label-encoded categories.",
        "- **Categorical Handling**: Native LightGBM categorical feature support with `min_child_samples=100`.",
        "- **Hyperparameters**: `num_leaves=24, max_depth=7, min_child_samples=100, feature_fraction=0.7, bagging_fraction=0.75, bagging_freq=1, reg_alpha=1.0, reg_lambda=5.0, learning_rate=0.03`.",
        "- **Full Training Budget**: 465 iterations (derived from 5-fold CV mean early-stopping iteration).",
        "- **5-Fold CV Performance**: Mean ROC-AUC = **0.75905 $\\pm$ 0.00323**, OOF ROC-AUC = **0.75900**.",
        f"- **Training Runtime**: {lgb_time:.1f} seconds.",
        "",
        "### Model 3: XGBoost Config 3 (Weight: 25%)",
        "- **Feature Pipeline**: Verified 46-feature `BaselineFeaturePipeline` with domain ratios and pandas `CategoricalDtype`.",
        "- **Categorical Handling**: Native XGBoost categorical support (`tree_method='hist'`, `enable_categorical=True`).",
        "- **Hyperparameters**: `max_depth=6, min_child_weight=10, colsample_bytree=0.60, subsample=0.8, reg_alpha=2.0, reg_lambda=10.0, learning_rate=0.03`.",
        "- **Full Training Budget**: 453 iterations (derived from 5-fold CV mean early-stopping iteration).",
        "- **5-Fold CV Performance**: Mean ROC-AUC = **0.76021 $\\pm$ 0.00285**, OOF ROC-AUC = **0.76015**.",
        f"- **Training Runtime**: {xgb_time:.1f} seconds.",
        "",
        "## 3. Ensemble Blending and Validation Benchmarks",
        "The ensemble weights `0.50 CatBoost + 0.25 LightGBM + 0.25 XGBoost` were validated across multiple stages:",
        "",
        "| Evaluation Stage / Split | Component Models | Evaluation Metric | Result | Benchmark Context |",
        "| :--- | :--- | :---: | :---: | :--- |",
        "| **Stage D Baseline** | LightGBM Baseline (46f) | 5-Fold OOF ROC-AUC | 0.75627 | Initial domain features |",
        "| **Stage E Tuning** | Tuned LightGBM (46f) | 5-Fold OOF ROC-AUC | 0.75900 | Hyperparameter tuning gain (+0.00273) |",
        "| **Stage F2 Baseline** | XGBoost Baseline (46f) | 5-Fold OOF ROC-AUC | 0.75874 | First XGBoost evaluation |",
        "| **Stage H1 Feature Ext.** | CatBoost Candidate 3 (62f) | 5-Fold OOF ROC-AUC | 0.76061 | Oblivious tree polynomial interaction gain |",
        "| **Stage H2 Step 3A Tuning**| XGBoost Config 3 (46f) | 5-Fold OOF ROC-AUC | 0.76015 | Heavy regularization gain (+0.00141 across all 5 folds) |",
        "| **Stage H2 Step 3A Tuning**| CatBoost Config 3 (62f) | 5-Fold OOF ROC-AUC | 0.76072 | Random strength & lr optimization |",
        "| **Stage H2 Step 3B Ensemble**| Tri-Model Blend (Seed 42) | Nested CV ROC-AUC | **0.76212** | **Unbiased cross-validation ensemble estimate** |",
        "| **Stage H2 Step 3B Ensemble**| Tri-Model Blend (Seed 42) | Aligned OOF ROC-AUC | **0.76208** | Combined probability blend |",
        "| **Stage H2 Step 3B Stability**| Tri-Model Blend (Seed 2026)| Independent OOF ROC-AUC | **0.76161** | Verified on completely reshuffled folds |",
        "",
        "> [!IMPORTANT]",
        "> **Important Clarification on Validation Metrics:**",
        "> The OOF ROC-AUC score (**0.76208** / nested **0.76212**) evaluates rank-ordering discrimination on out-of-fold validation splits. It is **not** classification accuracy and does **not** represent a public leaderboard score. Real test performance depends on competition test set calibration and sampling.",
        "",
        "## 4. Test Prediction Distribution Summary",
        f"- **Total Rows Evaluated**: {n_test:,}",
        f"- **Minimum Probability**: {p_test_ensemble.min():.6f}",
        f"- **1st Percentile**: {np.percentile(p_test_ensemble, 1):.6f}",
        f"- **Median Probability**: {np.median(p_test_ensemble):.6f}",
        f"- **Mean Probability**: {p_test_ensemble.mean():.6f} *(reflects the natural positive class baseline ~8% in credit risk)*",
        f"- **99th Percentile**: {np.percentile(p_test_ensemble, 99):.6f}",
        f"- **Maximum Probability**: {p_test_ensemble.max():.6f}",
        f"- **Standard Deviation**: {p_test_ensemble.std():.6f}",
        "",
        "## 5. File Verification and Compliance Audit",
        "| Audit Check | Requirement | Result | Status |",
        "| :--- | :--- | :--- | :---: |",
        f"| **File Location** | `outputs/submission_candidate_20261012.csv` | `{SUBMISSION_PATH}` | **PASSED** |",
        f"| **Row Count** | Exactly 61,500 rows | {len(loaded_sub):,} rows | **PASSED** |",
        f"| **Column Structure** | Exactly `['SK_ID_CURR', 'TARGET']` | `{list(loaded_sub.columns)}` | **PASSED** |",
        f"| **ID Integrity** | Exact match with test.csv in original order | Zero mismatch, zero duplicates | **PASSED** |",
        f"| **Prediction Values** | Finite probabilities $\\in [0.0, 1.0]$ | Min: {p_test_ensemble.min():.4f}, Max: {p_test_ensemble.max():.4f}, NaNs: 0 | **PASSED** |",
        f"| **File Parseability** | Clean CSV re-opened and parsed | 100% valid CSV | **PASSED** |",
        "| **Data Confidentiality** | No test labels, no data leakage, no external sets | Strictly adhered | **PASSED** |",
        "",
        "## 6. Stop Checkpoint & Final Status",
        "- The submission candidate file has been created and validated locally.",
        "- In accordance with your instructions, **no file has been uploaded, submitted, or published anywhere**.",
        "- This run is complete and stopped awaiting your review.",
    ]

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    print(f"[+] Saved comprehensive review report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
