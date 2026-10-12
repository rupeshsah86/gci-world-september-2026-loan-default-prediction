# Final Submission Candidate Review Report

## 1. Executive Summary
- **Submission Artifact**: [`outputs/submission_candidate_20261012.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/outputs/submission_candidate_20261012.csv)
- **Generated At**: 2026-10-12 17:01:19 UTC
- **Total Training & Inference Runtime**: 92.0 seconds (~1.5 minutes) on Apple Silicon M2.
- **Ensemble Architecture**: Tri-model probability blend combining three distinct gradient boosting implementations.
- **Training Scope**: Fitted on 100% of the competition training dataset (171,202 rows).
- **Inference Scope**: Generated predictions for 100% of the competition test dataset (61,500 rows).

## 2. Component Model Specifications

### Model 1: CatBoost Config 3 (Weight: 50%)
- **Feature Pipeline**: Verified 62-feature `ExtendedFeaturePipeline` (financial cash flow ratios, temporal & bureau ratios, and external score polynomial interactions).
- **Categorical Handling**: Native CatBoost categorical handling with string representation via `cat_features` pool.
- **Hyperparameters**: `depth=6, l2_leaf_reg=5.0, learning_rate=0.04, random_strength=2.0, loss_function='Logloss', eval_metric='AUC'`.
- **Full Training Budget**: 1,103 iterations (derived from 5-fold CV mean early-stopping iteration).
- **5-Fold CV Performance**: Mean ROC-AUC = **0.76076 $\pm$ 0.00263**, OOF ROC-AUC = **0.76072**.
- **Training Runtime**: 82.7 seconds.

### Model 2: Tuned LightGBM (Weight: 25%)
- **Feature Pipeline**: Verified 46-feature `BaselineFeaturePipeline` with domain ratios and integer label-encoded categories.
- **Categorical Handling**: Native LightGBM categorical feature support with `min_child_samples=100`.
- **Hyperparameters**: `num_leaves=24, max_depth=7, min_child_samples=100, feature_fraction=0.7, bagging_fraction=0.75, bagging_freq=1, reg_alpha=1.0, reg_lambda=5.0, learning_rate=0.03`.
- **Full Training Budget**: 465 iterations (derived from 5-fold CV mean early-stopping iteration).
- **5-Fold CV Performance**: Mean ROC-AUC = **0.75905 $\pm$ 0.00323**, OOF ROC-AUC = **0.75900**.
- **Training Runtime**: 3.9 seconds.

### Model 3: XGBoost Config 3 (Weight: 25%)
- **Feature Pipeline**: Verified 46-feature `BaselineFeaturePipeline` with domain ratios and pandas `CategoricalDtype`.
- **Categorical Handling**: Native XGBoost categorical support (`tree_method='hist'`, `enable_categorical=True`).
- **Hyperparameters**: `max_depth=6, min_child_weight=10, colsample_bytree=0.60, subsample=0.8, reg_alpha=2.0, reg_lambda=10.0, learning_rate=0.03`.
- **Full Training Budget**: 453 iterations (derived from 5-fold CV mean early-stopping iteration).
- **5-Fold CV Performance**: Mean ROC-AUC = **0.76021 $\pm$ 0.00285**, OOF ROC-AUC = **0.76015**.
- **Training Runtime**: 4.9 seconds.

## 3. Ensemble Blending and Validation Benchmarks
The ensemble weights `0.50 CatBoost + 0.25 LightGBM + 0.25 XGBoost` were validated across multiple stages:

| Evaluation Stage / Split | Component Models | Evaluation Metric | Result | Benchmark Context |
| :--- | :--- | :---: | :---: | :--- |
| **Stage D Baseline** | LightGBM Baseline (46f) | 5-Fold OOF ROC-AUC | 0.75627 | Initial domain features |
| **Stage E Tuning** | Tuned LightGBM (46f) | 5-Fold OOF ROC-AUC | 0.75900 | Hyperparameter tuning gain (+0.00273) |
| **Stage F2 Baseline** | XGBoost Baseline (46f) | 5-Fold OOF ROC-AUC | 0.75874 | First XGBoost evaluation |
| **Stage H1 Feature Ext.** | CatBoost Candidate 3 (62f) | 5-Fold OOF ROC-AUC | 0.76061 | Oblivious tree polynomial interaction gain |
| **Stage H2 Step 3A Tuning**| XGBoost Config 3 (46f) | 5-Fold OOF ROC-AUC | 0.76015 | Heavy regularization gain (+0.00141 across all 5 folds) |
| **Stage H2 Step 3A Tuning**| CatBoost Config 3 (62f) | 5-Fold OOF ROC-AUC | 0.76072 | Random strength & lr optimization |
| **Stage H2 Step 3B Ensemble**| Tri-Model Blend (Seed 42) | Nested CV ROC-AUC | **0.76212** | **Unbiased cross-validation ensemble estimate** |
| **Stage H2 Step 3B Ensemble**| Tri-Model Blend (Seed 42) | Aligned OOF ROC-AUC | **0.76208** | Combined probability blend |
| **Stage H2 Step 3B Stability**| Tri-Model Blend (Seed 2026)| Independent OOF ROC-AUC | **0.76161** | Verified on completely reshuffled folds |

> [!IMPORTANT]
> **Important Clarification on Validation Metrics:**
> The OOF ROC-AUC score (**0.76208** / nested **0.76212**) evaluates rank-ordering discrimination on out-of-fold validation splits. It is **not** classification accuracy and does **not** represent a public leaderboard score. Real test performance depends on competition test set calibration and sampling.

## 4. Test Prediction Distribution Summary
- **Total Rows Evaluated**: 61,500
- **Minimum Probability**: 0.002989
- **1st Percentile**: 0.008474
- **Median Probability**: 0.058145
- **Mean Probability**: 0.088585 *(reflects the natural positive class baseline ~8% in credit risk)*
- **99th Percentile**: 0.430466
- **Maximum Probability**: 0.811547
- **Standard Deviation**: 0.087920

## 5. File Verification and Compliance Audit
| Audit Check | Requirement | Result | Status |
| :--- | :--- | :--- | :---: |
| **File Location** | `outputs/submission_candidate_20261012.csv` | `/Users/rupeshmacbook/Documents/GCI_World/outputs/submission_candidate_20261012.csv` | **PASSED** |
| **Row Count** | Exactly 61,500 rows | 61,500 rows | **PASSED** |
| **Column Structure** | Exactly `['SK_ID_CURR', 'TARGET']` | `['SK_ID_CURR', 'TARGET']` | **PASSED** |
| **ID Integrity** | Exact match with test.csv in original order | Zero mismatch, zero duplicates | **PASSED** |
| **Prediction Values** | Finite probabilities $\in [0.0, 1.0]$ | Min: 0.0030, Max: 0.8115, NaNs: 0 | **PASSED** |
| **File Parseability** | Clean CSV re-opened and parsed | 100% valid CSV | **PASSED** |
| **Data Confidentiality** | No test labels, no data leakage, no external sets | Strictly adhered | **PASSED** |

## 6. Stop Checkpoint & Final Status
- The submission candidate file has been created and validated locally.
- In accordance with your instructions, **no file has been uploaded, submitted, or published anywhere**.
- This run is complete and stopped awaiting your review.