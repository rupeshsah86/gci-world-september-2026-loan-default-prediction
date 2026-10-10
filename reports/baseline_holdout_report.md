# Baseline Holdout Validation Report

## Overview
- **Evaluation Split**: 70% Train (119841 rows) / 30% Valid (51361 rows), Stratified, Seed = 42
- **Features Used**: All 32 permitted raw features + 3 domain anomaly indicators (35 features total)
- **Leakage Guards**: Preprocessing fitted solely on the training fold; target and `SK_ID_CURR` excluded.

## Performance Comparison
| Model | Features | Train ROC-AUC | Valid ROC-AUC | Improvement vs Tutorial | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tutorial Logistic Regression** | 5 | — | 0.66307 | Baseline | — |
| **Tutorial Random Forest** | 5 | — | 0.66879 | +0.00572 | — |
| **Random Forest (All Features)** | 35 | 0.89264 | **0.73353** | **+0.06474** | 2.4s |
| **LightGBM Baseline** | 35 | 0.83498 | **0.74719** | **+0.07840** | 2.8s |

## Classification Metrics at Threshold 0.5 (LightGBM)
- **ROC-AUC**: 0.74719
- **Accuracy**: 91.9180%
- **Precision**: 48.0315%
- **Recall**: 1.4713%
- **F1 Score**: 2.8551%
- **Confusion Matrix**: `TN=47149`, `FP=66`, `FN=4085`, `TP=61`

## Observations
1. Moving from 5 features to all available features dramatically boosts ROC-AUC from **~0.668** to **0.74719**.
2. LightGBM handles high cardinality categorical variables and native missing values cleanly and swiftly (2.8s training time).
3. The next step is validating this performance across 5-fold Stratified Cross-Validation and testing engineered financial domain ratios.
