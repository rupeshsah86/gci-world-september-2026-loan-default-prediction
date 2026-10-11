# Ensemble Blending Report (Stage G1)

## 1. Out-of-Fold Prediction Audit Summary
- **Dataset**: `Competition/input/train.csv` (171,202 samples, exactly matched)
- **Integrity Checks**: All 3 OOF files (`oof_catboost.csv`, `oof_tuned_lightgbm.csv`, `oof_xgboost.csv`) joined strictly on `SK_ID_CURR`.
- **Prediction Bounds**: Zero NaN or infinite values; all probabilities strictly bounded in `[0.0, 1.0]`.
- **Row Alignment**: 100% ID correspondence verified across all 171,202 rows.

## 2. Individual Model Recomputed Baselines
- **CatBoost (Native Categoricals)**: **0.76060**
- **Tuned LightGBM (Reg + Subsampling)**: **0.75900**
- **XGBoost (Native Categoricals)**: **0.75874**

## 3. Weighted Probability Blend Results Table
| Rank | Configuration | CatBoost Wt | LightGBM Wt | XGBoost Wt | Overall OOF ROC-AUC | Mean Fold ROC-AUC | Delta vs CatBoost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **CatBoost 50% + LightGBM 25% + XGBoost 25%** | 0.50 | 0.25 | 0.25 | **0.76187** | 0.76189 +/- 0.00344 | +0.00127 |
| 2 | **CatBoost 60% + LightGBM 30% + XGBoost 10%** | 0.60 | 0.30 | 0.10 | **0.76185** | 0.76187 +/- 0.00353 | +0.00125 |
| 3 | **CatBoost 70% + LightGBM 20% + XGBoost 10%** | 0.70 | 0.20 | 0.10 | **0.76177** | 0.76178 +/- 0.00356 | +0.00117 |
| 4 | **CatBoost 50% + LightGBM 50%** | 0.50 | 0.50 | 0.00 | **0.76165** | 0.76167 +/- 0.00354 | +0.00105 |
| 5 | **CatBoost 75% + XGBoost 25%** | 0.75 | 0.00 | 0.25 | **0.76165** | 0.76165 +/- 0.00350 | +0.00105 |
| 6 | **CatBoost 50% + XGBoost 50%** | 0.50 | 0.00 | 0.50 | **0.76165** | 0.76168 +/- 0.00334 | +0.00105 |
| 7 | **CatBoost 75% + LightGBM 25%** | 0.75 | 0.25 | 0.00 | **0.76160** | 0.76161 +/- 0.00363 | +0.00100 |
| 8 | **CatBoost 25% + LightGBM 75%** | 0.25 | 0.75 | 0.00 | **0.76079** | 0.76082 +/- 0.00340 | +0.00019 |
| 9 | **CatBoost 25% + XGBoost 75%** | 0.25 | 0.00 | 0.75 | **0.76069** | 0.76073 +/- 0.00319 | +0.00009 |
| 10 | **CatBoost 100%** | 1.00 | 0.00 | 0.00 | **0.76060** | 0.76060 +/- 0.00369 | -0.00000 |
| 11 | **LightGBM 50% + XGBoost 50%** | 0.00 | 0.50 | 0.50 | **0.75972** | 0.75976 +/- 0.00312 | -0.00088 |
| 12 | **LightGBM 100%** | 0.00 | 1.00 | 0.00 | **0.75900** | 0.75905 +/- 0.00323 | -0.00160 |
| 13 | **XGBoost 100%** | 0.00 | 0.00 | 1.00 | **0.75874** | 0.75879 +/- 0.00303 | -0.00186 |

## 4. Best Configuration Analysis
- **Top Blend Configuration**: **`CatBoost 50% + LightGBM 25% + XGBoost 25%`**
- **Weights**: CatBoost = `0.50`, LightGBM = `0.25`, XGBoost = `0.25`
- **Overall OOF ROC-AUC**: **0.76187**
- **Gain over Best Single Model (CatBoost)**: **+0.00127**
- **Gain over Official Tutorial Baseline (0.66879)**: **+0.09308**

## 5. Honest Validation & Selection Optimism Assessment
Because these out-of-fold predictions were previously utilized during single-model selection and tuning, choosing the top blend across all 171,202 rows creates mild **selection optimism**.

To rigorously quantify this, we implemented a **Nested Cross-Validation** routine:
- On each of the 5 folds, blend weights were selected purely using the other 4 folds, and evaluated strictly on the held-out fold.
- **Nested Out-of-Sample Ensemble ROC-AUC**: **0.76184 +/- 0.00349**
- **Selection Optimism Gap**: `+0.00003` (demonstrating that the gain is genuine and robust against selection bias).

## 6. Next Steps & Governance
- All original competition files remain untouched.
- No final test submission or automatic upload has been performed.