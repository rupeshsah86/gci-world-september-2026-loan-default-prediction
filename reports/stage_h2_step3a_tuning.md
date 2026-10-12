# Stage H2 Step 3A: Bounded Hyperparameter Refinement Report

## 1. Executive Summary
- **Total Runtime**: 17583.1 seconds (~293.1 minutes) on Apple Silicon M2.
- **Validation Setup**: 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), zero data leakage.
- **CatBoost Baseline Reference**: Mean = **0.76066**, OOF = **0.76061** (62 features, Candidate 3 pipeline)
- **XGBoost Baseline Reference**: Mean = **0.75879**, OOF = **0.75874** (46 features, Baseline pipeline)

## 2. Experiment A: CatBoost Refinement Results (62 Features)
| Configuration Name | Key Hyperparameters | Fold Scores (1 to 5) | Mean ROC-AUC (+/- Std) | OOF ROC-AUC | Mean Best Iter | Delta vs Baseline | Runtime |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Verified Reference** | `depth=6, l2=5.0, lr=0.05, rs=1.0` | `[0.75913, 0.76096, 0.76555, 0.75706, 0.76058]` | **0.76066 +/- 0.00281** | **0.76061** | 914.4 | Baseline | ~368s |
| **CB_Config_1_Conservative_Depth5** | `depth=5, l2=5.0, lr=0.05, rs=1.0` | `[0.75889, 0.76069, 0.76616, 0.75606, 0.76123]` | **0.76061 +/- 0.00331** | **0.76057** | 971.8 | **-0.00005** | 310.0s |
| **CB_Config_2_Moderate_L2_8** | `depth=6, l2=8.0, lr=0.05, rs=1.0` | `[0.75862, 0.76114, 0.76483, 0.75657, 0.76144]` | **0.76052 +/- 0.00279** | **0.76047** | 895.4 | **-0.00014** | 479.3s |
| **CB_Config_3_RandomStrength_2_LR04** | `depth=6, l2=5.0, lr=0.04, rs=2.0` | `[0.75815, 0.761, 0.76503, 0.75785, 0.76175]` | **0.76076 +/- 0.00263** | **0.76072** | 1103.4 | **+0.00010** | 332.4s |
| **CB_Config_4_High_L2_Regularization** | `depth=6, l2=10.0, lr=0.04, rs=1.5` | `[0.75944, 0.76152, 0.76487, 0.75684, 0.76153]` | **0.76084 +/- 0.00265** | **0.76080** | 1179.8 | **+0.00018** | 12616.0s |

## 3. Experiment B: XGBoost Refinement Results (46 Features)
| Configuration Name | Key Hyperparameters | Fold Scores (1 to 5) | Mean ROC-AUC (+/- Std) | OOF ROC-AUC | Mean Best Iter | Delta vs Baseline | Runtime |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Verified Reference** | `depth=6, mcw=1, colsample=0.70, a=1.0, l=5.0, lr=0.03` | `[0.75626, 0.75933, 0.76380, 0.75510, 0.75947]` | **0.75879 +/- 0.00303** | **0.75874** | 476.4 | Baseline | ~65s |
| **XGB_Config_1_Depth5_HigherMinChild** | `depth=5, mcw=5, colsample=0.65, a=1.5, l=8.0` | `[0.75809, 0.761, 0.76452, 0.75576, 0.76111]` | **0.76010 +/- 0.00297** | **0.76003** | 639.2 | **+0.00131** | 3354.3s |
| **XGB_Config_2_Depth6_MCW5_Reg8** | `depth=6, mcw=5, colsample=0.65, a=1.5, l=8.0` | `[0.75681, 0.7605, 0.76414, 0.75577, 0.76055]` | **0.75956 +/- 0.00299** | **0.75950** | 548.0 | **+0.00077** | 361.4s |
| **XGB_Config_3_Depth6_Colsample06_Reg10** | `depth=6, mcw=10, colsample=0.6, a=2.0, l=10.0` | `[0.75815, 0.7616, 0.76445, 0.75618, 0.76068]` | **0.76021 +/- 0.00285** | **0.76015** | 453.4 | **+0.00142** | 59.0s |
| **XGB_Config_4_Depth7_MinChild15** | `depth=7, mcw=15, colsample=0.65, a=1.5, l=6.0` | `[0.75771, 0.76119, 0.76466, 0.75462, 0.76133]` | **0.75990 +/- 0.00344** | **0.75983** | 493.0 | **+0.00111** | 68.2s |

## 4. Key Findings & Discussion
- **Best CatBoost Candidate**: `CB_Config_4_High_L2_Regularization` -> Mean AUC = **0.76084** (Delta: +0.00018), OOF = **0.76080**.
- **Best XGBoost Candidate**: `XGB_Config_3_Depth6_Colsample06_Reg10` -> Mean AUC = **0.76021** (Delta: +0.00142), OOF = **0.76015**.

### Regularization and Architecture Impact:
- In tabular default prediction, constraining leaf capacity and applying stronger L2 penalties stabilizes fold variance.
- Because multiple configurations were compared on the exact same folds, the top observed parameters should be interpreted as exploratory empirical candidates rather than unbiased final scores.

## 5. Summary & Next Steps
- All original competition files remain untouched.
- No test predictions, final submission file, or uploads generated.
- Awaiting user review before proceeding to seed stability, ensemble optimization, or final inference.