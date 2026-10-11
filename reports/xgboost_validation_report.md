# XGBoost 5-Fold Cross-Validation Report (Stage F2)

## Overview
- **Model**: `XGBClassifier` with Native Categorical Splits (`enable_categorical=True`, `tree_method="hist"`)
- **Dataset**: `Competition/input/train.csv` (171,202 rows, 8.07% positive default rate)
- **Features Used**: Identical 46 domain-engineered features
- **Categorical Columns (8)**: ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE'] formatted as pandas `CategoricalDtype` fitted strictly per fold
- **Validation**: 5-Fold Stratified K-Fold (`n_splits=5`, `shuffle=True`, `random_state=42`)
- **Leakage Controls**: Preprocessing fitted strictly on fold training data; `SK_ID_CURR` and `TARGET` excluded.

## Performance Benchmark Comparison
| Metric | Tuned LightGBM | CatBoost | XGBoost (Stage F2) | Delta vs LGB | Delta vs CatBoost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fold 1 ROC-AUC** | 0.75530 | 0.75713 | **0.75626** | +0.00096 | -0.00087 |
| **Fold 2 ROC-AUC** | 0.76093 | 0.76118 | **0.75933** | -0.00160 | -0.00185 |
| **Fold 3 ROC-AUC** | 0.76386 | 0.76646 | **0.76380** | -0.00006 | -0.00266 |
| **Fold 4 ROC-AUC** | 0.75571 | 0.75621 | **0.75510** | -0.00061 | -0.00111 |
| **Fold 5 ROC-AUC** | 0.75944 | 0.76200 | **0.75947** | +0.00003 | -0.00253 |
| **Mean Fold ROC-AUC** | **0.75905** | **0.76060** | **0.75879 +/- 0.00303** | **-0.00026** | **-0.00181** |
| **Overall OOF ROC-AUC** | **0.75900** | **0.76060** | **0.75874** | **-0.00026** | **-0.00186** |
| **Total Runtime** | ~35s | ~269s | 65.1s | — | — |

## Feature Handling & Implementation Details
1. **Native Categorical Partitioning**: Handled via XGBoost's `tree_method="hist"` with `enable_categorical=True`. Categories are represented without arbitrary ordinal assumptions.
2. **Missing Values**: Missing values in numerical columns are natively routed by XGBoost to the default split direction maximizing gain.
3. **Data Verification**:
   - Every training row received exactly one out-of-fold prediction.
   - Out-of-fold predictions align row-for-row with `SK_ID_CURR`.
   - Predictions strictly bounded in continuous probability range `[0.0, 1.0]`.
