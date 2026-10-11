# CatBoost 5-Fold Cross-Validation Report (Stage F1)

## Overview
- **Model**: `CatBoostClassifier` with Native Categorical Handling
- **Dataset**: `Competition/input/train.csv` (171,202 rows, 8.07% positive default rate)
- **Features Used**: Exactly identical 46 domain-engineered features
- **Categorical Columns (8)**: ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE'] passed directly as string categories into `cat_features`
- **Validation**: 5-Fold Stratified K-Fold (`n_splits=5`, `shuffle=True`, `random_state=42`)
- **Leakage Controls**: Preprocessing fitted strictly on fold training data; `SK_ID_CURR` and `TARGET` excluded.

## Performance Benchmark Comparison
| Metric | Tuned LightGBM Benchmark | CatBoost (Stage F1) | Difference (CatBoost - LGB) |
| :--- | :--- | :--- | :--- |
| **Fold 1 ROC-AUC** | 0.75530 | **0.75713** | +0.00183 |
| **Fold 2 ROC-AUC** | 0.76093 | **0.76118** | +0.00025 |
| **Fold 3 ROC-AUC** | 0.76386 | **0.76646** | +0.00260 |
| **Fold 4 ROC-AUC** | 0.75571 | **0.75621** | +0.00050 |
| **Fold 5 ROC-AUC** | 0.75944 | **0.76200** | +0.00256 |
| **Mean Fold ROC-AUC** | **0.75905 +/- 0.00323** | **0.76060 +/- 0.00369** | **+0.00155** |
| **Overall OOF ROC-AUC** | **0.75900** | **0.76060** | **+0.00160** |
| **Total Runtime** | ~35 seconds | 269.2 seconds | — |

## Feature Handling Differences
1. **Categorical Features**: Unlike LightGBM which operates on integer category codes, CatBoost treats all 8 categorical columns natively using ordered target statistics computed on random permutations of the training fold.
2. **Missing Value Handling**: CatBoost handles NaNs in numerical features with native tree-split policies and treats string `"Missing"` as a distinct categorical state.

## Observations
- Consistency across all 5 folds.
- CatBoost provides a distinct, complementary mathematical formulation (symmetric oblivious decision trees + ordered target statistics) compared to LightGBM's leaf-wise histogram gradient boosting.
