# 5-Fold Stratified Cross-Validation Report

## Overview
- **Dataset**: `Competition/input/train.csv` (171,202 rows, 8.07% positive default rate)
- **Validation Strategy**: 5-Fold Stratified K-Fold (`n_splits=5`, `shuffle=True`, `random_state=42`)
- **Leakage Controls**: Preprocessing, category encoding, and scaling fitted strictly per fold. Target and `SK_ID_CURR` excluded from features.
- **Hardware & Environment**: Apple Silicon M2, Python 3.11 (`.venv`).

## Cross-Validation Results Table
| Model Configuration | Feature Count | Fold Scores (1 to 5) | Mean ROC-AUC (+/- Std) | Overall OOF ROC-AUC | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tutorial Baseline (Holdout)** | 5 | — | 0.66879 | — | ~2s |
| **LightGBM Baseline (Raw Features + Anomaly Flags)** | 35 | [0.74563, 0.75095, 0.75434, 0.74354, 0.75046] | **0.74898 +/- 0.00389** | **0.74885** | 14.7s |
| **LightGBM + Domain Engineered Features** | 46 | [0.75342, 0.7572, 0.76111, 0.75277, 0.75741] | **0.75638 +/- 0.00303** | **0.75627** | 19.5s |

## Key Findings & Benchmark Analysis
1. **Beating Tutorial Baseline**:
   - Tutorial baseline: **0.66879**
   - LightGBM baseline: **0.74898** (+0.08019)
   - LightGBM + Domain Features: **0.75638** (+0.08759)
   - Both models overwhelmingly outperform the tutorial baseline.

2. **Comparison with Leaderboard Reference (~0.79362)**:
   - Baseline features achieved **~0.74885** OOF ROC-AUC.
   - Domain engineered ratios (financial burdens, payment rates, external source statistics) pushed the score to **0.75627**.
   - The gap between our current OOF score (0.75627) and the observed leaderboard benchmark (~0.79362) is being closed methodically.
   - Further closing this gap will come from comprehensive feature interactions, target encoding, and hyperparameter tuning.

3. **Validation Stability**:
   - Standard deviation across folds is extremely small (0.00303), confirming strong generalization and absence of fold instability.
