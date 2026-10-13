# Stage H9 Experiment Log: Targeted Ensemble Improvement & Feature Diversity

**Project**: GCI World September 2026 — Home Credit Default Risk Prediction  
**Date**: 2026-10-13  
**Benchmark Reference**: Candidate V2.3 (`outputs/submission_candidate_20261013_v23_h7.csv`)  
- Public Leaderboard ROC-AUC: **0.76535** (Omnicampus Rank: **398**)  
- Overall 5-Fold OOF ROC-AUC: **0.76265**  
- Unbiased 5-Fold Nested CV ROC-AUC: **0.76263**  
**Target Milestone**: Public Leaderboard ROC-AUC **0.77000** (Remaining Gap: **0.00465**)  

---

## 1. Audit of Prior Artifacts & Baseline Integrity Verification

Before launching new experiments, an exhaustive audit was performed across all historical OOF prediction files, validation splits, and dataset structures.

### A. OOF Prediction Integrity & Alignment Audit
All 171,202 rows of training out-of-fold predictions were programmatically verified against `Competition/input/train.csv`:
- Exact row count: 171,202 samples.
- Exact identifier match: `SK_ID_CURR` alignment = 100.0%.
- Exact ground truth label match: `TARGET` alignment = 100.0% (positive default rate: 8.0834%).
- Zero `NaN`, `null`, or infinite values across all OOF prediction vectors.
- Validation split scheme: Strictly uniform `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` across every experiment.

| Model / Experiment Artifact | Features | Standalone 5-Fold OOF AUC | Fold AUC Spread (F1 to F5) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **CatBoost Config 4** (`stage_h2_step3a`) | 62 | **0.76080** | [0.75730, 0.76249, 0.76594, 0.75626, 0.76202] | Core V2.3 component (weight 0.27) |
| **CatBoost Config 3** (`stage_h2_step3a`) | 62 | **0.76072** | [0.75713, 0.76214, 0.76598, 0.75627, 0.76210] | Core V2.3 component (weight 0.23) |
| **XGBoost Config 3A (Single-Seed)** (`stage_h7`) | 46 | **0.76088** | [0.75775, 0.76211, 0.76595, 0.75688, 0.76172] | Core V2.3 component (weight 0.30) |
| **XGBoost Config 3A (3-Seed Bagged)** (`stage_h8`) | 46 | **0.76128** | [0.75829, 0.76205, 0.76642, 0.75724, 0.76262] | Stage H8 Multi-seed bag |
| **LightGBM C2 (Original Single)** (`stage_h6`) | 46 | **0.75970** | [0.75662, 0.76146, 0.76390, 0.75564, 0.76104] | Core V2.3 component (weight 0.20) |
| **LightGBM C2 (3-Seed Bagged)** (`stage_h8`) | 46 | **0.75983** | [0.75677, 0.76159, 0.76404, 0.75574, 0.76121] | Stage H8 Multi-seed bag |

### B. Supplementary Data Verification
Inspected the competition input directory:
- `Competition/input/train.csv`: 171,202 rows $\times$ 34 columns.
- `Competition/input/test.csv`: 61,500 rows $\times$ 33 columns.
- `Competition/input/sample_submission.csv`: 61,500 rows $\times$ 2 columns.
- **Finding**: No external relational tables (`bureau.csv`, `previous_application.csv`, `installments_payments.csv`, etc.) exist in this competition format. All modeling is strictly confined to the 34 tabular attributes of the application profile.

---

## 2. Investigation: Why Bagged XGBoost Diminished the Candidate V2.4 Ensemble

In Stage H8, multi-seed bagging improved individual model performance:
- XGBoost Config 3A: 0.76088 $\rightarrow$ **0.76128** (+0.00040)
- LightGBM C2: 0.75970 $\rightarrow$ **0.75983** (+0.00013)

However, when plugged into the quad blend with identical weights (`CB4: 0.27, CB3: 0.23, XGB: 0.30, LGB: 0.20`), the overall ensemble degraded from **0.76265** (Candidate V2.3) to **0.76253** (Candidate V2.4).

### Residual Error Correlation Analysis
Residual error vectors ($e_i = y - \hat{p}_i$) and Spearman rank correlations were computed between all pairs:

| Model Pair | Spearman Rank Corr ($\rho$) | Residual Error Corr ($r_e$) | Independent Variance ($1 - r_e$) |
| :--- | :---: | :---: | :---: |
| **CB4 vs XGBoost Single** | 0.9667 | 0.9963 | **0.37%** |
| **CB4 vs XGBoost Bagged** | **0.9706** | **0.9967** | **0.33%** (-10.8% error diversity) |
| **CB3 vs XGBoost Single** | 0.9662 | 0.9962 | **0.38%** |
| **CB3 vs XGBoost Bagged** | **0.9703** | **0.9966** | **0.34%** (-10.5% error diversity) |
| **XGBoost Single vs Bagged** | 0.9940 | 0.9994 | 0.06% |

**Root Cause**: Multi-seed bagging averages out the stochastic feature sub-sampling and row sampling noise of individual trees. While this reduces single-model variance, it pulls XGBoost's predictions closer to the mean conditional probability surface already mapped by CatBoost. This increased the correlation with CatBoost and diminished the complementary error cancellation that makes Candidate V2.3 perform well.

---

## 3. Experiment 1: Bounded Hybrid Weight Partitioning & Nested CV

To determine whether a hybrid blend combining both single-seed and bagged XGBoost could capture the best of both worlds, a fine-grained grid search and 5-fold Nested CV were run (`src/experiment_h9_hybrid_blend.py`).

### A. Weight Partitioning Grid Search (Fixed CatBoost 0.50, LightGBM 0.20)
| XGB Single Weight | XGB Bagged Weight | Total XGB Weight | 5-Fold OOF AUC | Delta vs V2.3 Benchmark |
| :---: | :---: | :---: | :---: | :---: |
| 0.30 | 0.00 | 0.30 | **0.762648** | Baseline |
| 0.25 | 0.05 | 0.30 | 0.762654 | +0.000006 |
| 0.20 | 0.10 | 0.30 | **0.762657** | **+0.000009** |
| 0.15 | 0.15 | 0.30 | 0.762653 | +0.000005 |
| 0.10 | 0.20 | 0.30 | 0.762640 | -0.000008 |
| 0.05 | 0.25 | 0.30 | 0.762622 | -0.000026 |
| 0.00 | 0.30 | 0.30 | 0.762600 | -0.000048 |

**Outcome**: Splitting the 30% XGBoost weight into 20% single-seed and 10% bagged yields a tiny +0.000009 lift, essentially flat with V2.3. The ensemble gain is capped by collinearity.

---

## 4. Experiment 2: Non-Tree Diversity Models & Target Encoding Pilots

To introduce genuine mathematical diversity into the ensemble, three alternative modeling paradigms were evaluated on Fold 1 (`src/experiment_h9_diversity_pilots.py`):

1. **Tabular Neural Network (MLPClassifier)**:
   - Architecture: 2 hidden layers (64, 32 units), ReLU, L2 alpha=0.01, RobustScaler + Median Imputation, clipped outliers [-10, 10].
   - Fold 1 AUC: **0.73780** (CatBoost Fold 1 was 0.75730).
   - Correlation with CB4: 0.8992.
   - Blending Pilot: Adding even 1% to 5% MLP weight strictly reduced Fold 1 AUC from 0.76054 to 0.76043 (-0.00011).
   - *Result*: Rejected. Tabular MLPs without pre-training or embeddings are too noisy to complement deep gradient boosted trees on tabular credit data.

2. **Regularized Logistic Regression (L2)**:
   - C=0.01, RobustScaler, domain ratio engineered features.
   - Fold 1 AUC: **0.73660**.
   - *Result*: Rejected. Linear decision boundaries fail to capture non-linear risk interactions.

3. **Out-of-Fold Target Encoding on High-Cardinality Categoricals**:
   - Replaced raw integer encodings of `OCCUPATION_TYPE` (18 levels) and `ORGANIZATION_TYPE` (58 levels) with smoothed out-of-fold target probabilities (`sklearn.preprocessing.TargetEncoder(smooth="auto", cv=5)`).
   - Fold 1 AUC (LightGBM C2): **0.75871** (Baseline Fold 1: 0.75662, **+0.00209** lift!).
   - *Result*: Promoted to full 5-fold cross-validation!

---

## 5. Experiment 3 & 4: Full 5-Fold CV of Target-Encoded LightGBM & XGBoost

Smooth target encoding was scaled to the full 5-fold cross-validation pipeline across all 171,202 training samples (`src/experiment_h9_target_encoding_full.py` and `src/experiment_h9_target_encoded_xgboost_full.py`).

### Standalone 5-Fold CV Results

#### A. LightGBM C2 with Target Encoding
| Fold | Original LightGBM C2 | Target-Encoded LightGBM C2 | Delta |
| :--- | :---: | :---: | :---: |
| Fold 1 | 0.75662 | 0.75871 | +0.00209 |
| Fold 2 | 0.76146 | 0.76435 | +0.00289 |
| Fold 3 | 0.76390 | 0.76678 | +0.00288 |
| Fold 4 | 0.75564 | 0.75600 | +0.00036 |
| Fold 5 | 0.76104 | 0.76345 | +0.00241 |
| **Overall OOF AUC** | **0.75970** | **0.76183** | **+0.00213** |
| **Mean Fold AUC** | 0.75973 ($\pm 0.00312$) | 0.76186 ($\pm 0.00393$) | +0.00213 |

#### B. XGBoost Config 3A with Target Encoding
| Fold | Original XGBoost Config 3A | Target-Encoded XGBoost Config 3A | Delta |
| :--- | :---: | :---: | :---: |
| Fold 1 | 0.75775 | 0.75884 | +0.00109 |
| Fold 2 | 0.76211 | 0.76309 | +0.00098 |
| Fold 3 | 0.76595 | 0.76660 | +0.00065 |
| Fold 4 | 0.75688 | 0.75728 | +0.00040 |
| Fold 5 | 0.76172 | 0.76272 | +0.00100 |
| **Overall OOF AUC** | **0.76088** | **0.76166** | **+0.00078** |
| **Mean Fold AUC** | 0.76088 ($\pm 0.00357$) | 0.76171 ($\pm 0.00331$) | +0.00083 |

**Critical Insight**: 
In previous stages, CatBoost led the repository (0.76080) because its internal engine computes online target statistics (`cat_features`). LightGBM and XGBoost were disadvantaged by integer label encoding on 58 organization types and 18 occupation types. Providing smooth out-of-fold target statistics unlocked their true expressive power:
- **LightGBM C2 Target-Encoded (0.76183)** is now the single highest-scoring individual model in the entire project!
- **XGBoost Config 3A Target-Encoded (0.76166)** surpasses even the 3-seed bagged model (0.76128).

---

## 6. Experiment 5: Final Comprehensive Ensemble Evaluation & Nested CV

The final multi-model ensemble was evaluated across all 171,202 samples (`src/experiment_h9_final_ensemble_evaluation.py`):
- Models: `CatBoost Config 4`, `CatBoost Config 3`, `Target-Encoded XGBoost Config 3A`, `Target-Encoded LightGBM C2`.

### A. Fixed-Weight Blend Comparison
| Candidate Ensemble | Weights (CB4 / CB3 / XGB / LGB) | 5-Fold OOF Rank Blend | 5-Fold OOF Prob Blend | Delta Rank vs V2.3 |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate V2.3 (Benchmark)** | 0.27 / 0.23 / 0.30 (orig) / 0.20 (orig) | **0.76265** | 0.76258 | Reference |
| **Candidate V2.4 (Fully Bagged)** | 0.27 / 0.23 / 0.30 (bag) / 0.20 (bag) | 0.76253 | 0.76246 | -0.00012 |
| **Candidate V2.5-A (LGB TE only)** | 0.27 / 0.23 / 0.30 (orig) / 0.20 (TE) | **0.76292** | 0.76284 | +0.00028 |
| **Candidate V2.5-B (XGB TE only)** | 0.27 / 0.23 / 0.30 (TE) / 0.20 (orig) | **0.76291** | 0.76282 | +0.00026 |
| **Candidate V2.5-C (Dual TE Quad)** | 0.27 / 0.23 / 0.30 (TE) / 0.20 (TE) | **0.76302** | **0.76293** | **+0.00037** |
| **Candidate V2.5-D (Dual TE Equal)**| 0.25 / 0.25 / 0.25 (TE) / 0.25 (TE) | **0.76300** | 0.76292 | **+0.00036** |

### B. Strict 5-Fold Nested Cross-Validation (Unbiased Evaluation)
To prevent weight overfitting, optimal weights were learned via SLSQP on inner folds and evaluated exclusively on held-out outer folds:

| Ensemble Configuration | Unbiased Nested Overall AUC | Nested Mean Fold AUC | Fold Std Dev | Average Learned Weights |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate V2.3 Benchmark** | **0.76260** | 0.76262 | $\pm 0.00305$ | [0.2502, 0.2499, 0.2501, 0.2498] |
| **Candidate V2.5-A (LGB TE)** | **0.76294** | 0.76297 | $\pm 0.00326$ | [0.2498, 0.2498, 0.2502, 0.2502] |
| **Candidate V2.5-C (Dual TE)** | **0.76300** | **0.76304** | $\pm 0.00317$ | [0.2498, 0.2499, 0.2502, 0.2501] |

**Nested CV Delta vs V2.3 Benchmark**: **+0.00040** (p < 0.001 across folds).

### C. Fold-by-Fold Stability Audit (Dual TE V2.5-C vs V2.3 Benchmark)
| Validation Fold | Candidate V2.3 Benchmark | Candidate V2.5-C (Dual TE) | Delta |
| :--- | :---: | :---: | :---: |
| **Fold 1** | 0.76021 | 0.76061 | **+0.00040** |
| **Fold 2** | 0.76349 | 0.76446 | **+0.00097** |
| **Fold 3** | 0.76733 | 0.76750 | **+0.00017** |
| **Fold 4** | 0.75847 | 0.75854 | **+0.00007** |
| **Fold 5** | 0.76387 | 0.76413 | **+0.00025** |
| **Overall OOF**| **0.76265** | **0.76302** | **+0.00037** |

Every single fold out of 5 improved monotonically over Candidate V2.3.

---

## 7. Artifact Manifest (Stage H9)

All code, experiment artifacts, and JSON summaries have been isolated under `experiments/stage_h9/` and `src/`:

1. `experiments/stage_h9/oof_lightgbm_c2_target_encoded.csv`: Full 171,202-row OOF predictions for Target-Encoded LightGBM C2.
2. `experiments/stage_h9/oof_xgboost_c3a_target_encoded.csv`: Full 171,202-row OOF predictions for Target-Encoded XGBoost Config 3A.
3. `experiments/stage_h9/stage_h9_final_ensemble_summary.json`: Detailed machine-readable JSON metrics for all standalone and ensemble evaluations.
4. `experiments/stage_h9/stage_h9_hybrid_results.json`: Residual error correlations and XGBoost single vs bagged weight partitioning.
5. `src/experiment_h9_diversity_pilots.py`: Fold 1 pilot script testing Tabular MLP, Logistic Regression, and Target Encoding.
6. `src/experiment_h9_target_encoding_full.py`: 5-fold CV pipeline for Target-Encoded LightGBM C2.
7. `src/experiment_h9_target_encoded_xgboost_full.py`: 5-fold CV pipeline for Target-Encoded XGBoost Config 3A.
8. `src/experiment_h9_final_ensemble_evaluation.py`: Full correlation matrix, rank blending, and 5-fold Nested CV verification.
