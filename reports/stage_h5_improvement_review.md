# Stage H5: Advanced Model Improvement & Ensemble Audit Report

## 1. Executive Summary & Baseline Establishment

- **Project**: GCI World 2026 September — Home Credit Default Risk Prediction
- **Current Confirmed Best Submission**: [`outputs/submission_candidate_20261012_v2_quad_blend.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/outputs/submission_candidate_20261012_v2_quad_blend.csv) (Public Leaderboard: **0.76482**, Local OOF: **0.76224**)
- **Rank-Normalized Pilot Candidate**: [`outputs/submission_candidate_20261013_v21_rank_blend.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/outputs/submission_candidate_20261013_v21_rank_blend.csv) (Local OOF: **0.76230**)
- **Evaluation Methodology**: Strict 5-fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) on the exact 171,202 training samples, evaluated purely on ROC-AUC.
- **Stage H5 Objective**: Investigate complementary model architectures (ExtraTrees, HistGradientBoosting), blend diversity, ensemble weight re-optimization with nested cross-validation, feature engineering hypotheses, and deep validation leakage audits.
- **Primary Finding & Verdict**: 
  - Complementary tree models (**ExtraTrees**, **HistGradientBoosting**) introduce diversity (Spearman correlation $\sim 0.910 - 0.966$), but their substantially lower individual discriminative power (**0.74099** and **0.75590**) **dilutes the quad ensemble** at any positive blend weight ($w \ge 0.01$).
  - A feature engineering pilot testing debt-per-capita burden (`ANNUITY_TO_INCOME_PER_CAPITA`) degraded 5-fold LightGBM performance by **-0.00046** (0.75900 $\rightarrow$ 0.75854) due to tree split dilution.
  - Bounded quad-blend weight optimization shows mild in-sample OOF sensitivity (+0.00002), but **Nested 5-Fold Cross-Validation** confirms that the unbiased generalization estimate is **0.76228 $\pm$ 0.00281**, proving that adjusting weights beyond the established (0.27, 0.23, 0.30, 0.20) baseline would be selection bias.
  - **Recommendation**: Strictly **retain Candidate V2** (Public LB: **0.76482**) and **Candidate V2.1** (Local OOF: **0.76230**). Do not generate a degraded or unvalidated candidate CSV.

---

## 2. Experiment A: ExtraTrees & Complementary Tree Model Evaluation

To test whether non-gradient-boosted tree architectures or alternative implementations add orthogonal signal, we evaluated two distinct models across the identical 5 stratified folds:

### Model Specifications
1. **ExtraTreesClassifier (Extremely Randomized Trees)**:
   - Hyperparameters: `n_estimators=150, max_depth=18, min_samples_leaf=25, max_features="sqrt", class_weight="balanced_subsample", random_state=42+fold`
   - Preprocessing: Baseline 46 features with median imputation fit strictly inside each training fold.
2. **HistGradientBoostingClassifier**:
   - Hyperparameters: `max_iter=350, learning_rate=0.03, max_depth=6, min_samples_leaf=50, l2_regularization=5.0, random_state=42+fold`
   - Preprocessing: Baseline 46 features with native missing value handling and integer category encoding.

### Cross-Validation Results Matrix
| Model Architecture | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean Fold $\pm$ Std | 5-Fold OOF ROC-AUC | Delta vs Best Single Model (CB4: 0.76080) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost Config 4** (Baseline) | 0.75944 | 0.76152 | 0.76487 | 0.75684 | 0.76153 | 0.76084 $\pm$ 0.00265 | **0.76080** | Reference Baseline |
| **XGBoost Config 3** (Baseline)  | 0.75815 | 0.76160 | 0.76445 | 0.75618 | 0.76068 | 0.76021 $\pm$ 0.00285 | **0.76015** | -0.00065 |
| **Tuned LightGBM** (Baseline)    | 0.75530 | 0.76093 | 0.76386 | 0.75571 | 0.75944 | 0.75905 $\pm$ 0.00323 | **0.75900** | -0.00180 |
| **HistGradientBoosting** (H5 Exp A) | 0.75353 | 0.75708 | 0.75980 | 0.75335 | 0.75625 | 0.75600 $\pm$ 0.00240 | **0.75590** | **-0.00490** |
| **ExtraTreesClassifier** (H5 Exp A) | 0.73990 | 0.74098 | 0.74705 | 0.73800 | 0.73933 | 0.74105 $\pm$ 0.00315 | **0.74099** | **-0.01981** |

### Observations:
- **ExtraTrees** trained rapidly (~21.8s across all 5 folds), but its lack of gradient-based residual fitting resulted in an OOF AUC of **0.74099** (~0.020 below CatBoost).
- **HistGradientBoosting** achieved **0.75590**, performing competently but falling ~0.0031 behind Tuned LightGBM due to fixed binning and less aggressive regularizer parameterization.

---

## 3. Experiment B: Blend Diversity & Impact Testing

### Spearman Rank Correlation Matrix
| Model Pair | Spearman Correlation ($r_s$) | Diversity Interpretation |
| :--- | :---: | :--- |
| **ExtraTrees vs CatBoost Config 4** | **0.9235** | Moderate-low correlation (substantial diversity) |
| **ExtraTrees vs Tuned LightGBM**    | **0.9100** | Highest diversity observed across any tree pair |
| **ExtraTrees vs Candidate V2.1**    | **0.9279** | Divergent decision boundaries vs existing ensemble |
| **HistGradientBoosting vs CB4**     | **0.9657** | High correlation with gradient-boosted models |
| **HistGradientBoosting vs XGB3**    | **0.9699** | High correlation |
| **HistGradientBoosting vs V2.1**    | **0.9760** | Near-redundant with existing GBDT ensemble |

### Blend Inclusion Grid Analysis
We tested adding the new models into the existing baseline blends at incremental weights $w \in [0.01, 0.10]$:

#### Case 1: ExtraTrees Added to Ensemble
| ET Weight ($w$) | Prob Blend OOF AUC | Delta vs V2 Prob (0.76224) | Rank Blend OOF AUC | Delta vs V2.1 Rank (0.76230) |
| :---: | :---: | :---: | :---: | :---: |
| **$w = 0.00$** | **0.76224** | Baseline | **0.76230** | Baseline |
| **$w = 0.01$** | 0.76209 | -0.00015 | 0.76225 | -0.00005 |
| **$w = 0.02$** | 0.76191 | -0.00033 | 0.76220 | -0.00010 |
| **$w = 0.03$** | 0.76171 | -0.00053 | 0.76215 | -0.00015 |
| **$w = 0.05$** | 0.76128 | -0.00096 | 0.76203 | -0.00027 |
| **$w = 0.10$** | 0.76008 | -0.00216 | 0.76166 | -0.00064 |

#### Case 2: HistGradientBoosting Added to Ensemble
| HGB Weight ($w$) | Prob Blend OOF AUC | Delta vs V2 Prob (0.76224) | Rank Blend OOF AUC | Delta vs V2.1 Rank (0.76230) |
| :---: | :---: | :---: | :---: | :---: |
| **$w = 0.00$** | **0.76224** | Baseline | **0.76230** | Baseline |
| **$w = 0.01$** | 0.76223 | -0.00001 | 0.76229 | -0.00001 |
| **$w = 0.02$** | 0.76222 | -0.00002 | 0.76227 | -0.00003 |
| **$w = 0.03$** | 0.76220 | -0.00004 | 0.76226 | -0.00004 |
| **$w = 0.05$** | 0.76217 | -0.00007 | 0.76222 | -0.00008 |
| **$w = 0.10$** | 0.76207 | -0.00017 | 0.76211 | -0.00019 |

### Verdict on Model Inclusion:
> [!WARNING]
> **Ensemble Dilution Effect**:
> Diversity alone is insufficient for competitive blending; the candidate model must meet a threshold of individual predictive signal. Because ExtraTrees is ~0.020 lower in ROC-AUC than the GBDT components, adding even a 1% weight injects more ranking noise than complementary signal. HistGradientBoosting is also sub-optimal compared to Tuned LightGBM. Both models are **firmly rejected** for the final ensemble.

---

## 4. Experiment C: Quad-Blend Sensitivity & Nested Cross-Validation

To test whether the existing blend weights `(0.27 CB4, 0.23 CB3, 0.30 XGB3, 0.20 LGB)` can be improved, we evaluated a fine-grained grid of 62 valid non-negative weight vectors:

- **Top In-Sample OOF Configuration**: `(0.30 CB4, 0.25 CB3, 0.30 XGB3, 0.15 LGB)` $\rightarrow$ OOF ROC-AUC = **0.762326** (+0.000022 over 0.762304).
- **Nested 5-Fold Cross-Validation**: To guard against weight-selection optimism, weights were chosen purely on 4 training folds and evaluated on the held-out outer fold:
  - Fold 1 Selected: `(0.27, 0.25, 0.25, 0.23)` $\rightarrow$ Outer AUC: **0.75988**
  - Fold 2 Selected: `(0.35, 0.25, 0.30, 0.10)` $\rightarrow$ Outer AUC: **0.76338**
  - Fold 3 Selected: `(0.30, 0.25, 0.35, 0.10)` $\rightarrow$ Outer AUC: **0.76654**
  - Fold 4 Selected: `(0.35, 0.20, 0.30, 0.15)` $\rightarrow$ Outer AUC: **0.75855**
  - Fold 5 Selected: `(0.30, 0.25, 0.30, 0.15)` $\rightarrow$ Outer AUC: **0.76303**
  - **Nested CV Unbiased Mean**: **0.76228 $\pm$ 0.00281**

### Verdict on Weight Optimization:
The nested out-of-sample estimate (**0.76228**) matches the established baseline (**0.76228** mean fold score). The nominal in-sample +0.00002 gain is an artifact of mild sample overfitting. The existing `(0.27, 0.23, 0.30, 0.20)` allocation remains the optimal, stable balance.

---

## 5. Experiment D: Full Data, Pipeline & Validation Leakage Audit

A comprehensive audit was performed across all training, validation, and inference pipelines:

1. **Target Leakage**: **PASSED**.
   - Verified that `BaselineFeaturePipeline` and `ExtendedFeaturePipeline` fit all mappings (label encodings, category sets, sentinel thresholds) strictly on training folds.
   - Target labels (`TARGET`) are never used inside any transformer or imputer.
2. **Train/Test Feature Alignment**: **PASSED**.
   - Exact 100% correspondence of feature names, column count (46 for LGB/XGB, 62 for CB), and data types between `train.csv` and `test.csv`.
3. **ID Correspondence & Duplicates**: **PASSED**.
   - Training IDs (`SK_ID_CURR`) exactly match `[0, ..., 171201]` in sequential order (171,202 unique records, zero duplicates).
   - Test IDs exactly match `[171202, ..., 232701]` in sequential order (61,500 unique records, zero duplicates).
4. **Prediction Bounds & Finiteness**: **PASSED**.
   - All OOF predictions and candidate submissions contain strictly finite, non-NaN, non-inf continuous probabilities bounded within $[0.0, 1.0]$.
5. **OOF vs. Full-Training Test Pipeline Consistency**: **PASSED**.
   - Unnormalized full-test probability blend reproduced Candidate V2 with a maximum difference of $9.97 \times 10^{-17}$ (machine precision zero).
   - Full-train iteration numbers match CV early stopping averages (LightGBM: 465, XGBoost: 453, CB3: 1103, CB4: 1180).

---

## 6. Feature Engineering Pilot: Annuity-to-Income per Capita

- **Hypothesis**: In credit risk, a borrower's debt burden is heavily influenced by household size. A borrower with 4 dependents faces substantially higher default risk for the same income than a single borrower.
- **Formulation**:
  $$\text{ANNUITY\_TO\_INCOME\_PER\_CAPITA} = \frac{\text{AMT\_ANNUITY}}{\frac{\text{AMT\_INCOME\_TOTAL}}{\text{CNT\_FAM\_MEMBERS}} + 10^{-5}}$$
- **Validation**: Evaluated on 5-fold Stratified LightGBM with identical hyperparameters:
  - 46-Feature Baseline OOF ROC-AUC: **0.75900** (Mean: **0.75905 $\pm$ 0.00323**)
  - Pilot Feature OOF ROC-AUC: **0.75854** (Mean: **0.75858 $\pm$ 0.00316**)
  - Delta: **-0.00046** (Degraded on 4 of 5 folds: Fold 1: -0.00001, Fold 2: -0.00186, Fold 3: -0.00047, Fold 4: -0.00075, Fold 5: +0.00073).
- **Verdict**: **REJECTED**. The high collinearity with `ANNUITY_TO_INCOME_RATIO` and `INCOME_PER_FAMILY_MEMBER` dilutes split selection under feature subsampling (`colsample=0.70`).

---

## 7. Candidate Status & Strategic Recommendations

In strict accordance with competition governance rules:
> *"If no candidate shows a convincing improvement, do not pretend that one exists. Report the best validated candidate and recommend whether to retain V2."*

### Final Status Matrix:
| Candidate File | Method / Architecture | OOF ROC-AUC | Public LB | Status | Recommendation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| [`outputs/submission_candidate_20261012_v2_quad_blend.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/outputs/submission_candidate_20261012_v2_quad_blend.csv) | Quad-Model Probability Blend (`27/23/30/20`) | **0.76224** | **0.76482** | **CONFIRMED BEST** | **RETAIN AS PRIMARY** |
| [`outputs/submission_candidate_20261013_v21_rank_blend.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/outputs/submission_candidate_20261013_v21_rank_blend.csv) | Quad-Model Percentile-Rank Blend (`27/23/30/20`) | **0.76230** | — | **VALIDATED PILOT** | **RETAIN AS CONTENDER** |
| Stage H5 ExtraTrees Blend | 5-Model Blend (+ ExtraTrees) | 0.76209 | — | **REJECTED** | Dilutes ensemble (-0.00015) |
| Stage H5 HistGradientBoosting Blend | 5-Model Blend (+ HistGBM) | 0.76223 | — | **REJECTED** | Dilutes ensemble (-0.00001) |
| Stage H5 Re-tuned Quad Weights | Overfitted OOF Weights (`30/25/30/15`) | 0.76228 (nested) | — | **REJECTED** | Optimism artifact; no genuine gain |

### Strategic Conclusions:
1. **No new candidate CSV is generated for Stage H5**, preserving the integrity of your submission portfolio and avoiding leaderboard degradation.
2. The quad-model ensemble combining **CatBoost Config 4 (27%)**, **CatBoost Config 3 (23%)**, **XGBoost Config 3 (30%)**, and **Tuned LightGBM (20%)** represents the peak architecture for this feature space.
3. Both **Candidate V2** (probability-calibrated, LB: **0.76482**) and **Candidate V2.1** (rank-normalized, OOF: **0.76230**) remain fully verified, compliant, and ready.

---

## 8. Reproduction Commands & Experiment Artifacts

All Stage H5 artifacts are permanently tracked and reproducible:
- **Execution Script**: [`src/train_submission_v22_h5.py`](file:///Users/rupeshmacbook/Documents/GCI_World/src/train_submission_v22_h5.py)
- **ExtraTrees OOF Predictions**: [`experiments/stage_h5/oof_extratrees.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/experiments/stage_h5/oof_extratrees.csv)
- **HistGradientBoosting OOF Predictions**: [`experiments/stage_h5/oof_hist_gradient_boosting.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/experiments/stage_h5/oof_hist_gradient_boosting.csv)
- **Structured Experiment Results**: [`experiments/stage_h5/stage_h5_results.json`](file:///Users/rupeshmacbook/Documents/GCI_World/experiments/stage_h5/stage_h5_results.json)

**Reproduction Command**:
```bash
./.venv/bin/python src/train_submission_v22_h5.py
```
*(Total runtime: 53.6 seconds on Apple Silicon M2)*
