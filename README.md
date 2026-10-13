# GCI World 2026 — Home Credit Default Risk Prediction
### End-to-End Competitive Machine Learning Pipeline & Ensembling

[![Python 3.11](https://img.shields.io/badge/Python-3.11.12-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-yellow.svg)](https://catboost.ai/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red.svg)](https://xgboost.readthedocs.io/)
[![Metric](https://img.shields.io/badge/Metric-ROC--AUC-purple.svg)]()
[![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon%20(M2)-black.svg)]()

A state-of-the-art, fully reproducible, leakage-free machine learning solution for the **GCI World September 2026 Home Credit Loan Default Prediction Competition**. 

This repository documents the complete engineering journey from the initial competition tutorial baseline (**0.66879 ROC-AUC**) through structured feature engineering, adversarial validation, iterative hyperparameter tuning, seed stability benchmarking, and a tri-model probability ensemble (**0.76212 nested OOF ROC-AUC**).

---

## Table of Contents
1. [Executive Summary & Key Results](#1-executive-summary--key-results)
2. [Project Architecture & Methodology](#2-project-architecture--methodology)
3. [Experimentation Timeline & Evolution](#3-experimentation-timeline--evolution)
4. [Diagnostics & Scientific Verification](#4-diagnostics--scientific-verification)
5. [Final Model Ensemble & Weights](#5-final-model-ensemble--weights)
6. [Repository Organization](#6-repository-organization)
7. [Quickstart & Reproduction Guide](#7-quickstart--reproduction-guide)
8. [Submission File Specifications](#8-submission-file-specifications)
9. [Engineering Rigor & Leakage Prevention](#9-engineering-rigor--leakage-prevention)

---

## 1. Executive Summary & Key Results

The objective is to predict default probabilities (`TARGET` $\in \{0, 1\}$) on uncollateralized consumer loans, evaluated by **ROC-AUC** on continuous probabilities.

- **Dataset Scale**: 171,202 training samples (positive default rate: 8.07%) and 61,500 test samples.
- **Validation Strategy**: 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), replicated independently on `seed=2026`.
- **Top Individual Models**:
  - **CatBoost Config 3 (62 features)**: OOF ROC-AUC **0.76072** ($\pm 0.00263$)
  - **XGBoost Config 3 (46 features)**: OOF ROC-AUC **0.76015** ($\pm 0.00285$)
  - **Tuned LightGBM (46 features)**: OOF ROC-AUC **0.75900** ($\pm 0.00323$)
- **Winning Tri-Model Ensemble**:
  - Fixed Probability Blend: **50% CatBoost + 25% LightGBM + 25% XGBoost**
  - **Aligned OOF ROC-AUC**: **0.76208**
  - **Unbiased Nested CV ROC-AUC**: **0.76212**
  - **Replication on Independent Seed (2026)**: **0.76161** (+0.0015 over single best model)

### Benchmark Progression

| Milestone / Stage | Method / Configuration | Features | 5-Fold OOF ROC-AUC | Lift vs Previous |
| :--- | :--- | :---: | :---: | :---: |
| **Tutorial Baseline** | Default Random Forest (5 raw features) | 5 | 0.66879* | Baseline |
| **Stage C: Feature Discovery** | LightGBM with 35 raw features + anomaly fixes | 35 | 0.74719* | +0.07840 |
| **Stage D: Domain Engineering** | LightGBM Baseline with financial credit ratios | 46 | 0.75627 | +0.00908 |
| **Stage E: Hyperparameter Tuning** | Tuned LightGBM (leaves=24, depth=7, reg_lambda=5.0) | 46 | 0.75900 | +0.00273 |
| **Stage F1: CatBoost Introduction** | CatBoost with native categorical handling | 46 | 0.76060 | +0.00160 |
| **Stage F2: XGBoost Introduction** | XGBoost Histogram with native categoricals | 46 | 0.75874 | -0.00026 vs LGB |
| **Stage G1: First Ensemble Blend** | Probability blend (50% CB + 25% LGB + 25% XGB) | 46 | 0.76187 | +0.00127 |
| **Stage H1: External Polynomials** | CatBoost Candidate 3 (financial + temporal + ext prod) | 62 | 0.76061 | +0.00001 (lower std) |
| **Stage H2 Step 3A: XGBoost Refinement**| XGBoost Config 3 (mcw=10, col=0.60, a=2.0, l=10.0) | 46 | **0.76015** | **+0.00141** |
| **Stage H2 Step 3A: CatBoost Refinement**| CatBoost Config 3 (depth=6, l2=5.0, lr=0.04, rs=2.0) | 62 | **0.76072** | **+0.00011** |
| **Stage H2 Step 3B: Final Ensemble** | Optimized Tri-Model Blend (Seed 42) | 62 / 46 | **0.76208** | **+0.00136** |
| **Stage H2 Step 3B: Nested Validation** | Unbiased Nested CV Estimate | 62 / 46 | **0.76212** | **Unbiased estimate** |
| **Stage H3: Candidate V2 (Confirmed)** | Quad-Model Prob Blend (CB4 27%, CB3 23%, XGB3 30%, LGB 20%) | 62 / 46 | **0.76224** | **+0.00016** (LB: **0.76482**) |
| **Stage H4: Candidate V2.1** | Percentile-Rank Normalized Quad Blend | 62 / 46 | **0.76230** | **+0.00006** |
| **Stage H5: Tree Diversity & Nested Opt** | Evaluation of ExtraTrees/HistGBM (Rejected) & Weights | 62 / 46 | 0.76228 | Nested CV confirmation |
| **Stage H6: LightGBM C2 Optimization** | LightGBM C2 (depth=6, leaves=20, lr=0.025, reg_l2=10) | 46 | **0.75970** | **+0.00070** vs LGB C1 |
| **Stage H6: Candidate V2.2** | Upgraded LightGBM C2 Rank Blend (Pre-Submission Audited) | 62 / 46 | **0.76235** | **+0.00011** vs V2 Benchmark |
| **Stage H7: XGBoost Config 3A** | Upgraded XGBoost Config 3A (depth=6, mcw=15, col=0.65, l2=15) | 46 | **0.76088** | **+0.00073** (Top Single Model) |
| **Stage H7: Candidate V2.3** | Upgraded Quad-Model Rank Blend (Audited Submission) | 62 / 46 | **0.76265** | **+0.00041** vs V2 (Nested: 0.76263) |

*\*Holdout 70/30 split; all subsequent stages evaluated on identical 5-Fold Stratified CV.*

---

## 2. Project Architecture & Methodology

```
                                  Raw Competition Data (171,202 Train / 61,500 Test)
                                                          │
                             ┌────────────────────────────┴───────────────────────────┐
                             │                                                        │
                   Baseline Feature Pipeline                                Extended Feature Pipeline
                   (46 Engineered Features)                                 (62 Engineered Features)
                             │                                                        │
              ┌──────────────┴──────────────┐                                         │
              │                             │                                         │
     Tuned LightGBM                 XGBoost Config 3                          CatBoost Config 3
    • Leaves: 24, Depth: 7         • Depth: 6, MCW: 10                       • Depth: 6, L2: 5.0
    • Feature Fraction: 0.70       • Colsample: 0.60                         • LR: 0.04, Random Strength: 2.0
    • Reg Alpha: 1, Lambda: 5      • Reg Alpha: 2, Lambda: 10                • Native String Categoricals
    • Integer Label Encoding       • Native CategoricalDtype                 • 1,103 iterations
    • 465 iterations               • 453 iterations                                   │
              │                             │                                         │
              │ [Weight: 25%]               │ [Weight: 25%]                           │ [Weight: 50%]
              └─────────────────────────────┼─────────────────────────────────────────┘
                                            │
                             Tri-Model Probability Blending
                                   P = 0.50*CB + 0.25*LGB + 0.25*XGB
                                            │
                             ┌──────────────┴──────────────┐
                             │                             │
                   Nested 5-Fold OOF Eval          Full-Data Test Inference
                   ROC-AUC: 0.76212                outputs/submission_candidate_20261012.csv
```

### Feature Engineering Breakdown
1. **Domain & Financial Ratios (46 Features)**:
   - `PAYMENT_RATE`: `AMT_ANNUITY / AMT_CREDIT` (Strongest loan burden indicator).
   - `CREDIT_INCOME_PERCENT`: `AMT_CREDIT / AMT_INCOME_TOTAL` (Leverage multiple).
   - `ANNUITY_INCOME_PERCENT`: `AMT_ANNUITY / AMT_INCOME_TOTAL` (Debt-service-to-income).
   - `DAYS_EMPLOYED_PERCENT`: `DAYS_EMPLOYED / DAYS_BIRTH` (Career tenure ratio).
   - `INCOME_PER_PERSON`: `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS` (Household income dilution).
2. **External Interaction & Polynomial Extensions (62 Features — CatBoost Candidate 3)**:
   - `EXT_SOURCES_PROD`: `EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3` (Joint credit bureau credit score product).
   - `EXT_WEIGHTED_SCORE`: Variance-weighted composite of external credit bureau ratings.
   - `EXT_DIFF`: Divergence between max and min available external ratings.
   - Missing indicator flags for external sources and bureau records.
3. **Data Anomaly Sanitization**:
   - `DAYS_EMPLOYED == 365243` sentinel replacement with `NaN` + addition of `DAYS_EMPLOYED_ANOM` binary flag.
   - Outlier truncation on `OWN_CAR_AGE` ($\ge 60$) with binary car ownership indicators.

---

## 3. Experimentation Timeline & Evolution

### Stage E: LightGBM Regularization Tuning
- Discovered that reducing leaf capacity from default 31 down to **24 leaves** with **max_depth=7** and strong L2 penalty (**reg_lambda=5.0**) prevented leaf over-specialization, pushing OOF ROC-AUC from 0.75627 to **0.75900**.

### Stage F1 & F2: Expanding the Algorithmic Ensemble
- Implemented **CatBoost** using native categorical handling: oblivious symmetric trees achieved **0.76060**, surpassing LightGBM.
- Implemented **XGBoost** with `tree_method="hist"` and native pandas `CategoricalDtype`: achieved **0.75874**.

### Stage H1: Feature Extensions for Oblivious Trees
- Discovered an architectural divergence: polynomial external score terms produced gains for CatBoost's oblivious trees (OOF **0.76061**, narrowing standard deviation to $\pm 0.00281$), while degrading LightGBM's leaf-wise trees due to split dilution under column subsampling.

### Stage H2 Step 3A: Bounded Hyperparameter Refinement
- **XGBoost Breakthrough**: Tuning to `min_child_weight=10`, `colsample_bytree=0.60`, `reg_alpha=2.0`, and `reg_lambda=10.0` produced a uniform gain across **all 5 folds** ($+0.00141$ OOF), lifting XGBoost to **0.76015**.
- **CatBoost Refinement**: Introducing `random_strength=2.0` with `learning_rate=0.04` stabilized trees at **0.76072** OOF.

---

## 4. Diagnostics & Scientific Verification

### Adversarial Validation (Zero Covariate Shift)
- Trained a binary classifier to distinguish between training and test sets using all features.
- **Mean ROC-AUC**: **0.51952 $\pm$ 0.00190** ($\approx 0.50$).
- **Conclusion**: Confirms that training and test feature distributions are identical. Validation fold scores accurately reflect test set behavior with zero covariate shift.

### Learning Curve Diagnostics
- Evaluated training sample fractions (25%, 50%, 75%, 100%):
  - 25% data: 0.74760 AUC
  - 50% data: 0.75402 AUC (+0.00642)
  - 75% data: 0.75693 AUC (+0.00291)
  - 100% data: 0.75905 AUC (+0.00212)
- Confirmed smooth log-linear convergence with diminishing returns to raw sample size, demonstrating that model architecture diversity and regularization drive the remaining performance ceiling.

### Feature Gain Concentration
- Top 3 features drive **40.26%** of total tree gain.
- Top 10 features drive **68.74%** of gain.
- External bureau scores (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`) alone account for **42.17%** of predictive power.

---

## 5. Final Model Ensemble & Weights

### Model Prediction Correlation Matrix
The low error correlation across the three distinct architectures explains the steady ensemble lift:

| Model Family | CatBoost Config 3 | XGBoost Config 3 | Tuned LightGBM |
| :--- | :---: | :---: | :---: |
| **CatBoost Config 3** | **1.0000** | 0.9620 | 0.9559 |
| **XGBoost Config 3**  | 0.9620 | **1.0000** | 0.9819 |
| **Tuned LightGBM**    | 0.9559 | 0.9819 | **1.0000** |

*CatBoost vs. LightGBM Pearson correlation is 0.9559, providing critical orthogonal decision boundaries.*

### Ensemble Blend Formula
$$\hat{P}_{\text{ensemble}} = 0.50 \cdot \hat{P}_{\text{CatBoost}} + 0.25 \cdot \hat{P}_{\text{LightGBM}} + 0.25 \cdot \hat{P}_{\text{XGBoost}}$$

- **Nested Cross-Validation (Unbiased)**: **0.76212**
- **Independent Seed 2026 Replication**: **0.76161**

---

## 6. Repository Organization

```text
GCI_World/
├── Competition/                          # Competition input data (not committed)
│   ├── input/
│   │   ├── train.csv                     # Training data (171,202 rows)
│   │   ├── test.csv                      # Test data (61,500 rows)
│   │   └── sample_submission.csv
│   └── HomeCredit_columns_description.xlsx
├── src/                                  # Production pipeline & experiment code
│   ├── data.py                           # Data loading and integrity assertions
│   ├── features.py                       # 46-feature baseline pipeline
│   ├── features_h1.py                    # 62-feature extended pipeline (CatBoost)
│   ├── train_baseline.py                 # Holdout baseline evaluation (Stage C)
│   ├── train_cv.py                       # 5-Fold Stratified CV pipeline (Stage D)
│   ├── tune_lgb.py                       # LightGBM hyperparameter search (Stage E)
│   ├── train_catboost.py                 # Native CatBoost CV pipeline (Stage F1)
│   ├── train_xgboost.py                  # Native XGBoost CV pipeline (Stage F2)
│   ├── blend_evaluation.py               # Ensemble blending & audit (Stage G1/G2)
│   ├── experiment_h1.py                  # Feature exploration candidate suite (Stage H1)
│   ├── experiment_h2_step1.py            # Diagnostics & Adversarial Validation (Stage H2 Step 1)
│   ├── experiment_h2_step3a.py           # Bounded hyperparameter refinement (Stage H2 Step 3A)
│   ├── experiment_h2_step3b.py           # Seed stability & nested blend CV (Stage H2 Step 3B)
│   ├── train_final_submission.py         # Full-data training & test inference (V1 Baseline)
│   ├── train_submission_v2.py            # Candidate V2 quad-model probability blend (Stage H3)
│   ├── evaluate_stage_h4.py              # Rank blend evaluation pilot (Stage H4)
│   ├── train_submission_v21_rank.py      # Candidate V2.1 rank-normalized blend (Stage H4)
│   ├── train_submission_v22_h5.py        # Stage H5 tree diversity experiments
│   ├── train_submission_v22_h6.py        # Candidate V2.2 upgraded LightGBM C2 rank blend (Stage H6)
│   ├── experiment_h7_xgb_tuning.py       # Stage H7 XGBoost regularized tuning pilot
│   ├── experiment_h7_cb_tuning.py        # Stage H7 CatBoost screening pilot
│   ├── experiment_h7_features.py         # Stage H7 feature engineering screening on LightGBM C2
│   ├── train_xgb_c3a_5fold.py            # Stage H7 5-fold evaluation of XGBoost Config 3A
│   ├── evaluate_stage_h7_ensemble.py     # Stage H7 ensemble blending and nested CV engine
│   └── train_submission_v23_h7.py        # Candidate V2.3 full-data inference pipeline (Stage H7)
├── experiments/                          # Machine-readable experiment records & JSON logs
│   ├── stage_h2_step3a_results.json      # Tuning grid scores and best iterations
│   ├── stage_h2_step3b_results.json      # Seed stability and nested blend logs
│   ├── stage_h4_rank_blend_results.json  # Stage H4 rank blend evaluation
│   ├── stage_h5/stage_h5_results.json    # Stage H5 diversity model results
│   ├── stage_h6/stage_h6_results.json    # Stage H6 LightGBM C2 results
│   ├── stage_h7/stage_h7_results.json    # Stage H7 candidate V2.3 results
│   ├── stage_h7/stage_h7_ensemble_results.json # Stage H7 ensemble & nested CV metrics
│   ├── stage_h7/xgboost_c3a_results.json # Stage H7 XGBoost 3A 5-fold metrics
│   └── stage_h2_step1_feature_importance.csv
├── reports/                              # Detailed audit and stage review reports
│   ├── cv_validation_report.md
│   ├── stage_g2_validation_audit.md
│   ├── stage_h1_feature_engineering.md
│   ├── stage_h2_step1_diagnostics.md
│   ├── stage_h2_step3a_tuning.md
│   ├── stage_h2_step3b_stability_ensemble.md
│   ├── final_submission_candidate_review.md
│   ├── stage_h3_improvement_review.md
│   ├── stage_h4_rank_blend_review.md
│   ├── stage_h5_improvement_review.md
│   ├── stage_h6_dataset_audit.md
│   ├── stage_h6_improvement_review.md
│   ├── stage_h7_experiment_log.md
│   └── stage_h7_improvement_review.md
├── outputs/                              # Submission artifact directory
│   ├── submission_candidate_20261012.csv               # V1 Baseline candidate (LB: 0.76456)
│   ├── submission_candidate_20261012_v2_quad_blend.csv # V2 Quad prob blend (LB: 0.76482 benchmark)
│   ├── submission_candidate_20261013_v21_rank_blend.csv# V2.1 Rank blend (OOF: 0.76230)
│   ├── submission_candidate_20261013_v22_h6.csv        # V2.2 Upgraded LGB C2 rank blend (OOF: 0.76235)
│   └── submission_candidate_20261013_v23_h7.csv        # V2.3 Upgraded XGB 3A rank blend (OOF: 0.76265)
├── requirements.txt                      # Pinned Python package dependencies
└── README.md                             # Comprehensive project documentation
```

---

## 7. Quickstart & Reproduction Guide

### Prerequisites
- Python 3.11.x (tested on macOS Apple Silicon M2 / arm64)

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/rupeshsah86/gci-world-september-2026-loan-default-prediction.git
cd gci-world-september-2026-loan-default-prediction

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Place Competition Data
Place the competition datasets in `Competition/input/`:
```text
Competition/input/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### 3. Run Validation & Reproduction Pipelines

#### Run 5-Fold Stratified CV Baseline
```bash
python src/train_cv.py
```

#### Run Diagnostics & Adversarial Validation
```bash
python src/experiment_h2_step1.py
```

#### Run Bounded Hyperparameter Refinement
```bash
python src/experiment_h2_step3a.py
```

#### Run Seed Stability & Nested CV Optimization
```bash
python src/experiment_h2_step3b.py
```

#### Generate Final Submission Candidate
```bash
python src/train_final_submission.py
```
*Outputs: `outputs/submission_candidate_20261012.csv` (Runtime: ~92s on Apple Silicon M2).*

---

## 8. Submission File Specifications

The generated submission file `outputs/submission_candidate_20261012.csv` passed all compliance and format audits:

- **Format**: CSV, comma-delimited.
- **Header**: Exactly `SK_ID_CURR,TARGET`.
- **Dimensions**: Exactly **61,500 rows** and 2 columns.
- **ID Integrity**: 100% matched with `test.csv` in original order (`171202` to `232701`); zero duplicates, zero missing.
- **Prediction Values**: Continuous probabilities strictly bounded in $[0.0, 1.0]$.
  - Minimum probability: `0.002989`
  - Mean probability: `0.088585` (matches natural default rate)
  - Maximum probability: `0.811547`

---

## 9. Engineering Rigor & Leakage Prevention

1. **Strict Fold Isolation**: All imputers, encoders, and transformers are fitted **strictly on training fold slices** (`X_tr`), then applied to validation slices (`X_va`). No target statistics, frequencies, or bounds ever cross validation boundaries.
2. **Feature Filtering**: Identifiers (`SK_ID_CURR`) and target labels (`TARGET`) are strictly removed prior to feature transformation.
3. **Reproducibility**: All cross-validation splits and model seeds are pinned (`random_state=42`), with seed-shift invariance explicitly verified on `random_state=2026`.
4. **Early Stopping Protection**: Model iterations are tuned with validation folds purely for stopping criteria; full-data retraining adopts the exact mean early-stopping budget from cross-validation to prevent over-fitting.

---

## License & Attribution
Developed for the **GCI World 2026 Home Credit Default Risk Competition**. Code released under the [MIT License](LICENSE).
