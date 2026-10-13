# Stage H7: Advanced ROC-AUC Improvement Experiment Log

**Project**: GCI World September 2026 — Home Credit Default Risk Prediction  
**Timestamp**: 2026-10-13  
**Target Metric**: ROC-AUC on 5-Fold Stratified Cross-Validation (`random_state=42`)  
**Objective**: Pursue evidence-based ROC-AUC improvement toward 0.77000  

---

## 1. Verified Baseline Status & Benchmarks

### Public Leaderboard Milestones
- **Candidate V1** (`outputs/submission_candidate_20261012.csv`): Public LB **0.76456**
- **Candidate V2** (`outputs/submission_candidate_20261012_v2_quad_blend.csv`): Public LB **0.76482**
- **Latest Displayed Public Leaderboard**: **0.76506** (Reported by user after submitting candidate V2.1 or V2.2)

### 5-Fold Cross-Validation Baselines (Replicated Locally)
| Model / Pipeline | Feature Set | Hyperparameters / Setup | 5-Fold OOF ROC-AUC | Mean +/- Std |
| :--- | :---: | :--- | :---: | :---: |
| **CatBoost Config 4** | 62 features | `depth=6, l2=10.0, rs=1.5, lr=0.04` | **0.76080** | 0.76084 +/- 0.00265 |
| **CatBoost Config 3** | 62 features | `depth=6, l2=5.0, rs=2.0, lr=0.04` | **0.76072** | 0.76076 +/- 0.00263 |
| **XGBoost Config 3** | 46 features | `depth=6, mcw=10, colsample=0.60, a=2.0, l=10.0, lr=0.03` | **0.76015** | 0.76021 +/- 0.00285 |
| **LightGBM C1 (Original)** | 46 features | `num_leaves=24, depth=7, mcw=100, lr=0.03, l=5.0` | **0.75900** | 0.75905 +/- 0.00323 |
| **LightGBM C2 (Stage H6 Upgraded)**| 46 features | `num_leaves=20, depth=6, mcw=120, lr=0.025, a=2.0, l=10.0` | **0.75970** | 0.75973 +/- 0.00312 |

### Ensemble Baseline Scores
| Ensemble Combination | Weights / Formula | OOF ROC-AUC | Lift vs V2 Baseline |
| :--- | :--- | :---: | :---: |
| **Candidate V2 Prob Blend** | 27% CB4 + 23% CB3 + 30% XGB3 + 20% LGB_C1 | **0.76224** | Benchmark (LB: 0.76482) |
| **Candidate V2.1 Rank Blend** | 27% r_CB4 + 23% r_CB3 + 30% r_XGB3 + 20% r_LGB_C1 | **0.76230** | +0.00006 |
| **Candidate V2.2 Prob Blend** | 27% CB4 + 23% CB3 + 30% XGB3 + 20% LGB_C2 | **0.76228** | +0.00004 |
| **Candidate V2.2 Rank Blend** | 27% r_CB4 + 23% r_CB3 + 30% r_XGB3 + 20% r_LGB_C2 | **0.76235** | +0.00011 |

---

## 2. Experimental Hypotheses for Stage H7

- **Hypothesis H7-A (XGBoost Optimization)**: Re-tune XGBoost with tighter regularization (`reg_lambda=15.0`, `min_child_weight=15`), lower learning rate (0.025), and test depth-5 vs depth-6 to match the regularization gains discovered in LightGBM C2.
- **Hypothesis H7-B (CatBoost Refinement)**: High L2 regularization (`l2_leaf_reg=15.0`) with lower learning rate (0.035) showed +0.00048 lift on Fold 1 in Stage H6. Evaluate across all 5 folds.
- **Hypothesis H7-C (Feature Candidate Pilots)**: Test non-collinear interaction ratios (such as relative credit strength ratios and employment stability indicators) on LightGBM C2 across 5-fold CV.
- **Hypothesis H7-D (Ensemble Weight Optimization & Rank Blending)**: Test Dirichlet / SLSQP optimization under Nested 5-Fold CV to evaluate whether an updated weight allocation provides genuine generalization gains.

---

## 3. Experiment Execution & Measured Results

### Experiment A: XGBoost Hyperparameter Refinement
- **Fold 1 Screening**:
  - `Config_3_Baseline`: 0.75713 (437 iters)
  - `Config_3A_HighL2_MCW15_LR025` (`depth=6, mcw=15, colsample=0.65, subsample=0.85, a=3.0, l=15.0, lr=0.025`): **0.75793** (+0.00080 gain)
  - `Config_3B_Depth5_MCW15_LR03`: 0.75783 (+0.00070 gain)
  - `Config_3C_Col055_Sub08_L2_12`: 0.75746 (+0.00033 gain)
- **Full 5-Fold Cross-Validation of XGBoost Config 3A**:
  - Fold 1: **0.75812** (Best Iter: 663)
  - Fold 2: **0.76142** (Best Iter: 615)
  - Fold 3: **0.76673** (Best Iter: 787; +0.00228 massive lift)
  - Fold 4: **0.75636** (Best Iter: 497; +0.00018 lift)
  - Fold 5: **0.76194** (Best Iter: 939; +0.00126 lift)
  - **Overall OOF ROC-AUC**: **0.76088** (Fold Mean: 0.76091 +/- 0.00357)
  - **Gain vs Baseline XGBoost Config 3 (0.76015)**: **+0.00073** (Now highest individual model in repo!)
  - Status: **CONFIRMED & ADOPTED**

### Experiment B: CatBoost Hyperparameter Refinement
- **Fold 1 Screening**:
  - `Config_4_Baseline`: 0.75911
  - `Config_5_L2_15_LR035`: 0.75880 (-0.00031)
  - `Config_6_L2_20_RS2_LR030` (`depth=6, l2=20.0, rs=2.0, lr=0.030, iters=1800`): **0.76001** (+0.00090 gain on Fold 1)
  - `Config_7_Depth7_L2_20_LR03`: 0.75957 (+0.00046 gain on Fold 1)
- **Full 5-Fold Cross-Validation of CatBoost Config 6**:
  - Fold 1: 0.75938 (Best Iter: 1793)
  - Fold 2: 0.76143 (Best Iter: 1889)
  - Fold 3: 0.76490 (Best Iter: 1564)
  - Fold 4: 0.75699 (Best Iter: 1569)
  - Fold 5: 0.76086 (Best Iter: 1145)
  - **Overall OOF ROC-AUC**: **0.76067** (Fold Mean: 0.76071 +/- 0.00260)
  - **Comparison vs Baseline CatBoost Config 4 (0.76080)**: -0.00013. Very strong, but does not beat Config 4 overall.
  - Status: **RETAIN Config 4 & Config 3 in Ensemble**

### Experiment C: Feature Engineering Pilots on LightGBM C2
- **Fold 1 Screening**:
  - `Baseline_46`: 0.75662
  - `Pilot_1_Ext_Interactions_51`: 0.75509 (-0.00153, collinear dilution) -> **REJECTED**
  - `Pilot_2_Financial_Ratios_50`: 0.75678 (+0.00016 on Fold 1) -> Passed to 5-fold CV
  - `Pilot_3_All_Extended_62`: 0.75611 (-0.00051) -> **REJECTED**
  - `Pilot_4_Novel_Interactions_50`: 0.75590 (-0.00072) -> **REJECTED**
- **Full 5-Fold CV of Pilot 2 (Financial Ratios, 50 Feats)**:
  - Overall OOF ROC-AUC: **0.75955** (Fold Mean: 0.75958 +/- 0.00274)
  - Delta vs Baseline LightGBM C2 (0.75970): -0.00015
  - Status: **REJECTED** (Fold 1 gain did not generalize across all folds; preserve clean 46-feature set)

### Experiment D: Ensemble Optimization & Nested Cross-Validation
- **Ensemble Evaluations**:
  - `Candidate V2 Prob Blend` (CB4 27%, CB3 23%, XGB3 30%, LGB_C1 20%): **0.76224** (LB: 0.76482)
  - `Candidate V2.1 Rank Blend`: **0.76230**
  - `Candidate V2.2 Prob Blend` (with LGB C2): **0.76228**
  - `Candidate V2.2 Rank Blend` (with LGB C2): **0.76235**
  - `Stage H7 Prob Blend` (CB4 27%, CB3 23%, **XGB 3A 30%**, LGB C2 20%): **0.76258** (+0.00034 vs V2)
  - `Stage H7 Rank Blend` (Candidate V2.3): **0.76265** (+0.00030 vs V2.2; +0.00035 vs V2.1)
  - `Nested 5-Fold Cross-Validation (Unbiased)`: **0.76263**
- Status: **CONFIRMED BREAKTHROUGH**

---

## 4. Submission Candidate V2.3 Generation & Audit

- **File**: `outputs/submission_candidate_20261013_v23_h7.csv`
- **Rows**: Exactly 61,500 rows, 2 columns (`SK_ID_CURR`, `TARGET`).
- **ID Integrity**: 100% matched against `Competition/input/test.csv` and `sample_submission.csv` (range 171,202 to 232,701).
- **Values**: Continuous probabilities strictly bounded in $[0.000049, 0.999985]$ (mean 0.500008). Zero NaNs, zero Infs.
- **Spearman Correlation vs V2**: 0.99892
- **Spearman Correlation vs V2.2**: 0.99929
- **OOF Score**: **0.76265** (Rank Blend) / **0.76258** (Prob Blend) / **0.76263** (Nested CV)

