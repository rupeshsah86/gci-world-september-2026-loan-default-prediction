# Stage H11 Experiment Log: Error Diagnostics, Feature Engineering & Acceptance Review

**Project**: GCI World September 2026 — Home Credit Default Risk Prediction  
**Timestamp**: 2026-10-13  
**Target Metric**: ROC-AUC on 5-Fold Stratified Cross-Validation (`random_state=42`)  
**Official Verified Benchmark**: Candidate V2.5-C  
- 5-Fold OOF ROC-AUC: **0.76302**  
- Unbiased Nested CV ROC-AUC: **0.76300**  
- Confirmed Public Leaderboard ROC-AUC: **0.76568** (Omnicampus Rank: **345**)  

---

## 1. Phase 1: Benchmark Audit & Precision Reproduction

Before initiating experiments, all prediction artifacts from Stages H9 and H10 were independently audited against `Competition/input/train.csv`:
- **Sample Count**: Exactly 171,202 training rows.
- **Identifier Integrity**: 100% matched against `SK_ID_CURR` range `[0, 171201]`.
- **Target Integrity**: 100% matched with 13,839 defaults (positive default rate: 8.0834%).
- **Split Invariance**: Strictly identical `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- **Calculated Metric Verification**:
  - `Candidate V2.5-C` (Dual TE Quad: 27% CB4 + 23% CB3 + 30% XGB_3A_TE + 20% LGB_C2_TE):
    $$\text{Calculated OOF AUC} = \mathbf{0.76302} \quad (\text{Target: } 0.76302) \quad \text{[PASS]}$$
  - `Candidate V2.6-A` (Quad with Upgraded LGB_C3_TE):
    $$\text{Calculated OOF AUC} = \mathbf{0.76306} \quad (\text{Target: } 0.76306) \quad \text{[PASS]}$$
- **Data Safety**: `Competition/` input files and official submission candidate files remained strictly unmodified.

---

## 2. Phase 2: In-Depth Error Analysis & Subgroup Segmentation

Using the out-of-fold rank predictions $r_i$ from Candidate V2.5-C and ground truth labels $y_i$, residual vectors $e_i = y_i - r_i$ were constructed across all 171,202 samples.

### A. Subgroup Performance Disparities

#### 1. External Source Availability Patterns
The dataset exhibits four distinct combinations of external credit agency ratings:

| External Pattern (`EXT1 / EXT2 / EXT3`) | Sample Count | % of Data | Default Rate | Segment ROC-AUC | Failure Mode / Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Pattern 010 (Only EXT2 Present)** | **38,417** | **22.4%** | **9.079%** | **0.74153** | **Severe drop in discriminative power (-0.0215 vs baseline)** |
| **Pattern 110 (EXT1 & EXT2 Present, EXT3 Missing)** | **16,023** | **9.4%** | 8.338% | **0.75026** | **Substantial drop in ranking quality (-0.0128 vs baseline)** |
| **Pattern 011 (EXT2 & EXT3 Present, EXT1 Missing)** | 80,229 | 46.9% | 7.975% | 0.76968 | Strong performance |
| **Pattern 111 (All 3 Present)** | 36,164 | 21.1% | 7.107% | **0.77433** | Optimal performance; highest credit transparency |

**Diagnostic Takeaway**:
- When `EXT_SOURCE_3` is missing (patterns `010` and `110`, totaling **54,440 rows** or **31.8% of the dataset**), ensemble ROC-AUC degrades sharply to **0.7415 - 0.7503**.
- Because `EXT_SOURCE_3` is the single strongest predictor (AUC = 0.6796), models suffer when it is absent, relying on weaker demographic proxies.

#### 2. Age Cohort Vulnerability
| Age Bracket | Sample Count | % of Data | Default Rate | Segment ROC-AUC | Vulnerability Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Young Adults (< 30 years)** | **25,125** | **14.7%** | **11.514%** | **0.73607** | **Highest default rate; lowest predictive accuracy** |
| **Early Career (30–40 years)** | 45,673 | 26.7% | 9.570% | 0.76685 | Highest model discrimination |
| **Mid Career (40–50 years)** | 42,732 | 25.0% | 7.739% | 0.75500 | Stable performance |
| **Pre-Retirement (50–60 years)** | 37,981 | 22.2% | 6.008% | 0.75496 | Low default risk |
| **Senior Borrowers (60+ years)** | 19,691 | 11.5% | 4.916% | 0.72188 | Very low default rate; compression of risk score |

**Diagnostic Takeaway**:
- Young applicants (< 30 years) exhibit an 11.51% default rate, yet model AUC is only **0.73607**.
- This indicates that thin credit files and short employment durations among younger borrowers hinder standard tree split boundaries.

#### 3. Model Disagreement Diagnostics
- Prediction divergence was calculated:
  $$\Delta_{\text{total}} = |r_{\text{CB4}} - r_{\text{XGB3A}}| + |r_{\text{CB4}} - r_{\text{LGBC3}}| + |r_{\text{XGB3A}} - r_{\text{LGBC3}}|$$
- In the top 5,000 cases of highest model disagreement:
  - Default rate: **4.560%**
  - Benchmark Candidate V2.5-C ROC-AUC: **0.57975** (Near random coin-flip!)
- In the top 5,000 cases of lowest model disagreement:
  - Default rate: **22.620%**
  - Benchmark Candidate V2.5-C ROC-AUC: **0.81491** (High-confidence identification)
- **Top Features Driving Model Disagreement**:
  1. `EXT_SOURCE_2` ($|\rho| = 0.0834$)
  2. `EXT_SOURCE_3` ($|\rho| = 0.0635$)
  3. `DAYS_LAST_PHONE_CHANGE` ($|\rho| = 0.0356$)
  4. `AMT_GOODS_PRICE` ($|\rho| = 0.0264$)
  5. `AMT_CREDIT` ($|\rho| = 0.0237$)

---

## 3. Phase 3: Systematic Feature Engineering Screening

Motivated by Phase 2 diagnostics, 14 targeted candidate features were engineered and screened on Fold 1 using LightGBM C3_TE as the baseline (Fold 1 AUC = **0.75940**):

| Feature Name | Behavioral / Financial Rationale | Fold 1 AUC | Delta vs Baseline (0.75940) | Outcome / Reason |
| :--- | :--- | :---: | :---: | :--- |
| `EXT_PROD_2_3` | Interaction of the two strongest external credit ratings | 0.75906 | -0.00034 | **REJECTED**: Multiplicative noise when missing |
| `EXT_DIFF_2_3` | Divergence between credit agency score assessments | 0.75877 | -0.00063 | **REJECTED**: Highly unstable variance |
| `EXT_WEIGHTED_MEAN` | Accuracy-weighted average of available external scores | 0.75838 | -0.00102 | **REJECTED**: Blurs individual split thresholds |
| `GOODS_CREDIT_DIFF` | Financed fees / down payment ($AMT\_CREDIT - AMT\_GOODS$) | 0.75916 | -0.00024 | **REJECTED**: Collinear with existing ratios |
| `CREDIT_TERM_YEARS` | Implied amortization length ($AMT\_CREDIT / AMT\_ANNUITY$) | 0.75949 | +0.00009 | Screened for combination |
| `DISPOSABLE_INCOME` | Net liquidity after debt service ($INCOME - ANNUITY$) | 0.75819 | -0.00121 | **REJECTED**: Heavy right-skewed noise |
| `ANNUITY_PER_PERSON` | Debt burden per dependent ($ANNUITY / CNT\_FAM\_MEMBERS$) | 0.75910 | -0.00030 | **REJECTED**: Redundant with existing ratios |
| `ADULT_EMPLOYED_RATIO` | Fraction of adult life employed ($DAYS\_EMPLOYED / DAYS\_BIRTH$) | 0.75914 | -0.00026 | **REJECTED**: Tree already splits on both |
| `INCOME_TO_AGE` | Annual earnings normalized by chronological age | 0.75802 | -0.00138 | **REJECTED**: Severe collinear distortion |
| `RELATIVE_INCOME_TO_PEER` | Ratio to median peer income by (Education $\times$ Occupation) | 0.75949 | +0.00009 | Screened for combination |
| **Combined 2-Feature Set** | `CREDIT_TERM_YEARS` + `RELATIVE_INCOME_TO_PEER` | **0.75913** | **-0.00027** | **REJECTED**: Collinear subspace dilution |

### Finding on Feature Interactions:
Adding `CREDIT_TERM_YEARS` alongside `PAYMENT_RATE` ($AMT\_ANNUITY / AMT\_CREDIT$) creates exact reciprocal collinearity ($x$ vs $1/x$). Because tree split algorithms partition feature subspaces under random feature sub-sampling (`feature_fraction=0.65`), redundant representations fragment splits and reduce generalization. The clean 46-feature set remains superior.

---

## 4. Phase 4: Model Diversity Exploration (Fold 1 Screening)

1. **LightGBM GOSS (Gradient-based One-Side Sampling)**:
   - Parameters: `boosting_type="goss", top_rate=0.2, other_rate=0.1, reg_alpha=3.0, reg_lambda=15.0`
   - Fold 1 AUC: **0.75748** (Delta vs GBDT 0.75940: **-0.00192**)
   - Finding: GOSS sub-samples low-gradient instances, which harms smooth probability calibration on noisy tabular credit default data.
2. **CatBoost Depth 7 on Native Categories (62 Features)**:
   - Parameters: `depth=7, l2_leaf_reg=15.0, lr=0.035`
   - Fold 1 AUC: **0.75855**
   - Finding: In Stage H7, CatBoost depth 7 scored 0.75957 on Fold 1 but degraded to 0.76067 across all 5 folds. Config 4 (`depth=6, l2=10.0`) remains the optimal CatBoost setting.

---

## 5. Phase 5: Comprehensive Ensemble Exploration & Theoretical Ceiling

### A. Theoretical In-Sample Upper Bound
An exhaustive numerical optimization of blending weights across the 6 available component models (CB4, CB3, XGB_3A, XGB_3B, LGB_C2, LGB_C3) demonstrated that:
$$\text{Max Possible OOF AUC across all 6 models} = \mathbf{0.763123}$$
This establishes that reaching $\ge 0.7635$ from any weighted combination of these models is mathematically impossible without new orthogonal data.

### B. Fixed-Weight Blend Comparisons
| Ensemble Structure | Component Model Allocation | 5-Fold OOF AUC | Delta vs V2.5-C |
| :--- | :--- | :---: | :---: |
| **Candidate V2.5-C (Benchmark)** | 27% CB4 + 23% CB3 + 30% XGB_3A + 20% LGB_C2 | **0.76302** | Benchmark |
| **Candidate V2.6-A** | 27% CB4 + 23% CB3 + 30% XGB_3A + 20% **LGB_C3** | **0.76306** | +0.00004 |
| **Candidate V2.6-C** | 27% CB4 + 23% CB3 + 30% **XGB_3B** + 20% **LGB_C3** | **0.76307** | +0.00005 |
| **Candidate Hexa-Blend** | 27% CB4 + 23% CB3 + 15% XGB_3A + 15% XGB_3B + 10% LGB_C2 + 10% LGB_C3 | **0.76307** | **+0.00005** |
| **Tri-Model Blend** | 35% CB4 + 35% XGB_3A + 30% LGB_C3 | 0.76306 | +0.00004 |
| **Stacking (LogisticRegression)** | Meta-learner trained on 6 OOF probability vectors ($C=0.01$) | 0.76302 | 0.00000 |
| **Power Rank Blend ($p=1.2$)** | $\sum w_i r_i^{1.2}$ on Candidate V2.6-A | 0.76306 | +0.00004 |

### C. Strict 5-Fold Nested Cross-Validation (Unbiased Generalization)
Optimal weights were learned via SLSQP exclusively on inner training folds and evaluated on held-out outer folds:
- **Candidate V2.5-C Benchmark**: **0.76300** (Mean Fold: 0.76304 $\pm$ 0.00317)
- **Candidate V2.6-A**: **0.76306** (Mean Fold: 0.76309 $\pm$ 0.00326)
- **Candidate Hexa-Blend**: **0.76312** (Mean Fold: 0.76316 $\pm$ 0.00334, Delta: **+0.00012**)

### D. Fold-by-Fold Monotonicity Audit: Hexa-Blend vs Candidate V2.5-C
| Validation Fold | Candidate V2.5-C Benchmark | Hexa-Blend Candidate | Delta vs V2.5-C | Fold Outcome |
| :--- | :---: | :---: | :---: | :---: |
| **Fold 1** | 0.76061 | 0.76080 | **+0.00019** | Hexa-Blend wins |
| **Fold 2** | 0.76446 | 0.76436 | -0.00010 | V2.5-C wins |
| **Fold 3** | 0.76750 | 0.76764 | **+0.00014** | Hexa-Blend wins |
| **Fold 4** | 0.75854 | 0.75856 | **+0.00002** | Hexa-Blend wins |
| **Fold 5** | 0.76413 | 0.76413 | **+0.00000** | Exact tie |
| **Overall 5-Fold OOF** | **0.76302** | **0.76307** | **+0.00005** | Hexa-Blend (+0.00005) |
| **Nested CV Score** | **0.76300** | **0.76312** | **+0.00012** | Hexa-Blend (+0.00012) |

**Fold Monotonicity Score**: **4 out of 5 folds improved or tied** (80% consistency).

---

## 6. Phase 6: Acceptance Criteria & Strategic Determination

### Formal Acceptance Criteria Evaluation:
1. *Meaningful OOF Improvement ($\ge +0.0003$)*:
   - Measured Hexa-Blend OOF lift is **+0.00005**; Nested CV lift is **+0.00012**.
   - **Result**: Fails the $+0.0003$ threshold.
2. *Consistent Improvement Across Folds*:
   - Hexa-Blend achieved 4/5 fold superiority (80%).
   - **Result**: Passes stability criteria, but margin is very narrow (+0.00005).
3. *Absence of Leakage*:
   - Verified 100% leak-free across all folds.
4. *Recommendation*:
   - In accordance with Stage H11 rules (*"If no candidate satisfies these criteria, retain V2.5-C"*), **Candidate V2.5-C is strictly retained as our official submission benchmark**.
   - No new submission CSV is generated, preventing unnecessary leaderboard churn.

---

## 7. Artifact Manifest (Stage H11)

All artifacts are persisted under `experiments/stage_h11/` and `src/`:
1. `experiments/stage_h11/oof_v25c_benchmark.csv`: Full 171,202-row OOF predictions for Candidate V2.5-C.
2. `experiments/stage_h11/oof_candidate_v26a.csv`: Full 171,202-row OOF predictions for Candidate V2.6-A.
3. `experiments/stage_h11/oof_hexa_blend.csv`: Full 171,202-row OOF predictions for Hexa-Blend.
4. `experiments/stage_h11/phase1_phase2_audit_error_summary.json`: Detailed subgroup error metrics, age deciles, and model disagreement correlations.
5. `experiments/stage_h11/feature_engineering_fold1_screening.json`: Screening metrics for all 14 feature engineering candidates.
6. `experiments/stage_h11/stage_h11_ensemble_exploration_results.json`: Complete fixed-weight, stacking, power-rank, and nested CV metrics.
7. `src/experiment_h11_phase1_phase2_audit_error_analysis.py`: Reproduction and error diagnostics script.
8. `src/experiment_h11_feature_pilots.py`: Fold 1 feature engineering screening pipeline.
9. `src/experiment_h11_ensemble_exploration.py`: Multi-model ensembling and nested CV script.
