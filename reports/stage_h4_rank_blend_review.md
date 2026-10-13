# Stage H4: Rank-Normalized Quad-Model Blend Pilot Review Report

## 1. Executive Summary

- **Pilot Candidate Artifact**: [`outputs/submission_candidate_20261013_v21_rank_blend.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/outputs/submission_candidate_20261013_v21_rank_blend.csv)
- **Current Best Baseline**: [`outputs/submission_candidate_20261012_v2_quad_blend.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/outputs/submission_candidate_20261012_v2_quad_blend.csv) (Public Leaderboard: **0.76482**, OOF ROC-AUC: **0.76224**)
- **Initial Submission Reference**: `outputs/submission_candidate_20261012.csv` (Public Leaderboard: **0.76456**, OOF ROC-AUC: **0.76208**)
- **Generated At**: 2026-10-13 12:46:44 UTC
- **Hypothesis Evaluation**: Percentile-rank normalization was evaluated on the exact same 171,202 out-of-fold (OOF) rows, labels, and folds.
  - Reported Rank-Normalized OOF ROC-AUC: **0.76230** (Reproduced exactly: `0.76230357`)
  - Baseline Probability Blend OOF ROC-AUC: **0.76224** (Reproduced exactly: `0.76224232`)
  - OOF Delta: **+0.00006** (Gain observed across 4 out of 5 folds)
- **Component Models & Weights**:
  - **CatBoost Config 4** (62 features): **27%**
  - **CatBoost Config 3** (62 features): **23%**
  - **XGBoost Config 3** (46 features): **30%**
  - **Tuned LightGBM** (46 features): **20%**
- **Governance & Safety**:
  - Existing candidate files (`submission_candidate_20261012.csv`, `submission_candidate_20261012_v2_quad_blend.csv`) remain completely untouched.
  - Raw competition data in `Competition/input/` remains untouched.
  - No submissions uploaded to Omnicampus, and no changes pushed to GitHub.

---

## 2. Artifact Inspection & Data Integrity Audit

All four component out-of-fold prediction artifacts were inspected, verified, and audited prior to computing blend metrics:

| Model Component | OOF Artifact File Path | Record Count | Prediction Column | Min Pred | Max Pred | Mean Pred | Ties Count |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost Config 4** | [`experiments/stage_h2_step3a_oof_CB_Config_4_High_L2_Regularization.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/experiments/stage_h2_step3a_oof_CB_Config_4_High_L2_Regularization.csv) | 171,202 | `oof_pred` | 0.001786 | 0.849718 | 0.079917 | 0 |
| **CatBoost Config 3** | [`experiments/stage_h2_step3a_oof_CB_Config_3_RandomStrength_2_LR04.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/experiments/stage_h2_step3a_oof_CB_Config_3_RandomStrength_2_LR04.csv) | 171,202 | `oof_pred` | 0.001583 | 0.922818 | 0.080009 | 0 |
| **XGBoost Config 3**  | [`experiments/stage_h2_step3a_oof_XGB_Config_3_Depth6_Colsample06_Reg10.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/experiments/stage_h2_step3a_oof_XGB_Config_3_Depth6_Colsample06_Reg10.csv) | 171,202 | `oof_pred` | 0.002679 | 0.729325 | 0.080220 | 1,035 |
| **Tuned LightGBM**    | [`experiments/oof_tuned_lightgbm.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/experiments/oof_tuned_lightgbm.csv) | 171,202 | `oof_best_tuned_lgb` | 0.002069 | 0.707789 | 0.080266 | 0 |

### Integrity & Leakage Verification:
1. **Row Count Alignment**: Exactly 171,202 rows across all four artifacts, matching `Competition/input/train.csv` row-for-row.
2. **ID Correspondence & Ordering**: 100% sequential match on `SK_ID_CURR` with zero missing IDs and zero duplicate IDs (`nunique = 171,202`).
3. **Target Integrity**: Exact match with `TARGET` labels across all rows (157,511 negative instances, 13,691 positive instances, default rate 7.9970%).
4. **Finite Values**: Zero `NaN` and zero `inf` values across all four prediction sets; all values strictly bounded within $[0.0, 1.0]$.
5. **Leakage Guard**: All OOF predictions were generated strictly on held-out validation folds using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` with isolated preprocessing fit only on training folds.

---

## 3. Rank Normalization Methodology & Formulation

### Rationale
In gradient-boosted decision trees, different objective approximations, leaf structures, and regularizers lead to heterogeneous probability calibrations:
- CatBoost produces predictions with higher tail dispersion (max probability 0.85–0.92).
- XGBoost and LightGBM with strong L2 penalties produce compressed probability tails (max probability 0.71–0.73).

When taking a weighted arithmetic average of raw probabilities, models with wider dispersion exert disproportionate leverage on high-risk rankings. Percentile-rank normalization maps each model's predictions onto a uniform distribution $U(0, 1]$ while preserving 100% of the internal monotonic ranking of each model.

### Mathematical Specification
For model $m \in \{1, 2, 3, 4\}$ with prediction vector $\mathbf{p}^{(m)} \in \mathbb{R}^N$:
$$\text{rank\_pct}(p_i^{(m)}) = \frac{\text{rank}_{\text{average}}(p_i^{(m)})}{N}$$

- **Tie Resolution**: Ties are resolved using the **average rank** method (`scipy.stats.rankdata(..., method='average')` or `pandas.Series.rank(pct=True, method='average')`). In the case of XGBoost Config 3 (1,035 tied values across shallow discrete tree leaves), tied samples receive the exact mean of their potential rank positions, ensuring unbiased symmetric ranking.
- **Ensemble Combination**:
$$\hat{y}_{\text{rank\_blend}} = 0.27 \cdot \text{rank\_pct}(\mathbf{p}^{(\text{CB4})}) + 0.23 \cdot \text{rank\_pct}(\mathbf{p}^{(\text{CB3})}) + 0.30 \cdot \text{rank\_pct}(\mathbf{p}^{(\text{XGB3})}) + 0.20 \cdot \text{rank\_pct}(\mathbf{p}^{(\text{LGB})})$$

Because $\sum_m w_m = 1.0$, the resulting score is strictly bounded in $(0.0, 1.0]$ and represents an aligned percentile consensus.

---

## 4. Empirical Evaluation: Probability Blend vs Rank-Normalized Blend

### Overall and Fold-by-Fold Performance Matrix
All evaluations performed on the exact same 5 stratified folds (`seed=42`, 171,202 samples):

| Model / Ensemble Strategy | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean Fold $\pm$ Std | Overall OOF ROC-AUC | Delta vs Single Best |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost Config 4** (27%) | 0.75944 | 0.76152 | 0.76487 | 0.75684 | 0.76153 | 0.76084 $\pm$ 0.00265 | **0.76080** | Reference |
| **CatBoost Config 3** (23%) | 0.75815 | 0.76100 | 0.76503 | 0.75785 | 0.76175 | 0.76076 $\pm$ 0.00263 | **0.76072** | -0.00008 |
| **XGBoost Config 3** (30%)  | 0.75815 | 0.76160 | 0.76445 | 0.75618 | 0.76068 | 0.76021 $\pm$ 0.00285 | **0.76015** | -0.00065 |
| **Tuned LightGBM** (20%)    | 0.75530 | 0.76093 | 0.76386 | 0.75571 | 0.75944 | 0.75905 $\pm$ 0.00323 | **0.75900** | -0.00180 |
| **Probability Blend (Candidate V2)** | 0.75983 | 0.76345 | 0.76663 | 0.75863 | 0.76287 | 0.76228 $\pm$ 0.00283 | **0.76224** | +0.00144 |
| **Rank-Normalized Blend (Candidate V2.1)** | **0.75997** | **0.76352** | **0.76665** | 0.75859 | **0.76297** | **0.76234 $\pm$ 0.00283** | **0.76230** | **+0.00150** |

### Fold-by-Fold Consistency Analysis:
- **Fold 1**: $0.75983 \rightarrow \mathbf{0.75997}$ (**+0.00014**)
- **Fold 2**: $0.76345 \rightarrow \mathbf{0.76352}$ (**+0.00007**)
- **Fold 3**: $0.76663 \rightarrow \mathbf{0.76665}$ (**+0.00002**)
- **Fold 4**: $0.75863 \rightarrow 0.75859$ (**-0.00004**)
- **Fold 5**: $0.76287 \rightarrow \mathbf{0.76297}$ (**+0.00010**)
- **Net Consistency**: **4 out of 5 folds improved**, demonstrating that the +0.00006 gain is not concentrated in an outlier split.

---

## 5. Model Correlation & Rank Alignment Analysis

### Spearman Rank Correlation Matrix (OOF Predictions)
| Model | CatBoost C4 | CatBoost C3 | XGBoost C3 | Tuned LGB | Probability Blend | Rank Blend |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost C4** | 1.00000 | 0.99204 | 0.96907 | 0.96369 | 0.99135 | 0.99144 |
| **CatBoost C3** | 0.99204 | 1.00000 | 0.96886 | 0.96365 | 0.99095 | 0.99104 |
| **XGBoost C3**  | 0.96907 | 0.96886 | 1.00000 | 0.98216 | 0.99067 | 0.99072 |
| **Tuned LGB**    | 0.96369 | 0.96365 | 0.98216 | 1.00000 | 0.98613 | 0.98623 |
| **Probability Blend** | 0.99135 | 0.99095 | 0.99067 | 0.98613 | 1.00000 | **0.99992** |
| **Rank Blend**        | 0.99144 | 0.99104 | 0.99072 | 0.98623 | **0.99992** | 1.00000 |

### Key Correlation Insights:
1. **0.99992 Monotonic Concordance**: The Spearman correlation between the probability blend and the rank-normalized blend is **0.99992**. This confirms that rank normalization preserves global relative order while making micro-adjustments to tail leverage.
2. **Architectural Complementarity**: CatBoost and tree models (XGBoost/LightGBM) maintain a healthy ~0.963–0.969 rank correlation, confirming strong architectural diversity.

---

## 6. Submission Candidate V2.1 Audit & Verification

The newly generated candidate file [`outputs/submission_candidate_20261013_v21_rank_blend.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/outputs/submission_candidate_20261013_v21_rank_blend.csv) was subjected to rigorous pre-submission checks:

| Audit Check | Required Specification | Candidate V2.1 Actual | Status |
| :--- | :--- | :--- | :---: |
| **File Existence** | Present in `outputs/` | `outputs/submission_candidate_20261013_v21_rank_blend.csv` (1.54 MB) | **PASSED** |
| **Row Count** | Exactly 61,500 rows (+ header) | 61,500 data rows | **PASSED** |
| **Column Structure** | Exactly `['SK_ID_CURR', 'TARGET']` | `['SK_ID_CURR', 'TARGET']` | **PASSED** |
| **ID Alignment** | 100% sequential match with `test.csv` | Zero mismatch, zero duplicates | **PASSED** |
| **Value Domain** | Finite probabilities $\in [0.0, 1.0]$ | Min: 0.000054, Max: 0.999996 | **PASSED** |
| **Distribution** | Non-degenerate spread | Mean: 0.500008, Median: 0.499246, Std: 0.286450 | **PASSED** |
| **Candidate V2 Preservation** | Must not overwrite Candidate V2 | `outputs/submission_candidate_20261012_v2_quad_blend.csv` preserved | **PASSED** |
| **Baseline V1 Preservation** | Must not overwrite Baseline V1 | `outputs/submission_candidate_20261012.csv` preserved | **PASSED** |

### Reproduction Sanity Check:
Before applying test-set rank normalization, the raw full-test predictions generated by the four models were blended with probability weights:
$$\max |\hat{y}_{\text{prob\_test}} - \hat{y}_{\text{Candidate V2}}| = 9.97 \times 10^{-17}$$
This confirms that the models and test pipelines used for Candidate V2.1 are machine-precision identical to Candidate V2.

---

## 7. Risk Analysis & Honest Hypothesis Assessment

> [!IMPORTANT]
> ### Honest Evaluation of the +0.00006 Gain
>
> 1. **Local Statistical Significance**:
>    - The cross-validation fold standard deviation across the 5 folds is $\pm 0.00283$.
>    - The observed OOF improvement of $+0.00006$ is roughly **$1/47\text{th}$ of one standard deviation**.
>    - Although the gain appears on 4 of 5 folds, its magnitude is within the regime of statistical sampling noise.
>
> 2. **Public Leaderboard Sensitivity**:
>    - The public test set comprises only ~18,450 records (assuming the standard 30% split of 61,500 rows).
>    - The expected sampling variance on an 18,450-row split is approximately $\pm 0.0020$ to $\pm 0.0030$.
>    - A +0.00006 OOF improvement cannot guarantee a positive shift on the public leaderboard.
>
> 3. **Rank-Transformed Probability Metric Invariance**:
>    - The competition metric is **ROC-AUC**, which depends solely on the rank ordering of predictions rather than probability calibration (Brier score or Log Loss).
>    - The shift from mean probability 0.0886 to mean percentile rank 0.5000 has **zero adverse impact on ROC-AUC**, while removing non-linear dispersion leverage between models.
>
> 4. **Candidate Positioning**:
>    - Candidate V2 (`submission_candidate_20261012_v2_quad_blend.csv`) remains the primary, established submission candidate (Public LB: **0.76482**).
>    - Candidate V2.1 (`submission_candidate_20261013_v21_rank_blend.csv`) is safely preserved as a dedicated pilot candidate for controlled evaluation.

---

## 8. Summary & Stop Checkpoint

- All ten user tasks have been executed and verified.
- Candidate file: `outputs/submission_candidate_20261013_v21_rank_blend.csv`.
- In strict adherence to competition governance, **no files have been uploaded to Omnicampus, pushed to GitHub, or modified in the raw dataset**.
- Ready for user review.
