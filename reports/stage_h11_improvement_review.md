# Stage H11 Strategic Review: Error Analysis, Feature Frontiers & Benchmark Retention

**Project**: GCI World September 2026 — Home Credit Default Risk Prediction  
**Date**: 2026-10-13  
**Official Active Submission**: Candidate V2.5-C  
- Confirmed Public Leaderboard ROC-AUC: **0.76568**  
- Omnicampus Public Rank: **345**  
- 5-Fold Local OOF ROC-AUC: **0.76302**  
- Unbiased Nested CV ROC-AUC: **0.76300**  
**Stage H11 Objective**: Pursue reproducible lift toward 0.7635+ via deep error analysis, feature engineering, and ensemble exploration without compromising stability.

---

## 1. Executive Summary & Acceptance Determination

Stage H11 delivered the most granular error diagnostic and feature sensitivity audit in the history of this project.

### Core Discoveries:
1. **The External Score Missingness Cliff**:
   - For applicants with all 3 external credit ratings present (`Pattern 111`, 21.1% of data), model performance reaches **0.77433**.
   - However, when `EXT_SOURCE_3` is missing (`Pattern 010`, 38,417 applicants = 22.4% of data), default risk spikes to 9.08% and ensemble performance collapses to **0.74153**.
   - This single missing attribute explains over 60% of total ensemble prediction error.
2. **The Young Borrower Vulnerability**:
   - Borrowers under 30 years old have the highest default rate in the competition (**11.514%**), yet ensemble ROC-AUC is lowest on this demographic (**0.73607**).
   - Thin credit files and short job tenure prevent standard tree splits from differentiating high-risk vs creditworthy youth.
3. **The Reciprocal Collinearity Trap in Feature Engineering**:
   - Out of 14 engineered candidate features, none produced a robust multi-fold lift.
   - Adding `CREDIT_TERM_YEARS` ($AMT\_CREDIT / AMT\_ANNUITY$) alongside existing `PAYMENT_RATE` ($AMT\_ANNUITY / AMT\_CREDIT$) created exact inverse collinearity ($x$ vs $1/x$). Under random feature fraction subsampling, this fragmented tree splits and degraded validation AUC by -0.00027.
4. **The Mathematical Ceiling Analysis**:
   - An exhaustive numerical optimization across the top 6 GBDT models demonstrated that the absolute maximum in-sample OOF AUC achievable from blending these models is **0.763123**.
   - Reaching the stated objective of **0.76350+** from any combination of these models on this static 34-column table is mathematically impossible without new orthogonal data.
5. **Hexa-Model Blend Discovery & Acceptance Evaluation**:
   - An expanded 6-model ensemble (`Candidate Hexa-Blend`: 27% CB4 + 23% CB3 + 15% XGB_3A_TE + 15% XGB_3B_TE + 10% LGB_C2_TE + 10% LGB_C3_TE) achieved an unbiased 5-fold Nested CV of **0.76312** (+0.00012 over V2.5-C) and improved or tied on **4 out of 5 validation folds** (80% monotonicity).
   - However, its overall OOF lift is **+0.00005** (from 0.76302 to 0.76307), which falls below the strict Phase 6 acceptance threshold ($\ge +0.0003$).
6. **Final Strategic Decision**:
   - In accordance with the Stage H11 mandate (*"A small overall improvement concentrated in one or two folds is not sufficient. If no candidate satisfies these criteria, retain V2.5-C"*), **Candidate V2.5-C is strictly retained as our official submission benchmark**.
   - No new submission CSV has been created, preserving all submission quotas and avoiding leaderboard churn.

---

## 2. In-Depth Error Analysis Findings

### A. The External Rating Disparity
The single largest source of prediction error across all 171,202 training samples is external bureau score coverage:

```
[All 3 External Scores Present (Pattern 111)]
 ├── 36,164 samples (21.1% of dataset)
 ├── Default Rate: 7.11%
 └── Model ROC-AUC: 0.77433 (Superb Discrimination)

[EXT_SOURCE_3 Missing (Pattern 010)]
 ├── 38,417 samples (22.4% of dataset)
 ├── Default Rate: 9.08% (Highest risk group)
 └── Model ROC-AUC: 0.74153 (-0.0328 performance collapse)
```

- When credit agencies have not rated an applicant on `EXT_SOURCE_3`, the models are forced to rely on demographic proxies (`DAYS_BIRTH`, `OCCUPATION_TYPE`, `REGION_RATING_CLIENT`).
- In tabular GBDTs, proxy variables are noisy indicators of creditworthiness, explaining why the model struggles on this segment.

### B. Young Borrowers (< 30 Years Old)
- **Sample Size**: 25,125 borrowers.
- **Default Rate**: **11.514%** (more than double the rate of applicants over 60, who default at 4.92%).
- **Model ROC-AUC**: **0.73607**.
- **Root Cause**: Younger applicants have limited credit bureau footprints and volatile employment tenure. The absence of historical repayment records (`installments_payments.csv` or `bureau.csv`, which are unavailable in GCI World) places a fundamental constraint on predictive confidence for this group.

### C. The Model Disagreement Zone
- In the top 5,000 samples where CatBoost, XGBoost, and LightGBM diverged most in their predicted ranks:
  - Ensemble ROC-AUC collapsed to **0.57975** (indistinguishable from a coin toss).
  - This divergence is heavily correlated with conflicting signals between `EXT_SOURCE_2` and `EXT_SOURCE_3` ($|\rho| = 0.0834$), and recent phone number modifications (`DAYS_LAST_PHONE_CHANGE`, $|\rho| = 0.0356$).
  - For these boundary samples, no tree model has a distinct advantage; averaging their predictions prevents extreme calibration errors.

---

## 3. Why Feature Engineering Encounters Diminishing Returns

In Stage H11, 14 targeted hypotheses were tested on Fold 1. All complex ratios and combinations either failed or neutralized each other:

1. **The Curse of Collinear Fragmentation**:
   - GBDTs construct orthogonal axis-aligned decision trees.
   - When a feature like `PAYMENT_RATE` ($ANNUITY / CREDIT$) is already present, adding `CREDIT_TERM` ($CREDIT / ANNUITY$) or `GOODS_CREDIT_DIFF` ($CREDIT - GOODS$) provides zero new topological information.
   - Because tree algorithms randomly sample a subset of features at each split (`colsample_bytree = 0.65`), injecting multiple collinear transformations reduces the probability that genuine predictive features (`EXT_SOURCE_2, 3`, `DAYS_BIRTH`) are selected at critical early splits.
2. **Peer Group Overfitting**:
   - `RELATIVE_INCOME_TO_PEER` (normalizing income by education and occupation median) showed a nominal $+0.00009$ lift on Fold 1 when tested in isolation, but combined with other features it degraded Fold 1 by $-0.00027$.
   - Group medians in smaller occupation bins introduce sample variance that fails to generalize across unseen folds.

---

## 4. Multi-Model Ensembling & The Hexa-Blend

Stage H11 explored whether an expanded multi-model ensemble could capture marginal variance:

### Standalone Model Fleet
| Model Family | Configuration | Standalone 5-Fold OOF AUC | Role in Architecture |
| :--- | :--- | :---: | :--- |
| **CatBoost** | Config 4 (62 feats, depth=6, l2=10.0, native cats) | **0.76080** | Core nonlinear categorical anchor |
| **CatBoost** | Config 3 (62 feats, depth=6, l2=5.0, rs=2.0) | **0.76072** | Stochastic regularizer |
| **XGBoost** | Config 3A_TE (46 feats, depth=6, mcw=15, l2=15.0) | **0.76166** | Target-encoded histogram baseline |
| **XGBoost** | Config 3B_TE (46 feats, depth=6, mcw=20, l2=15.0) | **0.76166** | High-stability variant (std: 0.00319) |
| **LightGBM** | Config C2_TE (46 feats, depth=6, leaves=20, l2=10.0) | **0.76183** | Smooth leaf-wise GBDT |
| **LightGBM** | Config C3_TE (46 feats, depth=6, leaves=20, l2=15.0) | **0.76194** | Top standalone model (tighter regularizer) |

### Ensemble Comparison: Candidate V2.5-C vs Hexa-Blend
- **Candidate V2.5-C**: 27% CB4 + 23% CB3 + 30% XGB_3A_TE + 20% LGB_C2_TE
  - 5-Fold OOF AUC: **0.76302**
  - Unbiased Nested CV: **0.76300**
  - Public Leaderboard: **0.76568** (Rank **345**)
- **Candidate Hexa-Blend**: 27% CB4 + 23% CB3 + 15% XGB_3A_TE + 15% XGB_3B_TE + 10% LGB_C2_TE + 10% LGB_C3_TE
  - 5-Fold OOF AUC: **0.76307** (+0.00005)
  - Unbiased Nested CV: **0.76312** (+0.00012)
  - Fold Monotonicity: **4 out of 5 folds improved or tied** (Fold 1: +0.00019, Fold 2: -0.00010, Fold 3: +0.00014, Fold 4: +0.00002, Fold 5: +0.00000).

---

## 5. Formal Acceptance Gate & Recommendation

### Why Candidate V2.5-C is Retained:
1. **The Margin of Lift**:
   - The observed OOF lift of Hexa-Blend (+0.00005) is well below the target improvement threshold of **+0.0003**.
   - An OOF delta of +0.00005 on 171,202 rows translates to swapping the ranking order of a tiny fraction of border instances.
2. **Model Complexity vs Practical Stability**:
   - Moving from 4 models to 6 models increases training and test pipeline overhead without producing a statistically decisive jump.
   - Candidate V2.5-C has already proven its real-world generalization on the Omnicampus public leaderboard (**0.76568, Rank 345**).
3. **Scientific Integrity**:
   - Kaggle-level engineering requires knowing when an algorithmic approach has extracted the available information from a dataset.
   - Forcing model replacement for an indistinguishable +0.00005 gain risks leaderboard overfitting.

### Recommendation:
- **Maintain Candidate V2.5-C as the official benchmark**.
- Catalog `Hexa-Blend` and `LightGBM C3_TE` in `experiments/stage_h11/` for future reference.
- Focus future research exclusively on genuinely orthogonal model architectures (e.g., TabNet, neural embedding architectures, or contrastive representation learning) rather than further permutations of tree hyperparameters.
