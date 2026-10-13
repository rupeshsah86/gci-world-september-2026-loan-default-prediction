# Stage H9 Final Review: Targeted Ensemble Improvement & The Path Toward 0.77000

**Project**: GCI World September 2026 — Home Credit Default Risk Prediction  
**Date**: 2026-10-13  
**Current Public Leaderboard Standing**: **0.76535** (Rank **398** on Omnicampus, achieved by Candidate V2.3)  
**Target Goal**: Public ROC-AUC **0.77000** (Remaining gap: **0.00465**)  

---

## 1. Executive Summary & Key Milestones Achieved

Stage H9 was designed to investigate the negative result of Stage H8 (where bagged XGBoost degraded the Candidate V2.4 ensemble despite improving individual performance) and test whether a targeted hybrid or feature diversity hypothesis could break past the Candidate V2.3 ceiling.

### Major Breakthroughs in Stage H9
1. **Resolution of the "Bagging vs Diversity" Paradox**:
   - Demonstrated mathematically that multi-seed bagging of XGBoost smoothed tree variance toward the common conditional expectation, increasing its residual error correlation with CatBoost from **0.9963** to **0.9967** (a 10.8% loss in inter-model error independence).
   - This explained why the fully bagged Candidate V2.4 (0.76253) lagged Candidate V2.3 (0.76265).
2. **First Genuine Standalone Breakthrough Across Both GBDT Families**:
   - Replaced raw integer encodings of high-cardinality nominals (`OCCUPATION_TYPE`, 18 levels; `ORGANIZATION_TYPE`, 58 levels) with smooth out-of-fold Bayesian target encodings (`TargetEncoder(smooth="auto", cv=5)`).
   - **LightGBM C2**: Jumped from **0.75970 to 0.76183** (**+0.00213** lift, +0.00213 mean fold AUC).
   - **XGBoost Config 3A**: Jumped from **0.76088 to 0.76166** (**+0.00078** lift, beating even the 3-seed bagged model).
   - This closed the structural gap that previously caused LightGBM and XGBoost to trail CatBoost's native target statistics.
3. **New Best Ensemble Benchmark (Candidate V2.5-C)**:
   - Combining CatBoost Config 4, CatBoost Config 3, Target-Encoded XGBoost Config 3A, and Target-Encoded LightGBM C2 produced an overall 5-fold OOF ROC-AUC of **0.76302** (**+0.00037** over Candidate V2.3).
   - Unbiased 5-fold Nested Cross-Validation confirmed an out-of-fold generalization score of **0.76300** (**+0.00040** over Candidate V2.3's 0.76260).
   - The lift is **100% monotonic across all 5 validation folds** (+0.00040, +0.00097, +0.00017, +0.00007, +0.00025).

---

## 2. Exhaustive Audit & Baseline Verification

All models, data files, and validation splits were audited:
- **Ground Truth & Row Alignment**: Exactly 171,202 training samples and 61,500 test samples. 100% ID and label match across all 9 generated OOF files.
- **Split Scheme**: Strictly uniform `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- **Supplementary Data Inspection**: Checked `Competition/input/`. Only `train.csv`, `test.csv`, and `sample_submission.csv` exist. No secondary relational tables (`bureau.csv`, `previous_application.csv`, `installments_payments.csv`) are available in this competition.

---

## 3. Comparative Summary: Standalone Models & Ensembles

### A. Standalone Model Standings (5-Fold OOF ROC-AUC)
| Model Architecture | Feature Pipeline | Standalone OOF AUC | Delta vs Baseline LightGBM | Status |
| :--- | :---: | :---: | :---: | :--- |
| **LightGBM C1 (Original Baseline)** | 46 feats (integer cats) | 0.75900 | Reference | Stage H0 baseline |
| **LightGBM C2 (Tuned Depth 6)** | 46 feats (integer cats) | 0.75970 | +0.00070 | Stage H6 baseline |
| **LightGBM C2 (3-Seed Bagged)** | 46 feats (integer cats) | 0.75983 | +0.00083 | Stage H8 experiment |
| **CatBoost Config 3** | 62 feats (native cats) | 0.76072 | +0.00172 | Stage H2 component |
| **CatBoost Config 4** | 62 feats (native cats) | 0.76080 | +0.00180 | Stage H2 component |
| **XGBoost Config 3A (Single)** | 46 feats (cat dtype) | 0.76088 | +0.00188 | Stage H7 component |
| **XGBoost Config 3A (3-Seed Bagged)**| 46 feats (cat dtype) | 0.76128 | +0.00228 | Stage H8 experiment |
| **XGBoost Config 3A (Target-Encoded)**| 46 feats (OOF TE cats) | **0.76166** | +0.00266 | **Stage H9 breakthrough** |
| **LightGBM C2 (Target-Encoded)** | 46 feats (OOF TE cats) | **0.76183** | **+0.00283** | **Stage H9 breakthrough (Top Model)** |

### B. Ensemble Configurations (5-Fold OOF & Nested CV)
| Ensemble Name | Description / Weights | OOF Rank Blend | OOF Prob Blend | Unbiased Nested CV | Monotonicity (Folds Beat V2.3) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Candidate V2** | CB4 (0.27) + CB3 (0.23) + XGB3 (0.30) + LGB_C1 (0.20) | 0.76224 | 0.76218 | 0.76219 | - |
| **Candidate V2.1** | Rank-normalized quad blend | 0.76230 | - | 0.76228 | - |
| **Candidate V2.2** | Upgraded LightGBM C2 quad blend | 0.76235 | 0.76228 | 0.76233 | - |
| **Candidate V2.3 (Current Best)**| Upgraded XGBoost Config 3A quad blend | **0.76265** | 0.76258 | **0.76260** | **Benchmark (LB 0.76535)** |
| **Candidate V2.4 (Fully Bagged)**| 3-seed Bagged XGB + 3-seed Bagged LGB | 0.76253 | 0.76246 | 0.76250 | 0 / 5 folds |
| **Candidate V2.5-A** | V2.3 + Target-Encoded LightGBM C2 | **0.76292** | 0.76284 | **0.76294** | 4 / 5 folds |
| **Candidate V2.5-C (Dual TE)** | V2.3 + Target-Encoded XGB + LGB | **0.76302** | **0.76293** | **0.76300** | **5 / 5 folds (100%)** |

---

## 4. Honest Assessment of the Target (0.77000) & The Empirical Data Ceiling

The primary strategic question is: **Can this tabular dataset reach 0.77000 on the public leaderboard?**

### The Statistical Reality:
1. **Scope of Available Features**:
   - In the classic Kaggle Home Credit Default Risk competition, public leaderboard scores $\ge 0.79000$ were achieved by aggregating millions of historical loan records from `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `installments_payments.csv`, and `credit_card_balance.csv`.
   - In GCI World September 2026, **only the single application table (`train.csv`) is provided** (34 columns, reduced to ~18 demographic/financial fields plus `EXT_SOURCE_1, 2, 3`).
   - All models in this repository are operating strictly on this single static profile table.
2. **Bayes Error Rate on a Single Static Table**:
   - Single-model performance on this table tops out around **0.7618** (achieved by Target-Encoded LightGBM C2).
   - Multi-model ensembling of diverse trees (CatBoost, XGBoost, LightGBM) pushes performance to **0.76302** OOF and **0.76535** public leaderboard.
   - The remaining gap from 0.76535 to 0.77000 is **0.00465**.
   - An increase of +0.00465 ROC-AUC on a single application table without external payment histories requires new predictive variance that simply does not exist in the 34 raw columns.
   - Candidate V2.5-C represents an expected public leaderboard score of **~0.76570 to 0.76590** (an improvement of +0.00035 to +0.00055 over Candidate V2.3), which will solidify our position within the top 300-350 on Omnicampus.
   - However, reaching **0.77000** without additional external tables or leakage is mathematically improbable under standard supervised tabular modeling on this single table.

---

## 5. Next Steps & Decision Gate

In strict adherence to the safety rules:
- **No submission candidate file has been created yet.**
- **No file in `Competition/` has been touched.**
- **No code has been pushed or committed to Git.**
- **Candidate V2.3 (`outputs/submission_candidate_20261013_v23_h7.csv`) remains completely preserved as the benchmark.**

### Proposed Next Step (Upon User Approval):
If the user approves, we will generate the full test predictions for:
1. Full-data Target-Encoded LightGBM C2 (using smoothed training target stats applied to test).
2. Full-data Target-Encoded XGBoost Config 3A.
3. Combine them with the existing full-data predictions of CatBoost Config 4 and CatBoost Config 3 into **Candidate V2.5** (`outputs/submission_candidate_20261013_v25_h9.csv`).
4. Perform an independent audit on candidate row count (61,500), columns (`SK_ID_CURR`, `TARGET`), rank distribution, and lack of nulls before submission consideration.
