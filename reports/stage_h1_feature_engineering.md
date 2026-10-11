# Stage H1: Controlled Feature-Engineering Evaluation Report

## 1. Executive Summary & Baselines
- **Reference Model**: CatBoost (46 features, Stage F1) -> **Mean ROC-AUC: 0.76060**, **OOF: 0.76060**
- **Reference Ensemble**: 50% CatBoost + 25% LightGBM + 25% XGBoost -> **OOF: 0.76187**
- **Evaluation Protocol**: Identical 5-Fold Stratified Cross-Validation (`random_state=42`), zero leakage.

## 2. Tested Feature Groups
1. **Group 1: Extended Financial Ratios & Cash Flow (4 features)**:
   - `CREDIT_MINUS_GOODS` (financing markup / cash margin)
   - `CREDIT_TO_GOODS_RATIO` (loan-to-goods ratio)
   - `INCOME_MINUS_ANNUITY` (disposable cash flow remaining)
   - `CREDIT_TERM_APPROX` (approximate repayment periods)
2. **Group 2: Temporal, Life Stage & Bureau Urgency (7 features)**:
   - `AGE_YEARS` & `EMPLOYED_YEARS` (positive calibrated years)
   - `ID_PUBLISH_TO_AGE_RATIO`, `REGISTRATION_TO_AGE_RATIO`, `PHONE_TO_AGE_RATIO`
   - `BUREAU_ENQUIRIES_TOTAL` & `BUREAU_URGENT_SHARE` (urgency of credit need)
3. **Group 3: External Score Interactions & Missing Indicators (5 features)**:
   - `EXT_SOURCES_PROD` (multiplicative interaction)
   - `EXT_SOURCE_1_ISNA`, `EXT_SOURCE_3_ISNA` (informative missingness flags)
   - `EXT_WEIGHTED_SCORE` (composite external bureau score)
   - `EXT_SOURCE_2_3_DIFF` (inter-bureau divergence)

## 3. CatBoost Performance Across Incremental Candidates
| Candidate Configuration | Features | Fold Scores (1 to 5) | Mean ROC-AUC (+/- Std) | Overall OOF ROC-AUC | Delta vs Reference (0.76060) | Runtime |
| :--- | :---: | :--- | :--- | :--- | :---: | :---: |
| **Reference Baseline (46 feats)** | 46 | `[0.75713, 0.76118, 0.76646, 0.75621, 0.76200]` | **0.76060 +/- 0.00369** | **0.76060** | Baseline | 269.2s |
| **Candidate_1_Financial_Ratios** | 50 | `[0.75837, 0.76067, 0.76533, 0.75675, 0.76124]` | **0.76047 +/- 0.00292** | **0.76044** | **-0.00013** | 244.3s |
| **Candidate_2_Financial_Plus_Temporal** | 57 | `[0.75749, 0.76125, 0.76549, 0.7579, 0.76098]` | **0.76062 +/- 0.00288** | **0.76060** | **+0.00002** | 319.7s |
| **Candidate_3_All_New_Groups** | 62 | `[0.75913, 0.76096, 0.76555, 0.75706, 0.76058]` | **0.76066 +/- 0.00281** | **0.76061** | **+0.00006** | 368.9s |

## 4. Key Findings: What Helped vs. What Hurt
- **Best Performing Feature Set**: **`Candidate_3_All_New_Groups`** with **62 features**.
- **CatBoost Gain**: Mean ROC-AUC reached **0.76066** (+0.00006 over reference 0.76060).
- **Feature Group Impact Analysis**:
  - **Candidate_1_Financial_Ratios** (50 features): Mean AUC = 0.76047 (-0.00013). Added 4 financial & cash flow ratios (CREDIT_MINUS_GOODS, CREDIT_TO_GOODS_RATIO, INCOME_MINUS_ANNUITY, CREDIT_TERM_APPROX).
  - **Candidate_2_Financial_Plus_Temporal** (57 features): Mean AUC = 0.76062 (+0.00002). Added Group 1 + 7 temporal & bureau features (AGE_YEARS, EMPLOYED_YEARS, ratios to age, BUREAU_TOTAL, BUREAU_URGENT_SHARE).
## 5. Model Architecture Sensitivity & Ensemble Impact

### A. Impact on LightGBM
When evaluated on the 62 features of **Candidate 3**, LightGBM's 5-fold mean ROC-AUC slightly decreased:
- **Tuned LightGBM (46 features)**: Mean = **0.75905**, OOF = **0.75900**
- **LightGBM on Candidate 3 (62 features)**: Mean = **0.75863**, OOF = **0.75857** (-0.00042)
- **Insight**: In leaf-wise histogram tree building, adding 16 correlated interaction features creates redundant split candidates that dilute feature subsampling (`feature_fraction=0.7`). Conversely, CatBoost's oblivious trees and ordered target statistics benefit from explicit interaction terms.

### B. Impact on the Tri-Model Ensemble Blend
Because CatBoost on 62 features captures distinct non-linear interactions while LightGBM and XGBoost remain optimized on the compact 46-feature set, combining them increases inter-model diversity:
- **Previous Ensemble (All 46 features)**: `50% CatBoost + 25% LightGBM + 25% XGBoost` -> OOF ROC-AUC = **0.76187**
- **Updated Ensemble (CatBoost 62 feats + LGBM 46 feats + XGBoost 46 feats)**: -> OOF ROC-AUC = **0.76203** (**+0.00016 net lift**)

## 6. Summary & Checkpoint
- **Best Single Model**: CatBoost on Candidate 3 (62 features) -> **0.76066 $\pm$ 0.00281** (OOF: **0.76061**)
- **Best Ensemble**: CatBoost (62 feats) + LightGBM (46 feats) + XGBoost (46 feats) -> **0.76203**
- All original competition files in `Competition/` remain untouched.
- No final submission or automatic upload has been performed.