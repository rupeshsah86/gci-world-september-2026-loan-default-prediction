# Stage H6: Dataset Coverage & Availability Audit Report

## 1. Executive Summary

This dataset coverage audit inspects all data assets available in `Competition/input/` and `Competition/` for the **GCI World 2026 September Home Credit Default Risk Prediction** challenge.

- **Available Raw Data Files**:
  - `Competition/input/train.csv` (171,202 rows, 34 columns, 44.3 MB)
  - `Competition/input/test.csv` (61,500 rows, 33 columns, 15.8 MB)
  - `Competition/input/sample_submission.csv` (61,500 rows, 2 columns, 676.5 KB)
- **Auxiliary Data Sources Status**: **NONE AVAILABLE**.
  - No auxiliary relational tables (`bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `installments_payments.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`) exist in the competition input directory.
  - The competition dataset is strictly self-contained within the main application table.
  - All modeling and feature engineering must be legitimately derived from the 32 application-level features present in `train.csv` and `test.csv`.

---

## 2. Dataset Inventory & Schema Specification

| File Name | Location | Rows Count | Columns Count | Memory Footprint | Primary Key / ID | Target Present |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`train.csv`** | `Competition/input/train.csv` | 171,202 | 34 | 46.57 MB | `SK_ID_CURR` (`0` to `171,201`) | `TARGET` (Binary, Default rate: **8.0729%**) |
| **`test.csv`** | `Competition/input/test.csv` | 61,500 | 33 | 16.24 MB | `SK_ID_CURR` (`171,202` to `232,701`) | No |
| **`sample_submission.csv`** | `Competition/input/sample_submission.csv` | 61,500 | 2 | 0.68 MB | `SK_ID_CURR` (`171,202` to `232,701`) | Baseline probability format |

### Key Properties:
- **Key Uniqueness & Separation**:
  - `SK_ID_CURR` in `train.csv` is strictly unique (`nunique = 171,202`), spanning sequential integers `0` to `171,201`.
  - `SK_ID_CURR` in `test.csv` is strictly unique (`nunique = 61,500`), spanning sequential integers `171,202` to `232,701`.
  - **Zero Key Overlap**: $\text{Train IDs} \cap \text{Test IDs} = \emptyset$.
- **Target Distribution**:
  - Class `0` (Non-default): 157,381 applications (**91.9271%**)
  - Class `1` (Default): 13,821 applications (**8.0729%**)
  - Highly imbalanced binary classification task where ROC-AUC evaluates ranking calibration across classes.

---

## 3. Feature Coverage Profile & Current Pipeline Usage

The main table contains 32 raw features (8 categorical, 24 numerical) across several credit domains:

| Category | Raw Column Name | Data Type | Missingness (Train) | Missingness (Test) | Pipeline Usage (46-Feat / 62-Feat) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Contract** | `NAME_CONTRACT_TYPE` | Categorical | 0.0% | 0.0% | Label encoded (LGB/XGB), native string (CB) |
| **Demographics** | `CODE_GENDER` | Categorical | 0.0% (4 'XNA') | 0.0% | 'XNA' mapped to NaN; encoded/string |
| | `DAYS_BIRTH` | Integer (days) | 0.0% | 0.0% | Raw + `AGE_YEARS`, employment-to-age ratio |
| | `DAYS_EMPLOYED` | Integer (days) | 0.0% (18% 365243) | 0.0% | 365243 converted to NaN + `DAYS_EMPLOYED_ANOM` flag |
| | `CNT_CHILDREN` | Integer | 0.0% | 0.0% | Raw count |
| | `CNT_FAM_MEMBERS` | Float | 0.0% (2 NaNs) | 0.0% | Raw count, `INCOME_PER_FAMILY_MEMBER` |
| | `NAME_FAMILY_STATUS` | Categorical | 0.0% (2 'Unknown')| 0.0% | Encoded/string |
| | `NAME_HOUSING_TYPE` | Categorical | 0.0% | 0.0% | Encoded/string |
| **Financial** | `AMT_INCOME_TOTAL` | Float | 0.0% | 0.0% | Credit-to-income, Annuity-to-income ratios |
| | `AMT_CREDIT` | Float | 0.0% | 0.0% | Payment rate, credit-to-goods ratio |
| | `AMT_ANNUITY` | Float | 0.0% | 0.0% | Debt service burden ratios |
| | `AMT_GOODS_PRICE` | Float | 0.0% | 0.0% | Credit-to-goods price ratios |
| | `NAME_INCOME_TYPE` | Categorical | 0.0% | 0.0% | Encoded/string |
| | `NAME_EDUCATION_TYPE`| Categorical | 0.0% | 0.0% | Encoded/string |
| | `OCCUPATION_TYPE` | Categorical | 0.0% | 0.0% | Encoded/string |
| | `ORGANIZATION_TYPE` | Categorical | 0.0% | 0.0% | Encoded/string (58 categories) |
| **Assets & Region**| `OWN_CAR_AGE` | Float | **66.0%** | **66.5%** | Native NaN, `FLAG_OWN_CAR`, outlier flag |
| | `REGION_POPULATION_RELATIVE` | Float | 0.0% | 0.0% | Raw continuous density metric |
| | `REGION_RATING_CLIENT` | Integer | 0.0% | 0.0% | Ordinal rating (1, 2, 3) |
| **Flags & Urgency** | `FLAG_MOBIL`, `FLAG_EMP_PHONE`, `FLAG_WORK_PHONE` | Binary | 0.0% | 0.0% | Contactability indicators |
| | `DAYS_REGISTRATION` | Float | 0.0% | 0.0% | Raw + `REGISTRATION_YEARS` |
| | `DAYS_ID_PUBLISH` | Integer | 0.0% | 0.0% | Raw + `ID_PUBLISH_YEARS` |
| | `DAYS_LAST_PHONE_CHANGE` | Float | 0.0% | 0.0% | Raw + `PHONE_CHANGE_YEARS` |
| **External Scores**| `EXT_SOURCE_1` | Float | **69.5%** | **69.8%** | Critical credit score; summary stats & products |
| | `EXT_SOURCE_2` | Float | **0.2%** | **0.2%** | Critical credit score; summary stats & products |
| | `EXT_SOURCE_3` | Float | **31.9%** | **32.0%** | Critical credit score; summary stats & products |
| **Bureau Enquiries**| `AMT_REQ_CREDIT_BUREAU_HOUR` | Float | **13.5%** | **13.8%** | Enquiry counts |
| | `AMT_REQ_CREDIT_BUREAU_MON` | Float | **13.5%** | **13.8%** | Enquiry counts |
| | `AMT_REQ_CREDIT_BUREAU_QRT` | Float | **13.5%** | **13.8%** | Enquiry counts |
| | `AMT_REQ_CREDIT_BUREAU_YEAR`| Float | **13.5%** | **13.8%** | Enquiry counts; `BUREAU_TOTAL_CALLS`, urgency ratio |

---

## 4. Audit Findings on Auxiliary Tables & Aggregation Feasibility

1. **Absence of Relational Sub-Tables**:
   - In standard Kaggle Home Credit datasets, external history is supplied via secondary one-to-many tables.
   - In this competition environment, **none of these auxiliary tables exist on disk**.
   - No external files can be merged; all features must be derived strictly from the 32 application-level variables.
2. **Current Utilization Efficiency**:
   - The current baseline pipeline extracts 46 features (for LightGBM and XGBoost) and 62 features (for CatBoost).
   - These pipelines already exploit the primary financial interactions (payment rate, credit-to-income, credit-to-goods, per-family income) and external score aggregations (`EXT_SOURCES_MEAN`, `MIN`, `MAX`, `STD`, `NAN_COUNT`).
   - Prior experiments (Stage H2 Step 2 pilots, Stage H3 external score additions, and Stage H5 per-capita ratios) demonstrated that adding collinear polynomial or frequency features causes split dilution on gradient boosting models with feature subsampling (`colsample < 1.0`).
3. **Implications for Stage H6**:
   - Do not search for or fabricate missing auxiliary datasets.
   - Any feature experimentation in Stage H6 must focus strictly on:
     a) Unexploited non-linear interactions within the existing 32 columns.
     b) Targeted categorical interaction groupings.
     c) Algorithmic refinement, loss function calibration, and hyperparameter tuning on the existing strong model families (CatBoost, LightGBM, XGBoost).
