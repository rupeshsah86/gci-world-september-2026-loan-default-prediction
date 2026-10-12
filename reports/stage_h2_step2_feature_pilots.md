# Stage H2 Step 2: Feature Pilot Experiments Report

## 1. Executive Summary
- **Reference LightGBM Baseline**: Mean = **0.75905**, OOF = **0.75900** (46 features)
- **Reference Ensemble Baseline**: OOF = **0.76203**
- **Evaluation Setup**: Identical 5-Fold Stratified Cross-Validation (`random_state=42`), zero data leakage.
- **Total Runtime**: 128.2 seconds (~2.1 minutes) on Apple Silicon M2.

## 2. Pilot Experiment Configurations
1. **Pilot 1: Out-of-Fold Smoothed Target Encoding (F1)**:
   - Category targets encoded using an internal 5-fold cross-split inside each training fold (Bayesian smoothing $m=20$).
   - Evaluated on high-cardinality features: `ORGANIZATION_TYPE` and `OCCUPATION_TYPE`.
2. **Pilot 2: Non-Linear Cross-Interactions (F2/F3)**:
   - Non-linear cross-ratios combining the top gain drivers: `PAYMENT_RATE_X_EXT_MEAN`, `CREDIT_INCOME_TO_EXT_MEAN`, `ANNUITY_INCOME_TO_EXT_MEAN`, `EXT_MIN_TO_MAX_RATIO`, `PAYMENT_RATE_X_AGE`.
3. **Pilot 3: Frequency / Prevalence Encodings (F4)**:
   - Normalized category frequencies learned strictly from training folds for `ORGANIZATION_TYPE`, `OCCUPATION_TYPE`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`.
4. **Pilot 4: Combined Feature Set**:
   - Full unified pipeline incorporating Target Encoding, Cross-Interactions, and Frequency Encodings.

## 3. Cross-Validation Results Comparison Table
| Experiment Name | Feature Count | Folds ROC-AUC (1 to 5) | Mean ROC-AUC (+/- Std) | OOF ROC-AUC | Delta vs Reference (0.75905) | Runtime |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| **Reference LightGBM (46 feats)** | 46 | `[0.75530, 0.76093, 0.76386, 0.75571, 0.75944]` | **0.75905 +/- 0.00323** | **0.75900** | Baseline | 34.7s |
| **Pilot_1_Target_Encoding** | 48 | `[0.7554, 0.76001, 0.76288, 0.75484, 0.75856]` | **0.75834 +/- 0.00298** | **0.75831** | **-0.00071** | 28.6s |
| **Pilot_2_Cross_Interactions** | 51 | `[0.75585, 0.76047, 0.76395, 0.75432, 0.75885]` | **0.75869 +/- 0.00341** | **0.75867** | **-0.00036** | 33.0s |
| **Pilot_3_Frequency_Encodings** | 50 | `[0.75591, 0.76043, 0.76317, 0.75495, 0.75928]` | **0.75875 +/- 0.00301** | **0.75872** | **-0.00030** | 31.5s |
| **Pilot_4_Combined_Best_Features** | 57 | `[0.7555, 0.75964, 0.76374, 0.7551, 0.7585]` | **0.75850 +/- 0.00314** | **0.75845** | **-0.00055** | 34.4s |

## 4. Key Findings: What Helped vs. What Hurt
- **Best Feature Set**: **`Pilot_3_Frequency_Encodings`** (50 features) with Mean ROC-AUC of **0.75875** (-0.00030) and OOF of **0.75872**.

### Detailed Breakdown by Feature Group:
- **Pilot_1_Target_Encoding**: Mean = **0.75834** (-0.00071). Smoothed OOF target encoding for ORGANIZATION_TYPE and OCCUPATION_TYPE (m=20).
- **Pilot_2_Cross_Interactions**: Mean = **0.75869** (-0.00036). 5 non-linear interactions combining top gain drivers (PAYMENT_RATE, CREDIT_INCOME, EXT_SOURCES).
- **Pilot_3_Frequency_Encodings**: Mean = **0.75875** (-0.00030). Frequency / prevalence encodings for high-cardinality categoricals.
- **Pilot_4_Combined_Best_Features**: Mean = **0.75850** (-0.00055). Full combination of OOF Target Encoding, Cross-Interactions, and Frequency Encodings.

## 5. Summary & Next Steps
- All original competition files remain untouched.
- No final submission or upload performed.
- Results presented for user review before proceeding to CatBoost evaluation or Stage H2 Step 3.