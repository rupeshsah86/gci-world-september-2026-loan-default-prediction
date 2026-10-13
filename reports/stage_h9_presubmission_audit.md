# Stage H9 Pre-Submission Audit Report: Candidate V2.5-C

**Candidate File**: `outputs/submission_candidate_20261013_v25_h9.csv`  
**Candidate Name**: Candidate V2.5-C (Dual Target-Encoded Quad-Model Rank Blend)  
**Date of Audit**: 2026-10-13  
**Auditor**: Senior Machine Learning Engineer / Antigravity Agent  
**Current Confirmed Leaderboard Benchmark**: Candidate V2.3 (Public ROC-AUC: **0.76535**, Omnicampus Rank: **398**)  
**Audit Determination**: **READY FOR MY SUBMISSION REVIEW**  

---

## 1. Phase 1: Independent Audit of Target Encoding Implementation

A rigorous code and mathematical audit was performed on the target encoding implementation used in both LightGBM C2 and XGBoost Config 3A (`src/experiment_h9_target_encoding_full.py`, `src/experiment_h9_target_encoded_xgboost_full.py`, and `src/train_submission_v25_h9.py`).

### Verification of the 7 Audit Criteria:
1. **Fit Exclusively on Training Folds**:
   - *Verified*: In all 5 cross-validation folds, `TargetEncoder` was instantiated and fitted exclusively inside `skf.split()` on `X_tr[high_card_cats]` and `y_tr`. Validation splits `(X_va, y_va)` were strictly sequestered prior to fitting.
2. **Out-of-Fold Training Encodings (Leakage-Safe)**:
   - *Verified*: Used `TargetEncoder(smooth="auto", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))`. The `fit_transform()` method splits the training fold into 5 internal sub-folds, computing smoothed target encodings for each sub-fold using only the remaining 4 sub-folds.
3. **No Row Uses Its Own Target**:
   - *Verified*: Because internal cross-validation is enforced, no sample's target label is included in the numerator or denominator of its own encoded probability.
4. **Validation Encodings Never Fitted on Validation Targets**:
   - *Verified*: Validation features were transformed using `te.transform(X_va[high_card_cats])`. The `transform()` method is strictly deterministic and univariate, mapping levels using the already-fitted empirical Bayes m-estimate parameters from the training set. It takes no target argument `y_va`.
5. **Smoothing & Unseen Levels Configuration**:
   - *Verified*: `smooth="auto"` applies empirical Bayes shrinkage towards the prior class mean ($p \approx 0.0808$). Categories with low sample frequency or unseen test categories are safely shrunken towards the global prior default rate without causing numerical instability.
6. **Full-Data Training & Test Inference Pipeline**:
   - *Verified*: In `src/train_submission_v25_h9.py`:
     - Full training set encodings were generated via `te.fit_transform(X_train[high_card_cats], y_train)` with internal 5-fold CV to prevent self-leakage during training.
     - Test set encodings were transformed via `te.transform(X_test[high_card_cats])` using only the full training set's smoothed parameters. No test targets exist or were referenced.
7. **Nested CV Independence Characterization**:
   - *Audit Finding*: The nested CV reported in Stage H9 was an *exploratory nested cross-validation on ensemble blending weights* applied to precomputed OOF rank predictions, rather than an end-to-end retraining nested CV.
   - *Characterization*: In strict accordance with the prompt guidelines, this metric (**0.76300**) is explicitly classified as an **exploratory nested estimate of weight stability**, while the 5-fold cross-validation OOF score (**0.76302**) serves as the primary local validation benchmark.

---

## 2. Phase 2: Reproduction of Reported Experimental Results

Using identical row ordering, IDs, target labels, and 5-fold CV splits across all 171,202 training samples, all reported metrics were independently reproduced:

### A. Standalone Component Reproducibility Audit
| Component Model | Feature Set | 5-Fold OOF AUC | Status |
| :--- | :---: | :---: | :--- |
| **CatBoost Config 4** (`stage_h2_step3a`) | 62 feats (native cats) | **0.76080** | 100% Reproduced |
| **CatBoost Config 3** (`stage_h2_step3a`) | 62 feats (native cats) | **0.76072** | 100% Reproduced |
| **XGBoost Config 3A (Target-Encoded)** | 46 feats (OOF TE cats) | **0.76166** | 100% Reproduced (+0.00078 vs single) |
| **LightGBM C2 (Target-Encoded)** | 46 feats (OOF TE cats) | **0.76183** | 100% Reproduced (+0.00213 vs single) |

### B. Candidate V2.5-C vs Candidate V2.3 Benchmark Comparison
- **Ensemble Blend Weights**:
  - `CatBoost Config 4`: 0.27
  - `CatBoost Config 3`: 0.23
  - `XGBoost Config 3A (Target-Encoded)`: 0.30
  - `LightGBM C2 (Target-Encoded)`: 0.20
  - Normalization: Percentile-rank normalized `(rank - 0.5) / N` before linear combination.

| Validation Fold | Candidate V2.3 (Benchmark) | Candidate V2.5-C (Dual TE) | Delta | Fold Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Fold 1** | 0.76021 | 0.76061 | **+0.00040** | Candidate V2.5-C |
| **Fold 2** | 0.76349 | 0.76446 | **+0.00097** | Candidate V2.5-C |
| **Fold 3** | 0.76733 | 0.76750 | **+0.00017** | Candidate V2.5-C |
| **Fold 4** | 0.75847 | 0.75854 | **+0.00007** | Candidate V2.5-C |
| **Fold 5** | 0.76387 | 0.76413 | **+0.00025** | Candidate V2.5-C |
| **Overall OOF ROC-AUC** | **0.76265** | **0.76302** | **+0.00037** | **Candidate V2.5-C (100% Monotonic)** |
| **Exploratory Nested CV** | **0.76260** | **0.76300** | **+0.00040** | **Candidate V2.5-C** |

**Conclusion**: The reported +0.00037 OOF improvement is completely genuine and verified across 100% of folds.

---

## 3. Phase 3: Candidate Generation Details

The final submission candidate was generated using [`src/train_submission_v25_h9.py`](file:///Users/rupeshmacbook/Documents/GCI_World/src/train_submission_v25_h9.py):
- **Full Training Data**: 171,202 samples.
- **Full Test Data**: 61,500 samples.
- **Model 1**: Target-Encoded XGBoost Config 3A trained for 751 rounds (mean best iterations across folds). Runtime: 5.6s. Test prediction mean: 0.0891.
- **Model 2**: Target-Encoded LightGBM C2 trained for 826 rounds (mean best iterations across folds). Runtime: 5.3s. Test prediction mean: 0.0888.
- **Model 3 & 4**: Cached full-data test predictions for CatBoost Config 4 and Config 3 loaded from `experiments/test_preds_quad_models.npz`. Test prediction mean: 0.0885.
- **Blending**: Percentile-rank normalized test predictions combined using the audited weights `[0.27, 0.23, 0.30, 0.20]`.
- **Candidate Output**: Saved to `outputs/submission_candidate_20261013_v25_h9.csv`.

---

## 4. Phase 4: Independent Verification of the Candidate CSV

Every required verification check was executed programmatically on `outputs/submission_candidate_20261013_v25_h9.csv`:

| Check | Specification | Measured Result | Pass / Fail |
| :--- | :--- | :--- | :---: |
| **File Existence** | Exists in `outputs/` | Present (`outputs/submission_candidate_20261013_v25_h9.csv`) | **PASS** |
| **Row Count** | Exactly 61,500 rows | Exactly **61,500** rows | **PASS** |
| **Column Names** | `SK_ID_CURR`, `TARGET` | Exactly `['SK_ID_CURR', 'TARGET']` | **PASS** |
| **Column Order** | ID first, TARGET second | Verified: `SK_ID_CURR` (col 0), `TARGET` (col 1) | **PASS** |
| **ID Alignment vs test.csv** | Identical 61,500 IDs and order | **100.0% Exact Match** | **PASS** |
| **ID Alignment vs sample_sub** | Identical 61,500 IDs and order | **100.0% Exact Match** | **PASS** |
| **Missing / Null IDs** | 0 nulls | **0 nulls** | **PASS** |
| **Duplicate IDs** | 0 duplicates (61,500 unique) | **61,500 unique IDs** | **PASS** |
| **NaN / Null Predictions** | 0 nulls | **0 nulls** | **PASS** |
| **Infinite Predictions** | 0 infinite values | **0 infinite values** | **PASS** |
| **Probability Bounds** | All values in [0.0, 1.0] | Min: **0.000044**, Max: **0.999983** | **PASS** |
| **Granularity / Continuous** | Non-trivial unique probabilities | **61,285 unique values** / 61,500 rows | **PASS** |
| **Index Column Artifact** | No extra index column | Zero extra columns | **PASS** |
| **Correlation vs V2.3** | High fidelity with prior candidate | Spearman: **0.99785**, Pearson: **0.99785** | **PASS** |
| **Preservation of Past Files** | Candidate V2.3 untouched | `outputs/submission_candidate_20261013_v23_h7.csv` intact | **PASS** |
| **Competition Directory** | Untouched | `Competition/` strictly unmodified | **PASS** |

### Test Distribution Summary
- **Minimum Prediction**: 0.000044
- **Maximum Prediction**: 0.999983
- **Mean Prediction**: 0.50001
- **Median Prediction**: 0.49928
- **Standard Deviation**: 0.28674

---

## 5. Final Audit Determination

All 4 phases have been successfully completed and independently audited.

### Final Status:
**READY FOR MY SUBMISSION REVIEW**
