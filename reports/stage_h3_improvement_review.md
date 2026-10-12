# Stage H3 Improvement & Candidate V2 Review Report

## 1. Executive Summary
- **New Candidate Artifact**: [`outputs/submission_candidate_20261012_v2_quad_blend.csv`](file:///Users/rupeshmacbook/Documents/GCI_World/outputs/submission_candidate_20261012_v2_quad_blend.csv)
- **Previous Verified Baseline**: `outputs/submission_candidate_20261012.csv` (Public Leaderboard: **0.76456**, Rank 555).
- **Generated At**: 2026-10-12 17:51:13 UTC
- **Total Training & Inference Runtime**: 183.4 seconds (~3.1 minutes) on Apple Silicon M2.
- **Ensemble Architecture**: Quad-model probability blend combining intra-model CatBoost regularization diversity with tuned LightGBM and regularized XGBoost.

## 2. Component Models and Rationale for Changes

| Component Model | Feature Pipeline | Key Hyperparameters | Full Iterations | 5-Fold OOF AUC | Blend Weight | Rationale |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **CatBoost Config 4** | 62 features | `depth=6, l2=10.0, rs=1.5, lr=0.04` | 1,180 | **0.76080** | **27%** | Peak individual model performer; strong L2 regularizer |
| **CatBoost Config 3** | 62 features | `depth=6, l2=5.0, rs=2.0, lr=0.04` | 1,103 | **0.76072** | **23%** | High split randomness (`rs=2.0`) providing intra-model diversity |
| **XGBoost Config 3**  | 46 features | `depth=6, mcw=10, col=0.60, a=2.0, l=10.0, lr=0.03` | 453 | **0.76015** | **30%** | Massive gain (+0.00141 across all folds); weight increased from 25% to 30% |
| **Tuned LightGBM**    | 46 features | `leaves=24, depth=7, mcw=100, col=0.70, a=1.0, l=5.0, lr=0.03` | 465 | **0.75900** | **20%** | Fast asymmetric leaf-wise architecture; vital diversity ($r \approx 0.956$) |

## 3. Validation Metrics & Fold-Level Consistency
Comparison on the exact same 171,202 aligned out-of-fold training samples:

| Ensemble Strategy | Component Weights | 5-Fold OOF ROC-AUC | Mean Fold Score | Fold Consistency vs Previous |
| :--- | :---: | :---: | :---: | :--- |
| **Previous Submission (V1)** | `50% CB3 / 25% LGB / 25% XGB` | **0.76208** | 0.76211 $\pm$ 0.00288 | Reference Baseline (LB: **0.76456**) |
| **3-Model Upgraded Blend** | `50% CB4 / 20% LGB / 30% XGB` | **0.76216** | 0.76219 $\pm$ 0.00284 | Improved on 4 of 5 folds (+0.00008) |
| **New Quad-Model Blend (V2)** | `27% CB4 / 23% CB3 / 20% LGB / 30% XGB` | **0.76224** | **0.76228 $\pm$ 0.00285** | **Improved on 4 of 5 folds (+0.00016)** |

### Fold-by-Fold Breakdown:
- **Fold 1**: $0.75922 \rightarrow \mathbf{0.75983}$ (+0.00061)
- **Fold 2**: $0.76322 \rightarrow \mathbf{0.76345}$ (+0.00023)
- **Fold 3**: $0.76654 \rightarrow \mathbf{0.76663}$ (+0.00009)
- **Fold 4**: $0.75882 \rightarrow 0.75863$ (-0.00019)
- **Fold 5**: $0.76274 \rightarrow \mathbf{0.76287}$ (+0.00013)
- **Net Result**: 4 out of 5 folds improved, demonstrating that the gain is structural across the dataset rather than an artifact of a single fold.

## 4. Feature Improvements Tested & Rejected
In strict compliance with validation safety rules:
1. **External Rating Summary Statistics (66 features)**: Added `nanmean`, `nanmin`, `nanmax`, and `nanstd` across `EXT_SOURCE_1/2/3` to CatBoost. While Fold 1 improved by +0.00112, the 5-fold OOF dropped from 0.76072 to 0.76055 (-0.00017) due to proxy split dilution. **Rejected**.
2. **62 Features on XGBoost Config 3**: Tested the extended pipeline on XGBoost. While OOF was nominally flat (+0.00003), standard deviation widened from $\pm 0.00285$ to $\pm 0.00342$. Keeping XGBoost on the cleaner 46-feature set preserved stability and model diversity. **Rejected**.

## 5. File Verification and Compliance Audit
| Audit Item | Required Specification | Candidate V2 Actual | Status |
| :--- | :--- | :--- | :---: |
| **File Location** | `outputs/submission_candidate_20261012_v2_quad_blend.csv` | `/Users/rupeshmacbook/Documents/GCI_World/outputs/submission_candidate_20261012_v2_quad_blend.csv` | **PASSED** |
| **Row Count** | Exactly 61,500 rows | 61,500 rows | **PASSED** |
| **Column Structure** | Exactly `['SK_ID_CURR', 'TARGET']` | `['SK_ID_CURR', 'TARGET']` | **PASSED** |
| **ID Integrity** | Exact match with test.csv in original order | Zero mismatch, zero duplicates | **PASSED** |
| **Prediction Values** | Finite probabilities $\in [0.0, 1.0]$ | Min: 0.0030, Mean: 0.0886, Max: 0.7913 | **PASSED** |
| **Previous Submission** | Untouched at `outputs/submission_candidate_20261012.csv` | Verified preserved | **PASSED** |

> [!IMPORTANT]
> **Validation Caveat:**
> The local OOF ROC-AUC (**0.76224**) is a discriminative ranking measure on cross-validation splits. While the previous 0.76208 candidate achieved **0.76456 on the public leaderboard**, public leaderboard scores depend on the unseen private/public test split calibration. Do not treat small OOF differences as guarantees of leaderboard gains.

## 6. Stop Checkpoint
- The new candidate file is saved and verified locally.
- In accordance with your instructions, **no file has been uploaded or submitted anywhere**.
- Ready for your review.