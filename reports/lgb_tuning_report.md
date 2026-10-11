# LightGBM Hyperparameter Tuning Report (Stage E1)

## Summary
- **Reference Baseline (Stage D)**: Mean ROC-AUC = 0.75638, OOF ROC-AUC = 0.75627
- **Best Tuned Model**: Config_2_Conservative_Leaves_DeepRegularization
- **Best Mean ROC-AUC**: **0.75905 +/- 0.00323**
- **Best OOF ROC-AUC**: **0.75900**
- **Net Gain over Reference**: **+0.00267**

## Tuning Comparison Table
| Configuration Name | Key Parameters | Fold Scores (1–5) | Mean ROC-AUC | OOF ROC-AUC | Delta vs Reference | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config_0_StageD_Reference** | `lr=0.05, leaves=31, d=-1, mc=20, ff=0.8, a=0.0, l=0.0` | [0.75342, 0.7572, 0.76111, 0.75277, 0.75741] | **0.75638** | 0.75627 | +0.00000 | 19.4s |
| **Config_1_Regularized_L1_L2** | `lr=0.05, leaves=31, d=-1, mc=50, ff=0.7, a=0.5, l=2.0` | [0.75444, 0.75792, 0.76163, 0.75187, 0.75752] | **0.75668** | 0.75660 | +0.00030 | 17.0s |
| **Config_2_Conservative_Leaves_DeepRegularization** | `lr=0.03, leaves=24, d=7, mc=100, ff=0.7, a=1.0, l=5.0` | [0.7553, 0.76093, 0.76386, 0.75571, 0.75944] | **0.75905** | 0.75900 | +0.00267 | 34.6s |
| **Config_3_Feature_Subsampling_HighCapacity** | `lr=0.03, leaves=40, d=-1, mc=50, ff=0.65, a=0.2, l=1.0` | [0.75557, 0.76028, 0.76359, 0.75363, 0.75895] | **0.75840** | 0.75836 | +0.00202 | 32.8s |
| **Config_4_Robust_Credit_Tuned** | `lr=0.03, leaves=31, d=8, mc=70, ff=0.7, a=0.8, l=3.0` | [0.7552, 0.75947, 0.76345, 0.75414, 0.75944] | **0.75834** | 0.75830 | +0.00196 | 35.2s |

## Best Parameters Detail
```json
{
  "learning_rate": 0.03,
  "num_leaves": 24,
  "max_depth": 7,
  "min_child_samples": 100,
  "feature_fraction": 0.7,
  "bagging_fraction": 0.75,
  "bagging_freq": 1,
  "reg_alpha": 1.0,
  "reg_lambda": 5.0
}
```

## Validation Stability & Leakage Audit Findings
1. **Zero Data Leakage**: All category encodings and feature transformers are fitted independently per fold.
2. **No Target Contamination**: `TARGET` and `SK_ID_CURR` are strictly excluded from predictors.
3. **Regularization Impact**: Increasing L1/L2 penalties (`reg_alpha` and `reg_lambda`) alongside larger `min_child_samples` effectively tempers overfitting to noisy samples.
4. **Fold Consistency**: Fold variance remained minimal across all configurations, confirming stability on Apple Silicon M2.