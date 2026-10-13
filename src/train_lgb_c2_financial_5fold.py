"""
Stage H7: 5-Fold Cross-Validation of LightGBM C2 + Group 1 Financial Ratios (50 Features).
Tests whether the +0.00016 Fold 1 gain holds across all 5 folds vs baseline LGB C2 (0.75970).
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features_h1 import ExtendedFeaturePipeline
from src.features import CATEGORICAL_COLS

STAGE_H7_DIR = PROJECT_ROOT / "experiments" / "stage_h7"
STAGE_H7_DIR.mkdir(parents=True, exist_ok=True)
OOF_LGB_C2_FIN_PATH = STAGE_H7_DIR / "oof_lightgbm_c2_financial_50f.csv"

def main():
    print("=" * 80)
    print("STAGE H7: 5-FOLD CV OF LIGHTGBM C2 + FINANCIAL RATIOS (50 FEATS)")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])
    n_train = len(train_df)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(X_raw, y_true))

    params_lgb = {
        "learning_rate": 0.025,
        "num_leaves": 20,
        "max_depth": 6,
        "min_child_samples": 120,
        "feature_fraction": 0.65,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "reg_alpha": 2.0,
        "reg_lambda": 10.0,
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbose": -1,
        "n_jobs": -1,
    }

    oof_preds = np.zeros(n_train)
    fold_scores = []
    best_iters = []
    t_start = time.time()

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, 1):
        pipe = ExtendedFeaturePipeline(
            include_group1_financial=True,
            include_group2_temporal=False,
            include_group3_ext_interact=False,
            encode_categories=True,
        )
        pipe.fit(X_raw.iloc[tr_idx])
        X_tr = pipe.transform(X_raw.iloc[tr_idx])
        X_va = pipe.transform(X_raw.iloc[va_idx])

        cat_cols = [c for c in CATEGORICAL_COLS if c in X_tr.columns]
        dtrain = lgb.Dataset(X_tr, label=y_true[tr_idx], categorical_feature=cat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_true[va_idx], reference=dtrain, categorical_feature=cat_cols, free_raw_data=False)

        m = lgb.train(
            {**params_lgb, "random_state": 42 + fold_idx},
            dtrain,
            num_boost_round=1500,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        preds_va = m.predict(X_va, num_iteration=m.best_iteration)
        oof_preds[va_idx] = preds_va
        score = float(roc_auc_score(y_true[va_idx], preds_va))
        fold_scores.append(score)
        best_iters.append(int(m.best_iteration))
        print(f"  Fold {fold_idx}: AUC = {score:.5f} (Best Iter: {m.best_iteration})")

    overall_auc = float(roc_auc_score(y_true, oof_preds))
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    mean_iter = int(round(np.mean(best_iters)))
    elapsed = time.time() - t_start

    print("\n[+] 5-Fold Evaluation Complete:")
    print(f"    Overall OOF ROC-AUC: {overall_auc:.5f}")
    print(f"    Fold Mean:          {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f"    Baseline LGB C2:    0.75970 (Mean: 0.75973 +/- 0.00312)")
    print(f"    Delta vs Baseline:  {overall_auc - 0.75970:+.5f}")
    print(f"    Mean Best Iter:     {mean_iter}")
    print(f"    Total Runtime:      {elapsed:.1f}s")

    # Save OOF predictions
    pd.DataFrame({
        ID_COL: train_df[ID_COL].values,
        TARGET_COL: y_true,
        "oof_pred": oof_preds,
    }).to_csv(OOF_LGB_C2_FIN_PATH, index=False)
    print(f"    Saved OOF to: {OOF_LGB_C2_FIN_PATH}")

    # Save JSON summary
    summary = {
        "model": "LightGBM_C2_Financial_50F",
        "n_features": 50,
        "params": params_lgb,
        "overall_oof_auc": overall_auc,
        "mean_fold_auc": mean_auc,
        "std_fold_auc": std_auc,
        "fold_scores": fold_scores,
        "best_iters": best_iters,
        "mean_iter": mean_iter,
        "delta_vs_baseline": overall_auc - 0.75970,
        "runtime_seconds": elapsed,
    }
    with open(STAGE_H7_DIR / "lgb_c2_financial_results.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
