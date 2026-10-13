"""
Stage H7: 5-Fold Cross-Validation of Upgraded XGBoost Config 3A.
Measures full 5-fold OOF ROC-AUC against verified baseline (0.76015).
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS

STAGE_H7_DIR = PROJECT_ROOT / "experiments" / "stage_h7"
STAGE_H7_DIR.mkdir(parents=True, exist_ok=True)
OOF_XGB_C3A_PATH = STAGE_H7_DIR / "oof_xgboost_c3a.csv"

def main():
    print("=" * 80)
    print("STAGE H7: 5-FOLD CV OF UPGRADED XGBOOST CONFIG 3A")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])
    n_train = len(train_df)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(X_raw, y_true))

    params_xgb_c3a = {
        "n_estimators": 1200,
        "learning_rate": 0.025,
        "max_depth": 6,
        "min_child_weight": 15,
        "subsample": 0.85,
        "colsample_bytree": 0.65,
        "reg_alpha": 3.0,
        "reg_lambda": 15.0,
        "tree_method": "hist",
        "enable_categorical": True,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }

    oof_preds = np.zeros(n_train)
    fold_scores = []
    best_iters = []
    t_start = time.time()

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, 1):
        pipe = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
        pipe.fit(X_raw.iloc[tr_idx])
        X_tr = pipe.transform(X_raw.iloc[tr_idx])
        X_va = pipe.transform(X_raw.iloc[va_idx])

        for col in CATEGORICAL_COLS:
            if col in X_tr.columns:
                cats = sorted(X_tr[col].dropna().unique().tolist())
                dt = pd.CategoricalDtype(categories=cats, ordered=False)
                X_tr[col] = X_tr[col].astype(dt)
                X_va[col] = X_va[col].where(X_va[col].isin(cats), np.nan).astype(dt)

        clf = xgb.XGBClassifier(
            **{**params_xgb_c3a, "random_state": 42 + fold_idx},
            early_stopping_rounds=50,
        )
        clf.fit(
            X_tr, y_true[tr_idx],
            eval_set=[(X_va, y_true[va_idx])],
            verbose=False,
        )
        preds_va = clf.predict_proba(X_va)[:, 1]
        oof_preds[va_idx] = preds_va
        score = float(roc_auc_score(y_true[va_idx], preds_va))
        fold_scores.append(score)
        best_iters.append(int(clf.best_iteration))
        print(f"  Fold {fold_idx}: AUC = {score:.5f} (Best Iter: {clf.best_iteration})")

    overall_auc = float(roc_auc_score(y_true, oof_preds))
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    mean_iter = int(round(np.mean(best_iters)))
    elapsed = time.time() - t_start

    print("\n[+] 5-Fold Evaluation Complete:")
    print(f"    Overall OOF ROC-AUC: {overall_auc:.5f}")
    print(f"    Fold Mean:          {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f"    Baseline Reference: 0.76015 (Mean: 0.76021 +/- 0.00285)")
    print(f"    Delta vs Baseline:  {overall_auc - 0.76015:+.5f}")
    print(f"    Mean Best Iter:     {mean_iter}")
    print(f"    Total Runtime:      {elapsed:.1f}s")

    # Save OOF predictions
    pd.DataFrame({
        ID_COL: train_df[ID_COL].values,
        TARGET_COL: y_true,
        "oof_pred": oof_preds,
    }).to_csv(OOF_XGB_C3A_PATH, index=False)
    print(f"    Saved OOF to: {OOF_XGB_C3A_PATH}")

    # Save JSON summary
    summary = {
        "model": "XGBoost_Config_3A_Regularized",
        "params": params_xgb_c3a,
        "overall_oof_auc": overall_auc,
        "mean_fold_auc": mean_auc,
        "std_fold_auc": std_auc,
        "fold_scores": fold_scores,
        "best_iters": best_iters,
        "mean_iter": mean_iter,
        "delta_vs_baseline": overall_auc - 0.76015,
        "runtime_seconds": elapsed,
    }
    with open(STAGE_H7_DIR / "xgboost_c3a_results.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
