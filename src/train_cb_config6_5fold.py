"""
Stage H7: 5-Fold Cross-Validation of Upgraded CatBoost Config 6.
Evaluates high L2 regularization (20.0) + random strength (2.0) on 62 features against Config 4 (0.76080).
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features_h1 import ExtendedFeaturePipeline
from src.features import CATEGORICAL_COLS

STAGE_H7_DIR = PROJECT_ROOT / "experiments" / "stage_h7"
STAGE_H7_DIR.mkdir(parents=True, exist_ok=True)
OOF_CB_CONFIG6_PATH = STAGE_H7_DIR / "oof_catboost_config6.csv"

def main():
    print("=" * 80)
    print("STAGE H7: 5-FOLD CV OF UPGRADED CATBOOST CONFIG 6")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])
    n_train = len(train_df)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(X_raw, y_true))

    params_cb6 = {
        "iterations": 2000,
        "learning_rate": 0.030,
        "depth": 6,
        "l2_leaf_reg": 20.0,
        "random_strength": 2.0,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }

    oof_preds = np.zeros(n_train)
    fold_scores = []
    best_iters = []
    t_start = time.time()

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, 1):
        pipe = ExtendedFeaturePipeline(
            include_group1_financial=True,
            include_group2_temporal=True,
            include_group3_ext_interact=True,
            encode_categories=False,
        )
        pipe.fit(X_raw.iloc[tr_idx])
        X_tr = pipe.transform(X_raw.iloc[tr_idx])
        X_va = pipe.transform(X_raw.iloc[va_idx])
        cat_cols = [c for c in CATEGORICAL_COLS if c in X_tr.columns]

        pool_tr = Pool(X_tr, y_true[tr_idx], cat_features=cat_cols)
        pool_va = Pool(X_va, y_true[va_idx], cat_features=cat_cols)

        clf = CatBoostClassifier(
            **{**params_cb6, "random_seed": 42 + fold_idx},
            early_stopping_rounds=60,
        )
        clf.fit(pool_tr, eval_set=pool_va, verbose=False)
        preds_va = clf.predict_proba(pool_va)[:, 1]
        oof_preds[va_idx] = preds_va
        score = float(roc_auc_score(y_true[va_idx], preds_va))
        fold_scores.append(score)
        best_iters.append(int(clf.get_best_iteration()))
        print(f"  Fold {fold_idx}: AUC = {score:.5f} (Best Iter: {clf.get_best_iteration()})")

    overall_auc = float(roc_auc_score(y_true, oof_preds))
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    mean_iter = int(round(np.mean(best_iters)))
    elapsed = time.time() - t_start

    print("\n[+] 5-Fold Evaluation Complete:")
    print(f"    Overall OOF ROC-AUC: {overall_auc:.5f}")
    print(f"    Fold Mean:          {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f"    Baseline Reference: 0.76080 (Mean: 0.76084 +/- 0.00265)")
    print(f"    Delta vs Baseline:  {overall_auc - 0.76080:+.5f}")
    print(f"    Mean Best Iter:     {mean_iter}")
    print(f"    Total Runtime:      {elapsed:.1f}s")

    # Save OOF predictions
    pd.DataFrame({
        ID_COL: train_df[ID_COL].values,
        TARGET_COL: y_true,
        "oof_pred": oof_preds,
    }).to_csv(OOF_CB_CONFIG6_PATH, index=False)
    print(f"    Saved OOF to: {OOF_CB_CONFIG6_PATH}")

    # Save JSON summary
    summary = {
        "model": "CatBoost_Config_6_Regularized",
        "params": params_cb6,
        "overall_oof_auc": overall_auc,
        "mean_fold_auc": mean_auc,
        "std_fold_auc": std_auc,
        "fold_scores": fold_scores,
        "best_iters": best_iters,
        "mean_iter": mean_iter,
        "delta_vs_baseline": overall_auc - 0.76080,
        "runtime_seconds": elapsed,
    }
    with open(STAGE_H7_DIR / "catboost_config6_results.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
