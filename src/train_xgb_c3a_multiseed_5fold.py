"""
Stage H8: Experiment A2 - Multi-Seed Bagging for XGBoost Config 3A.
Averages predictions across 3 distinct seeds [42, 2026, 777] for each fold to reduce tree subsampling variance.
Evaluates whether 5-fold OOF ROC-AUC improves over single-seed baseline (0.76088).
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

STAGE_H8_DIR = PROJECT_ROOT / "experiments" / "stage_h8"
STAGE_H8_DIR.mkdir(parents=True, exist_ok=True)
OOF_XGB_BAGGED_PATH = STAGE_H8_DIR / "oof_xgboost_c3a_bagged_3seed.csv"

def main():
    print("=" * 80)
    print("STAGE H8: 5-FOLD CV OF 3-SEED BAGGED XGBOOST CONFIG 3A")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])
    n_train = len(train_df)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(X_raw, y_true))

    params_base = {
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
        "n_jobs": -1,
    }

    seeds = [42, 2026, 777]
    oof_preds_bagged = np.zeros(n_train)
    fold_scores = []
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

        seed_preds = []
        for s in seeds:
            clf = xgb.XGBClassifier(
                **{**params_base, "random_state": s + fold_idx * 10},
                early_stopping_rounds=50,
            )
            clf.fit(
                X_tr, y_true[tr_idx],
                eval_set=[(X_va, y_true[va_idx])],
                verbose=False,
            )
            p_val = clf.predict_proba(X_va)[:, 1]
            seed_preds.append(p_val)

        bagged_val = np.mean(seed_preds, axis=0)
        oof_preds_bagged[va_idx] = bagged_val
        score = float(roc_auc_score(y_true[va_idx], bagged_val))
        fold_scores.append(score)
        print(f"  Fold {fold_idx}: Bagged AUC = {score:.5f} (Individual seeds: {[round(roc_auc_score(y_true[va_idx], p), 5) for p in seed_preds]})")

    overall_auc = float(roc_auc_score(y_true, oof_preds_bagged))
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    elapsed = time.time() - t_start

    print("\n[+] 5-Fold Bagged Evaluation Complete:")
    print(f"    Overall OOF ROC-AUC: {overall_auc:.5f}")
    print(f"    Fold Mean:          {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f"    Single-Seed 3A Ref: 0.76088 (Mean: 0.76091 +/- 0.00357)")
    print(f"    Delta vs Baseline:  {overall_auc - 0.76088:+.5f}")
    print(f"    Total Runtime:      {elapsed:.1f}s")

    # Save OOF predictions
    pd.DataFrame({
        ID_COL: train_df[ID_COL].values,
        TARGET_COL: y_true,
        "oof_pred": oof_preds_bagged,
    }).to_csv(OOF_XGB_BAGGED_PATH, index=False)
    print(f"    Saved OOF to: {OOF_XGB_BAGGED_PATH}")

    # Save JSON summary
    summary = {
        "model": "XGBoost_Config_3A_Bagged_3Seed",
        "seeds": seeds,
        "overall_oof_auc": overall_auc,
        "mean_fold_auc": mean_auc,
        "std_fold_auc": std_auc,
        "fold_scores": fold_scores,
        "delta_vs_baseline": overall_auc - 0.76088,
        "runtime_seconds": elapsed,
    }
    with open(STAGE_H8_DIR / "xgboost_c3a_bagged_results.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
