"""
Stage H9: Experiment 5 - Full 5-Fold Cross-Validation of Target-Encoded XGBoost Config 3A.
Tests whether replacing raw high-cardinality categorical variables
(OCCUPATION_TYPE, ORGANIZATION_TYPE) with smooth out-of-fold target encodings
improves 5-fold OOF AUC and enhances ensemble performance.
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS

def run_target_encoded_xgb():
    print("=" * 80)
    print("STAGE H9: 5-FOLD CV OF TARGET-ENCODED XGBOOST CONFIG 3A")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    sk_ids = train_df[ID_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])

    pipe_raw = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
    pipe_raw.fit(X_raw)
    X_proc = pipe_raw.transform(X_raw)

    high_card_cats = ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
    low_card_cats = [c for c in CATEGORICAL_COLS if c not in high_card_cats and c in X_proc.columns]

    params_xgb = {
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
        "early_stopping_rounds": 50,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df), dtype=float)
    fold_aucs = []
    best_iters = []

    t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_proc, y_true), 1):
        X_tr = X_proc.iloc[tr_idx].copy()
        X_va = X_proc.iloc[va_idx].copy()
        y_tr = y_true[tr_idx]
        y_va = y_true[va_idx]

        # Leakage-safe Target Encoding
        te = TargetEncoder(smooth="auto", cv=5, random_state=42)
        X_tr_te = te.fit_transform(X_tr[high_card_cats], y_tr)
        X_va_te = te.transform(X_va[high_card_cats])

        for idx, col in enumerate(high_card_cats):
            X_tr[f"{col}_TE"] = X_tr_te[:, idx]
            X_va[f"{col}_TE"] = X_va_te[:, idx]
            X_tr.drop(columns=[col], inplace=True)
            X_va.drop(columns=[col], inplace=True)

        for col in low_card_cats:
            cats = sorted(X_tr[col].dropna().unique().tolist())
            dt = pd.CategoricalDtype(categories=cats, ordered=False)
            X_tr[col] = X_tr[col].astype(dt)
            X_va[col] = X_va[col].where(X_va[col].isin(cats), np.nan).astype(dt)

        clf = xgb.XGBClassifier(**params_xgb)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        p_va = clf.predict_proba(X_va)[:, 1]
        oof_preds[va_idx] = p_va
        auc_f = float(roc_auc_score(y_va, p_va))
        fold_aucs.append(auc_f)
        best_iters.append(clf.best_iteration)
        print(f"  Fold {fold}: AUC = {auc_f:.5f} (Best Iter: {clf.best_iteration})")

    overall_auc = float(roc_auc_score(y_true, oof_preds))
    mean_fold_auc = float(np.mean(fold_aucs))
    elapsed = time.time() - t0

    print("-" * 60)
    print(f"Target-Encoded XGBoost Config 3A Overall OOF AUC: {overall_auc:.5f}")
    print(f"Mean Fold AUC: {mean_fold_auc:.5f} (+/- {np.std(fold_aucs):.5f})")
    print(f"Baseline XGBoost Config 3A Overall OOF AUC: 0.76088")
    print(f"Delta: {overall_auc - 0.76088:+.5f} (Elapsed: {elapsed:.1f}s)")
    print("-" * 60)

    out_dir = PROJECT_ROOT / "experiments" / "stage_h9"
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame({
        ID_COL: sk_ids,
        TARGET_COL: y_true,
        "oof_pred": oof_preds
    })
    oof_df.to_csv(out_dir / "oof_xgboost_c3a_target_encoded.csv", index=False)

    results = {
        "xgb_te_overall_oof": overall_auc,
        "xgb_te_fold_aucs": fold_aucs,
        "xgb_te_mean_fold": mean_fold_auc,
        "xgb_te_best_iters": best_iters,
        "baseline_xgb_oof": 0.76088,
        "xgb_delta": overall_auc - 0.76088,
        "elapsed_seconds": elapsed,
    }
    with open(out_dir / "target_encoded_xgb_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_target_encoded_xgb()
