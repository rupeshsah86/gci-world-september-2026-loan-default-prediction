"""
Stage H10: Full 5-Fold Cross-Validation of Promising Model Configurations.
Evaluates:
1. LightGBM C3_TE: Target-encoded + tighter regularization (reg_alpha=3.0, reg_lambda=15.0)
2. XGBoost C3B_TE: Target-encoded + higher leaf sample threshold (min_child_weight=20)
3. CatBoost C1_TE: Target-encoded (depth=6, l2=10.0, lr=0.04)

All models evaluated under strictly identical StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
Leakage-safe: internal 5-fold CV for TargetEncoder on training folds; pure transform on validation folds.
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
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS

def main():
    t_start = time.time()
    print("=" * 80)
    print("STAGE H10: FULL 5-FOLD CV OF UPGRADED COMPONENT MODELS")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    sk_ids = train_df[ID_COL].values
    n = len(train_df)
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])

    out_dir = PROJECT_ROOT / "experiments" / "stage_h10"
    out_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    high_card = ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
    low_card = [c for c in CATEGORICAL_COLS if c not in high_card]

    # Pre-generate feature matrices per fold to ensure identical feature inputs
    print("\n[Step 1/4] Preparing Leakage-Safe Target-Encoded Folds...")
    folds_data = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_raw, y_true), 1):
        pipe = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
        pipe.fit(X_raw.iloc[tr_idx])
        X_tr = pipe.transform(X_raw.iloc[tr_idx])
        X_va = pipe.transform(X_raw.iloc[va_idx])
        y_tr = y_true[tr_idx]
        y_va = y_true[va_idx]

        # Target encoding with internal 5-fold CV
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        te = TargetEncoder(smooth="auto", cv=skf_inner)
        X_tr_te = te.fit_transform(X_tr[high_card], y_tr)
        X_va_te = te.transform(X_va[high_card])

        for i, c in enumerate(high_card):
            X_tr[f"{c}_TE"] = X_tr_te[:, i]
            X_va[f"{c}_TE"] = X_va_te[:, i]
            X_tr.drop(columns=[c], inplace=True)
            X_va.drop(columns=[c], inplace=True)

        # LGB/CB format (integer low card)
        X_tr_lgb = X_tr.copy()
        X_va_lgb = X_va.copy()
        for col in low_card:
            cats = sorted(X_tr_lgb[col].dropna().unique().tolist())
            cat_map = {c: idx for idx, c in enumerate(cats)}
            X_tr_lgb[col] = X_tr_lgb[col].map(cat_map).fillna(-1).astype(int)
            X_va_lgb[col] = X_va_lgb[col].map(cat_map).fillna(-1).astype(int)

        # XGB format (categorical dtype)
        X_tr_xgb = X_tr.copy()
        X_va_xgb = X_va.copy()
        for col in low_card:
            cats = sorted(X_tr_xgb[col].dropna().unique().tolist())
            dt = pd.CategoricalDtype(categories=cats, ordered=False)
            X_tr_xgb[col] = X_tr_xgb[col].astype(dt)
            X_va_xgb[col] = X_va_xgb[col].where(X_va_xgb[col].isin(cats), np.nan).astype(dt)

        folds_data.append({
            "tr_idx": tr_idx,
            "va_idx": va_idx,
            "y_tr": y_tr,
            "y_va": y_va,
            "X_tr_lgb": X_tr_lgb,
            "X_va_lgb": X_va_lgb,
            "X_tr_xgb": X_tr_xgb,
            "X_va_xgb": X_va_xgb,
            "low_card": low_card
        })

    print(f"  [+] Folds prepared successfully. All 5 splits isolated.")

    # 1. Full 5-Fold CV of LightGBM C3_TE
    print("\n[Step 2/4] Running 5-Fold CV for LightGBM C3_TE (reg_alpha=3.0, reg_lambda=15.0)...")
    params_lgb_c3 = {
        "learning_rate": 0.025,
        "num_leaves": 20,
        "max_depth": 6,
        "min_child_samples": 120,
        "feature_fraction": 0.65,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "reg_alpha": 3.0,
        "reg_lambda": 15.0,
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 43,
    }

    oof_lgb_c3 = np.zeros(n, dtype=float)
    fold_aucs_lgb = []
    best_iters_lgb = []

    for fold, fd in enumerate(folds_data, 1):
        dtr = lgb.Dataset(fd["X_tr_lgb"], label=fd["y_tr"], categorical_feature=fd["low_card"], free_raw_data=False)
        dva = lgb.Dataset(fd["X_va_lgb"], label=fd["y_va"], reference=dtr, categorical_feature=fd["low_card"], free_raw_data=False)
        model = lgb.train(
            params_lgb_c3,
            dtr,
            num_boost_round=1500,
            valid_sets=[dtr, dva],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        p_va = model.predict(fd["X_va_lgb"], num_iteration=model.best_iteration)
        oof_lgb_c3[fd["va_idx"]] = p_va
        auc_f = float(roc_auc_score(fd["y_va"], p_va))
        fold_aucs_lgb.append(auc_f)
        best_iters_lgb.append(model.best_iteration)
        print(f"  Fold {fold}: AUC = {auc_f:.5f} (Best Iter: {model.best_iteration})")

    auc_lgb_overall = float(roc_auc_score(y_true, oof_lgb_c3))
    print(f"  --> LightGBM C3_TE Overall OOF AUC: {auc_lgb_overall:.5f} (Mean: {np.mean(fold_aucs_lgb):.5f} +/- {np.std(fold_aucs_lgb):.5f})")
    print(f"  --> Delta vs LightGBM C2_TE (0.76183): {auc_lgb_overall - 0.76183:+.5f}")

    # 2. Full 5-Fold CV of XGBoost C3B_TE (min_child_weight=20)
    print("\n[Step 3/4] Running 5-Fold CV for XGBoost C3B_TE (mcw=20, reg_alpha=3.0, reg_lambda=15.0)...")
    params_xgb_c3b = {
        "n_estimators": 1400,
        "learning_rate": 0.025,
        "max_depth": 6,
        "min_child_weight": 20,
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

    oof_xgb_c3b = np.zeros(n, dtype=float)
    fold_aucs_xgb = []
    best_iters_xgb = []

    for fold, fd in enumerate(folds_data, 1):
        clf = xgb.XGBClassifier(**params_xgb_c3b)
        clf.fit(fd["X_tr_xgb"], fd["y_tr"], eval_set=[(fd["X_va_xgb"], fd["y_va"])], verbose=False)
        p_va = clf.predict_proba(fd["X_va_xgb"])[:, 1]
        oof_xgb_c3b[fd["va_idx"]] = p_va
        auc_f = float(roc_auc_score(fd["y_va"], p_va))
        fold_aucs_xgb.append(auc_f)
        best_iters_xgb.append(clf.best_iteration)
        print(f"  Fold {fold}: AUC = {auc_f:.5f} (Best Iter: {clf.best_iteration})")

    auc_xgb_overall = float(roc_auc_score(y_true, oof_xgb_c3b))
    print(f"  --> XGBoost C3B_TE Overall OOF AUC: {auc_xgb_overall:.5f} (Mean: {np.mean(fold_aucs_xgb):.5f} +/- {np.std(fold_aucs_xgb):.5f})")
    print(f"  --> Delta vs XGBoost C3A_TE (0.76166): {auc_xgb_overall - 0.76166:+.5f}")

    # 3. Full 5-Fold CV of CatBoost C1_TE
    print("\n[Step 4/4] Running 5-Fold CV for CatBoost C1_TE (depth=6, l2=10.0, lr=0.04)...")
    oof_cb_te = np.zeros(n, dtype=float)
    fold_aucs_cb = []
    best_iters_cb = []

    for fold, fd in enumerate(folds_data, 1):
        cb_clf = CatBoostClassifier(
            depth=6,
            l2_leaf_reg=10.0,
            learning_rate=0.04,
            iterations=1600,
            eval_metric="AUC",
            random_seed=42,
            early_stopping_rounds=50,
            verbose=False,
            thread_count=-1
        )
        cb_clf.fit(fd["X_tr_lgb"], fd["y_tr"], eval_set=(fd["X_va_lgb"], fd["y_va"]), verbose=False)
        p_va = cb_clf.predict_proba(fd["X_va_lgb"])[:, 1]
        oof_cb_te[fd["va_idx"]] = p_va
        auc_f = float(roc_auc_score(fd["y_va"], p_va))
        fold_aucs_cb.append(auc_f)
        best_iters_cb.append(cb_clf.get_best_iteration())
        print(f"  Fold {fold}: AUC = {auc_f:.5f} (Best Iter: {cb_clf.get_best_iteration()})")

    auc_cb_overall = float(roc_auc_score(y_true, oof_cb_te))
    print(f"  --> CatBoost C1_TE Overall OOF AUC: {auc_cb_overall:.5f} (Mean: {np.mean(fold_aucs_cb):.5f} +/- {np.std(fold_aucs_cb):.5f})")
    print(f"  --> Delta vs Native CatBoost Config 4 (0.76080): {auc_cb_overall - 0.76080:+.5f}")

    # Save OOF Predictions
    print("\nSaving 5-fold OOF predictions to disk...")
    pd.DataFrame({ID_COL: sk_ids, TARGET_COL: y_true, "oof_pred": oof_lgb_c3}).to_csv(out_dir / "oof_lightgbm_c3_te.csv", index=False)
    pd.DataFrame({ID_COL: sk_ids, TARGET_COL: y_true, "oof_pred": oof_xgb_c3b}).to_csv(out_dir / "oof_xgboost_c3b_te.csv", index=False)
    pd.DataFrame({ID_COL: sk_ids, TARGET_COL: y_true, "oof_pred": oof_cb_te}).to_csv(out_dir / "oof_catboost_c1_te.csv", index=False)

    summary = {
        "lightgbm_c3_te": {
            "overall_oof_auc": auc_lgb_overall,
            "mean_fold_auc": float(np.mean(fold_aucs_lgb)),
            "std_fold_auc": float(np.std(fold_aucs_lgb)),
            "fold_aucs": fold_aucs_lgb,
            "best_iters": best_iters_lgb,
            "delta_vs_lgb_c2_te": auc_lgb_overall - 0.76183,
        },
        "xgboost_c3b_te": {
            "overall_oof_auc": auc_xgb_overall,
            "mean_fold_auc": float(np.mean(fold_aucs_xgb)),
            "std_fold_auc": float(np.std(fold_aucs_xgb)),
            "fold_aucs": fold_aucs_xgb,
            "best_iters": best_iters_xgb,
            "delta_vs_xgb_c3a_te": auc_xgb_overall - 0.76166,
        },
        "catboost_c1_te": {
            "overall_oof_auc": auc_cb_overall,
            "mean_fold_auc": float(np.mean(fold_aucs_cb)),
            "std_fold_auc": float(np.std(fold_aucs_cb)),
            "fold_aucs": fold_aucs_cb,
            "best_iters": best_iters_cb,
            "delta_vs_cb4_native": auc_cb_overall - 0.76080,
        },
        "elapsed_total_seconds": time.time() - t_start,
    }

    with open(out_dir / "stage_h10_full_cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] Full 5-Fold CV Completed in {time.time() - t_start:.1f}s.")

if __name__ == "__main__":
    main()
