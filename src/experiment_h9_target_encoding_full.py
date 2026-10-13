"""
Stage H9: Experiment 3 - Full 5-Fold Cross-Validation of Target-Encoded LightGBM C2.
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
import lightgbm as lgb
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS

def run_target_encoded_lgb():
    print("=" * 80)
    print("STAGE H9: 5-FOLD CV OF TARGET-ENCODED LIGHTGBM C2")
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
        "random_state": 43,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df), dtype=float)
    fold_aucs = []

    t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_proc, y_true), 1):
        X_tr = X_proc.iloc[tr_idx].copy()
        X_va = X_proc.iloc[va_idx].copy()
        y_tr = y_true[tr_idx]
        y_va = y_true[va_idx]

        # Leakage-safe Target Encoding: fit on training fold with internal CV, transform val fold
        te = TargetEncoder(smooth="auto", cv=5, random_state=42)
        X_tr_te = te.fit_transform(X_tr[high_card_cats], y_tr)
        X_va_te = te.transform(X_va[high_card_cats])

        for idx, col in enumerate(high_card_cats):
            X_tr[f"{col}_TE"] = X_tr_te[:, idx]
            X_va[f"{col}_TE"] = X_va_te[:, idx]
            X_tr.drop(columns=[col], inplace=True)
            X_va.drop(columns=[col], inplace=True)

        # Categorical integer mapping for remaining low-cardinality categories
        for col in low_card_cats:
            cats = sorted(X_tr[col].dropna().unique().tolist())
            cat_map = {c: i for i, c in enumerate(cats)}
            X_tr[col] = X_tr[col].map(cat_map).fillna(-1).astype(int)
            X_va[col] = X_va[col].map(cat_map).fillna(-1).astype(int)

        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=low_card_cats, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, categorical_feature=low_card_cats, free_raw_data=False)

        model = lgb.train(
            params_lgb,
            dtrain,
            num_boost_round=1500,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        p_va = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[va_idx] = p_va
        auc_f = float(roc_auc_score(y_va, p_va))
        fold_aucs.append(auc_f)
        print(f"  Fold {fold}: AUC = {auc_f:.5f} (Best Iter: {model.best_iteration})")

    overall_auc = float(roc_auc_score(y_true, oof_preds))
    mean_fold_auc = float(np.mean(fold_aucs))
    elapsed = time.time() - t0

    print("-" * 60)
    print(f"Target-Encoded LightGBM C2 Overall OOF AUC: {overall_auc:.5f}")
    print(f"Mean Fold AUC: {mean_fold_auc:.5f} (+/- {np.std(fold_aucs):.5f})")
    print(f"Baseline LightGBM C2 Overall OOF AUC: 0.75970")
    print(f"Delta: {overall_auc - 0.75970:+.5f} (Elapsed: {elapsed:.1f}s)")
    print("-" * 60)

    # Save OOF CSV
    out_dir = PROJECT_ROOT / "experiments" / "stage_h9"
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame({
        ID_COL: sk_ids,
        TARGET_COL: y_true,
        "oof_pred": oof_preds
    })
    oof_df.to_csv(out_dir / "oof_lightgbm_c2_target_encoded.csv", index=False)

    # Now evaluate ensemble impact with Candidate V2.3 components
    p_cb4 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_4_High_L2_Regularization.csv")["oof_pred"].values
    p_cb3 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_3_RandomStrength_2_LR04.csv")["oof_pred"].values
    p_xgb3a_single = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h7" / "oof_xgboost_c3a.csv")["oof_pred"].values
    p_lgb_orig = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h6" / "oof_lightgbm_c2.csv")["oof_pred"].values

    def to_rank(p):
        return (rankdata(p) - 0.5) / len(p)

    r_cb4 = to_rank(p_cb4)
    r_cb3 = to_rank(p_cb3)
    r_xgb = to_rank(p_xgb3a_single)
    r_lgb_orig = to_rank(p_lgb_orig)
    r_lgb_te = to_rank(oof_preds)

    # V2.3 weights: CB4: 0.27, CB3: 0.23, XGB3A: 0.30, LGB: 0.20
    v23_orig_rank_blend = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb + 0.20 * r_lgb_orig
    auc_v23_orig = float(roc_auc_score(y_true, v23_orig_rank_blend))

    v23_te_rank_blend = 0.27 * r_cb4 + 0.23 * r_cb3 + 0.30 * r_xgb + 0.20 * r_lgb_te
    auc_v23_te = float(roc_auc_score(y_true, v23_te_rank_blend))

    print(f"\nCandidate V2.3 with Original LightGBM C2 Rank Blend AUC: {auc_v23_orig:.5f}")
    print(f"Candidate V2.3 with Target-Encoded LightGBM C2 Rank Blend AUC: {auc_v23_te:.5f}")
    print(f"Ensemble Lift: {auc_v23_te - auc_v23_orig:+.5f}")

    results = {
        "lgb_te_overall_oof": overall_auc,
        "lgb_te_fold_aucs": fold_aucs,
        "lgb_te_mean_fold": mean_fold_auc,
        "baseline_lgb_oof": 0.75970,
        "lgb_delta": overall_auc - 0.75970,
        "ensemble_v23_orig_auc": auc_v23_orig,
        "ensemble_v23_with_te_auc": auc_v23_te,
        "ensemble_lift": auc_v23_te - auc_v23_orig,
        "elapsed_seconds": elapsed,
    }

    with open(out_dir / "target_encoded_lgb_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_target_encoded_lgb()
