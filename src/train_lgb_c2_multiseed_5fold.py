"""
Stage H8: Experiment A1 - Multi-Seed Bagging for LightGBM C2.
Averages predictions across 3 distinct seeds [42, 2026, 777] for each fold to reduce tree subsampling variance.
Evaluates whether 5-fold OOF ROC-AUC improves over single-seed baseline (0.75970).
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
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS

STAGE_H8_DIR = PROJECT_ROOT / "experiments" / "stage_h8"
STAGE_H8_DIR.mkdir(parents=True, exist_ok=True)
OOF_LGB_BAGGED_PATH = STAGE_H8_DIR / "oof_lightgbm_c2_bagged_3seed.csv"

def main():
    print("=" * 80)
    print("STAGE H8: 5-FOLD CV OF 3-SEED BAGGED LIGHTGBM C2")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])
    n_train = len(train_df)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(X_raw, y_true))

    params_base = {
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

    seeds = [42, 2026, 777]
    oof_preds_bagged = np.zeros(n_train)
    fold_scores = []
    t_start = time.time()

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, 1):
        pipe = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=True)
        pipe.fit(X_raw.iloc[tr_idx])
        X_tr = pipe.transform(X_raw.iloc[tr_idx])
        X_va = pipe.transform(X_raw.iloc[va_idx])
        cat_cols = [c for c in CATEGORICAL_COLS if c in pipe.feature_names]

        dtrain = lgb.Dataset(X_tr, label=y_true[tr_idx], categorical_feature=cat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_true[va_idx], reference=dtrain, categorical_feature=cat_cols, free_raw_data=False)

        seed_preds = []
        for s in seeds:
            m = lgb.train(
                {**params_base, "random_state": s + fold_idx * 10},
                dtrain,
                num_boost_round=1500,
                valid_sets=[dtrain, dval],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            p_val = m.predict(X_va, num_iteration=m.best_iteration)
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
    print(f"    Single-Seed C2 Ref: 0.75970 (Mean: 0.75973 +/- 0.00312)")
    print(f"    Delta vs Baseline:  {overall_auc - 0.75970:+.5f}")
    print(f"    Total Runtime:      {elapsed:.1f}s")

    # Save OOF predictions
    pd.DataFrame({
        ID_COL: train_df[ID_COL].values,
        TARGET_COL: y_true,
        "oof_pred": oof_preds_bagged,
    }).to_csv(OOF_LGB_BAGGED_PATH, index=False)
    print(f"    Saved OOF to: {OOF_LGB_BAGGED_PATH}")

    # Save JSON summary
    summary = {
        "model": "LightGBM_C2_Bagged_3Seed",
        "seeds": seeds,
        "overall_oof_auc": overall_auc,
        "mean_fold_auc": mean_auc,
        "std_fold_auc": std_auc,
        "fold_scores": fold_scores,
        "delta_vs_baseline": overall_auc - 0.75970,
        "runtime_seconds": elapsed,
    }
    with open(STAGE_H8_DIR / "lgb_c2_bagged_results.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
