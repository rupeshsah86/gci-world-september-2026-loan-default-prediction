"""
Stage H2 Step 3A: Bounded Hyperparameter Refinement for CatBoost and XGBoost.
- Experiment A: CatBoost on 62 features (Candidate 3 pipeline)
- Experiment B: XGBoost on 46 features (Verified baseline pipeline)
Evaluates bounded configurations across 5 stratified folds with early stopping.
"""

import sys
from pathlib import Path
import json
import time
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
import xgboost as xgb

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import CATEGORICAL_COLS, BaselineFeaturePipeline
from src.features_h1 import ExtendedFeaturePipeline


# =====================================================================
# EXPERIMENT A: CATBOOST TUNING (62 Features)
# =====================================================================
def evaluate_catboost_config(
    cfg_name: str,
    cb_params: Dict[str, Any],
    X_raw: pd.DataFrame,
    y: np.ndarray,
    skf_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, Any], np.ndarray]:
    oof_preds = np.zeros(len(y))
    fold_scores = []
    fold_times = []
    best_iters = []

    start_time = time.time()
    print(f"\n[CatBoost Tuning] Evaluating: {cfg_name}")
    print(f"  Params: {cb_params}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf_folds, 1):
        f_start = time.time()
        X_tr_raw = X_raw.iloc[train_idx]
        y_tr = y[train_idx]
        X_va_raw = X_raw.iloc[val_idx]
        y_va = y[val_idx]

        # 62-feature pipeline fitted strictly on training fold
        pipe = ExtendedFeaturePipeline(
            include_group1_financial=True,
            include_group2_temporal=True,
            include_group3_ext_interact=True,
            encode_categories=False,
        )
        pipe.fit(X_tr_raw)
        X_tr = pipe.transform(X_tr_raw)
        X_va = pipe.transform(X_va_raw)

        cat_cols = [c for c in CATEGORICAL_COLS if c in X_tr.columns]
        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        val_pool = Pool(X_va, y_va, cat_features=cat_cols)

        params = {
            **cb_params,
            "eval_metric": "AUC",
            "loss_function": "Logloss",
            "random_seed": 42 + fold_idx,
            "early_stopping_rounds": 50,
            "thread_count": -1,
            "verbose": False,
            "task_type": "CPU",
        }

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        preds = model.predict_proba(val_pool)[:, 1]
        oof_preds[val_idx] = preds

        auc = float(roc_auc_score(y_va, preds))
        fold_scores.append(auc)
        best_iters.append(int(model.get_best_iteration()))
        f_elapsed = time.time() - f_start
        fold_times.append(f_elapsed)

        print(f"  Fold {fold_idx}: AUC = {auc:.5f} (Best Iter: {model.get_best_iteration()}, Time: {f_elapsed:.1f}s)")

    total_time = time.time() - start_time
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    oof_auc = float(roc_auc_score(y, oof_preds))

    res = {
        "model": "CatBoost",
        "cfg_name": cfg_name,
        "features": 62,
        "params": cb_params,
        "fold_scores": [round(s, 5) for s in fold_scores],
        "mean_auc": round(mean_auc, 5),
        "std_auc": round(std_auc, 5),
        "oof_auc": round(oof_auc, 5),
        "best_iterations": best_iters,
        "mean_best_iteration": round(float(np.mean(best_iters)), 1),
        "total_runtime_seconds": round(total_time, 2),
    }

    print(f"--> {cfg_name}: Mean AUC = {mean_auc:.5f} +/- {std_auc:.5f} | OOF = {oof_auc:.5f} | Time: {total_time:.1f}s")
    return res, oof_preds


# =====================================================================
# EXPERIMENT B: XGBOOST TUNING (46 Features)
# =====================================================================
def evaluate_xgboost_config(
    cfg_name: str,
    xgb_params: Dict[str, Any],
    X_raw: pd.DataFrame,
    y: np.ndarray,
    skf_folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, Any], np.ndarray]:
    oof_preds = np.zeros(len(y))
    fold_scores = []
    fold_times = []
    best_iters = []

    start_time = time.time()
    print(f"\n[XGBoost Tuning] Evaluating: {cfg_name}")
    print(f"  Params: {xgb_params}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf_folds, 1):
        f_start = time.time()
        X_tr_raw = X_raw.iloc[train_idx]
        y_tr = y[train_idx]
        X_va_raw = X_raw.iloc[val_idx]
        y_va = y[val_idx]

        # 46-feature baseline pipeline with native categories
        pipe = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
        pipe.fit(X_tr_raw)
        X_tr = pipe.transform(X_tr_raw)
        X_va = pipe.transform(X_va_raw)

        # Convert to CategoricalDtype
        for col in CATEGORICAL_COLS:
            if col in X_tr.columns:
                train_cats = X_tr[col].dropna().unique().tolist()
                train_cats.sort()
                cat_dtype = pd.CategoricalDtype(categories=train_cats, ordered=False)
                X_tr[col] = X_tr[col].astype(cat_dtype)
                X_va[col] = X_va[col].where(X_va[col].isin(train_cats), np.nan).astype(cat_dtype)

        params = {
            **xgb_params,
            "tree_method": "hist",
            "enable_categorical": True,
            "eval_metric": "auc",
            "early_stopping_rounds": 50,
            "random_state": 42 + fold_idx,
            "n_jobs": -1,
        }

        clf = xgb.XGBClassifier(**params)
        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_va, y_va)],
            verbose=False,
        )

        preds = clf.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = preds

        auc = float(roc_auc_score(y_va, preds))
        fold_scores.append(auc)
        best_iters.append(int(clf.best_iteration))
        f_elapsed = time.time() - f_start
        fold_times.append(f_elapsed)

        print(f"  Fold {fold_idx}: AUC = {auc:.5f} (Best Iter: {clf.best_iteration}, Time: {f_elapsed:.1f}s)")

    total_time = time.time() - start_time
    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    oof_auc = float(roc_auc_score(y, oof_preds))

    res = {
        "model": "XGBoost",
        "cfg_name": cfg_name,
        "features": 46,
        "params": xgb_params,
        "fold_scores": [round(s, 5) for s in fold_scores],
        "mean_auc": round(mean_auc, 5),
        "std_auc": round(std_auc, 5),
        "oof_auc": round(oof_auc, 5),
        "best_iterations": best_iters,
        "mean_best_iteration": round(float(np.mean(best_iters)), 1),
        "total_runtime_seconds": round(total_time, 2),
    }

    print(f"--> {cfg_name}: Mean AUC = {mean_auc:.5f} +/- {std_auc:.5f} | OOF = {oof_auc:.5f} | Time: {total_time:.1f}s")
    return res, oof_preds


def main():
    overall_start = time.time()
    print("=" * 70)
    print("STAGE H2 STEP 3A: BOUNDED HYPERPARAMETER REFINEMENT")
    print("=" * 70)

    train_df, _, _ = load_raw_data(verbose=False)
    y = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf_folds = list(skf.split(X_raw, y))

    # Reference baselines
    ref_cb_mean = 0.76066
    ref_cb_oof = 0.76061
    ref_xgb_mean = 0.75879
    ref_xgb_oof = 0.75874

    # -----------------------------------------------------------------
    # Bounded Search Grids (4 configs each)
    # -----------------------------------------------------------------
    # CatBoost candidates (around depth=6, l2=5.0, lr=0.05, rs=1.0 on 62 features)
    cb_grid = [
        (
            "CB_Config_1_Conservative_Depth5",
            {"iterations": 1200, "learning_rate": 0.05, "depth": 5, "l2_leaf_reg": 5.0, "random_strength": 1.0},
            "Shallower tree depth=5 to prevent overfitting on noisy credit records",
        ),
        (
            "CB_Config_2_Moderate_L2_8",
            {"iterations": 1200, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 8.0, "random_strength": 1.0},
            "Isolates increased L2 penalty (l2=8.0 vs baseline 5.0) with unchanged depth=6 and lr=0.05",
        ),
        (
            "CB_Config_3_RandomStrength_2_LR04",
            {"iterations": 1300, "learning_rate": 0.04, "depth": 6, "l2_leaf_reg": 5.0, "random_strength": 2.0},
            "Higher random_strength=2.0 for tree split diversity with slightly lower lr=0.04",
        ),
        (
            "CB_Config_4_High_L2_Regularization",
            {"iterations": 1300, "learning_rate": 0.04, "depth": 6, "l2_leaf_reg": 10.0, "random_strength": 1.5},
            "Conservative joint regularization (l2=10.0, random_strength=1.5, lr=0.04)",
        ),
    ]

    # XGBoost candidates (around max_depth=6, colsample=0.7, reg_alpha=1.0, reg_lambda=5.0, lr=0.03 on 46 features)
    xgb_grid = [
        (
            "XGB_Config_1_Depth5_HigherMinChild",
            {
                "n_estimators": 1200, "learning_rate": 0.03, "max_depth": 5,
                "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.65,
                "reg_alpha": 1.5, "reg_lambda": 8.0,
            },
            "Depth=5 with min_child_weight=5 and stronger L2 regularization (reg_lambda=8.0)",
        ),
        (
            "XGB_Config_2_Depth6_MCW5_Reg8",
            {
                "n_estimators": 1200, "learning_rate": 0.03, "max_depth": 6,
                "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.65,
                "reg_alpha": 1.5, "reg_lambda": 8.0,
            },
            "Depth=6 with min_child_weight=5 and higher regularization (alpha=1.5, lambda=8.0)",
        ),
        (
            "XGB_Config_3_Depth6_Colsample06_Reg10",
            {
                "n_estimators": 1200, "learning_rate": 0.03, "max_depth": 6,
                "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.60,
                "reg_alpha": 2.0, "reg_lambda": 10.0,
            },
            "Depth=6 with tight feature subsampling (0.60) and high regularization (alpha=2.0, lambda=10.0)",
        ),
        (
            "XGB_Config_4_Depth7_MinChild15",
            {
                "n_estimators": 1300, "learning_rate": 0.025, "max_depth": 7,
                "min_child_weight": 15, "subsample": 0.8, "colsample_bytree": 0.65,
                "reg_alpha": 1.5, "reg_lambda": 6.0,
            },
            "Depth=7 with large min_child_weight=15 and lr=0.025 to prevent small leaf overfitting",
        ),
    ]

    # Execute CatBoost Refinements
    print("\n" + "=" * 70)
    print("STARTING EXPERIMENT A: CATBOOST BOUNDED REFINEMENT (4 CONFIGS)")
    print("=" * 70)
    cb_results = []
    cb_best_res = None
    cb_best_oof = None
    cb_best_name = None

    for name, params, desc in cb_grid:
        res, oof = evaluate_catboost_config(name, params, X_raw, y, skf_folds)
        res["description"] = desc
        res["delta_vs_baseline"] = round(res["mean_auc"] - ref_cb_mean, 5)
        cb_results.append(res)

        # Save OOF predictions for this config
        oof_path = PROJECT_ROOT / "experiments" / f"stage_h2_step3a_oof_{name}.csv"
        pd.DataFrame({
            ID_COL: train_df[ID_COL],
            TARGET_COL: y,
            "oof_pred": oof,
        }).to_csv(oof_path, index=False)
        print(f"  [+] Saved OOF predictions to: {oof_path}")

        if cb_best_res is None or res["mean_auc"] > cb_best_res["mean_auc"]:
            cb_best_res = res
            cb_best_oof = oof
            cb_best_name = name

    # Execute XGBoost Refinements
    print("\n" + "=" * 70)
    print("STARTING EXPERIMENT B: XGBOOST BOUNDED REFINEMENT (4 CONFIGS)")
    print("=" * 70)
    xgb_results = []
    xgb_best_res = None
    xgb_best_oof = None
    xgb_best_name = None

    for name, params, desc in xgb_grid:
        res, oof = evaluate_xgboost_config(name, params, X_raw, y, skf_folds)
        res["description"] = desc
        res["delta_vs_baseline"] = round(res["mean_auc"] - ref_xgb_mean, 5)
        xgb_results.append(res)

        # Save OOF predictions for this config
        oof_path = PROJECT_ROOT / "experiments" / f"stage_h2_step3a_oof_{name}.csv"
        pd.DataFrame({
            ID_COL: train_df[ID_COL],
            TARGET_COL: y,
            "oof_pred": oof,
        }).to_csv(oof_path, index=False)
        print(f"  [+] Saved OOF predictions to: {oof_path}")

        if xgb_best_res is None or res["mean_auc"] > xgb_best_res["mean_auc"]:
            xgb_best_res = res
            xgb_best_oof = oof
            xgb_best_name = name

    # Save JSON summary
    total_time = time.time() - overall_start
    json_path = PROJECT_ROOT / "experiments" / "stage_h2_step3a_results.json"
    full_json = {
        "catboost_tuning": {
            "baseline_reference": {"mean_auc": ref_cb_mean, "oof_auc": ref_cb_oof, "features": 62},
            "candidates": cb_results,
            "best": cb_best_res,
        },
        "xgboost_tuning": {
            "baseline_reference": {"mean_auc": ref_xgb_mean, "oof_auc": ref_xgb_oof, "features": 46},
            "candidates": xgb_results,
            "best": xgb_best_res,
        },
        "total_runtime_seconds": round(total_time, 2),
    }
    with open(json_path, "w") as f:
        json.dump(full_json, f, indent=2)
    print(f"\n[+] Saved full results JSON to: {json_path}")

    # Generate Markdown Report
    report_path = PROJECT_ROOT / "reports" / "stage_h2_step3a_tuning.md"
    report_lines = [
        "# Stage H2 Step 3A: Bounded Hyperparameter Refinement Report",
        "",
        "## 1. Executive Summary",
        f"- **Total Runtime**: {total_time:.1f} seconds (~{total_time/60:.1f} minutes) on Apple Silicon M2.",
        "- **Validation Setup**: 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), zero data leakage.",
        f"- **CatBoost Baseline Reference**: Mean = **{ref_cb_mean:.5f}**, OOF = **{ref_cb_oof:.5f}** (62 features, Candidate 3 pipeline)",
        f"- **XGBoost Baseline Reference**: Mean = **{ref_xgb_mean:.5f}**, OOF = **{ref_xgb_oof:.5f}** (46 features, Baseline pipeline)",
        "",
        "## 2. Experiment A: CatBoost Refinement Results (62 Features)",
        "| Configuration Name | Key Hyperparameters | Fold Scores (1 to 5) | Mean ROC-AUC (+/- Std) | OOF ROC-AUC | Mean Best Iter | Delta vs Baseline | Runtime |",
        "| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |",
        f"| **Verified Reference** | `depth=6, l2=5.0, lr=0.05, rs=1.0` | `[0.75913, 0.76096, 0.76555, 0.75706, 0.76058]` | **{ref_cb_mean:.5f} +/- 0.00281** | **{ref_cb_oof:.5f}** | 914.4 | Baseline | ~368s |",
    ]

    for r in cb_results:
        p = r["params"]
        p_str = f"depth={p['depth']}, l2={p['l2_leaf_reg']}, lr={p['learning_rate']}, rs={p['random_strength']}"
        folds_str = str(r["fold_scores"])
        report_lines.append(
            f"| **{r['cfg_name']}** | `{p_str}` | `{folds_str}` | **{r['mean_auc']:.5f} +/- {r['std_auc']:.5f}** | **{r['oof_auc']:.5f}** | {r['mean_best_iteration']} | **{r['delta_vs_baseline']:+.5f}** | {r['total_runtime_seconds']:.1f}s |"
        )

    report_lines.extend([
        "",
        "## 3. Experiment B: XGBoost Refinement Results (46 Features)",
        "| Configuration Name | Key Hyperparameters | Fold Scores (1 to 5) | Mean ROC-AUC (+/- Std) | OOF ROC-AUC | Mean Best Iter | Delta vs Baseline | Runtime |",
        "| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |",
        f"| **Verified Reference** | `depth=6, mcw=1, colsample=0.70, a=1.0, l=5.0, lr=0.03` | `[0.75626, 0.75933, 0.76380, 0.75510, 0.75947]` | **{ref_xgb_mean:.5f} +/- 0.00303** | **{ref_xgb_oof:.5f}** | 476.4 | Baseline | ~65s |",
    ])

    for r in xgb_results:
        p = r["params"]
        p_str = f"depth={p['max_depth']}, mcw={p['min_child_weight']}, colsample={p['colsample_bytree']}, a={p['reg_alpha']}, l={p['reg_lambda']}"
        folds_str = str(r["fold_scores"])
        report_lines.append(
            f"| **{r['cfg_name']}** | `{p_str}` | `{folds_str}` | **{r['mean_auc']:.5f} +/- {r['std_auc']:.5f}** | **{r['oof_auc']:.5f}** | {r['mean_best_iteration']} | **{r['delta_vs_baseline']:+.5f}** | {r['total_runtime_seconds']:.1f}s |"
        )

    report_lines.extend([
        "",
        "## 4. Key Findings & Discussion",
        f"- **Best CatBoost Candidate**: `{cb_best_name}` -> Mean AUC = **{cb_best_res['mean_auc']:.5f}** (Delta: {cb_best_res['delta_vs_baseline']:+.5f}), OOF = **{cb_best_res['oof_auc']:.5f}**.",
        f"- **Best XGBoost Candidate**: `{xgb_best_name}` -> Mean AUC = **{xgb_best_res['mean_auc']:.5f}** (Delta: {xgb_best_res['delta_vs_baseline']:+.5f}), OOF = **{xgb_best_res['oof_auc']:.5f}**.",
        "",
        "### Regularization and Architecture Impact:",
        "- In tabular default prediction, constraining leaf capacity and applying stronger L2 penalties stabilizes fold variance.",
        "- Because multiple configurations were compared on the exact same folds, the top observed parameters should be interpreted as exploratory empirical candidates rather than unbiased final scores.",
        "",
        "## 5. Summary & Next Steps",
        "- All original competition files remain untouched.",
        "- No test predictions, final submission file, or uploads generated.",
        "- Awaiting user review before proceeding to seed stability, ensemble optimization, or final inference.",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"[+] Saved comprehensive tuning report to: {report_path}")


if __name__ == "__main__":
    main()

