"""
Stage H9: Experiment 2 - Non-Tree Diversity Models & Target Encoding Pilots.
Tests:
1. TargetEncoder on high-cardinality categoricals for LightGBM C2.
2. Tabular Neural Network (MLPClassifier) with RobustScaler.
3. Regularized Logistic Regression with Domain Ratios and RobustScaler.
Evaluates Fold 1 AUC and Spearman rank correlation with CatBoost & XGBoost.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data, TARGET_COL, ID_COL
from src.features import BaselineFeaturePipeline, CATEGORICAL_COLS

def run_pilots():
    print("=" * 80)
    print("STAGE H9: DIVERSITY MODEL & TARGET ENCODING PILOTS (FOLD 1)")
    print("=" * 80)

    train_df, _, _ = load_raw_data(verbose=False)
    y_true = train_df[TARGET_COL].values
    X_raw = train_df.drop(columns=[TARGET_COL, ID_COL])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(skf.split(X_raw, y_true))
    y_tr, y_va = y_true[tr_idx], y_true[va_idx]

    # Reference Predictions for Correlation
    p_cb4 = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h2_step3a_oof_CB_Config_4_High_L2_Regularization.csv")["oof_pred"].values[va_idx]
    p_xgb = pd.read_csv(PROJECT_ROOT / "experiments" / "stage_h7" / "oof_xgboost_c3a.csv")["oof_pred"].values[va_idx]

    # -------------------------------------------------------------
    # PILOT 1: Target Encoding on LightGBM C2
    # -------------------------------------------------------------
    print("\n--- Pilot 1: Target Encoding for High-Cardinality Categoricals (LightGBM C2) ---")
    pipe_raw = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=False)
    pipe_raw.fit(X_raw.iloc[tr_idx])
    X_tr_p1 = pipe_raw.transform(X_raw.iloc[tr_idx])
    X_va_p1 = pipe_raw.transform(X_raw.iloc[va_idx])

    high_card_cats = ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
    low_card_cats = [c for c in CATEGORICAL_COLS if c not in high_card_cats and c in X_tr_p1.columns]

    te = TargetEncoder(smooth="auto", cv=5, random_state=42)
    X_tr_te = te.fit_transform(X_tr_p1[high_card_cats], y_tr)
    X_va_te = te.transform(X_va_p1[high_card_cats])

    for idx, col in enumerate(high_card_cats):
        X_tr_p1[f"{col}_TE"] = X_tr_te[:, idx]
        X_va_p1[f"{col}_TE"] = X_va_te[:, idx]
        X_tr_p1 = X_tr_p1.drop(columns=[col])
        X_va_p1 = X_va_p1.drop(columns=[col])

    # Integer encode low-card categoricals
    for col in low_card_cats:
        cats = sorted(X_tr_p1[col].dropna().unique().tolist())
        cat_map = {c: i for i, c in enumerate(cats)}
        X_tr_p1[col] = X_tr_p1[col].map(cat_map).fillna(-1).astype(int)
        X_va_p1[col] = X_va_p1[col].map(cat_map).fillna(-1).astype(int)

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

    dtrain_p1 = lgb.Dataset(X_tr_p1, label=y_tr, categorical_feature=low_card_cats, free_raw_data=False)
    dval_p1 = lgb.Dataset(X_va_p1, label=y_va, reference=dtrain_p1, categorical_feature=low_card_cats, free_raw_data=False)
    m_p1 = lgb.train(params_lgb, dtrain_p1, num_boost_round=1500, valid_sets=[dtrain_p1, dval_p1],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
    preds_p1 = m_p1.predict(X_va_p1, num_iteration=m_p1.best_iteration)
    auc_p1 = float(roc_auc_score(y_va, preds_p1))
    print(f"  Target-Encoded LightGBM C2 Fold 1 AUC: {auc_p1:.5f} (Baseline: 0.75662 | Delta: {auc_p1 - 0.75662:+.5f})")

    # -------------------------------------------------------------
    # PILOT 2: Tabular Neural Network (MLPClassifier)
    # -------------------------------------------------------------
    print("\n--- Pilot 2: Tabular Neural Network (MLPClassifier) ---")
    pipe_num = BaselineFeaturePipeline(add_domain_ratios=True, encode_categories=True)
    pipe_num.fit(X_raw.iloc[tr_idx])
    X_tr_mlp = pipe_num.transform(X_raw.iloc[tr_idx])
    X_va_mlp = pipe_num.transform(X_raw.iloc[va_idx])

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_tr_scaled = scaler.fit_transform(imputer.fit_transform(X_tr_mlp))
    X_va_scaled = scaler.transform(imputer.transform(X_va_mlp))

    # Clip extreme outliers to prevent gradient explosion
    X_tr_scaled = np.clip(X_tr_scaled, -10.0, 10.0)
    X_va_scaled = np.clip(X_va_scaled, -10.0, 10.0)

    t0 = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=0.01,
        batch_size=512,
        learning_rate_init=0.003,
        max_iter=40,
        early_stopping=True,
        random_state=42,
    )
    mlp.fit(X_tr_scaled, y_tr)
    preds_mlp = mlp.predict_proba(X_va_scaled)[:, 1]
    auc_mlp = float(roc_auc_score(y_va, preds_mlp))
    rho_mlp_cb, _ = spearmanr(preds_mlp, p_cb4)
    rho_mlp_xgb, _ = spearmanr(preds_mlp, p_xgb)
    print(f"  MLPClassifier Fold 1 AUC: {auc_mlp:.5f} (Time: {time.time() - t0:.1f}s)")
    print(f"  Correlation vs CB4: {rho_mlp_cb:.4f} | vs XGB: {rho_mlp_xgb:.4f}")

    # Test blending a tiny amount of MLP with baseline
    p_blend_base = 0.50 * p_cb4 + 0.30 * p_xgb + 0.20 * preds_p1
    base_auc = roc_auc_score(y_va, p_blend_base)
    print(f"  Pre-MLP Blend AUC (Fold 1): {base_auc:.5f}")
    for w in [0.01, 0.02, 0.03, 0.05]:
        p_blend_mlp = (1 - w) * p_blend_base + w * preds_mlp
        auc_with_mlp = roc_auc_score(y_va, p_blend_mlp)
        print(f"    Weight {w:.2f} MLP -> AUC: {auc_with_mlp:.5f} ({auc_with_mlp - base_auc:+.5f})")

    # -------------------------------------------------------------
    # PILOT 3: Regularized Logistic Regression
    # -------------------------------------------------------------
    print("\n--- Pilot 3: Regularized Logistic Regression (L2) ---")
    t0 = time.time()
    lr = LogisticRegression(C=0.01, max_iter=200, random_state=42)
    lr.fit(X_tr_scaled, y_tr)
    preds_lr = lr.predict_proba(X_va_scaled)[:, 1]
    auc_lr = float(roc_auc_score(y_va, preds_lr))
    rho_lr_cb, _ = spearmanr(preds_lr, p_cb4)
    print(f"  Logistic Regression Fold 1 AUC: {auc_lr:.5f} (Time: {time.time() - t0:.1f}s)")
    print(f"  Correlation vs CB4: {rho_lr_cb:.4f}")

if __name__ == "__main__":
    run_pilots()
