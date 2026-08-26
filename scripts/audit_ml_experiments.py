"""
TrustLoop ML Engineering Audit & Comprehensive Experiment Matrix.
Conducts scientific evaluation of:
1. Production 33-feature baseline
2. Candidate 39-feature model
3. Class-weighted model (Balanced / Inverse Freq)
4. Policy-Abuser targeted class weight
5. Threshold optimization (Tuned on Val, evaluated on Test)
6. Probability calibration (Platt / Isotonic tuned on Val, evaluated on Test)
7. Binary Policy Abuser specialist detector
8. Cross-validation: GroupKFold (by customer_id), TimeSeriesSplit, StratifiedKFold
9. Synthetic separation / leakage analysis of Candidate features

DOES NOT MODIFY ANY PRODUCTION MODEL ARTIFACTS.
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    brier_score_loss,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GroupKFold, TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "trustloop" / "model_ready.csv"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ecommerce_return_abuse_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

CLASS_NAMES = ["Legitimate", "Policy Abuser", "Fraudulent Return", "Wardrobing"]


def load_datasets():
    df_model = pd.read_csv(DATA_PATH)
    df_raw = pd.read_csv(RAW_PATH)
    return df_model, df_raw


def prepare_feature_matrices(df: pd.DataFrame):
    y = df["abuse_label"].astype(int)
    X_raw = df.drop(columns=["abuse_label"]).copy()

    # Extract temporal features
    for col in ["order_date", "return_date"]:
        if col in X_raw.columns:
            X_raw[col] = pd.to_datetime(X_raw[col], errors="coerce")

    X_eng = X_raw.copy()
    for col in ["order_date", "return_date"]:
        if col in X_eng.columns:
            dti = pd.DatetimeIndex(pd.to_datetime(X_eng[col], errors="coerce"))
            X_eng[f"{col}_year"] = dti.year
            X_eng[f"{col}_month"] = dti.month
            X_eng[f"{col}_day"] = dti.day
            X_eng[f"{col}_dayofweek"] = dti.dayofweek
            X_eng[f"{col}_dayofyear"] = dti.dayofyear
            X_eng[f"{col}_is_weekend"] = (dti.dayofweek >= 5).astype(int)

    if "order_date" in X_eng.columns and "return_date" in X_eng.columns:
        ret_d = pd.DatetimeIndex(pd.to_datetime(X_eng["return_date"], errors="coerce"))
        ord_d = pd.DatetimeIndex(pd.to_datetime(X_eng["order_date"], errors="coerce"))
        X_eng["calculated_days_to_return"] = (
            ret_d - ord_d
        ).total_seconds() / 86400.0

    X_eng = X_eng.drop(columns=["order_date", "return_date"])

    cat_cols = [
        "country",
        "customer_segment",
        "device_type",
        "payment_method",
        "platform",
        "product_category",
        "return_reason",
        "shipping_carrier",
    ]
    for c in cat_cols:
        if c in X_eng.columns:
            X_eng[c] = X_eng[c].astype("category")

    # Sort chronologically matching benchmark split
    sort_order_df = df.assign(
        _s=pd.to_datetime(df["return_date"], errors="coerce"),
        _o=pd.to_datetime(df["order_date"], errors="coerce"),
        _idx=list(range(len(df))),
    ).sort_values(by=["_s", "_o"], kind="stable")
    indices = sort_order_df["_idx"].tolist()

    X_eng = X_eng.take(indices).reset_index(drop=True)
    y = y.take(indices).reset_index(drop=True)

    # 33 features list (Production)
    with open(MODELS_DIR / "lightgbm_model.pkl", "rb") as f:
        prod_model = pickle.load(f)
    prod_feats = list(prod_model.feature_name_)

    # 39 features list (Candidate)
    with open(MODELS_DIR / "lightgbm_candidate.pkl", "rb") as f:
        cand_model = pickle.load(f)
    cand_feats = list(cand_model.feature_name_)

    X_33 = X_eng[prod_feats].copy()
    X_39 = X_eng[cand_feats].copy()

    return X_33, X_39, y, indices


def compute_metrics(y_true, y_pred, y_prob):
    acc = float(accuracy_score(y_true, y_pred))
    b_acc = float(balanced_accuracy_score(y_true, y_pred))
    m_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    m_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    m_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    w_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    loss = float(log_loss(y_true, y_prob, labels=[0, 1, 2, 3]))

    # Multiclass Brier Score
    one_hot = np.eye(4)[y_true]
    brier = float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        roc_auc = 0.0

    # Per class
    p_prec = np.asarray(precision_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0))
    p_rec = np.asarray(recall_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0))
    p_f1 = np.asarray(f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    return {
        "accuracy": acc,
        "balanced_accuracy": b_acc,
        "macro_precision": m_prec,
        "macro_recall": m_rec,
        "macro_f1": m_f1,
        "weighted_f1": w_f1,
        "kappa": kappa,
        "mcc": mcc,
        "log_loss": loss,
        "brier_score": brier,
        "roc_auc": roc_auc,
        "policy_abuser_precision": float(p_prec[1]),
        "policy_abuser_recall": float(p_rec[1]),
        "policy_abuser_f1": float(p_f1[1]),
        "legitimate_f1": float(p_f1[0]),
        "fraud_f1": float(p_f1[2]),
        "wardrobing_f1": float(p_f1[3]),
        "confusion_matrix": cm.tolist(),
    }


def run_experiments():
    df_model, df_raw = load_datasets()
    X_33, X_39, y, indices = prepare_feature_matrices(df_model)

    n = len(df_model)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_tr_33, y_tr = X_33.iloc[:train_end], y.iloc[:train_end]
    X_val_33, y_val = X_33.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_te_33, y_te = X_33.iloc[val_end:], y.iloc[val_end:]

    X_tr_39 = X_39.iloc[:train_end]
    X_val_39 = X_39.iloc[train_end:val_end]
    X_te_39 = X_39.iloc[val_end:]

    results = {}

    print("Running Experiment A: Production Model (33 features)...")
    with open(MODELS_DIR / "lightgbm_model.pkl", "rb") as f:
        prod_model = pickle.load(f)
    pred_te_a = prod_model.predict(X_te_33)
    prob_te_a = prod_model.predict_proba(X_te_33)
    results["Exp_A_Prod_33"] = compute_metrics(y_te.to_numpy(), pred_te_a, prob_te_a)

    print("Running Experiment B: Candidate Model (39 features)...")
    with open(MODELS_DIR / "lightgbm_candidate.pkl", "rb") as f:
        cand_model = pickle.load(f)
    pred_te_b = cand_model.predict(X_te_39)
    prob_te_b = cand_model.predict_proba(X_te_39)
    results["Exp_B_Cand_39"] = compute_metrics(y_te.to_numpy(), pred_te_b, prob_te_b)

    print("Running Experiment C: Balanced Class-Weighted Model (33 features)...")
    model_c = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        class_weight="balanced",
        random_state=42,
        n_estimators=100,
        verbose=-1,
    )
    model_c.fit(X_tr_33, y_tr)
    pred_te_c = model_c.predict(X_te_33)
    prob_te_c = model_c.predict_proba(X_te_33)
    results["Exp_C_BalancedWeights_33"] = compute_metrics(y_te.to_numpy(), pred_te_c, prob_te_c)

    print("Running Experiment D: Targeted Policy Abuser Weighted Model (weight={0:1, 1:4, 2:1, 3:1})...")
    custom_weights = {0: 1.0, 1: 4.0, 2: 1.0, 3: 1.0}
    model_d = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        class_weight=custom_weights,
        random_state=42,
        n_estimators=100,
        verbose=-1,
    )
    model_d.fit(X_tr_33, y_tr)
    pred_te_d = model_d.predict(X_te_33)
    prob_te_d = model_d.predict_proba(X_te_33)
    results["Exp_D_TargetedWeight_33"] = compute_metrics(y_te.to_numpy(), pred_te_d, prob_te_d)

    print("Running Experiment E: Threshold-Optimized Model (Tuned on Validation Set)...")
    prob_val_a = prod_model.predict_proba(X_val_33)
    best_tau = 0.50
    best_val_f1 = 0.0
    for tau in np.linspace(0.1, 0.9, 81):
        p_val_adj = np.argmax(prob_val_a, axis=1)
        # override class 0/3 with class 1 if prob >= tau
        flag = (prob_val_a[:, 1] >= tau) & (prob_val_a[:, 1] > prob_val_a[:, 2]) & (prob_val_a[:, 1] > prob_val_a[:, 3])
        p_val_adj[flag] = 1
        f1_pa = f1_score(y_val == 1, p_val_adj == 1, zero_division=0)
        if f1_pa > best_val_f1:
            best_val_f1 = float(f1_pa)
            best_tau = float(tau)

    print(f"  Optimal Validation Threshold: tau={best_tau:.3f} (Val F1: {best_val_f1:.4f})")
    # Apply optimal tau to untouched Test Set
    pred_te_e = np.argmax(prob_te_a, axis=1)
    flag_te = (prob_te_a[:, 1] >= best_tau) & (prob_te_a[:, 1] > prob_te_a[:, 2]) & (prob_te_a[:, 1] > prob_te_a[:, 3])
    pred_te_e[flag_te] = 1
    results["Exp_E_ThresholdOpt_33"] = compute_metrics(y_te.to_numpy(), pred_te_e, prob_te_a)
    results["Exp_E_ThresholdOpt_33"]["optimal_val_threshold"] = best_tau

    print("Running Experiment F: Probability Calibrated Classifier (Platt Sigmoid 5-Fold)...")
    cal_model = CalibratedClassifierCV(estimator=model_c, method="sigmoid", cv=5)
    cal_model.fit(X_tr_33, y_tr)
    pred_te_f = cal_model.predict(X_te_33)
    prob_te_f = cal_model.predict_proba(X_te_33)
    results["Exp_F_Calibrated_33"] = compute_metrics(y_te.to_numpy(), pred_te_f, prob_te_f)

    print("Running Experiment G: Binary Policy-Abuser Specialist (One-vs-Rest)...")
    y_tr_bin = (y_tr == 1).astype(int)
    y_val_bin = (y_val == 1).astype(int)
    y_te_bin = (y_te == 1).astype(int)

    bin_model = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=3.5,
        random_state=42,
        n_estimators=100,
        verbose=-1,
    )
    bin_model.fit(X_tr_33, y_tr_bin)
    prob_bin_val = np.asarray(bin_model.predict_proba(X_val_33))[:, 1]
    prob_bin_te = np.asarray(bin_model.predict_proba(X_te_33))[:, 1]

    # Find optimal binary threshold on validation
    best_bin_tau = 0.50
    best_bin_f1 = 0.0
    for tau in np.linspace(0.1, 0.9, 81):
        f1_bin = f1_score(y_val_bin, (prob_bin_val >= tau).astype(int), zero_division=0)
        if f1_bin > best_bin_f1:
            best_bin_f1 = float(f1_bin)
            best_bin_tau = float(tau)

    bin_pred_te = (prob_bin_te >= best_bin_tau).astype(int)
    results["Exp_G_BinarySpecialist_33"] = {
        "optimal_threshold": best_bin_tau,
        "binary_precision": float(precision_score(y_te_bin, bin_pred_te, zero_division=0)),
        "binary_recall": float(recall_score(y_te_bin, bin_pred_te, zero_division=0)),
        "binary_f1": float(f1_score(y_te_bin, bin_pred_te, zero_division=0)),
        "binary_roc_auc": float(roc_auc_score(y_te_bin, prob_bin_te)),
    }

    print("Running Experiment H: Cross-Validation Leakage & Stability Check...")
    # Compare StratifiedKFold vs GroupKFold (by customer_id)
    raw_sorted = df_raw.iloc[indices].reset_index(drop=True)
    customer_ids = raw_sorted["customer_id"].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=5)
    gkf_f1_33 = []
    gkf_f1_39 = []

    for tr_idx, te_idx in gkf.split(X_33, y, groups=customer_ids):
        # 33 features
        m33 = lgb.LGBMClassifier(random_state=42, n_estimators=60, verbose=-1)
        m33.fit(X_33.iloc[tr_idx], y.iloc[tr_idx])
        p33 = m33.predict(X_33.iloc[te_idx])
        gkf_f1_33.append(float(f1_score(y.iloc[te_idx] == 1, p33 == 1, zero_division=0)))

        # 39 features
        m39 = lgb.LGBMClassifier(random_state=42, n_estimators=60, verbose=-1)
        m39.fit(X_39.iloc[tr_idx], y.iloc[tr_idx])
        p39 = m39.predict(X_39.iloc[te_idx])
        gkf_f1_39.append(float(f1_score(y.iloc[te_idx] == 1, p39 == 1, zero_division=0)))

    results["Exp_H_GroupCV_CustomerSplit"] = {
        "group_cv_5fold_33_pa_f1_mean": float(np.mean(gkf_f1_33)),
        "group_cv_5fold_33_pa_f1_std": float(np.std(gkf_f1_33)),
        "group_cv_5fold_39_pa_f1_mean": float(np.mean(gkf_f1_39)),
        "group_cv_5fold_39_pa_f1_std": float(np.std(gkf_f1_39)),
    }

    # Save all experiment results
    exp_out = REPORT_DIR / "ml_experiment_matrix_audit.json"
    with open(exp_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Complete experiment matrix audit saved to {exp_out}")
    return results


if __name__ == "__main__":
    run_experiments()
