import pickle
import json
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
MODEL_P = ROOT / "models" / "lightgbm_model.pkl"
DATA_P = ROOT / "data" / "processed" / "trustloop" / "model_ready.csv"
print("MODEL_P", MODEL_P)
print("DATA_P", DATA_P)
model = pickle.load(open(MODEL_P, "rb"))
print("model type", type(model))
print("classes", getattr(model, "classes_", None))
print("feature_names", getattr(model, "feature_name_", getattr(model, "feature_names_in_", None)))
df = pd.read_csv(DATA_P)
print("rows", len(df))

# Sort chronologically by return_date and order_date
df["_temp_ret"] = pd.to_datetime(df["return_date"], errors="coerce")
df["_temp_ord"] = pd.to_datetime(df["order_date"], errors="coerce")
df = df.sort_values(by=["_temp_ret", "_temp_ord"]).reset_index(drop=True)
df = df.drop(columns=["_temp_ret", "_temp_ord"])

y = df["abuse_label"].astype(int)
print("target counts full", y.value_counts().sort_index().to_dict())
X = df.drop(columns=["abuse_label"]).copy()

for col in ["order_date", "return_date"]:
    if col in X.columns:
        dti = pd.DatetimeIndex(pd.to_datetime(X[col], errors="coerce"))
        X[f"{col}_year"] = dti.year
        X[f"{col}_month"] = dti.month
        X[f"{col}_day"] = dti.day
        X[f"{col}_dayofweek"] = dti.dayofweek
        X[f"{col}_dayofyear"] = dti.dayofyear
        X[f"{col}_is_weekend"] = (dti.dayofweek >= 5).astype(int)

if "order_date" in X.columns and "return_date" in X.columns:
    ret_d = pd.DatetimeIndex(pd.to_datetime(X["return_date"], errors="coerce"))
    ord_d = pd.DatetimeIndex(pd.to_datetime(X["order_date"], errors="coerce"))
    X["calculated_days_to_return"] = (ret_d - ord_d).total_seconds() / 86400.0

X = X.drop(columns=[c for c in ["order_date", "return_date"] if c in X.columns])
cat_cols = ["country", "customer_segment", "device_type", "payment_method", "platform", "product_category", "return_reason", "shipping_carrier"]
for c in cat_cols:
    if c in X.columns:
        X[c] = X[c].astype("category")

train_end = int(len(X) * 0.7)
val_end = int(len(X) * 0.85)
X_train = X.iloc[:train_end].copy()
y_train = y.iloc[:train_end].copy()
X_val = X.iloc[train_end:val_end].copy()
y_val = y.iloc[train_end:val_end].copy()
X_test = X.iloc[val_end:].copy()
y_test = y.iloc[val_end:].copy()
print("splits", len(X_train), len(X_val), len(X_test))
print("train counts", y_train.value_counts().sort_index().to_dict())
print("val counts", y_val.value_counts().sort_index().to_dict())
print("test counts", y_test.value_counts().sort_index().to_dict())

raw_feats = getattr(model, "feature_name_", getattr(model, "feature_names_in_", None))
feature_names = list(raw_feats) if raw_feats is not None else list(X_train.columns)
print("feature count model", len(feature_names))
X_test2 = X_test[feature_names]
y_pred = model.predict(X_test2)
probs = model.predict_proba(X_test2)

# compute confusion matrix using sklearn
cm_mat = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
print("confusion_matrix:\n")
print(cm_mat)

# compute precision, recall, f1 per class
metrics: dict[int, dict[str, Any]] = {}
for lab in [0, 1, 2, 3]:
    tp = float(cm_mat[lab, lab])
    fp = float(cm_mat[:, lab].sum()) - tp
    fn = float(cm_mat[lab, :].sum()) - tp
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    support = int(cm_mat[lab, :].sum())
    metrics[lab] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }
print("classification metrics per class:")
print(json.dumps(metrics, indent=2))
macro_f1 = float(np.mean([metrics[lab]["f1"] for lab in metrics]))
print("macro_f1", macro_f1)
print("wardrobing recall", metrics[3]["recall"])
print("policy abuser recall", metrics[1]["recall"])
print("fraudulent return recall", metrics[2]["recall"])
print("legitimate recall", metrics[0]["recall"])
fi = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
print("top features:\n", fi.head(20).to_string(index=False))

ward = df[df["abuse_label"] == 3]
print("ward count", len(ward))
print("ward sample (up to 3):")
print(ward.head(3).to_dict(orient="records"))

from backend.app.ml_feature_builder import build_model_dataframe
payload = {
    "case_id": "TEST-WARD-001",
    "customer_id": "CUST-WARD-001",
    "order_id": "ORD-WARD-001",
    "age": 30,
    "account_age_days": 200,
    "customer_segment": "Gold",
    "country": "US",
    "platform": "Mobile App",
    "device_type": "iPhone",
    "payment_method": "Credit Card",
    "product_category": "Clothing",
    "avg_order_value_usd": 80.0,
    "is_high_value_item": 0,
    "discount_used": 1,
    "days_to_return": 10,
    "return_reason": "Too Small",
    "shipping_carrier": "FedEx",
    "multiple_accounts_flag": 0,
    "wishlist_to_cart_time_hrs": 0.5,
    "customer_return_count_prior": 4,
    "returns_last_30d_prior": 2,
    "returns_last_90d_prior": 5,
    "total_returns_lifetime_prior": 9,
    "order_date": "2026-06-20",
    "return_date": "2026-06-30",
    "item_condition": "new",
    "refund_amount": 80.0,
}
X_payload = build_model_dataframe(payload)
print("payload features sample:\n", X_payload.to_dict(orient="records"))

if len(ward) > 0:
    w = ward.copy()
    for c in ["order_date", "return_date"]:
        if c in w.columns:
            dti_w = pd.DatetimeIndex(pd.to_datetime(w[c], errors="coerce"))
            w[f"{c}_year"] = dti_w.year
            w[f"{c}_month"] = dti_w.month
            w[f"{c}_day"] = dti_w.day
            w[f"{c}_dayofweek"] = dti_w.dayofweek
            w[f"{c}_dayofyear"] = dti_w.dayofyear
            w[f"{c}_is_weekend"] = (dti_w.dayofweek >= 5).astype(int)
    if "order_date" in w.columns and "return_date" in w.columns:
        ret_w = pd.DatetimeIndex(pd.to_datetime(w["return_date"], errors="coerce"))
        ord_w = pd.DatetimeIndex(pd.to_datetime(w["order_date"], errors="coerce"))
        w["calculated_days_to_return"] = (ret_w - ord_w).total_seconds() / 86400.0
    w = w.drop(columns=[c for c in ["order_date", "return_date"] if c in w.columns])
    cmp_feats = ["age", "account_age_days", "is_high_value_item", "customer_return_count_prior", "total_returns_lifetime_prior", "calculated_days_to_return"]
    wnum = w[cmp_feats].fillna(0).astype(float)
    pnum = X_payload[[f for f in X_payload.columns if f in cmp_feats]].astype(float).iloc[0]
    dists = ((wnum - pnum) ** 2).sum(axis=1)
    nearest = dists.nsmallest(5).index.tolist()
    print("nearest ward indices", nearest)
    print("nearest ward numeric", wnum.loc[nearest].to_dict(orient="records"))
print("done")
