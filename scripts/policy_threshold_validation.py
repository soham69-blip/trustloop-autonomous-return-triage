import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

DATA_PATH = "data/processed/trustloop/model_ready.csv"
MODEL_PATH = "models/lightgbm_model.pkl"

print("=" * 70)
print("TRUSTLOOP - THRESHOLD VALIDATION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

TARGET = "abuse_label"

categorical = [
    "country",
    "customer_segment",
    "device_type",
    "payment_method",
    "platform",
    "product_category",
    "return_reason",
    "shipping_carrier"
]

for col in ["order_date", "return_date"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

def add_temporal_features(data, col):
    data[col + "_year"] = data[col].dt.year
    data[col + "_month"] = data[col].dt.month
    data[col + "_day"] = data[col].dt.day
    data[col + "_dayofweek"] = data[col].dt.dayofweek
    data[col + "_dayofyear"] = data[col].dt.dayofyear
    data[col + "_is_weekend"] = (
        data[col].dt.dayofweek.isin([5, 6])
    ).astype(int)

add_temporal_features(df, "order_date")
add_temporal_features(df, "return_date")

df["calculated_days_to_return"] = (
    pd.to_datetime(df["return_date"]) - pd.to_datetime(df["order_date"])
).dt.total_seconds() / 86400

feature_cols = [
    c for c in df.columns
    if c not in [TARGET, "order_date", "return_date"]
]

for col in categorical:
    if col in df.columns:
        df[col] = df[col].astype("category")

X = df[feature_cols]
y = df[TARGET].astype(int)

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_val = X.iloc[train_end:val_end].copy()
y_val = y.iloc[train_end:val_end].copy()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

proba = model.predict_proba(X_val)

base_pred = np.argmax(proba, axis=1)

thresholds = [
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]

results = []

for threshold in thresholds:

    pred = base_pred.copy()

    policy_mask = proba[:, 1] >= threshold

    pred[policy_mask] = 1

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_val, pred),
        "macro_f1": f1_score(
            y_val,
            pred,
            average="macro"
        ),
        "policy_precision": precision_score(
            y_val,
            pred,
            labels=[1],
            average="macro",
            zero_division=0
        ),
        "policy_recall": recall_score(
            y_val,
            pred,
            labels=[1],
            average="macro",
            zero_division=0
        ),
        "policy_f1": f1_score(
            y_val,
            pred,
            labels=[1],
            average="macro",
            zero_division=0
        )
    })

result_df = pd.DataFrame(results)

result_df.to_csv(
    "reports/policy_threshold_validation.csv",
    index=False
)

print()
print(result_df.round(4).to_string(index=False))

print()
print("Saved:")
print("reports/policy_threshold_validation.csv")

print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
