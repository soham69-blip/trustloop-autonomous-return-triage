import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import confusion_matrix

DATA_PATH = "data/processed/trustloop/model_ready.csv"
MODEL_PATH = "models/lightgbm_model.pkl"

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
    df["return_date"] - df["order_date"]
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

X_test = X.iloc[val_end:].copy()
y_test = y.iloc[val_end:].copy()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

proba = model.predict_proba(X_test)

pred = np.argmax(proba, axis=1)

print("=" * 70)
print("TRUSTLOOP - POLICY ABUSER ERROR ANALYSIS")
print("=" * 70)

print()
print("CONFUSION MATRIX")
print("-" * 70)

cm = confusion_matrix(
    y_test,
    pred,
    labels=[0, 1, 2, 3]
)

print(pd.DataFrame(
    cm,
    index=[
        "Actual Legitimate",
        "Actual Policy Abuser",
        "Actual Fraudulent Return",
        "Actual Wardrobing"
    ],
    columns=[
        "Pred Legitimate",
        "Pred Policy Abuser",
        "Pred Fraudulent Return",
        "Pred Wardrobing"
    ]
))

print()
print("POLICY ABUSER CASES")
print("-" * 70)

policy_mask = y_test.values == 1

actual_policy = proba[policy_mask]

policy_pred = pred[policy_mask]

print("Total Policy Abuser test cases:", len(actual_policy))
print()

print("Where Policy Abuser cases were predicted:")

for cls in [0, 1, 2, 3]:
    count = np.sum(policy_pred == cls)
    pct = count / len(policy_pred) * 100
    print(
        f"  Class {cls}: {count} ({pct:.2f}%)"
    )

print()
print("AVERAGE MODEL PROBABILITIES FOR TRUE POLICY ABUSERS")
print("-" * 70)

for cls, name in [
    (0, "Legitimate"),
    (1, "Policy Abuser"),
    (2, "Fraudulent Return"),
    (3, "Wardrobing")
]:
    print(
        f"{name:20s}: "
        f"{actual_policy[:, cls].mean():.4f}"
    )

print()
print("TOP POLICY ABUSER CASES BY POLICY PROBABILITY")
print("-" * 70)

top_indices = np.argsort(
    actual_policy[:, 1]
)[-20:][::-1]

for rank, idx in enumerate(top_indices, 1):
    print(
        f"{rank:2d}. "
        f"Policy probability = "
        f"{actual_policy[idx, 1]:.4f} | "
        f"Predicted = {policy_pred[idx]}"
    )

print()
print("=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)
