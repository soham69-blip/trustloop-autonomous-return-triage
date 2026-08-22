import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

print("=" * 70)
print("TRUSTLOOP - CLASS SPECIFIC THRESHOLD EXPERIMENT")
print("=" * 70)

df = pd.read_csv(
    "data/processed/trustloop/model_ready.csv"
)

target = "abuse_label"

for col in ["order_date", "return_date"]:
    df[col] = pd.to_datetime(
        df[col],
        errors="coerce"
    )

def temporal(data, col):

    data[col + "_year"] = data[col].dt.year
    data[col + "_month"] = data[col].dt.month
    data[col + "_day"] = data[col].dt.day

    data[col + "_dayofweek"] = (
        data[col].dt.dayofweek
    )

    data[col + "_dayofyear"] = (
        data[col].dt.dayofyear
    )

    data[col + "_is_weekend"] = (
        data[col].dt.dayofweek.isin([5, 6])
    ).astype(int)

temporal(df, "order_date")
temporal(df, "return_date")

df["calculated_days_to_return"] = (
    df["return_date"] -
    df["order_date"]
).dt.total_seconds() / 86400

feature_cols = [
    c for c in df.columns
    if c not in [
        target,
        "order_date",
        "return_date"
    ]
]

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

for col in categorical:
    if col in df.columns:
        df[col] = df[col].astype("category")

X = df[feature_cols]
y = df[target].astype(int)

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

with open(
    "models/lightgbm_model.pkl",
    "rb"
) as f:
    model = pickle.load(f)

proba = model.predict_proba(X_val)

results = []

policy_thresholds = [
    0.25,
    0.30,
    0.35
]

fraud_thresholds = [
    0.30,
    0.40,
    0.50
]

wardrobe_thresholds = [
    0.30,
    0.40,
    0.50
]

for pt in policy_thresholds:

    for ft in fraud_thresholds:

        for wt in wardrobe_thresholds:

            pred = np.zeros(len(proba), dtype=int)

            # Strongest classes get priority.
            fraud_mask = proba[:, 2] >= ft
            wardrobe_mask = proba[:, 3] >= wt
            policy_mask = proba[:, 1] >= pt

            pred[fraud_mask] = 2
            pred[wardrobe_mask] = 3

            remaining = (
                ~fraud_mask &
                ~wardrobe_mask
            )

            pred[
                remaining & policy_mask
            ] = 1

            # Remaining cases use normal
            # highest probability class.
            remaining_final = (
                ~fraud_mask &
                ~wardrobe_mask &
                ~policy_mask
            )

            pred[
                remaining_final
            ] = np.argmax(
                proba[
                    remaining_final
                ],
                axis=1
            )

            policy_precision = precision_score(
                y_val,
                pred,
                labels=[1],
                average="macro",
                zero_division=0
            )

            policy_recall = recall_score(
                y_val,
                pred,
                labels=[1],
                average="macro",
                zero_division=0
            )

            policy_f1 = f1_score(
                y_val,
                pred,
                labels=[1],
                average="macro",
                zero_division=0
            )

            results.append({
                "policy_threshold": pt,
                "fraud_threshold": ft,
                "wardrobing_threshold": wt,
                "accuracy": accuracy_score(
                    y_val,
                    pred
                ),
                "macro_f1": f1_score(
                    y_val,
                    pred,
                    average="macro"
                ),
                "policy_precision": policy_precision,
                "policy_recall": policy_recall,
                "policy_f1": policy_f1
            })

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    "macro_f1",
    ascending=False
)

result_df.to_csv(
    "reports/class_threshold_experiment.csv",
    index=False
)

print()
print(
    result_df.head(15).round(4).to_string(
        index=False
    )
)

print()
print("Saved:")
print(
    "reports/class_threshold_experiment.csv"
)

print()
print("=" * 70)
print("CLASS THRESHOLD EXPERIMENT COMPLETE")
print("=" * 70)
