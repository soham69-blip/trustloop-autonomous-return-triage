import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

MODEL_PATH = "models/lightgbm_model.pkl"
DATA_PATH = "data/processed/trustloop/model_ready.csv"

print("=" * 70)
print("TRUSTLOOP - POLICY ABUSER WEIGHT EXPERIMENT")
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

X_train = X.iloc[:train_end].copy()
y_train = y.iloc[:train_end].copy()

X_val = X.iloc[train_end:val_end].copy()
y_val = y.iloc[train_end:val_end].copy()

X_test = X.iloc[val_end:].copy()
y_test = y.iloc[val_end:].copy()

print("Rows:", n)
print("Features:", len(feature_cols))
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

with open(MODEL_PATH, "rb") as f:
    old_model = pickle.load(f)

params = old_model.get_params()

remove_params = [
    "class_weight",
    "n_jobs",
    "random_state",
    "eval_metric"
]

for key in remove_params:
    params.pop(key, None)

params["random_state"] = 42
params["n_jobs"] = -1

weights_to_test = [
    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
    {0: 1.0, 1: 1.25, 2: 1.0, 3: 1.0},
    {0: 1.0, 1: 1.50, 2: 1.0, 3: 1.0},
    {0: 1.0, 1: 1.75, 2: 1.0, 3: 1.0},
    {0: 1.0, 1: 2.00, 2: 1.0, 3: 1.0}
]

results = []

for weights in weights_to_test:

    print()
    print("-" * 70)
    print("TESTING POLICY ABUSER WEIGHT:", weights[1])

    model = lgb.LGBMClassifier(
        **params,
        class_weight=weights
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False)
        ]
    )

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    macro_f1 = f1_score(
        y_test,
        pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        pred,
        average="weighted"
    )

    precision = precision_score(
        y_test,
        pred,
        average=None,
        labels=[0, 1, 2, 3],
        zero_division=0
    )

    recall = recall_score(
        y_test,
        pred,
        average=None,
        labels=[0, 1, 2, 3],
        zero_division=0
    )

    class_f1 = f1_score(
        y_test,
        pred,
        average=None,
        labels=[0, 1, 2, 3],
        zero_division=0
    )

    result = {
        "policy_weight": weights[1],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "policy_precision": precision[1],
        "policy_recall": recall[1],
        "policy_f1": class_f1[1],
        "legitimate_f1": class_f1[0],
        "fraudulent_return_f1": class_f1[2],
        "wardrobing_f1": class_f1[3],
        "best_iteration": getattr(
            model,
            "best_iteration_",
            None
        )
    }

    results.append(result)

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print("Policy Precision:", round(precision[1], 4))
    print("Policy Recall:", round(recall[1], 4))
    print("Policy F1:", round(class_f1[1], 4))

results_df = pd.DataFrame(results)

results_df.to_csv(
    "reports/policy_abuser_weight_experiment.csv",
    index=False
)

print()
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

columns = [
    "policy_weight",
    "accuracy",
    "macro_f1",
    "policy_precision",
    "policy_recall",
    "policy_f1",
    "legitimate_f1",
    "fraudulent_return_f1",
    "wardrobing_f1"
]

print(results_df[columns].round(4).to_string(index=False))

print()
print("Saved:")
print("reports/policy_abuser_weight_experiment.csv")
