from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "trustloop" / "model_ready.csv"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TRUSTLOOP STAGE 2 — LIGHTGBM")
print("=" * 70)

print(f"Loading dataset: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# TARGET
# ============================================================

TARGET = "abuse_label"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

y = df[TARGET].astype(int)

X = df.drop(columns=[TARGET]).copy()


# ============================================================
# TEMPORAL FEATURES
# ============================================================

print("\nExtracting temporal features...")

datetime_columns = []

for col in ["order_date", "return_date"]:
    if col in X.columns:
        X[col] = pd.to_datetime(X[col], errors="coerce")
        datetime_columns.append(col)

for col in datetime_columns:
    X[f"{col}_year"] = X[col].dt.year
    X[f"{col}_month"] = X[col].dt.month
    X[f"{col}_day"] = X[col].dt.day
    X[f"{col}_dayofweek"] = X[col].dt.dayofweek

    X[f"{col}_dayofyear"] = X[col].dt.dayofyear

    X[f"{col}_is_weekend"] = (
        X[col].dt.dayofweek >= 5
    ).astype(int)

# Useful duration between order and return
if "order_date" in X.columns and "return_date" in X.columns:
    calculated_return_days = (X["return_date"] - X["order_date"]).dt.total_seconds() / 86400.0  # pyrefly: ignore[missing-attribute]

    X["calculated_days_to_return"] = calculated_return_days


# Remove raw datetime columns
datetime_columns = [
    col for col in datetime_columns
    if col in X.columns
]

X = X.drop(columns=datetime_columns)

print(f"Datetime columns removed: {datetime_columns}")
print(f"Feature count after temporal extraction: {len(X.columns)}")


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_columns = [
    "country",
    "customer_segment",
    "device_type",
    "payment_method",
    "platform",
    "product_category",
    "return_reason",
    "shipping_carrier",
]

categorical_columns = [
    col for col in categorical_columns
    if col in X.columns
]

for col in categorical_columns:
    X[col] = X[col].astype("category")


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("PRE-TRAINING VALIDATION")
print("=" * 70)

print(f"FEATURE COUNT: {len(X.columns)}")

# Datetime validation
datetime_remaining = X.select_dtypes(  # pyrefly: ignore[no-matching-overload]
    include=["datetime", "datetimetz"]
).columns.tolist()

datetime_pass = len(datetime_remaining) == 0

print(
    f"DATETIME CHECK: "
    f"{'PASS' if datetime_pass else 'FAIL'}"
)

if not datetime_pass:
    print("Remaining datetime columns:", datetime_remaining)
    raise ValueError("Datetime columns remain in X.")


# Numeric validation
numeric_columns = X.select_dtypes(  # pyrefly: ignore[no-matching-overload]
    include=["number"]
).columns.tolist()


numeric_pass = True

for col in numeric_columns:
    numeric_series = pd.to_numeric(
        X[col],
        errors="coerce"
    )

    if numeric_series.isna().any():
        numeric_pass = False
        print(f"Invalid numeric values in: {col}")

    if np.isinf(
        numeric_series.to_numpy(dtype="float64")
    ).any():
        numeric_pass = False
        print(f"Infinite values in: {col}")

print(
    f"NUMERIC COLUMNS CHECK: "
    f"{'PASS' if numeric_pass else 'FAIL'}"
)

if not numeric_pass:
    raise ValueError("Numeric validation failed.")


# Categorical validation
categorical_pass = True

for col in categorical_columns:
    if not isinstance(
        X[col].dtype,
        pd.CategoricalDtype
    ):
        categorical_pass = False
        print(f"Invalid categorical dtype: {col}")

print(
    f"CATEGORICAL COLUMNS CHECK: "
    f"{'PASS' if categorical_pass else 'FAIL'}"
)

if not categorical_pass:
    raise ValueError("Categorical validation failed.")


# Target validation
target_classes = sorted(y.unique().tolist())

target_pass = target_classes == [0, 1, 2, 3]

print(
    f"TARGET CHECK: "
    f"{'PASS' if target_pass else 'FAIL'}"
)

print("Target classes:", target_classes)

if not target_pass:
    raise ValueError(
        f"Expected target classes [0, 1, 2, 3], "
        f"got {target_classes}"
    )


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

print("\nCreating chronological split...")

# Use return_date from the original dataset
sort_dates = pd.to_datetime(
    df["return_date"],
    errors="coerce"
)

sort_order = np.lexsort((  # pyrefly: ignore[bad-argument-type]
    pd.to_datetime(df["order_date"], errors="coerce").astype("int64").to_numpy(),
    sort_dates.astype("int64").to_numpy(),
))

X = X.iloc[sort_order].reset_index(drop=True)  # pyrefly: ignore[bad-index]
y = y.iloc[sort_order].reset_index(drop=True)  # pyrefly: ignore[bad-index]


n = len(X)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end].copy()
y_train = y.iloc[:train_end].copy()

X_val = X.iloc[train_end:val_end].copy()
y_val = y.iloc[train_end:val_end].copy()

X_test = X.iloc[val_end:].copy()
y_test = y.iloc[val_end:].copy()

print(f"Train samples: {len(X_train)}")
print(f"Val samples:   {len(X_val)}")
print(f"Test samples:  {len(X_test)}")


# ============================================================
# FEATURE ALIGNMENT
# ============================================================

if list(X_train.columns) != list(X_val.columns):
    raise ValueError("Train/validation feature columns do not match.")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Train/test feature columns do not match.")

print("FEATURE ALIGNMENT CHECK: PASS")


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LIGHTGBM MODEL (CANDIDATE)")
print("=" * 70)

clf = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)

clf.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train, y_train),
        (X_val, y_val),
    ],
    categorical_feature=categorical_columns,
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(50),
    ],
)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred = clf.predict(X_test)


# ============================================================
# METRICS
# ============================================================

metrics = {
    "accuracy": float(
        accuracy_score(y_test, y_pred)
    ),
    "macro_precision": float(
        precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
    ),
    "macro_recall": float(
        recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
    ),
    "macro_f1": float(
        f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
    ),
    "weighted_f1": float(
        f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    ),
}


print("\nMODEL METRICS")
print("-" * 50)

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")


# ============================================================
# SAVE METRICS (CANDIDATE)
# ============================================================

with open(
    REPORT_DIR / "candidate_metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metrics,
        f,
        indent=2,
    )


# ============================================================
# CLASSIFICATION REPORT (CANDIDATE)
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report).transpose().to_csv(
    REPORT_DIR / "candidate_classification_report.csv"
)


# ============================================================
# CONFUSION MATRIX (CANDIDATE)
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,  # pyrefly: ignore[bad-argument-type]
)

pd.DataFrame(
    cm,
    index=[
        "Legitimate",
        "Policy Abuser",
        "Fraudulent Return",
        "Wardrobing",
    ],
    columns=[
        "Legitimate",
        "Policy Abuser",
        "Fraudulent Return",
        "Wardrobing",
    ],
).to_csv(
    REPORT_DIR / "candidate_confusion_matrix.csv"
)


# ============================================================
# FEATURE IMPORTANCE (CANDIDATE)
# ============================================================

importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": clf.feature_importances_,
})

importance = importance.sort_values(
    "importance",
    ascending=False,
)

importance.to_csv(
    REPORT_DIR / "candidate_feature_importance.csv",
    index=False,
)


# ============================================================
# SAVE CANDIDATE MODEL (SAFETY: NEVER OVERWRITE PRODUCTION)
# ============================================================

candidate_model_path = MODEL_DIR / "lightgbm_candidate.pkl"

with open(
    candidate_model_path,
    "wb",
) as f:
    pickle.dump(clf, f)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("LIGHTGBM CANDIDATE TRAINING COMPLETED")
print("=" * 70)

print(
    f"Train: {len(X_train)} | "
    f"Validation: {len(X_val)} | "
    f"Test: {len(X_test)}"
)

print(
    f"Features: {len(X_train.columns)}"
)

print(
    f"Best iteration: "
    f"{clf.best_iteration_}"
)

print("\nSaved Candidate Artifacts:")
print(f"- {candidate_model_path}")
print(f"- {REPORT_DIR / 'candidate_metrics.json'}")
print(f"- {REPORT_DIR / 'candidate_classification_report.csv'}")
print(f"- {REPORT_DIR / 'candidate_confusion_matrix.csv'}")
print(f"- {REPORT_DIR / 'candidate_feature_importance.csv'}")
print("\nPRODUCTION MODEL REMAINS UNTOUCHED: models/lightgbm_model.pkl")