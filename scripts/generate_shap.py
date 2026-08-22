from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trustloop"
    / "model_ready.csv"
)

MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_model.pkl"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("TRUSTLOOP — SHAP EXPLAINABILITY")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading dataset: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# TARGET
# ============================================================

TARGET = "abuse_label"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

y = df[TARGET].astype(int)

X = df.drop(
    columns=[TARGET]
).copy()


# ============================================================
# TEMPORAL FEATURES
# MUST MATCH TRAINING PIPELINE
# ============================================================

print("\nExtracting temporal features...")

datetime_columns = []

for col in ["order_date", "return_date"]:

    if col in X.columns:

        X[col] = pd.to_datetime(
            X[col],
            errors="coerce"
        )

        datetime_columns.append(col)


# Create the SAME temporal features
# used by the successful LightGBM training.

for col in datetime_columns:

    X[f"{col}_year"] = X[col].dt.year

    X[f"{col}_month"] = X[col].dt.month

    X[f"{col}_day"] = X[col].dt.day

    X[f"{col}_dayofweek"] = (
        X[col].dt.dayofweek
    )

    X[f"{col}_dayofyear"] = (
        X[col].dt.dayofyear
    )

    X[f"{col}_is_weekend"] = (
        X[col].dt.dayofweek >= 5
    ).astype(int)


# Same calculated return duration
# used during training.

if (
    "order_date" in X.columns
    and "return_date" in X.columns
):

    X["calculated_days_to_return"] = (
        X["return_date"]
        - X["order_date"]
    ).dt.total_seconds() / 86400.0


# Remove raw datetime columns.

X = X.drop(
    columns=datetime_columns
)


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


for col in categorical_columns:

    if col in X.columns:

        X[col] = X[col].astype(
            "category"
        )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading LightGBM model...")

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


with open(
    MODEL_PATH,
    "rb"
) as f:

    model = pickle.load(f)


# ============================================================
# FEATURE ALIGNMENT
# ============================================================

model_features = (
    model.booster_.feature_name()
)


print(
    f"Feature count: {len(X.columns)}"
)

print(
    f"Model feature count: "
    f"{len(model_features)}"
)


if list(X.columns) != list(
    model_features
):

    dataset_only = sorted(
        set(X.columns)
        - set(model_features)
    )

    model_only = sorted(
        set(model_features)
        - set(X.columns)
    )

    raise ValueError(
        "\nSHAP FEATURE MISMATCH\n"
        f"Dataset-only features: "
        f"{dataset_only}\n"
        f"Model-only features: "
        f"{model_only}"
    )


print(
    "FEATURE ALIGNMENT: PASS"
)


# ============================================================
# EXPLAIN TEST SET
# ============================================================

# The original training split was:
#
# 42,000 train
# 9,000 validation
# 9,000 test
#
# Therefore the final 9,000 rows correspond
# to the test portion after chronological ordering.

X_explain = X.iloc[-9000:].copy()

print(
    f"Rows explained: "
    f"{len(X_explain)}"
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

print(
    "\nCalculating SHAP values..."
)

explainer = shap.TreeExplainer(
    model
)

shap_values = (
    explainer.shap_values(
        X_explain
    )
)


# ============================================================
# INSPECT SHAP OUTPUT
# ============================================================

print(
    "\nSHAP output type:",
    type(shap_values)
)


if isinstance(
    shap_values,
    list
):

    print(
        "SHAP returned a list."
    )

    print(
        "Number of classes:",
        len(shap_values)
    )

    for i, value in enumerate(
        shap_values
    ):

        print(
            f"Class {i} shape:",
            np.asarray(value).shape
        )

elif isinstance(
    shap_values,
    np.ndarray
):

    print(
        "SHAP returned NumPy array."
    )

    print(
        "Raw SHAP shape:",
        shap_values.shape
    )

else:

    raise TypeError(
        "Unsupported SHAP output type: "
        f"{type(shap_values)}"
    )


# ============================================================
# NORMALIZE MULTICLASS SHAP OUTPUT
# ============================================================

n_rows = len(
    X_explain
)

n_features = len(
    X_explain.columns
)

n_classes = 4


# ------------------------------------------------------------
# CASE 1 — SHAP RETURNS LIST
# ------------------------------------------------------------

if isinstance(
    shap_values,
    list
):

    if len(shap_values) != n_classes:

        raise ValueError(
            f"Expected {n_classes} "
            f"SHAP classes but received "
            f"{len(shap_values)}"
        )

    arrays = [
        np.asarray(value)
        for value in shap_values
    ]

    for class_idx, arr in enumerate(
        arrays
    ):

        expected = (
            n_rows,
            n_features,
        )

        if arr.shape != expected:

            raise ValueError(
                f"Class {class_idx} SHAP "
                f"shape mismatch.\n"
                f"Received: {arr.shape}\n"
                f"Expected: {expected}"
            )

    # Result:
    # classes x rows x features

    shap_array = np.stack(
        arrays,
        axis=0
    )


# ------------------------------------------------------------
# CASE 2 — SHAP RETURNS NUMPY ARRAY
# ------------------------------------------------------------

elif isinstance(
    shap_values,
    np.ndarray
):

    raw_shape = shap_values.shape

    # Newer SHAP versions commonly return:
    #
    # rows x features x classes

    if raw_shape == (
        n_rows,
        n_features,
        n_classes,
    ):

        shap_array = np.transpose(
            shap_values,
            (2, 0, 1)
        )

    # Some versions return:
    #
    # classes x rows x features

    elif raw_shape == (
        n_classes,
        n_rows,
        n_features,
    ):

        shap_array = shap_values

    else:

        raise ValueError(
            "Unexpected SHAP dimensions.\n"
            f"Received: {raw_shape}\n"
            f"Expected either:\n"
            f"({n_rows}, "
            f"{n_features}, "
            f"{n_classes})\n"
            f"or:\n"
            f"({n_classes}, "
            f"{n_rows}, "
            f"{n_features})"
        )


# ============================================================
# FINAL SHAP SHAPE VALIDATION
# ============================================================

print(
    "\nNormalized SHAP shape:",
    shap_array.shape
)


expected_shape = (
    n_classes,
    n_rows,
    n_features,
)


if shap_array.shape != expected_shape:

    raise ValueError(
        "\nFINAL SHAP SHAPE MISMATCH\n"
        f"Received: {shap_array.shape}\n"
        f"Expected: {expected_shape}"
    )


print(
    "SHAP DIMENSION CHECK: PASS"
)


# ============================================================
# SAVE RAW SHAP VALUES
# ============================================================

np.save(
    REPORT_DIR / "shap_values.npy",
    shap_array
)

print(
    "Saved: reports/shap_values.npy"
)


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

global_importance = np.mean(
    np.abs(shap_array),
    axis=(0, 1)
)


print(
    "Global importance length:",
    len(global_importance)
)

print(
    "Feature count:",
    n_features
)


if len(global_importance) != n_features:

    raise ValueError(
        "SHAP importance length does "
        "not match feature count."
    )


# ============================================================
# GLOBAL SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "feature": list(
            X_explain.columns
        ),
        "mean_abs_shap": (
            global_importance
        ),
    }
)


summary = summary.sort_values(
    "mean_abs_shap",
    ascending=False
)


summary.to_csv(
    REPORT_DIR
    / "shap_summary.csv",
    index=False
)


print(
    "Saved: reports/shap_summary.csv"
)


# ============================================================
# CLASS-SPECIFIC SHAP IMPORTANCE
# ============================================================

class_names = {
    0: "Legitimate",
    1: "Policy Abuser",
    2: "Fraudulent Return",
    3: "Wardrobing",
}


class_rows = []


for class_idx in range(
    n_classes
):

    class_shap = (
        shap_array[class_idx]
    )

    class_importance = np.mean(
        np.abs(class_shap),
        axis=0
    )


    if len(
        class_importance
    ) != n_features:

        raise ValueError(
            f"Class {class_idx} "
            "SHAP feature count mismatch."
        )


    for feature, importance in zip(
        X_explain.columns,
        class_importance
    ):

        class_rows.append(
            {
                "class": class_idx,

                "class_name":
                    class_names[
                        class_idx
                    ],

                "feature":
                    feature,

                "mean_abs_shap":
                    float(
                        importance
                    ),
            }
        )


class_summary = pd.DataFrame(
    class_rows
)


class_summary = class_summary.sort_values(
    [
        "class",
        "mean_abs_shap"
    ],
    ascending=[
        True,
        False
    ]
)


class_summary.to_csv(
    REPORT_DIR
    / "shap_class_importance.csv",
    index=False
)


print(
    "Saved: "
    "reports/shap_class_importance.csv"
)


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SHAP COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)

print(
    f"Classes explained: {n_classes}"
)

print(
    f"Rows explained: {n_rows}"
)

print(
    f"Features explained: {n_features}"
)

print(
    "\nGenerated files:"
)

print(
    "  reports/shap_values.npy"
)

print(
    "  reports/shap_summary.csv"
)

print(
    "  reports/shap_class_importance.csv"
)

print(
    "\nTrustLoop SHAP stage: PASS"
)