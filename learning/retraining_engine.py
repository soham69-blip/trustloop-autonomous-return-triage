from pathlib import Path
from datetime import datetime, timezone
import json
import pickle
import shutil

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
LEARNING_DIR = PROJECT_ROOT / "learning"

CURRENT_MODEL = (
    MODEL_DIR / "lightgbm_model.pkl"
)

BACKUP_MODEL = (
    MODEL_DIR / "lightgbm_model_before_retraining.pkl"
)

CANDIDATE_MODEL = (
    MODEL_DIR / "lightgbm_candidate.pkl"
)

FEEDBACK_FILE = (
    LEARNING_DIR / "verified_feedback.jsonl"
)

ORIGINAL_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trustloop"
    / "model_ready.csv"
)

VERIFIED_DATASET = (
    LEARNING_DIR / "verified_training_data.csv"
)

RETRAINING_STATUS = (
    REPORT_DIR / "retraining_status.json"
)

CANDIDATE_REPORT = (
    REPORT_DIR / "candidate_model_evaluation.json"
)


# ============================================================
# SAFETY / PROMOTION PARAMETERS
# ============================================================

MINIMUM_FEEDBACK = 50

# Verified human feedback gets more influence than a normal
# original training example.
FEEDBACK_SAMPLE_WEIGHT = 15.0

# Candidate must improve macro F1 by at least 0.5 percentage
# points before qualifying for promotion.
MIN_MACRO_F1_IMPROVEMENT = 0.005

# A protected non-legitimate class cannot lose more than
# 2 percentage points of recall.
MIN_CLASS_RECALL_DROP = 0.02


# ============================================================
# LABELS
# ============================================================

TARGET_CLASSES = {
    0: "Legitimate",
    1: "Policy Abuser",
    2: "Fraudulent Return",
    3: "Wardrobing",
}

LABEL_TO_ID = {
    name: label
    for label, name in TARGET_CLASSES.items()
}

PROTECTED_CLASSES = [
    "Policy Abuser",
    "Fraudulent Return",
    "Wardrobing",
]


# ============================================================
# MODEL FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "age",
    "account_age_days",
    "customer_segment",
    "country",
    "platform",
    "device_type",
    "payment_method",
    "product_category",
    "avg_order_value_usd",
    "is_high_value_item",
    "discount_used",
    "days_to_return",
    "return_reason",
    "shipping_carrier",
    "multiple_accounts_flag",
    "wishlist_to_cart_time_hrs",
    "customer_return_count_prior",
    "returns_last_30d_prior",
    "returns_last_90d_prior",
    "total_returns_lifetime_prior",
    "order_date_year",
    "order_date_month",
    "order_date_day",
    "order_date_dayofweek",
    "order_date_dayofyear",
    "order_date_is_weekend",
    "return_date_year",
    "return_date_month",
    "return_date_day",
    "return_date_dayofweek",
    "return_date_dayofyear",
    "return_date_is_weekend",
    "calculated_days_to_return",
]

CATEGORICAL_COLUMNS = [
    "country",
    "customer_segment",
    "device_type",
    "payment_method",
    "platform",
    "product_category",
    "return_reason",
    "shipping_carrier",
]


# ============================================================
# UTILITIES
# ============================================================

def now_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_verified_feedback():
    """
    Load only explicitly verified feedback.

    The model's own prediction is never treated as verified
    ground truth.
    """

    if not FEEDBACK_FILE.exists():
        return []

    records = []

    with open(
        FEEDBACK_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("verified") is True:
                records.append(record)

    return records


def validate_feedback(records):

    valid = []
    problems = []
    seen_case_ids = set()

    for record in records:

        case_id = record.get(
            "case_id"
        )

        label = record.get(
            "verified_label"
        )

        features = record.get(
            "features"
        )

        if not case_id:

            problems.append(
                "Missing case_id."
            )

            continue

        if case_id in seen_case_ids:

            problems.append(
                f"Duplicate case_id: {case_id}"
            )

            continue

        seen_case_ids.add(case_id)

        if label not in LABEL_TO_ID:

            problems.append(
                f"{case_id}: invalid verified "
                f"label '{label}'."
            )

            continue

        if (
            not isinstance(
                features,
                dict
            )
            or not features
        ):

            problems.append(
                f"{case_id}: missing feature snapshot."
            )

            continue

        missing = [
            feature
            for feature in EXPECTED_FEATURES
            if feature not in features
        ]

        if missing:

            problems.append(
                f"{case_id}: missing features "
                f"{missing}"
            )

            continue

        valid.append(record)

    return {
        "valid_records": valid,
        "problems": problems,
    }


def label_distribution(records):

    result = {
        name: 0
        for name in TARGET_CLASSES.values()
    }

    for record in records:

        label = record.get(
            "verified_label"
        )

        if label in result:
            result[label] += 1

    return result


# ============================================================
# ORIGINAL DATASET
# ============================================================

def load_original_raw():

    if not ORIGINAL_DATASET.exists():

        raise FileNotFoundError(
            "Original training dataset not found:\n"
            f"{ORIGINAL_DATASET}"
        )

    return pd.read_csv(
        ORIGINAL_DATASET,
        low_memory=False
    )


def prepare_dataframe_from_raw(df):

    working = df.copy()

    datetime_columns = []

    for column in [
        "order_date",
        "return_date",
    ]:

        if column in working.columns:

            working[column] = pd.to_datetime(
                working[column],
                errors="coerce"
            )

            datetime_columns.append(
                column
            )

    for column in datetime_columns:

        working[
            f"{column}_year"
        ] = working[column].dt.year

        working[
            f"{column}_month"
        ] = working[column].dt.month

        working[
            f"{column}_day"
        ] = working[column].dt.day

        working[
            f"{column}_dayofweek"
        ] = working[column].dt.dayofweek

        working[
            f"{column}_dayofyear"
        ] = working[column].dt.dayofyear

        working[
            f"{column}_is_weekend"
        ] = (
            working[column]
            .dt.dayofweek
            .ge(5)
            .astype(int)
        )

    if (
        "order_date" in working.columns
        and "return_date" in working.columns
    ):

        working[
            "calculated_days_to_return"
        ] = (
            (
                working["return_date"]
                -
                working["order_date"]
            )
            .dt.total_seconds()
            / 86400.0
        )

    working = working.drop(
        columns=datetime_columns,
        errors="ignore"
    )

    missing = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in working.columns
    ]

    if missing:

        raise ValueError(
            "Dataset is missing model features: "
            f"{missing}"
        )

    if "abuse_label" not in working.columns:

        raise ValueError(
            "abuse_label column is missing."
        )

    return working[
        EXPECTED_FEATURES
        + ["abuse_label"]
    ].copy()


def prepare_original_dataset():

    raw = load_original_raw()

    return prepare_dataframe_from_raw(
        raw
    )


# ============================================================
# FEEDBACK DATAFRAME
# ============================================================

def feedback_to_dataframe(records):

    rows = []

    for record in records:

        row = {
            feature:
                record["features"][feature]
            for feature in EXPECTED_FEATURES
        }

        row["abuse_label"] = (
            LABEL_TO_ID[
                record["verified_label"]
            ]
        )

        row["case_id"] = record[
            "case_id"
        ]

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# ============================================================
# ORIGINAL TEMPORAL SPLIT
# ============================================================

def create_original_split():

    raw = load_original_raw()

    raw["order_date"] = pd.to_datetime(
        raw["order_date"],
        errors="coerce"
    )

    raw["return_date"] = pd.to_datetime(
        raw["return_date"],
        errors="coerce"
    )

    raw = raw.sort_values(
        by=[
            "return_date",
            "order_date",
        ],
        kind="mergesort"
    ).reset_index(
        drop=True
    )

    n = len(raw)

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.85
    )

    train_raw = raw.iloc[
        :train_end
    ].copy()

    validation_raw = raw.iloc[
        train_end:validation_end
    ].copy()

    test_raw = raw.iloc[
        validation_end:
    ].copy()

    train = prepare_dataframe_from_raw(
        train_raw
    )

    validation = prepare_dataframe_from_raw(
        validation_raw
    )

    test = prepare_dataframe_from_raw(
        test_raw
    )

    return (
        train,
        validation,
        test,
    )


# ============================================================
# CATEGORICAL ALIGNMENT
# ============================================================

def make_category_maps(*dataframes):

    category_maps = {}

    for column in CATEGORICAL_COLUMNS:

        values = []

        for df in dataframes:

            if column in df.columns:

                series = (
                    df[column]
                    .dropna()
                    .astype(str)
                )

                values.extend(
                    series.tolist()
                )

        categories = sorted(
            set(values)
        )

        category_maps[column] = categories

    return category_maps


def apply_category_maps(
    X,
    category_maps
):

    X = X.copy()

    for column in CATEGORICAL_COLUMNS:

        categories = category_maps[
            column
        ]

        X[column] = pd.Categorical(
            X[column].astype(str),
            categories=categories
        )

    return X


def prepare_feature_sets(
    training_df,
    validation_df,
    test_df
):

    X_train = training_df[
        EXPECTED_FEATURES
    ].copy()

    y_train = training_df[
        "abuse_label"
    ].astype(int)

    X_validation = validation_df[
        EXPECTED_FEATURES
    ].copy()

    y_validation = validation_df[
        "abuse_label"
    ].astype(int)

    X_test = test_df[
        EXPECTED_FEATURES
    ].copy()

    y_test = test_df[
        "abuse_label"
    ].astype(int)

    category_maps = make_category_maps(
        X_train,
        X_validation,
        X_test
    )

    X_train = apply_category_maps(
        X_train,
        category_maps
    )

    X_validation = apply_category_maps(
        X_validation,
        category_maps
    )

    X_test = apply_category_maps(
        X_test,
        category_maps
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


# ============================================================
# TRAINING WEIGHTS
# ============================================================

def build_training_weights(
    original_training,
    feedback_df
):

    original_weights = np.ones(
        len(original_training),
        dtype=float
    )

    feedback_weights = np.full(
        len(feedback_df),
        FEEDBACK_SAMPLE_WEIGHT,
        dtype=float
    )

    return np.concatenate(
        [
            original_weights,
            feedback_weights,
        ]
    )


# ============================================================
# TRAIN CANDIDATE
# ============================================================

def train_candidate(
    training_df,
    sample_weights
):
    """
    Train the candidate using original training data plus
    weighted verified feedback.

    IMPORTANT:
    We intentionally do NOT pass eval_set/eval_X/eval_y.

    The installed LightGBM version in this environment has
    rejected those validation arguments. Candidate evaluation
    is performed separately against the untouched original
    test set, which is the stronger deployment gate anyway.
    """

    X_train = training_df[
        EXPECTED_FEATURES
    ].copy()

    y_train = training_df[
        "abuse_label"
    ].astype(int)

    # All categorical features use pandas category dtype.
    category_maps = make_category_maps(
        X_train
    )

    X_train = apply_category_maps(
        X_train,
        category_maps
    )

    candidate = lgb.LGBMClassifier(

        objective="multiclass",

        num_class=4,

        n_estimators=500,

        learning_rate=0.05,

        num_leaves=31,

        max_depth=-1,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        n_jobs=-1,

        verbosity=-1,
    )

    candidate.fit(

        X_train,

        y_train,

        sample_weight=sample_weights,

        categorical_feature=CATEGORICAL_COLUMNS,
    )

    return candidate


# ============================================================
# PREDICTION PREPARATION
# ============================================================

def prepare_evaluation_features(
    dataset_df
):

    X = dataset_df[
        EXPECTED_FEATURES
    ].copy()

    for column in CATEGORICAL_COLUMNS:

        values = (
            X[column]
            .dropna()
            .astype(str)
        )

        categories = sorted(
            set(values.tolist())
        )

        X[column] = pd.Categorical(
            X[column].astype(str),
            categories=categories
        )

    return X


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    dataset_df
):

    X = prepare_evaluation_features(
        dataset_df
    )

    y = dataset_df[
        "abuse_label"
    ].astype(int)

    predictions = np.asarray(
        model.predict(X)
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y,
        predictions,
        labels=[
            0,
            1,
            2,
            3,
        ],
        target_names=[
            TARGET_CLASSES[0],
            TARGET_CLASSES[1],
            TARGET_CLASSES[2],
            TARGET_CLASSES[3],
        ],
        output_dict=True,
        zero_division=0
    )

    return {

        "accuracy":
            float(accuracy),

        "macro_f1":
            float(macro_f1),

        "weighted_f1":
            float(weighted_f1),

        "classification_report":
            report,

        "evaluated_records":
            len(dataset_df),
    }


# ============================================================
# SAFETY CHECKS
# ============================================================

def no_class_collapsed(
    metrics
):

    report = metrics[
        "classification_report"
    ]

    for label in PROTECTED_CLASSES:

        class_f1 = float(
            report[label][
                "f1-score"
            ]
        )

        if class_f1 <= 0.0:

            return False

    return True


def recall_deltas(
    production_metrics,
    candidate_metrics
):

    production_report = (
        production_metrics[
            "classification_report"
        ]
    )

    candidate_report = (
        candidate_metrics[
            "classification_report"
        ]
    )

    deltas = {}

    for label in [
        "Legitimate",
        "Policy Abuser",
        "Fraudulent Return",
        "Wardrobing",
    ]:

        production_recall = float(
            production_report[label][
                "recall"
            ]
        )

        candidate_recall = float(
            candidate_report[label][
                "recall"
            ]
        )

        deltas[label] = {

            "production_recall":
                production_recall,

            "candidate_recall":
                candidate_recall,

            "delta":
                candidate_recall
                -
                production_recall,
        }

    return deltas


def protected_class_recall_safe(
    production_metrics,
    candidate_metrics
):

    deltas = recall_deltas(
        production_metrics,
        candidate_metrics
    )

    for label in PROTECTED_CLASSES:

        drop = (
            -deltas[label]["delta"]
        )

        if drop > MIN_CLASS_RECALL_DROP:

            return False

    return True


def candidate_is_better(
    production_metrics,
    candidate_metrics
):

    production_macro = (
        production_metrics[
            "macro_f1"
        ]
    )

    candidate_macro = (
        candidate_metrics[
            "macro_f1"
        ]
    )

    macro_improvement = (
        candidate_macro
        -
        production_macro
    )

    class_safety = (
        no_class_collapsed(
            candidate_metrics
        )
    )

    recall_safety = (
        protected_class_recall_safe(
            production_metrics,
            candidate_metrics
        )
    )

    return (
        macro_improvement
        >= MIN_MACRO_F1_IMPROVEMENT
        and class_safety
        and recall_safety
    )


# ============================================================
# PRODUCTION MODEL
# ============================================================

def load_current_model():

    if not CURRENT_MODEL.exists():

        raise FileNotFoundError(
            "Production model not found:\n"
            f"{CURRENT_MODEL}"
        )

    with open(
        CURRENT_MODEL,
        "rb"
    ) as f:

        return pickle.load(f)


def backup_current_model():

    if not CURRENT_MODEL.exists():

        return False

    shutil.copy2(
        CURRENT_MODEL,
        BACKUP_MODEL
    )

    return True


def promote_candidate():

    if not CANDIDATE_MODEL.exists():

        raise FileNotFoundError(
            "Candidate model does not exist."
        )

    if not backup_current_model():

        raise RuntimeError(
            "Could not create production backup."
        )

    shutil.copy2(
        CANDIDATE_MODEL,
        CURRENT_MODEL
    )


# ============================================================
# MAIN RETRAINING ENGINE
# ============================================================

def run_retraining(
    allow_promotion=False
):

    print("=" * 70)

    print(
        "TRUSTLOOP CONTROLLED "
        "RETRAINING ENGINE V8"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # VERIFIED FEEDBACK
    # --------------------------------------------------------

    feedback_records = (
        load_verified_feedback()
    )

    print(
        f"\nVerified feedback records: "
        f"{len(feedback_records)}"
    )

    validation = validate_feedback(
        feedback_records
    )

    valid_feedback = validation[
        "valid_records"
    ]

    problems = validation[
        "problems"
    ]

    print(
        f"Valid verified records: "
        f"{len(valid_feedback)}"
    )

    distribution = (
        label_distribution(
            valid_feedback
        )
    )

    print(
        "\nVerified feedback distribution:"
    )

    for label, count in distribution.items():

        print(
            f"  {label}: {count}"
        )

    all_classes_present = all(
        count > 0
        for count in distribution.values()
    )

    print(
        "\nAll classes present:",
        all_classes_present
    )

    if problems:

        print(
            "\nValidation problems:"
        )

        for problem in problems[:20]:

            print(
                f"  - {problem}"
            )

    # --------------------------------------------------------
    # ORIGINAL DATASET
    # --------------------------------------------------------

    print(
        "\nLoading original dataset..."
    )

    original_df = (
        prepare_original_dataset()
    )

    print(
        "Original rows:",
        len(original_df)
    )

    # --------------------------------------------------------
    # ORIGINAL TEMPORAL SPLIT
    # --------------------------------------------------------

    print(
        "\nCreating original temporal split..."
    )

    (
        original_train,
        original_validation,
        original_test
    ) = create_original_split()

    print(
        "Original train:",
        len(original_train)
    )

    print(
        "Original validation:",
        len(original_validation)
    )

    print(
        "Original test:",
        len(original_test)
    )

    # --------------------------------------------------------
    # FEEDBACK DATAFRAME
    # --------------------------------------------------------

    feedback_df = feedback_to_dataframe(
        valid_feedback
    )

    feedback_df.to_csv(
        VERIFIED_DATASET,
        index=False
    )

    # --------------------------------------------------------
    # COMBINE TRAINING DATA
    # --------------------------------------------------------

    combined_training = pd.concat(
        [
            original_train,

            feedback_df[
                EXPECTED_FEATURES
                + ["abuse_label"]
            ],
        ],
        ignore_index=True
    )

    sample_weights = (
        build_training_weights(
            original_train,
            feedback_df
        )
    )

    print(
        "\nCandidate training rows:",
        len(combined_training)
    )

    print(
        "Original training rows:",
        len(original_train)
    )

    print(
        "Verified feedback rows:",
        len(feedback_df)
    )

    print(
        "Feedback sample weight:",
        FEEDBACK_SAMPLE_WEIGHT
    )

    # --------------------------------------------------------
    # TRAIN CANDIDATE
    # --------------------------------------------------------

    print(
        "\nTraining weighted candidate..."
    )

    candidate = train_candidate(
        combined_training,
        sample_weights
    )

    with open(
        CANDIDATE_MODEL,
        "wb"
    ) as f:

        pickle.dump(
            candidate,
            f
        )

    print(
        "Candidate saved:"
    )

    print(
        CANDIDATE_MODEL
    )

    # --------------------------------------------------------
    # LOAD PRODUCTION
    # --------------------------------------------------------

    production = (
        load_current_model()
    )

    # --------------------------------------------------------
    # EVALUATE PRODUCTION
    # --------------------------------------------------------

    print(
        "\nEvaluating production "
        "on untouched original test..."
    )

    production_metrics = evaluate_model(
        production,
        original_test
    )

    # --------------------------------------------------------
    # EVALUATE CANDIDATE
    # --------------------------------------------------------

    print(
        "\nEvaluating candidate "
        "on same untouched original test..."
    )

    candidate_metrics = evaluate_model(
        candidate,
        original_test
    )

    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    print(
        "\nPRODUCTION MODEL"
    )

    print(
        "Accuracy:",
        f"{production_metrics['accuracy']:.4f}"
    )

    print(
        "Macro F1:",
        f"{production_metrics['macro_f1']:.4f}"
    )

    print(
        "Weighted F1:",
        f"{production_metrics['weighted_f1']:.4f}"
    )

    print(
        "\nCANDIDATE MODEL"
    )

    print(
        "Accuracy:",
        f"{candidate_metrics['accuracy']:.4f}"
    )

    print(
        "Macro F1:",
        f"{candidate_metrics['macro_f1']:.4f}"
    )

    print(
        "Weighted F1:",
        f"{candidate_metrics['weighted_f1']:.4f}"
    )

    # --------------------------------------------------------
    # SAFETY METRICS
    # --------------------------------------------------------

    macro_improvement = (
        candidate_metrics["macro_f1"]
        -
        production_metrics["macro_f1"]
    )

    deltas = recall_deltas(
        production_metrics,
        candidate_metrics
    )

    class_safety = (
        no_class_collapsed(
            candidate_metrics
        )
    )

    recall_safety = (
        protected_class_recall_safe(
            production_metrics,
            candidate_metrics
        )
    )

    candidate_better = (
        candidate_is_better(
            production_metrics,
            candidate_metrics
        )
    )

    minimum_feedback_met = (
        len(valid_feedback)
        >= MINIMUM_FEEDBACK
    )

    promotion_ready = (
        minimum_feedback_met
        and all_classes_present
        and not problems
        and candidate_better
    )

    print(
        "\nMacro F1 improvement:",
        f"{macro_improvement:.4f}"
    )

    print(
        "Candidate class safety:",
        class_safety
    )

    print(
        "Protected recall safety:",
        recall_safety
    )

    print(
        "\nRecall deltas:"
    )

    for label, values in deltas.items():

        print(
            f"  {label}: "
            f"{values['delta']:+.4f}"
        )

    print(
        "\nCandidate better:",
        candidate_better
    )

    print(
        "Minimum feedback met:",
        minimum_feedback_met
    )

    print(
        "Promotion-ready:",
        promotion_ready
    )

    # --------------------------------------------------------
    # SAFE NON-PROMOTION
    # --------------------------------------------------------

    if not promotion_ready:

        reasons = []

        if not minimum_feedback_met:

            reasons.append(
                "minimum verified-feedback "
                "threshold not reached"
            )

        if not all_classes_present:

            reasons.append(
                "not all classes represented"
            )

        if problems:

            reasons.append(
                "feedback validation problems exist"
            )

        if not class_safety:

            reasons.append(
                "a protected class collapsed"
            )

        if not recall_safety:

            reasons.append(
                "protected-class recall "
                "decreased too much"
            )

        if (
            macro_improvement
            < MIN_MACRO_F1_IMPROVEMENT
        ):

            reasons.append(
                "macro-F1 improvement below "
                f"{MIN_MACRO_F1_IMPROVEMENT:.3f}"
            )

        if not reasons:

            reasons.append(
                "candidate did not satisfy "
                "promotion criteria"
            )

        reason = (
            "Candidate was not promoted: "
            + "; ".join(reasons)
            + "."
        )

        return {

            "status":
                "CANDIDATE_EVALUATED_NOT_PROMOTED",

            "reason":
                reason,

            "architecture":
                "weighted_original_plus_verified_feedback",

            "original_rows":
                len(original_df),

            "original_train_rows":
                len(original_train),

            "original_validation_rows":
                len(original_validation),

            "original_test_rows":
                len(original_test),

            "verified_feedback_rows":
                len(valid_feedback),

            "feedback_sample_weight":
                FEEDBACK_SAMPLE_WEIGHT,

            "minimum_feedback":
                MINIMUM_FEEDBACK,

            "minimum_feedback_met":
                minimum_feedback_met,

            "all_classes_present":
                all_classes_present,

            "macro_f1_improvement":
                macro_improvement,

            "minimum_macro_f1_improvement":
                MIN_MACRO_F1_IMPROVEMENT,

            "protected_class_recall_drop_limit":
                MIN_CLASS_RECALL_DROP,

            "candidate_class_safety":
                class_safety,

            "protected_recall_safety":
                recall_safety,

            "recall_deltas":
                deltas,

            "candidate_better":
                candidate_better,

            "promotion_ready":
                False,

            "production_metrics":
                production_metrics,

            "candidate_metrics":
                candidate_metrics,

            "candidate_model":
                str(CANDIDATE_MODEL),

            "production_model_changed":
                False,

            "timestamp":
                now_iso(),
        }

    # --------------------------------------------------------
    # EXPLICIT PROMOTION
    # --------------------------------------------------------

    if allow_promotion:

        print(
            "\nPROMOTING CANDIDATE..."
        )

        promote_candidate()

        return {

            "status":
                "CANDIDATE_PROMOTED",

            "reason":
                "Candidate passed all "
                "promotion gates.",

            "production_model_changed":
                True,

            "candidate_better":
                True,

            "promotion_ready":
                True,

            "production_metrics":
                production_metrics,

            "candidate_metrics":
                candidate_metrics,

            "timestamp":
                now_iso(),
        }

    # --------------------------------------------------------
    # PASSED BUT NOT EXPLICITLY PROMOTED
    # --------------------------------------------------------

    return {

        "status":
            "CANDIDATE_APPROVED_PENDING_PROMOTION",

        "reason":
            "Candidate passed all promotion "
            "gates, but promotion was not "
            "explicitly enabled.",

        "production_model_changed":
            False,

        "candidate_better":
            True,

        "promotion_ready":
            True,

        "production_metrics":
            production_metrics,

        "candidate_metrics":
            candidate_metrics,

        "timestamp":
            now_iso(),
    }


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    result
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RETRAINING_STATUS,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    with open(
        CANDIDATE_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    result = run_retraining(
        allow_promotion=False
    )

    save_reports(
        result
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRAINING RESULT"
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    print(
        "\nSaved:"
    )

    print(
        RETRAINING_STATUS
    )

    print(
        CANDIDATE_REPORT
    )

    print(
        "=" * 70
    )
