from pathlib import Path
import pickle
import sys
import json
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lightgbm_model.pkl"
)

LEARNING_DIR = PROJECT_ROOT / "learning"

PENDING_FEEDBACK_FILE = (
    LEARNING_DIR / "pending_feedback.jsonl"
)

LEARNING_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag.policy_agent import evaluate_policy
from vision.image_analyzer import analyze_image


# ============================================================
# CONSTANTS
# ============================================================

TARGET_CLASSES = {
    0: "Legitimate",
    1: "Policy Abuser",
    2: "Fraudulent Return",
    3: "Wardrobing",
}


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
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"LightGBM model not found:\n{MODEL_PATH}"
        )

    with open(
        MODEL_PATH,
        "rb"
    ) as f:

        model = pickle.load(f)

    actual_features = (
        model.booster_.feature_name()
    )

    if actual_features != EXPECTED_FEATURES:

        raise ValueError(
            "MODEL FEATURE MISMATCH\n\n"
            f"Expected:\n{EXPECTED_FEATURES}\n\n"
            f"Actual:\n{actual_features}"
        )

    return model


# ============================================================
# BUILD MODEL FEATURES
# ============================================================

def build_model_features(return_case):

    data = return_case.copy()

    order_date = pd.to_datetime(
        data.get("order_date"),
        errors="coerce"
    )

    return_date = pd.to_datetime(
        data.get("return_date"),
        errors="coerce"
    )

    if pd.isna(order_date):

        raise ValueError(
            "order_date is required."
        )

    if pd.isna(return_date):

        raise ValueError(
            "return_date is required."
        )

    features = {}

    features["age"] = data.get("age")

    features["account_age_days"] = data.get(
        "account_age_days"
    )

    features["customer_segment"] = data.get(
        "customer_segment"
    )

    features["country"] = data.get(
        "country"
    )

    features["platform"] = data.get(
        "platform"
    )

    features["device_type"] = data.get(
        "device_type"
    )

    features["payment_method"] = data.get(
        "payment_method"
    )

    features["product_category"] = data.get(
        "product_category"
    )

    features["avg_order_value_usd"] = data.get(
        "avg_order_value_usd"
    )

    features["is_high_value_item"] = data.get(
        "is_high_value_item"
    )

    features["discount_used"] = data.get(
        "discount_used"
    )

    features["days_to_return"] = data.get(
        "days_to_return"
    )

    features["return_reason"] = data.get(
        "return_reason"
    )

    features["shipping_carrier"] = data.get(
        "shipping_carrier"
    )

    features["multiple_accounts_flag"] = data.get(
        "multiple_accounts_flag"
    )

    features["wishlist_to_cart_time_hrs"] = data.get(
        "wishlist_to_cart_time_hrs"
    )

    features["customer_return_count_prior"] = data.get(
        "customer_return_count_prior",
        0
    )

    features["returns_last_30d_prior"] = data.get(
        "returns_last_30d_prior",
        0
    )

    features["returns_last_90d_prior"] = data.get(
        "returns_last_90d_prior",
        0
    )

    features["total_returns_lifetime_prior"] = data.get(
        "total_returns_lifetime_prior",
        0
    )

    features["order_date_year"] = order_date.year
    features["order_date_month"] = order_date.month
    features["order_date_day"] = order_date.day
    features["order_date_dayofweek"] = order_date.dayofweek
    features["order_date_dayofyear"] = order_date.dayofyear

    features["order_date_is_weekend"] = int(
        order_date.dayofweek >= 5
    )

    features["return_date_year"] = return_date.year
    features["return_date_month"] = return_date.month
    features["return_date_day"] = return_date.day
    features["return_date_dayofweek"] = return_date.dayofweek
    features["return_date_dayofyear"] = return_date.dayofyear

    features["return_date_is_weekend"] = int(
        return_date.dayofweek >= 5
    )

    features["calculated_days_to_return"] = (
        return_date - order_date
    ).total_seconds() / 86400.0

    X = pd.DataFrame(
        [features],
        columns=EXPECTED_FEATURES
    )

    for column in CATEGORICAL_COLUMNS:

        X[column] = X[column].astype(
            "category"
        )

    return X


# ============================================================
# SAVE PENDING LEARNING CASE
# ============================================================

def save_pending_feedback(
    return_case,
    X,
    ml_result,
    policy_result,
    vision_result,
    final_decision,
):
    """
    Stores the exact model input and prediction for
    later human verification.

    The model prediction is NOT treated as ground truth.
    """

    case_id = return_case.get(
        "case_id"
    )

    if not case_id:

        case_id = (
            "CASE-"
            + datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%d%H%M%S%f"
            )
        )

    feature_snapshot = {}

    for column in EXPECTED_FEATURES:

        value = X.iloc[0][column]

        if hasattr(value, "item"):

            try:
                value = value.item()

            except Exception:
                pass

        try:

            if pd.isna(value):
                value = None

        except Exception:
            pass

        feature_snapshot[column] = value

    record = {

        "case_id":
            case_id,

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "return_case":
            return_case,

        "features":
            feature_snapshot,

        "prediction":
            ml_result["predicted_label"],

        "predicted_class":
            ml_result["predicted_class"],

        "model_confidence":
            ml_result["confidence"],

        "probabilities":
            ml_result["probabilities"],

        "policy_status":
            policy_result.get(
                "policy_status"
            ),

        "vision_result":
            vision_result,

        "final_decision":
            final_decision,

        "verified":
            False,

        "verified_label":
            None,

        "reviewer":
            None,

        "review_notes":
            None,
    }

    # Avoid duplicate case IDs in pending data.
    existing = []

    if PENDING_FEEDBACK_FILE.exists():

        with open(
            PENDING_FEEDBACK_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:
                    old_record = json.loads(
                        line
                    )
                except json.JSONDecodeError:
                    continue

                if (
                    old_record.get("case_id")
                    != case_id
                ):

                    existing.append(
                        old_record
                    )

    existing.append(record)

    with open(
        PENDING_FEEDBACK_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for item in existing:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    default=str
                )
                + "\n"
            )

    return case_id


# ============================================================
# LIGHTGBM PREDICTION
# ============================================================

def predict_risk(
    model,
    X
):

    prediction = model.predict(X)

    probabilities = model.predict_proba(X)[0]

    predicted_class = int(
        prediction[0]
    )

    return {

        "predicted_class":
            predicted_class,

        "predicted_label":
            TARGET_CLASSES[
                predicted_class
            ],

        "probabilities": {

            TARGET_CLASSES[i]:
                float(
                    probabilities[i]
                )

            for i in range(
                len(probabilities)
            )
        },

        "confidence":
            float(
                probabilities[
                    predicted_class
                ]
            ),
    }


# ============================================================
# DECISION CONFIDENCE ENGINE
# ============================================================

def make_final_decision(
    ml_result,
    policy_result,
    vision_result=None,
):
    """
    Multi-signal final decision layer.

    ML confidence is NOT the same as decision confidence.

    The final decision considers:
        - ML classification
        - ML risk probability
        - policy status
        - visual evidence
        - physical damage
        - evidence consistency
        - vision confidence

    Decisions:
        AUTO_APPROVE
        AUTO_REJECT
        HUMAN_INVESTIGATION
    """

    predicted_class = int(
        ml_result["predicted_class"]
    )

    ml_confidence = float(
        ml_result["confidence"]
    )

    probabilities = ml_result.get(
        "probabilities",
        {}
    )

    ml_legitimate_probability = float(
        probabilities.get(
            "Legitimate",
            0.0
        )
    )

    ml_risk_probability = (
        1.0 - ml_legitimate_probability
    )

    policy_status = (
        policy_result.get(
            "policy_status",
            "UNKNOWN"
        )
    )

    # --------------------------------------------------------
    # POLICY SCORE
    # --------------------------------------------------------

    if policy_status == "POLICY_COMPLIANT":

        policy_score = 1.0

    elif policy_status == "POLICY_VIOLATION":

        policy_score = 0.0

    elif policy_status == "HUMAN_ESCALATION":

        policy_score = 0.0

    else:

        policy_score = 0.5


    # --------------------------------------------------------
    # VISION SIGNALS
    # --------------------------------------------------------

    vision_available = (
        vision_result is not None
    )

    vision_confidence = 0.0

    damage_detected = False

    evidence_consistent = None

    product_condition = "UNKNOWN"

    packaging_condition = "UNKNOWN"

    if vision_available:

        vision_confidence = float(
            vision_result.get(
                "confidence",
                0.0
            )
        )

        damage_detected = (
            vision_result.get(
                "damage_detected",
                False
            )
            is True
        )

        evidence_consistent = (
            vision_result.get(
                "evidence_consistent"
            )
        )

        product_condition = (
            vision_result.get(
                "product_condition",
                "UNKNOWN"
            )
        )

        packaging_condition = (
            vision_result.get(
                "packaging_condition",
                "UNKNOWN"
            )
        )


    # ========================================================
    # RULE 1 — POLICY ESCALATION
    # ========================================================

    if policy_status == "HUMAN_ESCALATION":

        return {

            "decision":
                "HUMAN_INVESTIGATION",

            "decision_confidence":
                90.0,

            "confidence":
                0.90,

            "reason":
                "Policy rules require "
                "additional human review.",

            "signals":
                {
                    "ml_legitimate_probability":
                        round(
                            ml_legitimate_probability,
                            4
                        ),

                    "ml_risk_probability":
                        round(
                            ml_risk_probability,
                            4
                        ),

                    "policy_score":
                        policy_score,

                    "vision_confidence":
                        round(
                            vision_confidence,
                            4
                        ),
                },
        }


    # ========================================================
    # RULE 2 — STRONG CONFLICTING VISION EVIDENCE
    # ========================================================

    if (
        vision_available
        and evidence_consistent is False
        and vision_confidence >= 0.60
    ):

        decision_confidence = (
            vision_confidence * 0.60
            +
            ml_risk_probability * 0.40
        )

        return {

            "decision":
                "HUMAN_INVESTIGATION",

            "decision_confidence":
                round(
                    decision_confidence * 100,
                    2
                ),

            "confidence":
                round(
                    decision_confidence,
                    4
                ),

            "reason":
                "Visual evidence conflicts "
                "with the stated return reason. "
                "Human investigation is required.",

            "signals":
                {
                    "ml_legitimate_probability":
                        round(
                            ml_legitimate_probability,
                            4
                        ),

                    "ml_risk_probability":
                        round(
                            ml_risk_probability,
                            4
                        ),

                    "policy_score":
                        policy_score,

                    "vision_confidence":
                        round(
                            vision_confidence,
                            4
                        ),

                    "evidence_consistent":
                        evidence_consistent,
                },
        }


    # ========================================================
    # RULE 3 — CONFIRMED PHYSICAL DAMAGE
    # ========================================================

    if (
        vision_available
        and damage_detected
        and evidence_consistent is True
        and vision_confidence >= 0.60
    ):

        decision_confidence = (
            ml_confidence * 0.45
            +
            vision_confidence * 0.40
            +
            policy_score * 0.15
        )

        return {

            "decision":
                "HUMAN_INVESTIGATION",

            "decision_confidence":
                round(
                    decision_confidence * 100,
                    2
                ),

            "confidence":
                round(
                    decision_confidence,
                    4
                ),

            "reason":
                "ML indicates a legitimate return "
                "and visual evidence confirms physical "
                "damage. The evidence is consistent, "
                "but human verification is required "
                "before approving the damaged return.",

            "signals":
                {
                    "ml_legitimate_probability":
                        round(
                            ml_legitimate_probability,
                            4
                        ),

                    "ml_risk_probability":
                        round(
                            ml_risk_probability,
                            4
                        ),

                    "policy_score":
                        policy_score,

                    "vision_confidence":
                        round(
                            vision_confidence,
                            4
                        ),

                    "damage_detected":
                        damage_detected,

                    "evidence_consistent":
                        evidence_consistent,

                    "product_condition":
                        product_condition,

                    "packaging_condition":
                        packaging_condition,
                },
        }


    # ========================================================
    # RULE 4 — WEAK VISION
    # ========================================================

    if (
        vision_available
        and vision_confidence < 0.50
    ):

        decision_confidence = (
            ml_confidence * 0.60
            +
            policy_score * 0.40
        )

        return {

            "decision":
                "HUMAN_INVESTIGATION",

            "decision_confidence":
                round(
                    decision_confidence * 100,
                    2
                ),

            "confidence":
                round(
                    decision_confidence,
                    4
                ),

            "reason":
                "Visual evidence is not reliable "
                "enough for a fully automated "
                "decision.",

            "signals":
                {
                    "ml_legitimate_probability":
                        round(
                            ml_legitimate_probability,
                            4
                        ),

                    "ml_risk_probability":
                        round(
                            ml_risk_probability,
                            4
                        ),

                    "policy_score":
                        policy_score,

                    "vision_confidence":
                        round(
                            vision_confidence,
                            4
                        ),
                },
        }


    # ========================================================
    # RULE 5 — HIGH-RISK ML
    # ========================================================

    if predicted_class in [1, 2, 3]:

        risk_score = (
            ml_confidence * 0.80
            +
            (1.0 - policy_score) * 0.20
        )

        if (
            ml_confidence >= 0.90
            and risk_score >= 0.85
        ):

            return {

                "decision":
                    "AUTO_REJECT",

                "decision_confidence":
                    round(
                        risk_score * 100,
                        2
                    ),

                "confidence":
                    round(
                        risk_score,
                        4
                    ),

                "reason":
                    "The model detected a "
                    "high-confidence return-risk "
                    "classification. Automatic "
                    "rejection is supported by "
                    "the current evidence hierarchy.",

                "signals":
                    {
                        "ml_legitimate_probability":
                            round(
                                ml_legitimate_probability,
                                4
                            ),

                        "ml_risk_probability":
                            round(
                                ml_risk_probability,
                                4
                            ),

                        "policy_score":
                            policy_score,

                        "vision_confidence":
                            round(
                                vision_confidence,
                                4
                            ),
                    },
            }


        risk_score = min(
            max(
                risk_score,
                0.0
            ),
            1.0
        )

        return {

            "decision":
                "HUMAN_INVESTIGATION",

            "decision_confidence":
                round(
                    risk_score * 100,
                    2
                ),

            "confidence":
                round(
                    risk_score,
                    4
                ),

            "reason":
                "The model detected return-risk "
                "signals that require human "
                "investigation.",

            "signals":
                {
                    "ml_legitimate_probability":
                        round(
                            ml_legitimate_probability,
                            4
                        ),

                    "ml_risk_probability":
                        round(
                            ml_risk_probability,
                            4
                        ),

                    "policy_score":
                        policy_score,

                    "vision_confidence":
                        round(
                            vision_confidence,
                            4
                        ),
                },
        }


    # ========================================================
    # RULE 6 — STRONG LEGITIMATE CASE
    # ========================================================

    legitimate_score = (
        ml_legitimate_probability * 0.75
        +
        policy_score * 0.25
    )

    if (
        legitimate_score >= 0.85
        and ml_confidence >= 0.80
    ):

        return {

            "decision":
                "AUTO_APPROVE",

            "decision_confidence":
                round(
                    legitimate_score * 100,
                    2
                ),

            "confidence":
                round(
                    legitimate_score,
                    4
                ),

            "reason":
                "ML classified the return as "
                "Legitimate with high confidence "
                "and policy checks passed. "
                "No conflicting physical evidence "
                "requires human investigation.",

            "signals":
                {
                    "ml_legitimate_probability":
                        round(
                            ml_legitimate_probability,
                            4
                        ),

                    "ml_risk_probability":
                        round(
                            ml_risk_probability,
                            4
                        ),

                    "policy_score":
                        policy_score,

                    "vision_confidence":
                        round(
                            vision_confidence,
                            4
                        ),
                },
        }


    # ========================================================
    # RULE 7 — AMBIGUOUS CASE
    # ========================================================

    return {

        "decision":
            "HUMAN_INVESTIGATION",

        "decision_confidence":
            round(
                legitimate_score * 100,
                2
            ),

        "confidence":
            round(
                legitimate_score,
                4
            ),

        "reason":
            "Available ML, policy and evidence "
            "signals are not sufficiently decisive "
            "for an automated decision.",

        "signals":
            {
                "ml_legitimate_probability":
                    round(
                        ml_legitimate_probability,
                        4
                    ),

                "ml_risk_probability":
                    round(
                        ml_risk_probability,
                        4
                    ),

                "policy_score":
                    policy_score,

                "vision_confidence":
                    round(
                        vision_confidence,
                        4
                    ),
            },
    }


# ============================================================
# END-TO-END ANALYSIS
# ============================================================

def analyze_return(
    return_case,
    image_path=None,
):

    model = load_model()

    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------

    X = build_model_features(
        return_case
    )

    ml_result = predict_risk(
        model,
        X
    )

    # --------------------------------------------------------
    # POLICY
    # --------------------------------------------------------

    policy_result = evaluate_policy(
        return_case
    )

    # --------------------------------------------------------
    # VISION
    # --------------------------------------------------------

    vision_result = None

    if image_path:

        vision_result = analyze_image(
            image_path,
            return_case.get(
                "return_reason"
            )
        )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    final_decision = make_final_decision(
        ml_result,
        policy_result,
        vision_result
    )

    # --------------------------------------------------------
    # SAVE FOR HUMAN VERIFICATION
    # --------------------------------------------------------

    case_id = save_pending_feedback(
        return_case=return_case,
        X=X,
        ml_result=ml_result,
        policy_result=policy_result,
        vision_result=vision_result,
        final_decision=final_decision,
    )

    return {

        "case_id":
            case_id,

        "ml_result":
            ml_result,

        "policy_result":
            policy_result,

        "vision_result":
            vision_result,

        "final_decision":
            final_decision,

        "learning":
            {
                "status":
                    "PENDING_HUMAN_VERIFICATION",

                "case_id":
                    case_id,

                "message":
                    "Prediction stored for "
                    "human verification. "
                    "It has NOT been added to "
                    "verified training data."
            },
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "TRUSTLOOP — ML + RAG + VISION TEST"
    )

    print("=" * 70)

    test_case = {

        "case_id":
            "TEST-DECISION-CONFIDENCE-001",

        "age":
            32,

        "account_age_days":
            900,

        "customer_segment":
            "regular",

        "country":
            "India",

        "platform":
            "web",

        "device_type":
            "mobile",

        "payment_method":
            "credit_card",

        "product_category":
            "electronics",

        "avg_order_value_usd":
            180.0,

        "is_high_value_item":
            0,

        "discount_used":
            0,

        "order_date":
            "2026-06-01",

        "return_date":
            "2026-06-10",

        "days_to_return":
            9,

        "return_reason":
            "Product arrived damaged",

        "shipping_carrier":
            "standard",

        "multiple_accounts_flag":
            0,

        "wishlist_to_cart_time_hrs":
            30.0,

        "customer_return_count_prior":
            0,

        "returns_last_30d_prior":
            0,

        "returns_last_90d_prior":
            0,

        "total_returns_lifetime_prior":
            0,
    }

    image_path = (
        PROJECT_ROOT
        / "test_images"
        / "return_test.jpg.png"
    )

    print(
        "\nRunning complete TrustLoop pipeline..."
    )

    result = analyze_return(
        test_case,
        str(image_path)
    )

    # ========================================================
    # ML RESULT
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "ML RESULT"
    )

    print(
        "Prediction:",
        result[
            "ml_result"
        ][
            "predicted_label"
        ]
    )

    print(
        "ML Confidence:",
        f"{result['ml_result']['confidence']:.4f}"
    )

    print(
        "\nProbabilities:"
    )

    for label, probability in (
        result[
            "ml_result"
        ][
            "probabilities"
        ].items()
    ):

        print(
            f"  {label}: "
            f"{probability:.4f}"
        )

    # ========================================================
    # POLICY
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "POLICY RESULT"
    )

    print(
        "Status:",
        result[
            "policy_result"
        ][
            "policy_status"
        ]
    )

    if result[
        "policy_result"
    ][
        "flags"
    ]:

        for flag in result[
            "policy_result"
        ][
            "flags"
        ]:

            print(
                f"  - {flag['reason']}"
            )

    else:

        print(
            "Flags: None"
        )

    # ========================================================
    # VISION
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "VISION RESULT"
    )

    vision = result[
        "vision_result"
    ]

    if vision:

        print(
            "Image quality:",
            vision[
                "image_quality"
            ]
        )

        print(
            "Product condition:",
            vision[
                "product_condition"
            ]
        )

        print(
            "Damage detected:",
            vision[
                "damage_detected"
            ]
        )

        print(
            "Packaging:",
            vision[
                "packaging_condition"
            ]
        )

        print(
            "Evidence consistent:",
            vision[
                "evidence_consistent"
            ]
        )

        print(
            "Vision confidence:",
            f"{vision['confidence']:.4f}"
        )

        print(
            "Explanation:",
            vision[
                "explanation"
            ]
        )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "FINAL TRUSTLOOP DECISION"
    )

    final = result[
        "final_decision"
    ]

    print(
        "Decision:",
        final[
            "decision"
        ]
    )

    print(
        "Decision Confidence:",
        f"{final['decision_confidence']:.2f}%"
    )

    print(
        "Reason:",
        final[
            "reason"
        ]
    )

    # ========================================================
    # DECISION SIGNALS
    # ========================================================

    print(
        "\nDecision Signals:"
    )

    for key, value in final.get(
        "signals",
        {}
    ).items():

        print(
            f"  {key}: {value}"
        )

    # ========================================================
    # LEARNING
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "SELF-LEARNING PIPELINE"
    )

    print(
        "Case ID:",
        result[
            "learning"
        ][
            "case_id"
        ]
    )

    print(
        "Status:",
        result[
            "learning"
        ][
            "status"
        ]
    )

    print(
        "Pending human verification."
    )

    print(
        "\nPending feedback file:"
    )

    print(
        PENDING_FEEDBACK_FILE
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TRUSTLOOP ML + RAG + VISION TEST COMPLETED"
    )

    print(
        "=" * 70
    )
