from pathlib import Path
import json

from learning.feedback_store import load_feedback
from learning.label_mapper import normalize_label, label_to_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EVALUATION_FILE = REPORT_DIR / "learning_evaluation.json"

MINIMUM_FEEDBACK_FOR_RETRAINING = 50


def evaluate_feedback() -> dict:

    feedback = load_feedback()

    if not feedback:
        result = {
            "status": "INSUFFICIENT_DATA",
            "feedback_count": 0,
            "valid_verified_records": 0,
            "invalid_records": 0,
            "minimum_required": MINIMUM_FEEDBACK_FOR_RETRAINING,
            "label_distribution": {
                "Legitimate": 0,
                "Policy Abuser": 0,
                "Fraudulent Return": 0,
                "Wardrobing": 0,
            },
            "all_classes_present": False,
            "retraining_ready": False,
            "reason": "No verified feedback exists yet.",
        }

        save_result(result)
        return result

    valid_records = []
    invalid_records = []

    for record in feedback:

        try:

            if not record.get("verified", False):
                invalid_records.append(record)
                continue

            verified_label = normalize_label(
                record["verified_label"]
            )

            record["normalized_label"] = verified_label
            record["label_name"] = label_to_name(
                verified_label
            )

            valid_records.append(record)

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            invalid_records.append(record)

    label_counts = {
        "Legitimate": 0,
        "Policy Abuser": 0,
        "Fraudulent Return": 0,
        "Wardrobing": 0,
    }

    for record in valid_records:

        label_name = record["label_name"]

        if label_name in label_counts:
            label_counts[label_name] += 1

    total_valid = len(valid_records)

    enough_data = (
        total_valid >= MINIMUM_FEEDBACK_FOR_RETRAINING
    )

    all_classes_present = all(
        count > 0
        for count in label_counts.values()
    )

    retraining_ready = (
        enough_data
        and all_classes_present
    )

    result = {
        "status": (
            "READY_FOR_RETRAINING"
            if retraining_ready
            else "INSUFFICIENT_DATA"
        ),
        "feedback_count": len(feedback),
        "valid_verified_records": total_valid,
        "invalid_records": len(invalid_records),
        "minimum_required": MINIMUM_FEEDBACK_FOR_RETRAINING,
        "label_distribution": label_counts,
        "all_classes_present": all_classes_present,
        "retraining_ready": retraining_ready,
    }

    if not enough_data:

        result["reason"] = (
            f"Need at least "
            f"{MINIMUM_FEEDBACK_FOR_RETRAINING} "
            f"verified records."
        )

    elif not all_classes_present:

        result["reason"] = (
            "Every TrustLoop class must have "
            "at least one verified example."
        )

    save_result(result)

    return result


def save_result(result: dict):

    with open(
        EVALUATION_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )


if __name__ == "__main__":

    print("=" * 70)
    print("TRUSTLOOP MODEL LEARNING EVALUATOR")
    print("=" * 70)

    result = evaluate_feedback()

    print(
        f"\nTotal feedback records: "
        f"{result['feedback_count']}"
    )

    print(
        f"Verified feedback: "
        f"{result['valid_verified_records']}"
    )

    print(
        f"Invalid/unverified records: "
        f"{result['invalid_records']}"
    )

    print(
        f"Minimum required: "
        f"{result['minimum_required']}"
    )

    print("\nLabel distribution:")

    for label, count in result[
        "label_distribution"
    ].items():

        print(
            f"  {label}: {count}"
        )

    print(
        "\nAll classes present:",
        result["all_classes_present"],
    )

    print(
        "Retraining ready:",
        result["retraining_ready"],
    )

    if "reason" in result:

        print(
            "\nReason:",
            result["reason"],
        )

    print(
        "\nEvaluation report:",
        EVALUATION_FILE,
    )
