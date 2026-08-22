from pathlib import Path
import json
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEEDBACK_FILE = (
    PROJECT_ROOT
    / "learning"
    / "verified_feedback.jsonl"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "feedback_quality.json"
)


VALID_LABELS = {
    "Legitimate",
    "Policy Abuser",
    "Fraudulent Return",
    "Wardrobing",
}


# ------------------------------------------------------------
# These reviewers/case prefixes are explicitly demo/test data.
# They must NEVER be treated as production learning evidence.
# ------------------------------------------------------------

DEMO_REVIEWERS = {
    "human-test",
    "synthetic-test",
    "scenario-test",
}

DEMO_CASE_PREFIXES = (
    "TEST-",
    "TRAIN-",
    "SCENARIO-",
)


def load_feedback():

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
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:
                continue

    return records


def classify_record(record):

    case_id = str(
        record.get(
            "case_id",
            ""
        )
    )

    reviewer = str(
        record.get(
            "reviewer",
            ""
        )
    ).strip()

    if (
        reviewer in DEMO_REVIEWERS
        or case_id.startswith(
            DEMO_CASE_PREFIXES
        )
    ):

        return "DEMO_TEST"

    if reviewer:

        return "PRODUCTION"

    return "UNKNOWN"


def validate_record(record):

    problems = []

    case_id = record.get(
        "case_id"
    )

    if not case_id:
        problems.append(
            "missing_case_id"
        )

    if record.get(
        "verified"
    ) is not True:

        problems.append(
            "not_verified"
        )

    label = record.get(
        "verified_label"
    )

    if label not in VALID_LABELS:

        problems.append(
            "invalid_verified_label"
        )

    features = record.get(
        "features"
    )

    if not isinstance(
        features,
        dict
    ) or not features:

        problems.append(
            "missing_features"
        )

    reviewer = record.get(
        "reviewer"
    )

    if not reviewer:

        problems.append(
            "missing_reviewer"
        )

    notes = (
        record.get("notes")
        or
        record.get("review_notes")
    )

    if not notes:

        problems.append(
            "missing_review_notes"
        )

    return problems


def main():

    records = load_feedback()

    production_records = []
    demo_records = []
    unknown_records = []

    quality_distribution = Counter()
    label_distribution = Counter()

    for record in records:

        classification = classify_record(
            record
        )

        problems = validate_record(
            record
        )

        case_id = record.get(
            "case_id"
        )

        label = record.get(
            "verified_label"
        )

        if classification == "PRODUCTION":

            target = production_records

        elif classification == "DEMO_TEST":

            target = demo_records

        else:

            target = unknown_records

        target.append({

            "case_id":
                case_id,

            "label":
                label,

            "reviewer":
                record.get(
                    "reviewer"
                ),

            "validation_problems":
                problems,
        })

        if not problems:

            quality_distribution[
                "HIGH_QUALITY"
            ] += 1

        else:

            quality_distribution[
                "NEEDS_REVIEW"
            ] += 1

        if label in VALID_LABELS:

            label_distribution[
                label
            ] += 1

    production_high_quality = [
        item
        for item in production_records
        if not item[
            "validation_problems"
        ]
    ]

    production_labels = Counter(
        item["label"]
        for item in production_high_quality
        if item["label"] in VALID_LABELS
    )

    production_all_classes_present = all(
        production_labels[label] > 0
        for label in VALID_LABELS
    )

    production_ready = (
        len(production_high_quality) >= 50
        and production_all_classes_present
    )

    report = {

        "total_feedback_records":
            len(records),

        "production_records":
            len(production_records),

        "production_high_quality_records":
            len(production_high_quality),

        "demo_test_records":
            len(demo_records),

        "unknown_records":
            len(unknown_records),

        "overall_quality_distribution":
            dict(
                quality_distribution
            ),

        "overall_label_distribution":
            dict(
                label_distribution
            ),

        "production_label_distribution":
            dict(
                production_labels
            ),

        "production_all_classes_present":
            production_all_classes_present,

        "minimum_production_records":
            50,

        "production_learning_ready":
            production_ready,

        "production_records_detail":
            production_records,

        "demo_test_records_detail":
            demo_records,

        "unknown_records_detail":
            unknown_records,

        "policy":
            {
                "demo_records_are_excluded":
                    True,

                "production_learning_requires":
                    [
                        "real reviewer identity",
                        "verified=true",
                        "valid verified label",
                        "feature snapshot",
                        "review notes",
                        "at least 50 production records",
                        "all four labels represented",
                    ]
            },
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print("=" * 70)

    print(
        "TRUSTLOOP FEEDBACK QUALITY / PROVENANCE REPORT"
    )

    print("=" * 70)

    print(
        "\nTotal feedback:",
        len(records)
    )

    print(
        "Production records:",
        len(production_records)
    )

    print(
        "Production high-quality:",
        len(production_high_quality)
    )

    print(
        "Demo/test records:",
        len(demo_records)
    )

    print(
        "Unknown records:",
        len(unknown_records)
    )

    print(
        "\nProduction label distribution:"
    )

    for label in sorted(
        VALID_LABELS
    ):

        print(
            f"  {label}: "
            f"{production_labels[label]}"
        )

    print(
        "\nAll production classes present:",
        production_all_classes_present
    )

    print(
        "Production learning ready:",
        production_ready
    )

    print(
        "\nSaved:"
    )

    print(
        REPORT_FILE
    )


if __name__ == "__main__":
    main()
