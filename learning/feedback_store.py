from pathlib import Path
import json
from datetime import datetime, timezone


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEARNING_DIR = PROJECT_ROOT / "learning"

PENDING_FILE = LEARNING_DIR / "pending_feedback.jsonl"
VERIFIED_FILE = LEARNING_DIR / "verified_feedback.jsonl"

LEARNING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALID LABELS
# ============================================================

VALID_LABELS = {
    "Legitimate",
    "Policy Abuser",
    "Fraudulent Return",
    "Wardrobing",
}


# ============================================================
# JSONL HELPERS
# ============================================================

def _read_jsonl(path: Path) -> list[dict]:

    if not path.exists():
        return []

    records = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(line)

                if isinstance(record, dict):
                    records.append(record)

            except json.JSONDecodeError:

                print(
                    f"WARNING: Skipping invalid JSON line "
                    f"in {path}"
                )

    return records


def _append_jsonl(
    path: Path,
    record: dict
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


def _write_jsonl(
    path: Path,
    records: list[dict]
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


# ============================================================
# PENDING FEEDBACK
# ============================================================

def save_pending_feedback(
    record: dict
) -> dict:

    if not isinstance(record, dict):

        raise TypeError(
            "Feedback record must be a dictionary."
        )

    case_id = record.get("case_id")

    if not case_id:

        raise ValueError(
            "case_id is required."
        )

    # --------------------------------------------------------
    # Check whether this case is already VERIFIED
    # --------------------------------------------------------

    verified_records = load_feedback()

    for verified in verified_records:

        if verified.get("case_id") == case_id:

            raise ValueError(
                f"Case {case_id} is already verified. "
                f"A verified case cannot be added again."
            )

    # --------------------------------------------------------
    # Check whether case is already pending
    # --------------------------------------------------------

    pending_records = load_pending_feedback()

    for pending in pending_records:

        if pending.get("case_id") == case_id:

            raise ValueError(
                f"Case {case_id} is already pending "
                f"human verification."
            )

    # --------------------------------------------------------
    # Prepare pending record
    # --------------------------------------------------------

    new_record = dict(record)

    new_record.setdefault(
        "timestamp",
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    new_record["verified"] = False
    new_record["verified_label"] = None
    new_record["reviewer"] = None
    new_record["review_notes"] = None

    # --------------------------------------------------------
    # APPEND ONLY
    # --------------------------------------------------------

    _append_jsonl(
        PENDING_FILE,
        new_record
    )

    return new_record


# ============================================================
# DIRECT VERIFIED FEEDBACK
# ============================================================

def save_feedback(
    case_id: str,
    prediction: str,
    verified_label: str,
    reviewer: str = "human",
    notes: str = "",
    features: dict | None = None,
) -> dict:

    if verified_label not in VALID_LABELS:

        raise ValueError(
            f"Invalid verified label: {verified_label}. "
            f"Expected one of: "
            f"{sorted(VALID_LABELS)}"
        )

    if not case_id:

        raise ValueError(
            "case_id is required."
        )

    # --------------------------------------------------------
    # Prevent duplicate verified cases
    # --------------------------------------------------------

    existing_records = load_feedback()

    for existing in existing_records:

        if existing.get("case_id") == case_id:

            raise ValueError(
                f"Case {case_id} is already present "
                f"in verified_feedback.jsonl."
            )

    if features is None:

        features = {}

    record = {

        "case_id":
            case_id,

        "prediction":
            prediction,

        "verified_label":
            verified_label,

        "reviewer":
            reviewer,

        "notes":
            notes,

        "features":
            features,

        "verified":
            True,

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    # --------------------------------------------------------
    # APPEND ONLY
    # --------------------------------------------------------

    _append_jsonl(
        VERIFIED_FILE,
        record
    )

    return record


# ============================================================
# LOAD VERIFIED FEEDBACK
# ============================================================

def load_feedback() -> list[dict]:

    return _read_jsonl(
        VERIFIED_FILE
    )


# ============================================================
# VERIFIED COUNT
# ============================================================

def feedback_count() -> int:

    return len(
        load_feedback()
    )


# ============================================================
# LOAD PENDING FEEDBACK
# ============================================================

def load_pending_feedback() -> list[dict]:

    return _read_jsonl(
        PENDING_FILE
    )


# ============================================================
# GET PENDING CASE
# ============================================================

def get_pending_case(
    case_id: str
) -> dict | None:

    for record in load_pending_feedback():

        if record.get("case_id") == case_id:

            return record

    return None


# ============================================================
# VERIFY CASE
# ============================================================

def verify_case(
    case_id: str,
    verified_label: str,
    reviewer: str = "human",
    notes: str = "",
) -> dict:

    # --------------------------------------------------------
    # Validate label
    # --------------------------------------------------------

    if verified_label not in VALID_LABELS:

        raise ValueError(
            f"Invalid verified label: {verified_label}. "
            f"Expected one of: "
            f"{sorted(VALID_LABELS)}"
        )

    # --------------------------------------------------------
    # Check if already verified
    # --------------------------------------------------------

    verified_records = load_feedback()

    for record in verified_records:

        if record.get("case_id") == case_id:

            raise ValueError(
                f"Case {case_id} is already verified."
            )

    # --------------------------------------------------------
    # Find pending case
    # --------------------------------------------------------

    pending_records = load_pending_feedback()

    target = None

    for record in pending_records:

        if record.get("case_id") == case_id:

            target = record
            break

    if target is None:

        available_ids = [
            record.get("case_id")
            for record in pending_records
        ]

        raise ValueError(
            f"No pending feedback found for case: "
            f"{case_id}\n"
            f"Pending file: {PENDING_FILE}\n"
            f"Available pending case IDs: "
            f"{available_ids}"
        )

    # --------------------------------------------------------
    # Validate feature snapshot
    # --------------------------------------------------------

    features = target.get(
        "features"
    )

    if not isinstance(
        features,
        dict
    ) or not features:

        raise ValueError(
            f"Case {case_id} does not contain "
            f"a valid feature snapshot."
        )

    # --------------------------------------------------------
    # Build verified record
    # --------------------------------------------------------

    verified_record = {

        "case_id":
            target.get(
                "case_id"
            ),

        "prediction":
            target.get(
                "prediction"
            ),

        "predicted_class":
            target.get(
                "predicted_class"
            ),

        "model_confidence":
            target.get(
                "model_confidence"
            ),

        "probabilities":
            target.get(
                "probabilities",
                {}
            ),

        "policy_status":
            target.get(
                "policy_status"
            ),

        "vision_result":
            target.get(
                "vision_result"
            ),

        "final_decision":
            target.get(
                "final_decision"
            ),

        "features":
            features,

        "verified_label":
            verified_label,

        "reviewer":
            reviewer,

        "notes":
            notes,

        "verified":
            True,

        "verified_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    # ========================================================
    # IMPORTANT:
    # APPEND verified case.
    #
    # NEVER rewrite the existing verified dataset.
    # ========================================================

    _append_jsonl(
        VERIFIED_FILE,
        verified_record
    )

    # --------------------------------------------------------
    # Remove ONLY this case from pending
    # --------------------------------------------------------

    remaining_records = [

        record

        for record in pending_records

        if record.get(
            "case_id"
        ) != case_id

    ]

    _write_jsonl(
        PENDING_FILE,
        remaining_records
    )

    return verified_record


# ============================================================
# DATASET SUMMARY
# ============================================================

def get_feedback_summary() -> dict:

    records = load_feedback()

    distribution = {

        label: 0

        for label in VALID_LABELS

    }

    for record in records:

        label = record.get(
            "verified_label"
        )

        if label in distribution:

            distribution[label] += 1

    return {

        "total_verified":
            len(records),

        "label_distribution":
            distribution,

        "pending":
            len(
                load_pending_feedback()
            ),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "TRUSTLOOP VERIFIED FEEDBACK STORE"
    )

    print("=" * 70)

    VERIFIED_FILE.touch(
        exist_ok=True
    )

    PENDING_FILE.touch(
        exist_ok=True
    )

    summary = get_feedback_summary()

    print(
        f"Verified cases: "
        f"{summary['total_verified']}"
    )

    print(
        f"Pending cases: "
        f"{summary['pending']}"
    )

    print(
        "\nLabel distribution:"
    )

    for label, count in (
        summary[
            "label_distribution"
        ].items()
    ):

        print(
            f"  {label}: {count}"
        )

    print(
        "\nPending file:"
    )

    print(
        PENDING_FILE
    )

    print(
        "\nVerified file:"
    )

    print(
        VERIFIED_FILE
    )