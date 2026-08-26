from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
from datetime import datetime, timezone
import uuid
import pandas as pd

from backend.app.schemas.feedback import (
    FeedbackSubmission,
    FeedbackRecord,
    FeedbackStats,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"

FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

VERIFIED_FEEDBACK_FILE = FEEDBACK_DIR / "verified_feedback.jsonl"
QUARANTINE_FEEDBACK_FILE = FEEDBACK_DIR / "quarantine_feedback.jsonl"

VALID_LABELS = {
    "Legitimate",
    "Policy Abuser",
    "Fraudulent Return",
    "Wardrobing",
}

LABEL_TO_ID = {
    "Legitimate": 0,
    "Policy Abuser": 1,
    "Fraudulent Return": 2,
    "Wardrobing": 3,
}

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


def validate_feedback_record(record: FeedbackRecord) -> Tuple[bool, str, List[str]]:
    """
    Strict Data Quality Gate for human feedback.
    Evaluates whether a feedback record is eligible for production model retraining.

    Returns:
        (is_eligible: bool, quality_status: str, quarantine_reasons: List[str])
    """
    reasons: List[str] = []

    # 1. Valid verified label check
    if not record.human_verified_label or record.human_verified_label not in VALID_LABELS:
        reasons.append(f"Invalid human verified label: '{record.human_verified_label}'")

    # 2. Reviewer legitimacy check
    reviewer = record.reviewer_id.strip()
    if not reviewer:
        reasons.append("Missing reviewer_id")
    elif reviewer in DEMO_REVIEWERS:
        reasons.append(f"Reviewer '{reviewer}' is marked as demo/test; not eligible for production learning")

    # 3. Case ID demo filter
    case_id = record.case_id.strip()
    if not case_id:
        reasons.append("Missing case_id")
    elif case_id.startswith(DEMO_CASE_PREFIXES):
        reasons.append(f"Case ID '{case_id}' has test/demo prefix; excluded from production training")

    # 4. Raw features completeness check
    if not record.raw_features:
        reasons.append("Missing raw features payload for model retraining")

    # 5. Traffic type check
    if record.traffic_type in ("test", "demo", "synthetic"):
        reasons.append(f"Traffic type is '{record.traffic_type}'; segregated from production training")

    if reasons:
        # Determine if rejected or quarantined
        if not record.human_verified_label or record.human_verified_label not in VALID_LABELS:
            return False, "REJECTED", reasons
        return False, "QUARANTINED", reasons

    return True, "ELIGIBLE", []


def _read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    from backend.app.core.persistence import locked_read_jsonl
    return locked_read_jsonl(path, limit=limit)


def _append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    from backend.app.core.persistence import locked_append_jsonl
    locked_append_jsonl(path, data)


def record_feedback(submission: FeedbackSubmission) -> FeedbackRecord:
    """
    Ingest, quality-check, and persist a human feedback submission with concurrency protection.
    """
    record = FeedbackRecord(
        feedback_id=str(uuid.uuid4()),
        case_id=submission.case_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        human_verified_label=submission.human_verified_label,
        human_decision=submission.human_decision,
        reviewer_id=submission.reviewer_id,
        feedback_source=submission.feedback_source,
        traffic_type=getattr(submission, "traffic_type", "production"),
        notes=submission.notes,
        raw_features=submission.raw_payload or {},
    )

    is_eligible, quality_status, quarantine_reasons = validate_feedback_record(record)
    record.training_eligible = is_eligible
    record.quality_status = quality_status
    record.quarantine_reasons = quarantine_reasons

    record_dict = record.model_dump()

    if is_eligible:
        _append_jsonl(VERIFIED_FEEDBACK_FILE, record_dict)
    else:
        _append_jsonl(QUARANTINE_FEEDBACK_FILE, record_dict)

    return record


def list_feedback(
    eligible_only: bool = False,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Retrieve stored feedback records with bounded limit.
    """
    if eligible_only:
        records = _read_jsonl(VERIFIED_FEEDBACK_FILE, limit=limit)
    else:
        verified = _read_jsonl(VERIFIED_FEEDBACK_FILE, limit=limit)
        quarantined = _read_jsonl(QUARANTINE_FEEDBACK_FILE, limit=limit)
        records = verified + quarantined

    return records[-limit:]


def get_feedback_statistics() -> FeedbackStats:
    """
    Summarize current feedback quality and distribution metrics.
    """
    verified = _read_jsonl(VERIFIED_FEEDBACK_FILE)
    quarantined = _read_jsonl(QUARANTINE_FEEDBACK_FILE)

    total = len(verified) + len(quarantined)
    eligible_count = len(verified)
    quarantined_count = sum(1 for r in quarantined if r.get("quality_status") == "QUARANTINED")
    rejected_count = sum(1 for r in quarantined if r.get("quality_status") == "REJECTED")

    class_dist: Dict[str, int] = {}
    source_dist: Dict[str, int] = {}

    for r in verified:
        lbl = r.get("human_verified_label", "Unknown")
        class_dist[lbl] = class_dist.get(lbl, 0) + 1
        src = r.get("feedback_source", "unknown")
        source_dist[src] = source_dist.get(src, 0) + 1

    return FeedbackStats(
        total_records=total,
        eligible_count=eligible_count,
        quarantined_count=quarantined_count,
        rejected_count=rejected_count,
        class_distribution=class_dist,
        source_distribution=source_dist,
    )


def export_verified_training_dataset(output_path: Optional[Path] = None) -> int:
    """
    Export all eligible feedback records as training-ready rows.
    """
    output_file = output_path or (FEEDBACK_DIR / "exported_verified_training.csv")
    records = _read_jsonl(VERIFIED_FEEDBACK_FILE)
    if not records:
        return 0

    rows = []
    for r in records:
        if not r.get("training_eligible"):
            continue
        feat = dict(r.get("raw_features", {}))
        label_str = r.get("human_verified_label")
        if label_str in LABEL_TO_ID:
            feat["abuse_label"] = LABEL_TO_ID[label_str]
            feat["feedback_id"] = r.get("feedback_id")
            rows.append(feat)

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    return len(df)
