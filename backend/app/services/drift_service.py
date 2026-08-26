from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Sequence
import math
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
VERIFIED_FEEDBACK_FILE = FEEDBACK_DIR / "verified_feedback.jsonl"
QUARANTINE_FEEDBACK_FILE = FEEDBACK_DIR / "quarantine_feedback.jsonl"
SHADOW_LOG_FILE = FEEDBACK_DIR / "shadow_log.jsonl"

# Training baseline class prior distribution from model_ready.csv (60k dataset)
BASELINE_CLASS_PRIORS: Dict[str, float] = {
    "Legitimate": 0.7010,
    "Policy Abuser": 0.1199,
    "Fraudulent Return": 0.1019,
    "Wardrobing": 0.0773,
}

# Configurable PSI & drift thresholds
PSI_STABLE_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25
MAX_QUARANTINE_RATIO_WARN = 0.30  # Warning if > 30% feedback is quarantined


def calculate_psi(
    expected_dist: Sequence[float],
    actual_dist: Sequence[float],
    epsilon: float = 1e-4
) -> float:

    """
    Calculate Population Stability Index (PSI) between expected and actual distributions.
    PSI < 0.10: Stable (no significant drift)
    0.10 <= PSI < 0.25: Moderate drift
    PSI >= 0.25: Significant drift
    """
    psi = 0.0
    for exp, act in zip(expected_dist, actual_dist):
        exp_p = max(epsilon, exp)
        act_p = max(epsilon, act)
        psi += (act_p - exp_p) * math.log(act_p / exp_p)
    return max(0.0, psi)


def calculate_numerical_feature_psi(
    expected_values: List[float],
    actual_values: List[float],
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> Tuple[float, str]:
    """
    Compute PSI on a continuous numerical feature across binned quantiles.
    """
    if len(expected_values) < 5 or len(actual_values) < 5:
        return 0.0, "INSUFFICIENT_DATA"

    # Define bins from expected distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected_values, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) <= 2:
        return 0.0, "CONSTANT_FEATURE"

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    exp_counts, _ = np.histogram(expected_values, bins=bin_edges)
    act_counts, _ = np.histogram(actual_values, bins=bin_edges)

    k = len(exp_counts)
    exp_dist = [(c + 1) / (len(expected_values) + k) for c in exp_counts]
    act_dist = [(c + 1) / (len(actual_values) + k) for c in act_counts]

    psi = calculate_psi(exp_dist, act_dist, epsilon=epsilon)

    if psi < PSI_STABLE_THRESHOLD:
        status = "STABLE"
    elif psi < PSI_CRITICAL_THRESHOLD:
        status = "MODERATE_DRIFT"
    else:
        status = "SIGNIFICANT_DRIFT"

    return round(psi, 4), status



def evaluate_prediction_drift(
    recent_predictions: List[str]
) -> Dict[str, Any]:
    """
    Assess whether runtime inference predictions have drifted from training baseline class priors.
    """
    if not recent_predictions:
        return {
            "sample_count": 0,
            "drift_detected": False,
            "drift_level": "INSUFFICIENT_DATA",
            "psi_score": 0.0,
            "status": "INSUFFICIENT_DATA",
            "current_distribution": {},
            "baseline_distribution": BASELINE_CLASS_PRIORS,
        }

    total = len(recent_predictions)
    counts: Dict[str, int] = {}
    for p in recent_predictions:
        counts[p] = counts.get(p, 0) + 1

    current_dist = {
        c: counts.get(c, 0) / total
        for c in BASELINE_CLASS_PRIORS.keys()
    }

    expected_list = [BASELINE_CLASS_PRIORS[c] for c in BASELINE_CLASS_PRIORS.keys()]
    actual_list = [current_dist[c] for c in BASELINE_CLASS_PRIORS.keys()]

    psi = calculate_psi(expected_list, actual_list)

    drift_detected = psi >= PSI_CRITICAL_THRESHOLD
    if psi < PSI_STABLE_THRESHOLD:
        drift_level = "STABLE"
    elif psi < PSI_CRITICAL_THRESHOLD:
        drift_level = "MODERATE_SHIFT"
    else:
        drift_level = "SIGNIFICANT_DRIFT"

    return {
        "sample_count": total,
        "drift_detected": drift_detected,
        "drift_level": drift_level,
        "psi_score": round(psi, 4),
        "current_distribution": {k: round(v, 4) for k, v in current_dist.items()},
        "baseline_distribution": BASELINE_CLASS_PRIORS,
    }


def evaluate_confidence_drift(
    recent_confidences: List[float],
    baseline_expected_confidence: float = 85.0,
    confidence_drop_warn_pct: float = 15.0
) -> Dict[str, Any]:
    """
    Assess whether average model confidence is degrading over time.
    """
    if not recent_confidences:
        return {
            "sample_count": 0,
            "mean_confidence": 0.0,
            "baseline_confidence": baseline_expected_confidence,
            "confidence_drop_pct": 0.0,
            "drift_detected": False,
            "status": "INSUFFICIENT_DATA",
        }

    mean_conf = float(np.mean(recent_confidences))
    delta = baseline_expected_confidence - mean_conf
    drop_pct = max(0.0, (delta / baseline_expected_confidence) * 100.0) if baseline_expected_confidence > 0 else 0.0

    drift_detected = drop_pct >= confidence_drop_warn_pct

    return {
        "sample_count": len(recent_confidences),
        "mean_confidence": round(mean_conf, 2),
        "baseline_confidence": baseline_expected_confidence,
        "confidence_drop_pct": round(drop_pct, 2),
        "drift_detected": drift_detected,
        "status": "CONFIDENCE_DEGRADED" if drift_detected else "STABLE",
    }


def evaluate_feedback_quality_drift(include_test_traffic: bool = False) -> Dict[str, Any]:
    """
    Assess quality and hygiene of incoming human audit feedback.
    Filters out test/demo records by default to prevent test contamination.
    """
    from backend.app.core.persistence import locked_read_jsonl

    verified_records = locked_read_jsonl(VERIFIED_FEEDBACK_FILE)
    quarantine_records = locked_read_jsonl(QUARANTINE_FEEDBACK_FILE)

    if not include_test_traffic:
        verified_records = [
            r for r in verified_records
            if r.get("traffic_type", "production") == "production"
            and not str(r.get("case_id", "")).startswith(("TEST-", "SCENARIO-", "TRAIN-"))
        ]
        quarantine_records = [
            r for r in quarantine_records
            if r.get("traffic_type", "production") == "production"
            and not str(r.get("case_id", "")).startswith(("TEST-", "SCENARIO-", "TRAIN-"))
        ]

    verified_count = len(verified_records)
    quarantined_count = len(quarantine_records)
    quarantine_reasons_summary: Dict[str, int] = {}

    for rec in quarantine_records:
        for r in rec.get("quarantine_reasons", []):
            quarantine_reasons_summary[r] = quarantine_reasons_summary.get(r, 0) + 1

    total = verified_count + quarantined_count
    quarantine_ratio = (quarantined_count / total) if total > 0 else 0.0
    quality_warning = quarantine_ratio >= MAX_QUARANTINE_RATIO_WARN if total >= 5 else False

    return {
        "total_feedback_submissions": total,
        "verified_eligible_count": verified_count,
        "quarantined_count": quarantined_count,
        "quarantine_ratio": round(quarantine_ratio, 4),
        "quality_alert": quality_warning,
        "quarantine_reasons": quarantine_reasons_summary,
        "status": "QUALITY_ALERT" if quality_warning else "HEALTHY",
    }


def generate_comprehensive_drift_report(include_test_traffic: bool = False) -> Dict[str, Any]:
    """
    Aggregate all drift dimensions into a unified monitoring report.
    By default evaluates strictly production traffic to prevent test contamination.
    """
    from backend.app.core.persistence import locked_read_jsonl
    from datetime import datetime, timezone

    # 1. Gather predictions and confidences from shadow log
    shadow_records = locked_read_jsonl(SHADOW_LOG_FILE)

    if not include_test_traffic:
        shadow_records = [
            r for r in shadow_records
            if r.get("traffic_type", "production") == "production"
            and not str(r.get("case_id", "")).startswith(("TEST-", "SCENARIO-", "TRAIN-"))
        ]

    recent_preds: List[str] = []
    recent_confs: List[float] = []

    for rec in shadow_records:
        if rec.get("production_label"):
            recent_preds.append(rec["production_label"])
        if rec.get("candidate_confidence") is not None:
            recent_confs.append(float(rec["candidate_confidence"]) * 100.0)

    pred_drift = evaluate_prediction_drift(recent_preds[-500:] if recent_preds else [])
    conf_drift = evaluate_confidence_drift(recent_confs[-500:] if recent_confs else [])
    feedback_quality = evaluate_feedback_quality_drift(include_test_traffic=include_test_traffic)

    overall_alert = (
        pred_drift.get("drift_detected", False)
        or conf_drift.get("drift_detected", False)
        or feedback_quality.get("quality_alert", False)
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_health": "DRIFT_ALERT" if overall_alert else "HEALTHY",
        "production_traffic_only": not include_test_traffic,
        "prediction_drift": pred_drift,
        "confidence_drift": conf_drift,
        "feedback_quality": feedback_quality,
    }


def reset_drift_telemetry(archive: bool = True) -> Dict[str, Any]:
    """
    Safely reset or rotate telemetry logs so drift monitoring begins with a clean slate.
    """
    from datetime import datetime, timezone
    import shutil

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_dir = FEEDBACK_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    archived_files = []
    for fpath in (SHADOW_LOG_FILE, VERIFIED_FEEDBACK_FILE, QUARANTINE_FEEDBACK_FILE):
        if fpath.exists():
            if archive:
                dest = archive_dir / f"{fpath.stem}_{ts}{fpath.suffix}"
                shutil.move(str(fpath), str(dest))
                archived_files.append(dest.name)
            else:
                fpath.unlink()

    return {
        "status": "TELEMETRY_RESET_SUCCESSFUL",
        "archived_files": archived_files,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_feedback_outcome_drift(feedback_labels: List[str]) -> Dict[str, Any]:
    """
    Assess whether verified human outcomes differ significantly from baseline expectations.
    """
    return evaluate_prediction_drift(feedback_labels)

