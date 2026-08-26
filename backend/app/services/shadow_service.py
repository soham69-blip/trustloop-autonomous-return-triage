from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import pickle
import hashlib
from datetime import datetime, timezone
import pandas as pd
from pydantic import BaseModel, Field

from backend.app.ml_feature_builder import build_model_dataframe, CANDIDATE_MODEL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

SHADOW_LOG_FILE = FEEDBACK_DIR / "shadow_log.jsonl"
CANDIDATE_MODEL_PATH = MODELS_DIR / "lightgbm_candidate.pkl"
SHADOW_CONFIG_FILE = FEEDBACK_DIR / "shadow_config.json"

TARGET_CLASSES = {
    0: "Legitimate",
    1: "Policy Abuser",
    2: "Fraudulent Return",
    3: "Wardrobing",
}

# In-memory candidate model cache
_cached_candidate_model = None
_cached_candidate_sha256 = None


def is_shadow_mode_enabled() -> bool:
    if not SHADOW_CONFIG_FILE.exists():
        return True  # Default enabled
    try:
        with open(SHADOW_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            return bool(cfg.get("enabled", True))
    except Exception:
        return True


def set_shadow_mode_enabled(enabled: bool) -> bool:
    SHADOW_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SHADOW_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"enabled": enabled, "updated_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    return enabled


def _get_candidate_model() -> Tuple[Any, str]:
    global _cached_candidate_model, _cached_candidate_sha256

    if not CANDIDATE_MODEL_PATH.exists():
        return None, ""

    if _cached_candidate_model is None:
        try:
            with open(CANDIDATE_MODEL_PATH, "rb") as f:
                content = f.read()
                sha = hashlib.sha256(content).hexdigest()
                _cached_candidate_model = pickle.loads(content)
                _cached_candidate_sha256 = sha
        except Exception:
            return None, ""

    return _cached_candidate_model, _cached_candidate_sha256 or ""


def _append_shadow_log(record: Dict[str, Any]) -> None:
    try:
        from backend.app.core.persistence import locked_append_jsonl
        locked_append_jsonl(SHADOW_LOG_FILE, record)
    except Exception:
        # Never fail calling pipeline due to logging
        pass


def evaluate_shadow_case(
    case_payload: Dict[str, Any],
    production_label: str,
    production_probabilities: Dict[str, float],
    production_confidence: float,
    production_risk_score: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run candidate model in silent shadow mode.
    Guaranteed NEVER to modify or influence the production decision.
    """
    if not is_shadow_mode_enabled():
        return None

    candidate_model, cand_sha = _get_candidate_model()
    if candidate_model is None:
        return {
            "status": "CANDIDATE_UNAVAILABLE",
            "evaluated": False,
        }

    case_id = str(case_payload.get("case_id", "UNKNOWN"))
    traffic_type = "test" if case_id.startswith(("TEST-", "SCENARIO-", "TRAIN-")) else "production"

    try:
        cand_feature_names = list(
            getattr(candidate_model, "feature_name_", getattr(candidate_model, "feature_names_in_", CANDIDATE_MODEL_FEATURES))
        )
        cand_df = build_model_dataframe(case_payload, feature_names=cand_feature_names)

        pred_idx = int(candidate_model.predict(cand_df)[0])
        cand_probs_arr = candidate_model.predict_proba(cand_df)[0]

        candidate_label = TARGET_CLASSES.get(pred_idx, "Unknown")
        cand_probs = {
            TARGET_CLASSES.get(i, str(i)): round(float(p), 6)
            for i, p in enumerate(cand_probs_arr)
        }
        cand_confidence = round(float(cand_probs_arr[pred_idx]), 4)

        disagreement = (production_label != candidate_label)
        confidence_delta = round(cand_confidence - (production_confidence / 100.0 if production_confidence > 1.0 else production_confidence), 4)

        shadow_record = {
            "case_id": case_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "traffic_type": traffic_type,
            "production_label": production_label,
            "production_probabilities": production_probabilities,
            "candidate_label": candidate_label,
            "candidate_probabilities": cand_probs,
            "candidate_confidence": cand_confidence,
            "disagreement": disagreement,
            "confidence_delta": confidence_delta,
            "candidate_sha256": cand_sha,
            "evaluated": True,
            "status": "SUCCESS",
        }

        _append_shadow_log(shadow_record)
        return shadow_record

    except Exception as exc:
        err_record = {
            "case_id": case_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "traffic_type": traffic_type,
            "production_label": production_label,
            "evaluated": False,
            "status": "ERROR",
            "error": str(exc),
        }
        _append_shadow_log(err_record)
        return err_record


def get_shadow_summary(production_only: bool = False) -> Dict[str, Any]:
    """
    Get aggregated shadow evaluation metrics with optional production filtering.
    """
    from backend.app.core.persistence import locked_read_jsonl
    records = locked_read_jsonl(SHADOW_LOG_FILE)
    if production_only:
        records = [r for r in records if r.get("traffic_type", "production") == "production"]

    if not records:
        return {
            "shadow_mode_enabled": is_shadow_mode_enabled(),
            "total_evaluated": 0,
            "disagreement_count": 0,
            "disagreement_rate_pct": 0.0,
            "class_disagreements": {},
            "candidate_active": CANDIDATE_MODEL_PATH.exists(),
        }

    total = 0
    disagreements = 0
    class_transitions: Dict[str, int] = {}

    for rec in records:
        if not rec.get("evaluated", False):
            continue
        total += 1
        if rec.get("disagreement", False):
            disagreements += 1
            p_lbl = rec.get("production_label", "Unknown")
            c_lbl = rec.get("candidate_label", "Unknown")
            trans_key = f"{p_lbl} -> {c_lbl}"
            class_transitions[trans_key] = class_transitions.get(trans_key, 0) + 1

    rate = round((disagreements / total) * 100.0, 2) if total > 0 else 0.0

    return {
        "shadow_mode_enabled": is_shadow_mode_enabled(),
        "total_evaluated": total,
        "disagreement_count": disagreements,
        "disagreement_rate_pct": rate,
        "class_disagreements": class_transitions,
    }


def get_shadow_disagreements(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve cases where candidate model prediction differed from production with bounded limit.
    """
    from backend.app.core.persistence import locked_read_jsonl
    records = locked_read_jsonl(SHADOW_LOG_FILE, filter_fn=lambda r: bool(r.get("disagreement", False)))
    return records[-limit:]
