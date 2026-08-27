"""
TrustLoop Counterfactual & Challenge Decision Service.

Enables interactive evidence toggling, recalculates multi-party responsibility,
and explains causal shifts in recommended action and responsibility percentages.
"""

from typing import Dict, Any, List, Optional
from backend.app.services.responsibility_service import calculate_responsibility
from pathlib import Path
import json
from datetime import datetime, timezone

OVERRIDE_FILE = Path(__file__).resolve().parents[3] / "data" / "feedback" / "human_overrides.jsonl"


def evaluate_challenge(
    case_payload: Dict[str, Any],
    disabled_signals: List[str],
    ml_probabilities: Optional[Dict[str, float]] = None,
    vision_result: Optional[Dict[str, Any]] = None,
    rag_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Recalculate case decision and responsibility distribution with toggled/disabled evidence signals.
    """
    # 1. Calculate baseline (all signals active)
    baseline_result = calculate_responsibility(
        case_data=case_payload,
        ml_probabilities=ml_probabilities,
        vision_result=vision_result,
        rag_result=rag_result,
        disabled_signals=[],
    )

    # 2. Calculate counterfactual (with requested disabled signals)
    counterfactual_result = calculate_responsibility(
        case_data=case_payload,
        ml_probabilities=ml_probabilities,
        vision_result=vision_result,
        rag_result=rag_result,
        disabled_signals=disabled_signals,
    )

    base_resp = baseline_result["responsibility"]
    cf_resp = counterfactual_result["responsibility"]

    # Compute deltas
    deltas = {
        party: {
            "before": base_resp[party],
            "after": cf_resp[party],
            "delta": cf_resp[party] - base_resp[party],
        }
        for party in ["customer", "seller", "courier", "unknown"]
    }

    # Generate explanatory rationale for the judge
    explanations: List[str] = []
    for signal in disabled_signals:
        if signal == "courier_incident_history":
            explanations.append(
                f"Removing courier incident history reduced courier responsibility by {abs(deltas['courier']['delta'])}% "
                f"because historical hub transit loss was a major contributing factor."
            )
        elif signal == "packaging_damage":
            explanations.append(
                f"Excluding packaging damage evidence decreased courier responsibility from {base_resp['courier']}% to {cf_resp['courier']}% "
                f"as physical box crushing was the primary indicator of in-transit mishandling."
            )
        elif signal == "sku_mismatch" or signal == "seller_defect_history":
            explanations.append(
                f"Disabling merchant dispatch signals shifted responsibility away from seller ({base_resp['seller']}% -> {cf_resp['seller']}%)."
            )
        elif signal == "customer_history" or signal == "identity_linkage":
            explanations.append(
                f"Excluding customer dispute and identity linkage history lowered customer suspicion from {base_resp['customer']}% to {cf_resp['customer']}%."
            )
        else:
            explanations.append(
                f"Disabling '{signal}' recalculated evidence weights across all parties."
            )

    # Determine recommended action shift
    base_action = _determine_action(baseline_result["dominant_party"], base_resp)
    cf_action = _determine_action(counterfactual_result["dominant_party"], cf_resp)

    return {
        "case_id": str(case_payload.get("case_id", "UNKNOWN")),
        "disabled_signals": disabled_signals,
        "baseline": {
            "responsibility": base_resp,
            "dominant_party": baseline_result["dominant_party"],
            "recommended_action": base_action,
        },
        "counterfactual": {
            "responsibility": cf_resp,
            "dominant_party": counterfactual_result["dominant_party"],
            "recommended_action": cf_action,
        },
        "deltas": deltas,
        "action_changed": (base_action != cf_action),
        "explanation": " ".join(explanations) if explanations else "No evidence signals were disabled.",
    }


def _determine_action(dominant_party: str, responsibility: Dict[str, int]) -> str:
    if dominant_party == "courier":
        return "AUTO_ACCEPT"
    elif dominant_party == "seller":
        return "AUTO_ACCEPT"
    elif dominant_party == "customer":
        if responsibility.get("customer", 0) >= 60:
            return "AUTO_RETURN"
        return "HUMAN_ESCALATION"
    else:
        if responsibility.get("unknown", 0) > 40:
            return "HUMAN_ESCALATION"
        return "AUTO_ACCEPT"


def store_human_override(case_id: str, ai_decision: str, human_decision: str, reason: str) -> Dict[str, Any]:
    """Persist an investigator's decision override for the learning loop."""
    if human_decision == "REJECT":
        human_decision = "RETURN"
    if human_decision not in {"APPROVE", "RETURN", "ESCALATE"}:
        raise ValueError("human_decision must be APPROVE, RETURN, or ESCALATE")
    OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {"case_id": case_id, "ai_decision": ai_decision, "human_decision": human_decision,
              "reason": reason, "status": "OVERRIDDEN" if ai_decision != human_decision else "CONFIRMED",
              "timestamp": datetime.now(timezone.utc).isoformat()}
    with OVERRIDE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record
