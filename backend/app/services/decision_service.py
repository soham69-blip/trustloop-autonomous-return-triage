"""
TrustLoop Decision Orchestration Service.

Acts as an adapter and orchestrator around existing decision components:
- backend.app.decision.decision_engine.make_decision (primary authoritative engine)
- rag.trustloop_decision_engine.analyze_decision (multi-signal evidence enrichment)

Strict Anti-Duplication Rule:
- Does NOT invent new decision thresholds or rewrite decision rules.
- Strictly preserves Phase 4 decision behavior while enriching with policy and vision evidence signals.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def evaluate_decision(
    case_data: Dict[str, Any],
    probabilities: Dict[str, float],
    ml_label: str,
    policy_result: Optional[Dict[str, Any]] = None,
    vision_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate case decision by combining ML classification, deterministic risk rules,
    policy compliance signals, and vision evidence.

    Args:
        case_data: Raw return case dictionary.
        probabilities: Class probabilities dictionary (e.g., {"Legitimate": 0.95, ...}).
        ml_label: Model's predicted class label string.
        policy_result: Optional RAG policy result dictionary.
        vision_result: Optional Vision evidence result dictionary.

    Returns:
        Structured decision result dictionary matching DecisionResultSchema / Phase 4 contract.
    """
    # 1. Authoritative base decision from backend.app.decision.decision_engine
    from backend.app.decision.decision_engine import make_decision

    base_decision = make_decision(
        case=case_data,
        probabilities=probabilities,
        ml_label=ml_label,
    )

    signals = list(base_decision.get("signals", []))

    # 2. Enrich signals from RAG policy evaluation (if provided)
    if policy_result and policy_result.get("available", False):
        policy_status = policy_result.get("policy_status")
        flags = policy_result.get("flags", [])

        if policy_status == "HUMAN_ESCALATION":
            signals.append("Policy evaluation requires human review")
            for flag in flags:
                rule_name = flag.get("rule", "policy_rule")
                reason = flag.get("reason", "")
                signals.append(f"Policy flag [{rule_name}]: {reason}")
        elif policy_status == "POLICY_VIOLATION":
            signals.append("Policy evaluation identified explicit return violation")

    # 3. Enrich signals from Vision evidence verification (if provided & verified)
    if vision_result and vision_result.get("available", False) and vision_result.get("verified", False):
        if vision_result.get("damage_detected"):
            signals.append("Visual evidence confirms physical damage")
        if vision_result.get("evidence_consistent") is True:
            signals.append("Visual evidence is consistent with return claim")
        elif vision_result.get("evidence_consistent") is False:
            signals.append("Visual evidence is inconsistent with stated claim")

    # Deduplicate signals preserving order
    deduped_signals = list(dict.fromkeys(signals))

    return {
        "risk_score": base_decision["risk_score"],
        "deterministic_risk": base_decision["deterministic_risk"],
        "ml_risk": base_decision["ml_risk"],
        "decision_confidence": base_decision["decision_confidence"],
        "decision": base_decision["decision"],
        "signals": deduped_signals,
        "risk_components": base_decision["risk_components"],
    }
