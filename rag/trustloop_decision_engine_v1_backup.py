# TrustLoop Decision Engine
# Conservative decision layer combining:
# ML risk + policy compliance + vision evidence + decision confidence

from dataclasses import dataclass
from typing import Optional


@dataclass
class DecisionResult:
    customer_risk: float
    claim_validity: float
    policy_compliance: float
    decision_confidence: float
    decision: str
    reason: str
    risk_factors: list
    evidence_policy_factors: list


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def calculate_customer_risk(
    ml_probabilities: dict,
    policy_flags: Optional[list] = None
):
    policy_flags = policy_flags or []

    legitimate = ml_probabilities.get("Legitimate", 0.0)
    policy_abuser = ml_probabilities.get("Policy Abuser", 0.0)
    fraudulent = ml_probabilities.get("Fraudulent Return", 0.0)
    wardrobing = ml_probabilities.get("Wardrobing", 0.0)

    risk = (
        legitimate * 0.05
        + policy_abuser * 0.65
        + fraudulent * 1.00
        + wardrobing * 0.95
    )

    if policy_flags:
        risk += min(0.15, 0.05 * len(policy_flags))

    return clamp(risk)


def calculate_claim_validity(vision_result):
    if not vision_result:
        return 0.0

    confidence = clamp(vision_result.get("confidence", 0.0))

    damage_detected = bool(
        vision_result.get("damage_detected", False)
    )

    evidence_consistent = bool(
        vision_result.get("evidence_consistent", False)
    )

    image_quality = str(
        vision_result.get("image_quality", "")
    ).upper()

    validity = confidence

    if evidence_consistent:
        validity += 0.15

    if damage_detected:
        validity += 0.05

    if image_quality in {"GOOD", "ACCEPTABLE"}:
        validity += 0.05

    return clamp(validity)


def calculate_policy_compliance(policy_status, policy_flags):
    policy_flags = policy_flags or []

    if policy_status == "POLICY_COMPLIANT" and not policy_flags:
        return 1.0

    if policy_status == "HUMAN_ESCALATION":
        return 0.50

    if policy_status == "POLICY_VIOLATION":
        return 0.0

    return clamp(1.0 - (0.20 * len(policy_flags)))


def analyze_decision(
    ml_probabilities,
    policy_status,
    policy_flags=None,
    vision_result=None
):
    policy_flags = policy_flags or []

    legitimate = ml_probabilities.get("Legitimate", 0.0)
    policy_abuser = ml_probabilities.get("Policy Abuser", 0.0)
    fraudulent = ml_probabilities.get("Fraudulent Return", 0.0)
    wardrobing = ml_probabilities.get("Wardrobing", 0.0)

    customer_risk = calculate_customer_risk(
        ml_probabilities,
        policy_flags
    )

    claim_validity = calculate_claim_validity(
        vision_result
    )

    policy_compliance = calculate_policy_compliance(
        policy_status,
        policy_flags
    )

    risk_factors = []
    evidence_policy_factors = []

    # ---------------------------------------------------------
    # RISK FACTORS
    # ---------------------------------------------------------

    if policy_abuser >= 0.30:
        risk_factors.append(
            "Elevated Policy Abuser probability"
        )

    if fraudulent >= 0.20:
        risk_factors.append(
            "Elevated Fraudulent Return probability"
        )

    if wardrobing >= 0.20:
        risk_factors.append(
            "Elevated Wardrobing probability"
        )

    if policy_flags:
        risk_factors.extend(policy_flags)

    # ---------------------------------------------------------
    # POLICY
    # ---------------------------------------------------------

    if policy_status == "HUMAN_ESCALATION":
        evidence_policy_factors.append(
            "Policy rules require human review"
        )

    if policy_status == "POLICY_VIOLATION":
        evidence_policy_factors.append(
            "Return violates an applicable policy rule"
        )

    if policy_status == "POLICY_COMPLIANT":
        evidence_policy_factors.append(
            "No policy escalation triggered"
        )

    # ---------------------------------------------------------
    # VISION
    # ---------------------------------------------------------

    damage_detected = False
    evidence_consistent = False
    vision_confidence = 0.0

    if vision_result:
        damage_detected = bool(
            vision_result.get("damage_detected", False)
        )

        evidence_consistent = bool(
            vision_result.get("evidence_consistent", False)
        )

        vision_confidence = clamp(
            vision_result.get("confidence", 0.0)
        )

        if evidence_consistent:
            evidence_policy_factors.append(
                "Visual evidence is consistent with the claim"
            )

        if damage_detected:
            evidence_policy_factors.append(
                "Reported damage is visually supported"
            )

    # ---------------------------------------------------------
    # DECISION CONFIDENCE
    # ---------------------------------------------------------

    decision_confidence = (
        (1.0 - customer_risk) * 0.40
        + claim_validity * 0.25
        + policy_compliance * 0.35
    )

    decision_confidence = clamp(decision_confidence)

    # ---------------------------------------------------------
    # DECISION LOGIC
    # ---------------------------------------------------------

    # 1. Strong fraud / wardrobing evidence
    if fraudulent >= 0.50 or wardrobing >= 0.50:
        decision = "AUTO_REJECT"
        reason = (
            "Strong model evidence indicates fraudulent "
            "or wardrobing behavior."
        )

    # 2. Strong policy abuse
    elif policy_abuser >= 0.50:
        decision = "HUMAN_ESCALATION"
        reason = (
            "Policy abuse probability is elevated and "
            "requires additional evidence review."
        )

    # 3. Policy escalation always overrides automatic approval
    elif policy_status == "HUMAN_ESCALATION":
        decision = "HUMAN_ESCALATION"
        reason = (
            "Policy rules require human review before "
            "a final refund decision."
        )

    # 4. Policy violation
    elif policy_status == "POLICY_VIOLATION":
        decision = "AUTO_REJECT"
        reason = (
            "The return violates an applicable policy rule."
        )

    # 5. Damage claims require conservative handling
    elif damage_detected:
        if (
            evidence_consistent
            and vision_confidence >= 0.85
            and customer_risk < 0.10
            and policy_compliance >= 0.95
        ):
            decision = "HUMAN_ESCALATION"
            reason = (
                "Damage is strongly supported by visual evidence, "
                "but physical-condition claims require evidence review "
                "before refund approval."
            )
        else:
            decision = "HUMAN_ESCALATION"
            reason = (
                "Physical damage was detected and the claim requires "
                "human evidence review."
            )

    # 6. Very low risk + compliant + strong overall confidence
    elif (
        customer_risk < 0.10
        and policy_compliance >= 0.95
        and decision_confidence >= 0.85
        and legitimate >= 0.70
    ):
        decision = "AUTO_APPROVE"
        reason = (
            "Model risk is low, policy requirements are satisfied, "
            "and decision confidence is high."
        )

    # 7. Everything uncertain goes to human review
    else:
        decision = "HUMAN_ESCALATION"
        reason = (
            "Evidence is insufficient for a sufficiently confident "
            "automated decision."
        )

    return DecisionResult(
        customer_risk=customer_risk,
        claim_validity=claim_validity,
        policy_compliance=policy_compliance,
        decision_confidence=decision_confidence,
        decision=decision,
        reason=reason,
        risk_factors=risk_factors,
        evidence_policy_factors=evidence_policy_factors,
    )


def print_result(result: DecisionResult):

    print("=" * 70)
    print("TRUSTLOOP — DECISION ENGINE")
    print("=" * 70)

    print()
    print("CUSTOMER RISK:")
    print(f"  {result.customer_risk:.2%}")

    print()
    print("CLAIM VALIDITY:")
    print(f"  {result.claim_validity:.2%}")

    print()
    print("POLICY COMPLIANCE:")
    print(f"  {result.policy_compliance:.2%}")

    print()
    print("DECISION CONFIDENCE:")
    print(f"  {result.decision_confidence:.2%}")

    print()
    print("FINAL DECISION:")
    print(f"  {result.decision}")

    print()
    print("REASON:")
    print(f"  {result.reason}")

    print()
    print("RISK FACTORS:")
    if result.risk_factors:
        for item in result.risk_factors:
            print(f"  - {item}")
    else:
        print("  None")

    print()
    print("EVIDENCE / POLICY FACTORS:")
    if result.evidence_policy_factors:
        for item in result.evidence_policy_factors:
            print(f"  - {item}")
    else:
        print("  None")

    print()
    print("=" * 70)


if __name__ == "__main__":

    # Conservative damaged-return test case.
    ml_probabilities = {
        "Legitimate": 0.8917,
        "Policy Abuser": 0.1075,
        "Fraudulent Return": 0.0005,
        "Wardrobing": 0.0003,
    }

    vision_result = {
        "image_quality": "GOOD",
        "product_condition": "DAMAGED",
        "damage_detected": True,
        "packaging_condition": "DAMAGED",
        "evidence_consistent": True,
        "confidence": 0.72,
        "explanation": (
            "Headphones show a snapped headband and exposed "
            "internal wiring, consistent with reported damage."
        ),
    }

    result = analyze_decision(
        ml_probabilities=ml_probabilities,
        policy_status="POLICY_COMPLIANT",
        policy_flags=[],
        vision_result=vision_result,
    )

    print_result(result)
