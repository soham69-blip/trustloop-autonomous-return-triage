# TrustLoop Decision Engine
# Production-oriented conservative decision layer.
#
# Combines:
#   1. ML customer-risk probabilities
#   2. RAG/policy result
#   3. Vision evidence when a physical-condition claim exists
#
# Important principle:
# Missing vision evidence is NOT negative evidence.
# Vision is only used when a vision analysis is actually supplied.


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


# ============================================================
# CUSTOMER RISK
# ============================================================

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

    # Policy flags add risk, but are not treated as proof of fraud.
    if policy_flags:
        risk += min(0.15, 0.05 * len(policy_flags))

    return clamp(risk)


# ============================================================
# CLAIM VALIDITY
# ============================================================

def calculate_claim_validity(vision_result):
    """
    Vision is optional.

    None means:
        No physical-condition evidence was supplied.

    That is NOT the same as:
        Claim is invalid.

    Therefore we return None when vision is unavailable.
    """

    if not vision_result:
        return None

    confidence = clamp(
        vision_result.get("confidence", 0.0)
    )

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


# ============================================================
# POLICY COMPLIANCE
# ============================================================

def calculate_policy_compliance(
    policy_status,
    policy_flags
):
    policy_flags = policy_flags or []

    if (
        policy_status == "POLICY_COMPLIANT"
        and not policy_flags
    ):
        return 1.0

    if policy_status == "HUMAN_ESCALATION":
        return 0.50

    if policy_status == "POLICY_VIOLATION":
        return 0.0

    return clamp(
        1.0 - (0.20 * len(policy_flags))
    )


# ============================================================
# DECISION ENGINE
# ============================================================

def analyze_decision(
    ml_probabilities,
    policy_status,
    policy_flags=None,
    vision_result=None
):
    policy_flags = policy_flags or []

    legitimate = ml_probabilities.get(
        "Legitimate", 0.0
    )

    policy_abuser = ml_probabilities.get(
        "Policy Abuser", 0.0
    )

    fraudulent = ml_probabilities.get(
        "Fraudulent Return", 0.0
    )

    wardrobing = ml_probabilities.get(
        "Wardrobing", 0.0
    )

    # --------------------------------------------------------
    # CORE SIGNALS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FACTORS
    # --------------------------------------------------------

    risk_factors = []
    evidence_policy_factors = []

    # --------------------------------------------------------
    # ML RISK FACTORS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # POLICY FACTORS
    # --------------------------------------------------------

    if policy_status == "HUMAN_ESCALATION":
        evidence_policy_factors.append(
            "Policy rules require human review"
        )

    elif policy_status == "POLICY_VIOLATION":
        evidence_policy_factors.append(
            "Return violates an applicable policy rule"
        )

    elif policy_status == "POLICY_COMPLIANT":
        evidence_policy_factors.append(
            "No policy escalation triggered"
        )

    # --------------------------------------------------------
    # VISION FACTORS
    # --------------------------------------------------------

    damage_detected = False
    evidence_consistent = False
    vision_confidence = None
    vision_available = vision_result is not None

    if vision_result:

        damage_detected = bool(
            vision_result.get(
                "damage_detected",
                False
            )
        )

        evidence_consistent = bool(
            vision_result.get(
                "evidence_consistent",
                False
            )
        )

        vision_confidence = clamp(
            vision_result.get(
                "confidence",
                0.0
            )
        )

        if evidence_consistent:
            evidence_policy_factors.append(
                "Visual evidence is consistent with the claim"
            )

        if damage_detected:
            evidence_policy_factors.append(
                "Reported damage is visually supported"
            )

        if not evidence_consistent:
            evidence_policy_factors.append(
                "Visual evidence is inconsistent or inconclusive"
            )

    # --------------------------------------------------------
    # DECISION CONFIDENCE
    #
    # Without vision:
    #   ML + policy determine confidence.
    #
    # With vision:
    #   ML + policy + claim validity contribute.
    # --------------------------------------------------------

    if claim_validity is None:

        decision_confidence = (
            (1.0 - customer_risk) * 0.55
            + policy_compliance * 0.45
        )

    else:

        decision_confidence = (
            (1.0 - customer_risk) * 0.40
            + claim_validity * 0.25
            + policy_compliance * 0.35
        )

    decision_confidence = clamp(
        decision_confidence
    )

    # ========================================================
    # DECISION PRIORITY
    #
    # Higher-priority safety rules execute first.
    # ========================================================

    # --------------------------------------------------------
    # 1. Strong fraud / wardrobing
    # --------------------------------------------------------

    if (
        fraudulent >= 0.50
        or wardrobing >= 0.50
    ):

        decision = "AUTO_REJECT"

        reason = (
            "Strong model evidence indicates fraudulent "
            "or wardrobing behavior."
        )

    # --------------------------------------------------------
    # 2. Strong policy abuse
    # --------------------------------------------------------

    elif policy_abuser >= 0.50:

        decision = "HUMAN_ESCALATION"

        reason = (
            "Policy abuse probability is elevated and "
            "requires additional evidence review."
        )

    # --------------------------------------------------------
    # 3. Policy escalation
    # --------------------------------------------------------

    elif policy_status == "HUMAN_ESCALATION":

        decision = "HUMAN_ESCALATION"

        reason = (
            "Policy rules require human review before "
            "a final refund decision."
        )

    # --------------------------------------------------------
    # 4. Explicit policy violation
    # --------------------------------------------------------

    elif policy_status == "POLICY_VIOLATION":

        decision = "AUTO_REJECT"

        reason = (
            "The return violates an applicable policy rule."
        )

    # --------------------------------------------------------
    # 5. Physical damage / vision claim
    # --------------------------------------------------------

    elif vision_available and damage_detected:

        if (
            evidence_consistent
            and vision_confidence is not None
            and vision_confidence >= 0.70
            and customer_risk < 0.15
            and policy_compliance >= 0.95
            and decision_confidence >= 0.85
            and legitimate >= 0.70
        ):

            decision = "AUTO_APPROVE"

            reason = (
                "Physical damage is supported by visual evidence, "
                "customer fraud risk is low, policy requirements are "
                "satisfied, and decision confidence is high."
            )

        elif (
            evidence_consistent
            and vision_confidence is not None
            and vision_confidence >= 0.50
            and customer_risk < 0.30
            and policy_compliance >= 0.80
        ):

            decision = "HUMAN_ESCALATION"

            reason = (
                "Physical damage is supported, but the combined "
                "evidence is not strong enough for automatic approval."
            )

        else:

            decision = "HUMAN_ESCALATION"

            reason = (
                "Physical damage was detected, but the evidence "
                "does not support a sufficiently confident "
                "automated approval."
            )

    # --------------------------------------------------------
    # 6. Strong legitimate normal return
    # --------------------------------------------------------

    elif (
        customer_risk < 0.10
        and policy_compliance >= 0.95
        and decision_confidence >= 0.85
        and legitimate >= 0.70
    ):

        decision = "AUTO_APPROVE"

        reason = (
            "Model risk is low, policy requirements are "
            "satisfied, and decision confidence is high."
        )

    # --------------------------------------------------------
    # 7. Uncertain case
    # --------------------------------------------------------

    else:

        decision = "HUMAN_ESCALATION"

        reason = (
            "Evidence is insufficient for a sufficiently "
            "confident automated decision."
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


# ============================================================
# DISPLAY
# ============================================================

def print_result(result: DecisionResult):

    print("=" * 70)
    print("TRUSTLOOP — DECISION ENGINE")
    print("=" * 70)

    print()
    print("CUSTOMER RISK:")
    print(f"  {result.customer_risk:.2%}")

    print()
    print("CLAIM VALIDITY:")

    if result.claim_validity is None:
        print("  N/A — no vision evidence supplied")
    else:
        print(
            f"  {result.claim_validity:.2%}"
        )

    print()
    print("POLICY COMPLIANCE:")
    print(
        f"  {result.policy_compliance:.2%}"
    )

    print()
    print("DECISION CONFIDENCE:")
    print(
        f"  {result.decision_confidence:.2%}"
    )

    print()
    print("FINAL DECISION:")
    print(
        f"  {result.decision}"
    )

    print()
    print("REASON:")
    print(
        f"  {result.reason}"
    )

    print()
    print("RISK FACTORS:")

    if result.risk_factors:

        for item in result.risk_factors:
            print(
                f"  - {item}"
            )

    else:

        print("  None")

    print()
    print("EVIDENCE / POLICY FACTORS:")

    if result.evidence_policy_factors:

        for item in result.evidence_policy_factors:
            print(
                f"  - {item}"
            )

    else:

        print("  None")

    print()
    print("=" * 70)


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

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
            "Headphones show a snapped headband and "
            "exposed internal wiring, consistent with "
            "reported damage."
        ),
    }

    result = analyze_decision(
        ml_probabilities=ml_probabilities,
        policy_status="POLICY_COMPLIANT",
        policy_flags=[],
        vision_result=vision_result,
    )

    print_result(result)

