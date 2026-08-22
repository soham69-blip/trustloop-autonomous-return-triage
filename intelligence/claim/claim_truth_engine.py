from dataclasses import dataclass


@dataclass
class ClaimTruthResult:
    truth_score: float
    status: str
    reasons: list
    contradictions: list
    evidence_strength: float
    policy_support: float


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def analyze_claim(
    claim_type,
    customer_claim,
    vision_result=None,
    policy_result=None,
):
    """
    TrustLoop Claim Truth Engine V2

    Produces a calibrated claim-support score using:
    - Vision confidence
    - Evidence consistency
    - Physical condition
    - Image quality
    - Packaging condition
    - Policy compliance
    - Contradictions
    """

    claim_type = str(claim_type).upper()

    reasons = []
    contradictions = []

    # ---------------------------------------------------------
    # 1. BASELINE
    # ---------------------------------------------------------

    evidence_strength = 0.0
    policy_support = 0.50

    if vision_result:
        vision_confidence = clamp(
            vision_result.get("confidence", 0.0)
        )

        image_quality = str(
            vision_result.get("image_quality", "UNKNOWN")
        ).upper()

        evidence_consistent = bool(
            vision_result.get("evidence_consistent", False)
        )

        damage_detected = bool(
            vision_result.get("damage_detected", False)
        )

        product_condition = str(
            vision_result.get("product_condition", "UNKNOWN")
        ).upper()

        packaging_condition = str(
            vision_result.get("packaging_condition", "UNKNOWN")
        ).upper()

        # -----------------------------------------------------
        # 2. VISION EVIDENCE STRENGTH
        # -----------------------------------------------------

        evidence_strength = vision_confidence

        if image_quality == "GOOD":
            evidence_strength += 0.05

        elif image_quality == "POOR":
            evidence_strength -= 0.20
            contradictions.append(
                "Image quality is insufficient for reliable evidence verification."
            )

        if evidence_consistent:
            evidence_strength += 0.10
            reasons.append(
                "Visual evidence is consistent with the customer claim."
            )
        else:
            evidence_strength -= 0.15
            contradictions.append(
                "Visual evidence is not sufficiently consistent with the claim."
            )

        # -----------------------------------------------------
        # 3. CLAIM-SPECIFIC VALIDATION
        # -----------------------------------------------------

        if claim_type == "DAMAGED":

            if damage_detected:
                evidence_strength += 0.15
                reasons.append(
                    "Physical damage detected in submitted evidence."
                )
            else:
                evidence_strength -= 0.25
                contradictions.append(
                    "No physical damage detected despite a damage claim."
                )

            if product_condition == "DAMAGED":
                evidence_strength += 0.10
                reasons.append(
                    "Detected product condition matches the claimed damage."
                )

            if packaging_condition == "DAMAGED":
                evidence_strength += 0.05
                reasons.append(
                    "Packaging damage provides additional supporting evidence."
                )

            elif packaging_condition == "INTACT":
                evidence_strength -= 0.05
                contradictions.append(
                    "Packaging appears intact despite the reported damage."
                )

        elif claim_type == "WRONG_ITEM":

            wrong_item = bool(
                vision_result.get("wrong_item_detected", False)
            )

            if wrong_item:
                evidence_strength += 0.20
                reasons.append(
                    "Visual evidence indicates the received item differs from the expected item."
                )
            else:
                evidence_strength -= 0.20
                contradictions.append(
                    "Submitted evidence does not establish that the wrong item was received."
                )

        elif claim_type == "MISSING_ITEM":

            missing_detected = bool(
                vision_result.get("missing_item_detected", False)
            )

            if missing_detected:
                evidence_strength += 0.20
                reasons.append(
                    "Evidence supports the reported missing-item condition."
                )
            else:
                evidence_strength -= 0.15
                contradictions.append(
                    "Evidence does not independently establish the missing-item claim."
                )

        # -----------------------------------------------------
        # 4. POLICY VALIDATION
        # -----------------------------------------------------

    if policy_result:

        compliant = policy_result.get("compliant")

        contradiction = bool(
            policy_result.get("contradiction", False)
        )

        if compliant is True:
            policy_support = 1.0
            reasons.append(
                "Claim is consistent with applicable policy."
            )

        elif compliant is False:
            policy_support = 0.0
            contradictions.append(
                "Claim does not satisfy applicable policy requirements."
            )

        else:
            policy_support = 0.50

        if contradiction:
            policy_support -= 0.40
            contradictions.append(
                "Policy evidence contradicts the claim."
            )

    # ---------------------------------------------------------
    # 5. NORMALIZE EVIDENCE
    # ---------------------------------------------------------

    evidence_strength = clamp(evidence_strength)
    policy_support = clamp(policy_support)

    # Evidence should dominate policy.
    #
    # 70% evidence
    # 30% policy

    truth_score = (
        evidence_strength * 0.70
        + policy_support * 0.30
    )

    truth_score = clamp(truth_score)

    # ---------------------------------------------------------
    # 6. CONTRADICTION PENALTY
    # ---------------------------------------------------------

    if contradictions:
        contradiction_penalty = min(
            0.35,
            len(contradictions) * 0.10
        )

        truth_score = clamp(
            truth_score - contradiction_penalty
        )

    # ---------------------------------------------------------
    # 7. STATUS
    # ---------------------------------------------------------

    if truth_score >= 0.85:
        status = "HIGHLY_SUPPORTED"

    elif truth_score >= 0.70:
        status = "SUPPORTED"

    elif truth_score >= 0.45:
        status = "UNCERTAIN"

    else:
        status = "CONTRADICTED"

    if not reasons:
        reasons.append(
            "Insufficient evidence was available to strongly support the claim."
        )

    return ClaimTruthResult(
        truth_score=round(truth_score, 4),
        status=status,
        reasons=reasons,
        contradictions=contradictions,
        evidence_strength=round(evidence_strength, 4),
        policy_support=round(policy_support, 4),
    )


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TRUSTLOOP — CURRENT CLAIM TRUTH ENGINE V2")
    print("=" * 70)

    result = analyze_claim(
        claim_type="DAMAGED",
        customer_claim="Product arrived damaged",

        vision_result={
            "damage_detected": True,
            "evidence_consistent": True,
            "confidence": 0.92,
            "image_quality": "GOOD",
            "product_condition": "DAMAGED",
            "packaging_condition": "DAMAGED",
        },

        policy_result={
            "compliant": True,
            "contradiction": False,
        },
    )

    print()
    print(f"Truth Score       : {result.truth_score:.2%}")
    print(f"Evidence Strength : {result.evidence_strength:.2%}")
    print(f"Policy Support    : {result.policy_support:.2%}")
    print(f"Status            : {result.status}")

    print()
    print("Supporting Evidence:")

    for reason in result.reasons:
        print(f"  + {reason}")

    if result.contradictions:
        print()
        print("Contradictions:")

        for contradiction in result.contradictions:
            print(f"  - {contradiction}")

    print()
    print("=" * 70)
    print("CLAIM TRUTH ENGINE V2 TEST COMPLETED")
    print("=" * 70)
