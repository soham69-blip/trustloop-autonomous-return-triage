from dataclasses import dataclass


@dataclass
class InvestigationResult:
    investigation_score: float
    priority: str
    actions: list
    reasons: list


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def investigate(
    customer_risk=0.0,
    identity_result=None,
    claim_truth_result=None,
    evidence_reuse_result=None,
):

    identity_result = identity_result or {}
    claim_truth_result = claim_truth_result or {}
    evidence_reuse_result = evidence_reuse_result or {}

    score = 0.0
    actions = []
    reasons = []

    fraud_ring = float(
        identity_result.get("fraud_ring_score", 0.0)
    )

    identity_link = float(
        identity_result.get("identity_link_score", 0.0)
    )

    claim_truth = float(
        claim_truth_result.get("truth_score", 0.50)
    )

    evidence_reuse = float(
        evidence_reuse_result.get("reuse_score", 0.0)
    )

    score += customer_risk * 0.30
    score += fraud_ring * 0.30
    score += identity_link * 0.15
    score += evidence_reuse * 0.25

    if fraud_ring >= 0.70:
        actions.append("Investigate linked customer cluster.")
        reasons.append("Strong fraud-ring signal detected.")

    if identity_link >= 0.60:
        actions.append("Review shared identity attributes.")
        reasons.append("Multiple identity attributes are strongly linked.")

    if evidence_reuse >= 0.50:
        actions.append("Compare reused evidence against previous claims.")
        reasons.append("Potential evidence reuse detected.")

    if claim_truth < 0.40:
        actions.append("Request additional claim evidence.")
        reasons.append("Claim truth score is weak.")

    score = clamp(score)

    if score >= 0.75:
        priority = "CRITICAL"
    elif score >= 0.50:
        priority = "HIGH"
    elif score >= 0.25:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    return InvestigationResult(
        investigation_score=score,
        priority=priority,
        actions=actions,
        reasons=reasons,
    )


if __name__ == "__main__":

    result = investigate(
        customer_risk=0.55,
        identity_result={
            "identity_link_score": 0.73,
            "fraud_ring_score": 0.80,
        },
        claim_truth_result={
            "truth_score": 0.70,
        },
        evidence_reuse_result={
            "reuse_score": 0.75,
        },
    )

    print("=" * 70)
    print("TRUSTLOOP — AUTONOMOUS INVESTIGATION AGENT")
    print("=" * 70)

    print(f"Investigation Score : {result.investigation_score:.2%}")
    print(f"Priority            : {result.priority}")

    print()
    print("Recommended Actions:")

    for action in result.actions:
        print(f"  - {action}")

    print()
    print("Reasons:")

    for reason in result.reasons:
        print(f"  - {reason}")

    print("=" * 70)
