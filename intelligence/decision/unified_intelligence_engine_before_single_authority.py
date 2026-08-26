from intelligence.claim.claim_truth_engine import analyze_claim
from intelligence.evidence.evidence_reuse_detector import EvidenceReuseDetector
from intelligence.graph.identity_graph_v2 import IdentityFraudGraphV2 as IdentityFraudGraph
from intelligence.investigation.investigation_agent import investigate as investigate_case
from intelligence.decision.expected_loss_engine import calculate_expected_loss as calculate_expected_losses


def calculate_unified_decision(
    customer_risk,
    claim_truth,
    identity_result,
    investigation_result,
    expected_loss_result,
    evidence_result,
):
    customer_risk = float(customer_risk)
    claim_truth = float(claim_truth)

    identity_risk = float(identity_result.get("identity_link_score", 0.0))
    fraud_ring_risk = float(identity_result.get("fraud_ring_score", 0.0))
    evidence_reuse_risk = float(evidence_result.get("reuse_score", 0.0))
    investigation_score = float(
        investigation_result.get("investigation_score", 0.0)
    )

    losses = {
        "AUTO_APPROVE": float(expected_loss_result.get("AUTO_APPROVE", float("inf"))),
        "AUTO_REJECT": float(expected_loss_result.get("AUTO_REJECT", float("inf"))),
        "HUMAN_ESCALATION": float(expected_loss_result.get("HUMAN_ESCALATION", float("inf"))),
    }

    lowest_loss_decision = min(losses, key=lambda k: losses[k])

    blockers = []
    explanation = []

    if fraud_ring_risk >= 0.70:
        blockers.append("HIGH_FRAUD_RING_SIGNAL")
        explanation.append(
            "Connected customer relationships indicate potential coordinated fraud."
        )

    if identity_risk >= 0.70:
        blockers.append("HIGH_IDENTITY_LINK_SIGNAL")
        explanation.append(
            "Strong identity linkage was detected across customer accounts."
        )

    if evidence_reuse_risk >= 0.70:
        blockers.append("HIGH_EVIDENCE_REUSE_SIGNAL")
        explanation.append(
            "Evidence appears to have been reused across previous claims."
        )

    if investigation_score >= 0.70:
        blockers.append("HIGH_INVESTIGATION_PRIORITY")
        explanation.append(
            "Autonomous investigation classified the case as high priority."
        )

    if customer_risk >= 0.80:
        blockers.append("HIGH_CUSTOMER_RISK")

    if claim_truth <= 0.30:
        blockers.append("LOW_CLAIM_TRUTH")

    # EXTREME FRAUD
    if (
        customer_risk >= 0.85
        or fraud_ring_risk >= 0.80
        or (
            customer_risk >= 0.70
            and claim_truth <= 0.40
        )
    ):
        decision = "AUTO_REJECT"
        confidence = max(
            customer_risk,
            fraud_ring_risk,
            1.0 - claim_truth,
        )
        reason = (
            "Fraud intelligence is sufficiently strong to support "
            "automatic rejection."
        )

    # STRONG LEGITIMATE CLAIM
    elif (
        customer_risk <= 0.15
        and claim_truth >= 0.85
        and identity_risk < 0.30
        and fraud_ring_risk < 0.30
        and evidence_reuse_risk < 0.30
        and investigation_score < 0.50
    ):
        decision = "AUTO_APPROVE"

        confidence = min(
            0.99,
            (
                (1.0 - customer_risk) * 0.45
                + claim_truth * 0.45
                + (1.0 - identity_risk) * 0.10
            ),
        )

        reason = (
            "Customer risk is low, claim evidence is strong, "
            "and no significant fraud-network signal is present."
        )

        explanation.append(
            "Claim evidence strongly supports a legitimate resolution."
        )

    # EVIDENCE REUSE
    elif evidence_reuse_risk >= 0.30:
        decision = "HUMAN_ESCALATION"

        confidence = max(
            evidence_reuse_risk,
            investigation_score,
        )

        reason = (
            "Evidence reuse creates uncertainty that requires "
            "additional investigation before a final decision."
        )

    # STRONG INVESTIGATION SIGNAL
    elif investigation_score >= 0.60:
        decision = "HUMAN_ESCALATION"

        confidence = max(
            investigation_score,
            identity_risk,
        )

        reason = (
            "Autonomous investigation identified material risk, "
            "but the evidence is not strong enough for automatic rejection."
        )

    # LOW CLAIM TRUTH
    elif claim_truth < 0.55:
        decision = "HUMAN_ESCALATION"

        confidence = 1.0 - claim_truth

        reason = (
            "Claim evidence is insufficient to safely approve or reject "
            "the claim automatically."
        )

    # MODERATE FRAUD RISK
    elif customer_risk >= 0.40:
        decision = "HUMAN_ESCALATION"

        confidence = customer_risk

        reason = (
            "Customer fraud risk is elevated but not sufficiently strong "
            "for automatic rejection."
        )

    # AMBIGUOUS CASE
    else:
        decision = "HUMAN_ESCALATION"

        confidence = max(
            customer_risk,
            1.0 - claim_truth,
            investigation_score,
        )

        reason = (
            "Evidence is not sufficiently decisive for a safe "
            "automatic decision."
        )

    if lowest_loss_decision != decision:
        explanation.append(
            f"Expected-loss model preferred {lowest_loss_decision}, "
            f"but evidence hierarchy selected {decision}."
        )

    return {
        "decision": decision,
        "confidence": round(
            min(max(confidence, 0.0), 1.0),
            4,
        ),
        "reason": reason,
        "blockers": blockers,
        "lowest_expected_loss_decision": lowest_loss_decision,
        "expected_losses": losses,
        "signals": {
            "customer_risk": round(customer_risk, 4),
            "claim_truth": round(claim_truth, 4),
            "identity_risk": round(identity_risk, 4),
            "fraud_ring_risk": round(fraud_ring_risk, 4),
            "evidence_reuse_risk": round(evidence_reuse_risk, 4),
            "investigation_score": round(investigation_score, 4),
        },
        "explanation": explanation,
    }


