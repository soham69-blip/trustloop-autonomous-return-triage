from intelligence.claim.claim_truth_engine import analyze_claim
from intelligence.graph.identity_graph_v2 import IdentityFraudGraphV2
from intelligence.evidence.evidence_reuse_detector import EvidenceReuseDetector
from intelligence.investigation.investigation_agent import investigate
from intelligence.decision.expected_loss_engine import calculate_expected_loss
from intelligence.decision.unified_intelligence_engine import calculate_unified_decision


def no_reuse():
    return {
        "reuse_score": 0.0,
        "status": "NO_REUSE",
    }


def reuse():
    return {
        "reuse_score": 0.50,
        "status": "POSSIBLE_REUSE",
    }


def losses(customer_risk, claim_truth, refund_amount=10000.0):

    result = calculate_expected_loss(
        fraud_probability=customer_risk,
        claim_truth=claim_truth,
        refund_amount=refund_amount,
        review_cost=50.0,
    )

    return {
        option.decision: option.expected_loss
        for option in result.options
    }


def run_case(
    name,
    customer_risk,
    claim_truth,
    identity_result,
    investigation_result,
    expected_losses,
    evidence_result,
):

    result = calculate_unified_decision(
        customer_risk=customer_risk,
        claim_truth=claim_truth,
        identity_result=identity_result,
        investigation_result=investigation_result,
        expected_loss_result=expected_losses,
        evidence_result=evidence_result,
    )

    print()
    print("=" * 78)
    print(name)
    print("=" * 78)

    print(f"Customer Risk       : {customer_risk:.2%}")
    print(f"Claim Truth         : {claim_truth:.2%}")
    print(
        f"Identity Risk       : "
        f"{identity_result.get('identity_link_score', 0):.2%}"
    )
    print(
        f"Fraud Ring Risk     : "
        f"{identity_result.get('fraud_ring_score', 0):.2%}"
    )
    print(
        f"Evidence Reuse      : "
        f"{evidence_result.get('reuse_score', 0):.2%}"
    )
    print(
        f"Investigation Score : "
        f"{investigation_result.get('investigation_score', 0):.2%}"
    )

    print()
    print(f"DECISION            : {result['decision']}")
    print(f"CONFIDENCE          : {result['confidence']:.2%}")
    print(
        f"LOWEST EXPECTED LOSS: "
        f"{result['lowest_expected_loss_decision']}"
    )
    print(f"REASON              : {result['reason']}")

    if result.get("explanation"):
        print("INTELLIGENCE EXPLANATION:")
        for item in result["explanation"]:
            print(f"  - {item}")
    if result["blockers"]:
        print("BLOCKERS:")
        for blocker in result["blockers"]:
            print(f"  - {blocker}")


def main():

    print()
    print("=" * 78)
    print("TRUSTLOOP — MULTI-SCENARIO INTELLIGENCE EVALUATION")
    print("=" * 78)

    # ---------------------------------------------------------
    # CASE 1 — Genuine Low-Risk Return
    # ---------------------------------------------------------

    run_case(
        "CASE 1 — Genuine Low-Risk Return",
        customer_risk=0.05,
        claim_truth=0.95,
        identity_result={
            "identity_link_score": 0.00,
            "fraud_ring_score": 0.00,
        },
        investigation_result={
            "investigation_score": 0.05,
        },
        expected_losses=losses(0.05, 0.95),
        evidence_result=no_reuse(),
    )

    # ---------------------------------------------------------
    # CASE 2 — Genuine Damaged Product
    # ---------------------------------------------------------

    run_case(
        "CASE 2 — Genuine Damaged Product",
        customer_risk=0.08,
        claim_truth=0.97,
        identity_result={
            "identity_link_score": 0.00,
            "fraud_ring_score": 0.00,
        },
        investigation_result={
            "investigation_score": 0.05,
        },
        expected_losses=losses(0.08, 0.97),
        evidence_result=no_reuse(),
    )

    # ---------------------------------------------------------
    # CASE 3 — Strong Fraud
    # ---------------------------------------------------------

    run_case(
        "CASE 3 — Strong Fraud",
        customer_risk=0.92,
        claim_truth=0.20,
        identity_result={
            "identity_link_score": 0.10,
            "fraud_ring_score": 0.05,
        },
        investigation_result={
            "investigation_score": 0.85,
        },
        expected_losses=losses(0.92, 0.20),
        evidence_result=no_reuse(),
    )

    # ---------------------------------------------------------
    # CASE 4 — Fraud Ring
    # ---------------------------------------------------------

    run_case(
        "CASE 4 — Fraud Ring",
        customer_risk=0.55,
        claim_truth=0.40,
        identity_result={
            "identity_link_score": 0.73,
            "fraud_ring_score": 0.80,
        },
        investigation_result={
            "investigation_score": 0.70,
        },
        expected_losses=losses(0.55, 0.40),
        evidence_result=no_reuse(),
    )

    # ---------------------------------------------------------
    # CASE 5 — Evidence Reuse
    # ---------------------------------------------------------

    run_case(
        "CASE 5 — Evidence Reuse",
        customer_risk=0.60,
        claim_truth=0.45,
        identity_result={
            "identity_link_score": 0.15,
            "fraud_ring_score": 0.10,
        },
        investigation_result={
            "investigation_score": 0.75,
        },
        expected_losses=losses(0.60, 0.45),
        evidence_result=reuse(),
    )

    # ---------------------------------------------------------
    # CASE 6 — Policy Ambiguity
    # ---------------------------------------------------------

    run_case(
        "CASE 6 — Policy Ambiguity",
        customer_risk=0.20,
        claim_truth=0.65,
        identity_result={
            "identity_link_score": 0.10,
            "fraud_ring_score": 0.05,
        },
        investigation_result={
            "investigation_score": 0.55,
        },
        expected_losses=losses(0.20, 0.65),
        evidence_result=no_reuse(),
    )

    # ---------------------------------------------------------
    # CASE 7 — High-Value Legitimate Customer
    # ---------------------------------------------------------

    run_case(
        "CASE 7 — High-Value Legitimate Customer",
        customer_risk=0.12,
        claim_truth=0.92,
        identity_result={
            "identity_link_score": 0.05,
            "fraud_ring_score": 0.00,
        },
        investigation_result={
            "investigation_score": 0.20,
        },
        expected_losses=losses(0.12, 0.92, 50000.0),
        evidence_result=no_reuse(),
    )

    # ---------------------------------------------------------
    # CASE 8 — Conflicting Evidence
    # ---------------------------------------------------------

    run_case(
        "CASE 8 — Conflicting Evidence",
        customer_risk=0.35,
        claim_truth=0.35,
        identity_result={
            "identity_link_score": 0.20,
            "fraud_ring_score": 0.10,
        },
        investigation_result={
            "investigation_score": 0.65,
        },
        expected_losses=losses(0.35, 0.35),
        evidence_result=no_reuse(),
    )

    print()
    print("=" * 78)
    print("EVALUATION COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    main()

