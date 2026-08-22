from dataclasses import dataclass


@dataclass
class DecisionOption:
    decision: str
    expected_loss: float
    rationale: str


@dataclass
class ExpectedLossResult:
    recommended_decision: str
    options: list


def calculate_expected_loss(
    fraud_probability,
    claim_truth,
    refund_amount,
    review_cost=50.0,
):

    fraud_probability = max(
        0.0,
        min(1.0, float(fraud_probability))
    )

    claim_truth = max(
        0.0,
        min(1.0, float(claim_truth))
    )

    refund_amount = float(refund_amount)
    review_cost = float(review_cost)

    legitimate_probability = (
        (1.0 - fraud_probability) * 0.50
        + claim_truth * 0.50
    )

    # ---------------------------------------------------------
    # AUTO APPROVE
    # ---------------------------------------------------------

    approve_loss = (
        fraud_probability
        * refund_amount
    )

    # ---------------------------------------------------------
    # AUTO REJECT
    # ---------------------------------------------------------

    reject_loss = (
        legitimate_probability
        * refund_amount
    )

    # ---------------------------------------------------------
    # HUMAN ESCALATION
    #
    # Human review is valuable for ambiguous cases,
    # but should NOT automatically dominate obvious cases.
    # ---------------------------------------------------------

    ambiguity = abs(
        fraud_probability - claim_truth
    )

    uncertainty = (
        1.0 - abs(
            fraud_probability
            + claim_truth
            - 1.0
        )
    )

    review_loss = (
        review_cost
        + uncertainty
        * refund_amount
        * 0.50
    )

    # Strong legitimate evidence:
    # approving is cheaper than unnecessary review.

    if (
        fraud_probability <= 0.15
        and claim_truth >= 0.85
    ):
        review_loss = min(
            review_loss,
            approve_loss + refund_amount * 0.05
        )

    # Strong fraud evidence:
    # rejection should dominate unnecessary review.

    if (
        fraud_probability >= 0.80
        and claim_truth <= 0.40
    ):
        review_loss = min(
            review_loss,
            reject_loss + refund_amount * 0.05
        )

    options = [

        DecisionOption(
            decision="AUTO_APPROVE",
            expected_loss=approve_loss,
            rationale=(
                "Expected financial loss from approving "
                "a potentially fraudulent claim."
            ),
        ),

        DecisionOption(
            decision="AUTO_REJECT",
            expected_loss=reject_loss,
            rationale=(
                "Expected customer and business loss from "
                "rejecting a potentially legitimate claim."
            ),
        ),

        DecisionOption(
            decision="HUMAN_ESCALATION",
            expected_loss=review_loss,
            rationale=(
                "Human review cost plus residual uncertainty."
            ),
        ),
    ]

    best = min(
        options,
        key=lambda option: option.expected_loss
    )

    return ExpectedLossResult(
        recommended_decision=best.decision,
        options=options,
    )


if __name__ == "__main__":

    result = calculate_expected_loss(
        fraud_probability=0.80,
        claim_truth=0.30,
        refund_amount=10000,
        review_cost=50,
    )

    print("=" * 70)
    print("TRUSTLOOP — EXPECTED LOSS & COUNTERFACTUAL ENGINE V3")
    print("=" * 70)

    print()

    for option in result.options:

        print(option.decision)
        print(
            f"Expected Loss : ₹{option.expected_loss:,.2f}"
        )
        print(
            f"Rationale     : {option.rationale}"
        )
        print()

    print(
        f"RECOMMENDED DECISION: "
        f"{result.recommended_decision}"
    )

    print("=" * 70)
