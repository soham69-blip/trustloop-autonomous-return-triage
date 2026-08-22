from pathlib import Path
import sys


# Allow importing retriever.py when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag.retriever import retrieve_policy


# ============================================================
# POLICY DECISION
# ============================================================

def evaluate_policy(
    return_case,
    top_k=5,
):
    """
    Retrieve relevant policy sections and produce
    a deterministic policy assessment.

    This layer does NOT make the final TrustLoop
    fraud decision.

    It only evaluates policy compliance.
    """

    query_parts = []

    if "days_to_return" in return_case:
        query_parts.append(
            f"return requested after "
            f"{return_case['days_to_return']} days"
        )

    if "product_category" in return_case:
        query_parts.append(
            f"product category "
            f"{return_case['product_category']}"
        )

    if "return_reason" in return_case:
        query_parts.append(
            f"return reason "
            f"{return_case['return_reason']}"
        )

    if return_case.get(
        "multiple_accounts_flag",
        False
    ):
        query_parts.append(
            "multiple customer accounts"
        )

    if return_case.get(
        "is_high_value_item",
        False
    ):
        query_parts.append(
            "high value product"
        )

    if return_case.get(
        "discount_used",
        False
    ):
        query_parts.append(
            "discounted order"
        )

    query = " ".join(
        query_parts
    )

    if not query:
        query = (
            "return refund eligibility "
            "policy requirements"
        )

    retrieved = retrieve_policy(
        query,
        top_k=top_k
    )

    # --------------------------------------------------------
    # RULE EVALUATION
    # --------------------------------------------------------

    policy_flags = []

    days_to_return = return_case.get(
        "days_to_return"
    )

    if (
        days_to_return is not None
        and days_to_return > 30
    ):
        policy_flags.append(
            {
                "rule": "return_window",
                "status": "REVIEW_REQUIRED",
                "reason": (
                    "Return requested after "
                    "the 30-day standard "
                    "return window."
                ),
            }
        )

    if return_case.get(
        "multiple_accounts_flag",
        False
    ):
        policy_flags.append(
            {
                "rule": "multiple_accounts",
                "status": "REVIEW_REQUIRED",
                "reason": (
                    "Multiple-account activity "
                    "requires additional review."
                ),
            }
        )

    if return_case.get(
        "is_high_value_item",
        False
    ):
        policy_flags.append(
            {
                "rule": "high_value_product",
                "status": "REVIEW_REQUIRED",
                "reason": (
                    "High-value merchandise "
                    "may require additional "
                    "verification."
                ),
            }
        )

    # --------------------------------------------------------
    # POLICY STATUS
    # --------------------------------------------------------

    if policy_flags:

        policy_status = (
            "HUMAN_ESCALATION"
        )

    else:

        policy_status = (
            "POLICY_COMPLIANT"
        )

    return {
        "query": query,
        "policy_status": policy_status,
        "flags": policy_flags,
        "retrieved_policy": retrieved,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TRUSTLOOP — POLICY AGENT")
    print("=" * 70)

    test_cases = [

        {
            "name": "Normal return",

            "days_to_return": 10,

            "product_category":
                "electronics",

            "return_reason":
                "changed_mind",

            "multiple_accounts_flag":
                False,

            "is_high_value_item":
                False,

            "discount_used":
                False,
        },

        {
            "name": "Late return",

            "days_to_return": 45,

            "product_category":
                "fashion",

            "return_reason":
                "changed_mind",

            "multiple_accounts_flag":
                False,

            "is_high_value_item":
                False,

            "discount_used":
                False,
        },

        {
            "name":
                "Multiple account + high value",

            "days_to_return": 12,

            "product_category":
                "electronics",

            "return_reason":
                "defective",

            "multiple_accounts_flag":
                True,

            "is_high_value_item":
                True,

            "discount_used":
                True,
        },
    ]


    for case in test_cases:

        print(
            "\n" + "-" * 70
        )

        print(
            "CASE:",
            case["name"]
        )

        result = evaluate_policy(
            case
        )

        print(
            "POLICY STATUS:",
            result["policy_status"]
        )

        if result["flags"]:

            print("\nFLAGS:")

            for flag in result["flags"]:

                print(
                    f"- {flag['rule']}: "
                    f"{flag['reason']}"
                )

        else:

            print(
                "FLAGS: None"
            )

        print(
            "\nRETRIEVED POLICY:"
        )

        for item in result[
            "retrieved_policy"
        ][:3]:

            print(
                f"\n[{item['score']:.6f}]"
            )

            print(
                item["text"]
            )


    print(
        "\n" + "=" * 70
    )

    print(
        "POLICY AGENT TEST COMPLETED"
    )

    print(
        "=" * 70
    )