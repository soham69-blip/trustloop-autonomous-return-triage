from rag.trustloop_decision_engine import analyze_decision


def run_case(
    name,
    ml,
    policy_status="POLICY_COMPLIANT",
    policy_flags=None,
    vision=None,
):
    result = analyze_decision(
        ml_probabilities=ml,
        policy_status=policy_status,
        policy_flags=policy_flags or [],
        vision_result=vision,
    )

    print("=" * 75)
    print(name)
    print("=" * 75)

    print(f"Customer Risk      : {result.customer_risk:.2%}")

    if result.claim_validity is None:
        print("Claim Validity     : N/A - no vision evidence")
    else:
        print(
            f"Claim Validity     : {result.claim_validity:.2%}"
        )

    print(
        f"Policy Compliance  : {result.policy_compliance:.2%}"
    )

    print(
        f"Decision Confidence: {result.decision_confidence:.2%}"
    )

    print(
        f"Decision            : {result.decision}"
    )

    print(
        f"Reason              : {result.reason}"
    )

    if result.risk_factors:
        print("Risk Factors:")
        for x in result.risk_factors:
            print(f"  - {x}")

    if result.evidence_policy_factors:
        print("Evidence/Policy:")
        for x in result.evidence_policy_factors:
            print(f"  - {x}")

    print()


CASES = [

    (
        "CASE 1 â€” Normal Legitimate Return",
        {
            "Legitimate": 0.96,
            "Policy Abuser": 0.03,
            "Fraudulent Return": 0.005,
            "Wardrobing": 0.005,
        },
        "POLICY_COMPLIANT",
        [],
        None,
    ),

    (
        "CASE 2 â€” Policy Abuser",
        {
            "Legitimate": 0.35,
            "Policy Abuser": 0.62,
            "Fraudulent Return": 0.01,
            "Wardrobing": 0.02,
        },
        "POLICY_COMPLIANT",
        [],
        None,
    ),

    (
        "CASE 3 â€” Strong Fraud",
        {
            "Legitimate": 0.05,
            "Policy Abuser": 0.05,
            "Fraudulent Return": 0.88,
            "Wardrobing": 0.02,
        },
        "POLICY_COMPLIANT",
        [],
        None,
    ),

    (
        "CASE 4 â€” Strong Wardrobing",
        {
            "Legitimate": 0.05,
            "Policy Abuser": 0.03,
            "Fraudulent Return": 0.02,
            "Wardrobing": 0.90,
        },
        "POLICY_COMPLIANT",
        [],
        None,
    ),

    (
        "CASE 5 â€” Late Return",
        {
            "Legitimate": 0.85,
            "Policy Abuser": 0.12,
            "Fraudulent Return": 0.01,
            "Wardrobing": 0.02,
        },
        "HUMAN_ESCALATION",
        ["return_window"],
        None,
    ),

    (
        "CASE 6 â€” Multiple Accounts",
        {
            "Legitimate": 0.80,
            "Policy Abuser": 0.17,
            "Fraudulent Return": 0.01,
            "Wardrobing": 0.02,
        },
        "HUMAN_ESCALATION",
        ["multiple_accounts"],
        None,
    ),

    (
        "CASE 7 â€” High Value Product",
        {
            "Legitimate": 0.82,
            "Policy Abuser": 0.14,
            "Fraudulent Return": 0.01,
            "Wardrobing": 0.03,
        },
        "HUMAN_ESCALATION",
        ["high_value_product"],
        None,
    ),

    (
        "CASE 8 â€” Damaged Product",
        {
            "Legitimate": 0.89,
            "Policy Abuser": 0.10,
            "Fraudulent Return": 0.005,
            "Wardrobing": 0.005,
        },
        "POLICY_COMPLIANT",
        [],
        {
            "image_quality": "GOOD",
            "product_condition": "DAMAGED",
            "damage_detected": True,
            "packaging_condition": "DAMAGED",
            "evidence_consistent": True,
            "confidence": 0.72,
        },
    ),

]


print()
print("=" * 75)
print("TRUSTLOOP â€” DECISION ENGINE SCENARIO TEST")
print("=" * 75)
print()

for case in CASES:
    run_case(*case)

print("=" * 75)
print("SCENARIO TEST COMPLETED")
print("=" * 75)


