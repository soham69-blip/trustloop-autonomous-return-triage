from rag.trustloop_decision_engine import analyze_decision


# Simulated output from Identity Fraud Graph V2
identity_result = {
    "identity_link_score": 0.73,
    "fraud_ring_score": 0.80,
    "risk_level": "HIGH",
    "recommendation": "FRAUD_RING_INVESTIGATION",
}


# Normal ML model output
ml_probabilities = {
    "Legitimate": 0.35,
    "Policy Abuser": 0.62,
    "Fraudulent Return": 0.01,
    "Wardrobing": 0.02,
}


# Identity graph should influence policy/risk investigation
policy_flags = [
    "identity_linked_accounts",
    "fraud_ring_signal",
]


result = analyze_decision(
    ml_probabilities=ml_probabilities,
    policy_status="HUMAN_ESCALATION",
    policy_flags=policy_flags,
    vision_result=None,
)


print("=" * 75)
print("TRUSTLOOP — IDENTITY GRAPH + DECISION ENGINE")
print("=" * 75)

print()
print("IDENTITY GRAPH")
print("-" * 75)
print(f"Identity Link Score : {identity_result['identity_link_score']:.2%}")
print(f"Fraud Ring Score    : {identity_result['fraud_ring_score']:.2%}")
print(f"Risk Level          : {identity_result['risk_level']}")
print(f"Recommendation      : {identity_result['recommendation']}")

print()
print("DECISION ENGINE")
print("-" * 75)
print(f"Customer Risk       : {result.customer_risk:.2%}")
print(f"Policy Compliance   : {result.policy_compliance:.2%}")
print(f"Decision Confidence : {result.decision_confidence:.2%}")
print(f"Final Decision      : {result.decision}")

print()
print("Reason:")
print(f"  {result.reason}")

print()
print("Risk Factors:")
for item in result.risk_factors:
    print(f"  - {item}")

print()
print("Evidence / Policy Factors:")
for item in result.evidence_policy_factors:
    print(f"  - {item}")

print()
print("=" * 75)
print("IDENTITY INTEGRATION TEST COMPLETED")
print("=" * 75)
