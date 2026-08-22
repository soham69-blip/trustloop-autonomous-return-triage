POLICY_ABUSER_THRESHOLD = 0.30

CLASS_NAMES = {
    0: "Legitimate",
    1: "Policy Abuser",
    2: "Fraudulent Return",
    3: "Wardrobing"
}

DECISION_CONFIDENCE_THRESHOLDS = {
    "AUTO_APPROVE": 0.85,
    "AUTO_REJECT": 0.90,
    "HUMAN_ESCALATION": 0.00
}
