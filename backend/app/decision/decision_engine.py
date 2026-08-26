from typing import Dict, Any


LABELS = {
    0: "Legitimate",
    1: "Policy Abuser",
    2: "Fraudulent Return",
    3: "Wardrobing",
}


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def calculate_deterministic_risk(case: Dict[str, Any]) -> tuple[float, list[str]]:
    risk = 0.0
    signals = []

    return_count = int(case.get("customer_return_count_prior", 0) or 0)
    returns_30 = int(case.get("returns_last_30d_prior", 0) or 0)
    returns_90 = int(case.get("returns_last_90d_prior", 0) or 0)
    lifetime_returns = int(case.get("total_returns_lifetime_prior", 0) or 0)

    orders = int(case.get("previous_orders", 0) or 0)

    days_to_return = float(case.get("days_to_return", 0) or 0)

    if return_count >= 8:
        risk += 20
        signals.append("Very high historical return activity")
    elif return_count >= 5:
        risk += 15
        signals.append("High historical return activity")
    elif return_count >= 3:
        risk += 8
        signals.append("Elevated historical return activity")

    if returns_30 >= 3:
        risk += 15
        signals.append("Multiple returns in last 30 days")
    elif returns_30 >= 2:
        risk += 8
        signals.append("Elevated returns in last 30 days")

    if returns_90 >= 6:
        risk += 15
        signals.append("High returns in last 90 days")
    elif returns_90 >= 4:
        risk += 8
        signals.append("Elevated returns in last 90 days")

    if lifetime_returns >= 10:
        risk += 15
        signals.append("High lifetime return count")

    if orders > 0:
        return_rate = lifetime_returns / max(orders, 1)

        if return_rate >= 0.70:
            risk += 20
            signals.append("Very high historical return rate")
        elif return_rate >= 0.40:
            risk += 10
            signals.append("Elevated historical return rate")

    if days_to_return <= 2:
        risk += 10
        signals.append("Very fast return")

    condition = str(case.get("item_condition", "") or "").lower()

    if condition in {"used", "worn"}:
        risk += 15
        signals.append("Item appears used or worn")

    elif condition in {"damaged", "missing"}:
        risk += 20
        signals.append("Item-condition concern")

    if int(case.get("multiple_accounts_flag", 0) or 0) == 1:
        risk += 20
        signals.append("Multiple-account activity")

    if int(case.get("is_high_value_item", 0) or 0) == 1:
        risk += 5
        signals.append("High-value item")

    return clamp(risk), signals


def calculate_ml_risk(
    probabilities: Dict[str, float]
) -> float:

    legitimate = probabilities.get("Legitimate", 0.0)
    policy = probabilities.get("Policy Abuser", 0.0)
    fraud = probabilities.get("Fraudulent Return", 0.0)
    wardrobe = probabilities.get("Wardrobing", 0.0)

    risk = (
        policy * 70.0
        + fraud * 100.0
        + wardrobe * 85.0
        + legitimate * 0.0
    )

    return clamp(risk)


def make_decision(
    case: Dict[str, Any],
    probabilities: Dict[str, float],
    ml_label: str,
) -> Dict[str, Any]:

    deterministic_risk, signals = calculate_deterministic_risk(case)

    ml_risk = calculate_ml_risk(probabilities)

    # ML is authoritative for classification,
    # while deterministic rules provide explainability.
    #
    # Give ML more weight because it was trained on historical
    # return-abuse patterns.
    combined_risk = (
        ml_risk * 0.70
        + deterministic_risk * 0.30
    )

    combined_risk = clamp(combined_risk)

    ml_confidence = max(probabilities.values()) if probabilities else 0.0

    # Confidence represents how strongly the model supports
    # the selected class, not how "good" the customer is.
    decision_confidence = clamp(
        50.0 + (ml_confidence * 50.0)
    )

    # Add ML explanation when it detects abuse.
    if ml_label != "Legitimate":
        signals.append(
            f"ML model classified case as {ml_label}"
        )

    # Decision policy.
    #
    # Legitimate with strong confidence remains approval-oriented.
    # Abuse classifications require meaningful confidence before
    # automatic rejection.
    if ml_label == "Legitimate" and ml_confidence >= 0.80:
        decision = "Auto-approve"

    elif ml_label == "Fraudulent Return" and ml_confidence >= 0.80:
        decision = "Auto-reject"

    elif ml_label == "Wardrobing" and ml_confidence >= 0.80:
        decision = "Manual review"

    elif ml_label == "Policy Abuser" and ml_confidence >= 0.80:
        decision = "Manual review"

    elif combined_risk >= 70:
        decision = "Human investigation"

    elif combined_risk >= 40:
        decision = "Manual review"

    else:
        decision = "Auto-approve"

    # Prevent obviously contradictory outputs.
    if decision == "Auto-reject" and combined_risk < 40:
        decision = "Manual review"

    if decision == "Auto-approve" and combined_risk >= 70:
        decision = "Human investigation"

    return {
        "risk_score": round(combined_risk, 2),
        "deterministic_risk": round(deterministic_risk, 2),
        "ml_risk": round(ml_risk, 2),
        "decision_confidence": round(decision_confidence, 2),
        "decision": decision,
        "signals": list(dict.fromkeys(signals)),
        "risk_components": {
            "deterministic_risk": round(deterministic_risk, 2),
            "ml_risk": round(ml_risk, 2),
            "ml_weight": 0.70,
            "deterministic_weight": 0.30,
        },
    }
