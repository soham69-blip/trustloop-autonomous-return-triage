from typing import Dict, Any, List, Tuple
import math


def calculate_probability_entropy(probabilities: Dict[str, float]) -> float:
    """
    Calculate normalized Shannon entropy of multi-class probabilities (0.0 = certain, 1.0 = maximum uncertainty).
    """
    probs = [p for p in probabilities.values() if p > 0.0]
    if len(probs) <= 1:
        return 0.0

    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(probabilities))
    if max_entropy <= 0.0:
        return 0.0

    return min(1.0, max(0.0, entropy / max_entropy))


def calculate_calibrated_confidence(
    probabilities: Dict[str, float],
    ml_label: str,
    deterministic_signals: List[str],
    policy_status: str,
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Calibrate decision confidence based on:
    1. Top probability magnitude.
    2. Probability margin: P(top1) - P(top2).
    3. Distribution entropy (uncertainty).
    4. Deterministic signal agreement.
    5. Policy evaluation clarity.

    Returns:
        (calibrated_confidence: float [0..100], uncertainty_level: str, metrics: Dict[str, Any])
    """
    if not probabilities:
        return 50.0, "HIGH", {"margin": 0.0, "entropy": 1.0}

    sorted_probs = sorted(probabilities.values(), reverse=True)
    top_p = sorted_probs[0]
    second_p = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = max(0.0, top_p - second_p)

    entropy = calculate_probability_entropy(probabilities)

    # Base statistical confidence derived from margin and peak probability
    raw_stat_conf = (top_p * 0.50) + (margin * 0.30) + ((1.0 - entropy) * 0.20)

    # Agreement modifier
    agreement_modifier = 0.0

    # If ML predicts Fraud/Abuse and deterministic signals align:
    has_risk_signals = len(deterministic_signals) > 0
    if ml_label in ("Fraudulent Return", "Policy Abuser", "Wardrobing"):
        if has_risk_signals:
            agreement_modifier += 0.05
        if policy_status in ("POLICY_VIOLATION", "HUMAN_ESCALATION"):
            agreement_modifier += 0.05
    elif ml_label == "Legitimate":
        if not has_risk_signals and policy_status == "POLICY_COMPLIANT":
            agreement_modifier += 0.05
        elif has_risk_signals:
            # Conflicting evidence reduces confidence
            agreement_modifier -= 0.10

    calibrated = min(100.0, max(10.0, (raw_stat_conf + agreement_modifier) * 100.0))

    if calibrated >= 85.0:
        uncertainty = "LOW"
    elif calibrated >= 65.0:
        uncertainty = "MEDIUM"
    else:
        uncertainty = "HIGH"

    metrics = {
        "top_probability": round(top_p, 4),
        "margin": round(margin, 4),
        "normalized_entropy": round(entropy, 4),
        "agreement_modifier": round(agreement_modifier, 4),
        "uncertainty_level": uncertainty,
    }

    return round(calibrated, 2), uncertainty, metrics
