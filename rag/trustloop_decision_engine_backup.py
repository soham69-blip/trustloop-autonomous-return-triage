from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TrustLoopScores:
    customer_risk: float
    claim_validity: float
    policy_compliance: float
    decision_confidence: float


@dataclass
class TrustLoopDecision:
    decision: str
    reason: str
    scores: TrustLoopScores
    risk_factors: List[str]
    evidence_factors: List[str]


class TrustLoopDecisionEngine:

    CLASS_NAMES = {
        0: "Legitimate",
        1: "Policy Abuser",
        2: "Fraudulent Return",
        3: "Wardrobing",
    }

    def _clamp(self, value: float) -> float:
        return max(0.0, min(100.0, float(value)))

    def calculate_customer_risk(
        self,
        probabilities: Dict[str, float],
        multiple_accounts: bool = False,
        prior_returns: float = 0,
        returns_30d: float = 0,
        returns_90d: float = 0,
    ):
        policy = probabilities.get("Policy Abuser", 0.0)
        fraud = probabilities.get("Fraudulent Return", 0.0)
        wardrobe = probabilities.get("Wardrobing", 0.0)

        risk = (
            policy * 45
            + fraud * 30
            + wardrobe * 25
        )

        factors = []

        if policy >= 0.30:
            factors.append(
                f"Policy-abuse probability is {policy:.1%}"
            )

        if fraud >= 0.30:
            factors.append(
                f"Fraudulent-return probability is {fraud:.1%}"
            )

        if wardrobe >= 0.30:
            factors.append(
                f"Wardrobing probability is {wardrobe:.1%}"
            )

        if multiple_accounts:
            risk += 15
            factors.append(
                "Multiple-account activity detected"
            )

        if returns_30d > 0:
            risk += min(10, returns_30d * 2)
            factors.append(
                f"{returns_30d:g} return(s) in the last 30 days"
            )

        if returns_90d > 0:
            risk += min(5, returns_90d)
            factors.append(
                f"{returns_90d:g} return(s) in the last 90 days"
            )

        return self._clamp(risk), factors

    def calculate_claim_validity(
        self,
        vision_result: Optional[Dict] = None,
        reported_damage: bool = False,
    ):
        if not vision_result:
            return 50.0, [
                "No visual evidence provided"
            ]

        confidence = float(
            vision_result.get("confidence", 0.0)
        )

        evidence_consistent = bool(
            vision_result.get(
                "evidence_consistent",
                False
            )
        )

        damage_detected = bool(
            vision_result.get(
                "damage_detected",
                False
            )
        )

        image_quality = str(
            vision_result.get(
                "image_quality",
                "UNKNOWN"
            )
        ).upper()

        score = confidence * 100
        factors = []

        if evidence_consistent:
            score += 15
            factors.append(
                "Visual evidence is consistent with the claim"
            )
        else:
            score -= 20
            factors.append(
                "Visual evidence is inconsistent or inconclusive"
            )

        if reported_damage and damage_detected:
            score += 10
            factors.append(
                "Reported damage is visually supported"
            )

        if image_quality in {"POOR", "UNUSABLE"}:
            score -= 25
            factors.append(
                "Image quality limits evidence reliability"
            )

        return self._clamp(score), factors

    def calculate_policy_compliance(
        self,
        policy_status: str,
        policy_flags: Optional[List[str]] = None,
    ):
        status = str(
            policy_status or ""
        ).upper()

        flags = policy_flags or []

        if status == "POLICY_COMPLIANT":
            score = 100.0
        elif status == "HUMAN_ESCALATION":
            score = 55.0
        else:
            score = 25.0

        score -= min(40, len(flags) * 10)

        factors = []

        if status == "POLICY_COMPLIANT":
            factors.append(
                "No policy escalation triggered"
            )
        else:
            factors.append(
                f"Policy status: {status}"
            )

        factors.extend(flags)

        return self._clamp(score), factors

    def calculate_decision_confidence(
        self,
        customer_risk: float,
        claim_validity: float,
        policy_compliance: float,
        vision_confidence: Optional[float] = None,
    ):
        risk_certainty = (
            max(customer_risk, 100 - customer_risk)
        )

        components = [
            risk_certainty,
            claim_validity,
            policy_compliance,
        ]

        if vision_confidence is not None:
            components.append(
                vision_confidence * 100
            )

        return self._clamp(
            sum(components) / len(components)
        )

    def decide(
        self,
        probabilities: Dict[str, float],
        policy_status: str,
        policy_flags: Optional[List[str]] = None,
        vision_result: Optional[Dict] = None,
        multiple_accounts: bool = False,
        prior_returns: float = 0,
        returns_30d: float = 0,
        returns_90d: float = 0,
        reported_damage: bool = False,
    ):

        customer_risk, risk_factors = (
            self.calculate_customer_risk(
                probabilities,
                multiple_accounts,
                prior_returns,
                returns_30d,
                returns_90d,
            )
        )

        claim_validity, evidence_factors = (
            self.calculate_claim_validity(
                vision_result,
                reported_damage,
            )
        )

        policy_compliance, policy_factors = (
            self.calculate_policy_compliance(
                policy_status,
                policy_flags,
            )
        )

        vision_confidence = None

        if vision_result:
            vision_confidence = float(
                vision_result.get(
                    "confidence",
                    0.0
                )
            )

        decision_confidence = (
            self.calculate_decision_confidence(
                customer_risk,
                claim_validity,
                policy_compliance,
                vision_confidence,
            )
        )

        fraud_probability = probabilities.get(
            "Fraudulent Return",
            0.0
        )

        wardrobe_probability = probabilities.get(
            "Wardrobing",
            0.0
        )

        policy_probability = probabilities.get(
            "Policy Abuser",
            0.0
        )

        reasons = []

        # Hard escalation conditions
        if policy_status == "HUMAN_ESCALATION":
            decision = "HUMAN_ESCALATION"
            reasons.append(
                "Policy requires human review"
            )

        elif vision_result:

            image_quality = str(
                vision_result.get(
                    "image_quality",
                    ""
                )
            ).upper()

            evidence_consistent = bool(
                vision_result.get(
                    "evidence_consistent",
                    False
                )
            )

            if image_quality in {
                "POOR",
                "UNUSABLE"
            }:
                decision = "HUMAN_ESCALATION"
                reasons.append(
                    "Visual evidence quality is insufficient"
                )

            elif reported_damage and not evidence_consistent:
                decision = "HUMAN_ESCALATION"
                reasons.append(
                    "Damage claim is not supported by visual evidence"
                )

            elif decision_confidence < 70:
                decision = "HUMAN_ESCALATION"
                reasons.append(
                    "Decision confidence is below the automated threshold"
                )

            elif fraud_probability >= 0.50:
                decision = "AUTO_REJECT"
                reasons.append(
                    "Strong fraudulent-return signal detected"
                )

            elif wardrobe_probability >= 0.50:
                decision = "AUTO_REJECT"
                reasons.append(
                    "Strong wardrobing signal detected"
                )

            elif (
                policy_probability >= 0.30
                and customer_risk >= 55
            ):
                decision = "HUMAN_ESCALATION"
                reasons.append(
                    "Elevated policy-abuse risk requires review"
                )

            elif (
                claim_validity >= 75
                and policy_compliance >= 80
                and customer_risk < 35
            ):
                decision = "AUTO_APPROVE"
                reasons.append(
                    "Claim evidence is strong and risk is low"
                )

            else:
                decision = "HUMAN_ESCALATION"
                reasons.append(
                    "Evidence and risk signals require additional review"
                )

        else:

            if fraud_probability >= 0.50:
                decision = "AUTO_REJECT"
                reasons.append(
                    "Strong fraudulent-return signal detected"
                )

            elif wardrobe_probability >= 0.50:
                decision = "AUTO_REJECT"
                reasons.append(
                    "Strong wardrobing signal detected"
                )

            elif (
                policy_probability >= 0.30
                and customer_risk >= 55
            ):
                decision = "HUMAN_ESCALATION"
                reasons.append(
                    "Elevated policy-abuse risk requires review"
                )

            elif (
                customer_risk < 35
                and policy_compliance >= 80
            ):
                decision = "AUTO_APPROVE"
                reasons.append(
                    "Low customer risk and policy compliant"
                )

            else:
                decision = "HUMAN_ESCALATION"
                reasons.append(
                    "Risk signals require additional review"
                )

        scores = TrustLoopScores(
            customer_risk=round(
                customer_risk,
                2
            ),
            claim_validity=round(
                claim_validity,
                2
            ),
            policy_compliance=round(
                policy_compliance,
                2
            ),
            decision_confidence=round(
                decision_confidence,
                2
            ),
        )

        all_factors = (
            risk_factors
            + evidence_factors
            + policy_factors
        )

        return TrustLoopDecision(
            decision=decision,
            reason="; ".join(reasons),
            scores=scores,
            risk_factors=risk_factors,
            evidence_factors=all_factors,
        )


if __name__ == "__main__":

    print("=" * 70)
    print("TRUSTLOOP — DECISION ENGINE TEST")
    print("=" * 70)

    engine = TrustLoopDecisionEngine()

    probabilities = {
        "Legitimate": 0.8917,
        "Policy Abuser": 0.1075,
        "Fraudulent Return": 0.0005,
        "Wardrobing": 0.0003,
    }

    vision = {
        "image_quality": "GOOD",
        "product_condition": "DAMAGED",
        "damage_detected": True,
        "packaging_condition": "DAMAGED",
        "evidence_consistent": True,
        "confidence": 0.72,
    }

    result = engine.decide(
        probabilities=probabilities,
        policy_status="POLICY_COMPLIANT",
        policy_flags=[],
        vision_result=vision,
        reported_damage=True,
    )

    print()
    print("CUSTOMER RISK:")
    print(
        f"  {result.scores.customer_risk:.2f}%"
    )

    print()
    print("CLAIM VALIDITY:")
    print(
        f"  {result.scores.claim_validity:.2f}%"
    )

    print()
    print("POLICY COMPLIANCE:")
    print(
        f"  {result.scores.policy_compliance:.2f}%"
    )

    print()
    print("DECISION CONFIDENCE:")
    print(
        f"  {result.scores.decision_confidence:.2f}%"
    )

    print()
    print("FINAL DECISION:")
    print(
        f"  {result.decision}"
    )

    print()
    print("REASON:")
    print(
        f"  {result.reason}"
    )

    print()
    print("RISK FACTORS:")

    if result.risk_factors:
        for item in result.risk_factors:
            print(f"  - {item}")
    else:
        print("  None")

    print()
    print("EVIDENCE / POLICY FACTORS:")

    if result.evidence_factors:
        for item in result.evidence_factors:
            print(f"  - {item}")
    else:
        print("  None")

    print()
    print("=" * 70)
    print("DECISION ENGINE TEST COMPLETED")
    print("=" * 70)
