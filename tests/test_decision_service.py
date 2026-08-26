import unittest

from backend.app.services.decision_service import evaluate_decision


class TestDecisionService(unittest.TestCase):

    def setUp(self):
        self.base_case = {
            "case_id": "CASE-DEC-001",
            "age": 30,
            "days_to_return": 5.0,
            "customer_return_count_prior": 0,
            "returns_last_30d_prior": 0,
            "returns_last_90d_prior": 0,
            "total_returns_lifetime_prior": 0,
            "multiple_accounts_flag": 0,
            "is_high_value_item": 0,
            "item_condition": "New",
        }

    def test_legitimate_case_auto_approve(self):
        probs = {"Legitimate": 0.96, "Policy Abuser": 0.03, "Fraudulent Return": 0.005, "Wardrobing": 0.005}
        result = evaluate_decision(
            case_data=self.base_case,
            probabilities=probs,
            ml_label="Legitimate",
        )
        self.assertEqual(result["decision"], "Auto-approve")
        self.assertTrue(result["risk_score"] < 40)
        self.assertIn("deterministic_risk", result["risk_components"])
        self.assertIn("ml_risk", result["risk_components"])

    def test_fraud_signal_auto_reject(self):
        fraud_case = dict(self.base_case)
        fraud_case["multiple_accounts_flag"] = 1
        fraud_case["customer_return_count_prior"] = 9
        probs = {"Legitimate": 0.01, "Policy Abuser": 0.05, "Fraudulent Return": 0.92, "Wardrobing": 0.02}
        result = evaluate_decision(
            case_data=fraud_case,
            probabilities=probs,
            ml_label="Fraudulent Return",
        )
        self.assertEqual(result["decision"], "Auto-reject")
        self.assertTrue(result["risk_score"] >= 70)
        self.assertTrue(any("ML model classified case as Fraudulent Return" in s for s in result["signals"]))

    def test_wardrobing_manual_review(self):
        ward_case = dict(self.base_case)
        ward_case["item_condition"] = "worn"
        probs = {"Legitimate": 0.02, "Policy Abuser": 0.03, "Fraudulent Return": 0.05, "Wardrobing": 0.90}
        result = evaluate_decision(
            case_data=ward_case,
            probabilities=probs,
            ml_label="Wardrobing",
        )
        self.assertEqual(result["decision"], "Manual review")
        self.assertTrue(any("Wardrobing" in s for s in result["signals"]))

    def test_policy_abuser_manual_review(self):
        abuser_case = dict(self.base_case)
        abuser_case["customer_return_count_prior"] = 6
        probs = {"Legitimate": 0.05, "Policy Abuser": 0.88, "Fraudulent Return": 0.04, "Wardrobing": 0.03}
        result = evaluate_decision(
            case_data=abuser_case,
            probabilities=probs,
            ml_label="Policy Abuser",
        )
        self.assertEqual(result["decision"], "Manual review")

    def test_policy_escalation_signal_enrichment(self):
        probs = {"Legitimate": 0.90, "Policy Abuser": 0.08, "Fraudulent Return": 0.01, "Wardrobing": 0.01}
        policy_result = {
            "available": True,
            "policy_status": "HUMAN_ESCALATION",
            "flags": [{"rule": "return_window", "status": "REVIEW_REQUIRED", "reason": "Exceeded 30 days"}],
        }
        result = evaluate_decision(
            case_data=self.base_case,
            probabilities=probs,
            ml_label="Legitimate",
            policy_result=policy_result,
        )
        self.assertTrue(any("Policy flag [return_window]" in s for s in result["signals"]))

    def test_vision_evidence_signal_enrichment(self):
        damage_case = dict(self.base_case)
        damage_case["item_condition"] = "damaged"
        probs = {"Legitimate": 0.85, "Policy Abuser": 0.10, "Fraudulent Return": 0.03, "Wardrobing": 0.02}
        vision_result = {
            "available": True,
            "verified": True,
            "damage_detected": True,
            "evidence_consistent": True,
        }
        result = evaluate_decision(
            case_data=damage_case,
            probabilities=probs,
            ml_label="Legitimate",
            vision_result=vision_result,
        )
        self.assertTrue(any("Visual evidence confirms physical damage" in s for s in result["signals"]))
        self.assertTrue(any("Visual evidence is consistent" in s for s in result["signals"]))

    def test_multiple_accounts_signal(self):
        multi_case = dict(self.base_case)
        multi_case["multiple_accounts_flag"] = 1
        probs = {"Legitimate": 0.75, "Policy Abuser": 0.20, "Fraudulent Return": 0.03, "Wardrobing": 0.02}
        result = evaluate_decision(
            case_data=multi_case,
            probabilities=probs,
            ml_label="Legitimate",
        )
        self.assertTrue(any("Multiple-account activity" in s for s in result["signals"]))

    def test_high_value_item_signal(self):
        hv_case = dict(self.base_case)
        hv_case["is_high_value_item"] = 1
        probs = {"Legitimate": 0.95, "Policy Abuser": 0.03, "Fraudulent Return": 0.01, "Wardrobing": 0.01}
        result = evaluate_decision(
            case_data=hv_case,
            probabilities=probs,
            ml_label="Legitimate",
        )
        self.assertTrue(any("High-value item" in s for s in result["signals"]))


if __name__ == "__main__":
    unittest.main()
