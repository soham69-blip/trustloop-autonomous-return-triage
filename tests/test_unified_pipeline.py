import json
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from backend.app.main import app


class TestUnifiedPipeline(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        with open("tests/payload_legitimate.json") as f:
            self.legit_payload = json.load(f)
        with open("tests/payload_policy_abuser.json") as f:
            self.policy_payload = json.load(f)
        with open("tests/payload_fraudulent.json") as f:
            self.fraud_payload = json.load(f)
        with open("tests/payload_wardrobing.json") as f:
            self.ward_payload = json.load(f)

    # 1. Legitimate Case
    def test_pipeline_legitimate_case(self):
        response = self.client.post("/api/v1/analyze", json=self.legit_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["case_id"], self.legit_payload["case_id"])
        self.assertEqual(data["decision"], "Auto-approve")
        self.assertIn("policy_analysis", data)
        self.assertIn("vision_analysis", data)
        self.assertIn("shap_explanation", data)
        self.assertTrue(data["shap_explanation"]["available"])
        self.assertEqual(data["shap_explanation"]["predicted_class"], "Legitimate")

    # 2. Policy Abuser Case
    def test_pipeline_policy_abuser_case(self):
        payload = dict(self.policy_payload)
        payload.update({
            "total_orders_lifetime": 20,
            "total_returns_lifetime": 8,
            "return_rate_pct": 40.0,
            "customer_support_contacts": 2,
            "previous_dispute_count": 1,
            "refund_amount_requested_usd": 45.0,
        })
        response = self.client.post("/api/v1/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["decision"], ["Manual review", "Human investigation"])
        self.assertTrue(data["feature_contract"]["contract_valid"])
        self.assertTrue(data["shap_explanation"]["available"])

    # 3. Fraud Case
    def test_pipeline_fraud_case(self):
        response = self.client.post("/api/v1/analyze", json=self.fraud_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["decision"], "Auto-reject")
        self.assertTrue(data["risk_score"] >= 70)
        self.assertEqual(data["ml_label"], "Fraudulent Return")

    # 4. Wardrobing Payload Case
    def test_pipeline_wardrobing_case(self):
        response = self.client.post("/api/v1/analyze", json=self.ward_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["decision"], ["Auto-approve", "Manual review", "Human investigation"])
        self.assertIn(data["ml_label"], ["Legitimate", "Wardrobing", "Policy Abuser"])
        self.assertTrue(data["shap_explanation"]["available"])


    # 5. Late Return Case (Policy RAG escalation)
    def test_pipeline_late_return_policy_escalation(self):
        late_payload = dict(self.legit_payload)
        late_payload["days_to_return"] = 45.0
        response = self.client.post("/api/v1/analyze", json=late_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["policy_analysis"]["available"])
        self.assertEqual(data["policy_analysis"]["policy_status"], "HUMAN_ESCALATION")
        self.assertTrue(any("Policy flag [return_window]" in s for s in data["signals"]))

    # 6. Multiple Accounts Case
    def test_pipeline_multiple_accounts_signal(self):
        multi_payload = dict(self.legit_payload)
        multi_payload["multiple_accounts_flag"] = 1
        response = self.client.post("/api/v1/analyze", json=multi_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(any("Multiple-account activity" in s for s in data["signals"]))
        self.assertEqual(data["policy_analysis"]["policy_status"], "HUMAN_ESCALATION")

    # 7. High Value Item Case
    def test_pipeline_high_value_case(self):
        hv_payload = dict(self.legit_payload)
        hv_payload["is_high_value_item"] = 1
        response = self.client.post("/api/v1/analyze", json=hv_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(any("High-value item" in s for s in data["signals"]))

    # 8. Case Without Vision Evidence (Missing Vision is NOT negative evidence)
    def test_pipeline_missing_vision_is_neutral(self):
        no_vision_payload = dict(self.legit_payload)
        no_vision_payload["image_path"] = None
        response = self.client.post("/api/v1/analyze", json=no_vision_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["vision_analysis"]["available"])
        self.assertFalse(data["vision_analysis"]["verified"])
        self.assertEqual(data["vision_analysis"]["reason"], "NO_IMAGE_EVIDENCE")
        # Should remain approved since missing vision != rejection
        self.assertEqual(data["decision"], "Auto-approve")

    # 9. Graceful SHAP Failure Recovery
    def test_pipeline_graceful_shap_failure(self):
        with patch("backend.app.main.explain_prediction", return_value={"available": False, "explanation_summary": "SHAP engine offline"}):
            response = self.client.post("/api/v1/analyze", json=self.legit_payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["decision"], "Auto-approve")
            self.assertFalse(data["shap_explanation"]["available"])

    # 10. Graceful RAG Failure Recovery
    def test_pipeline_graceful_rag_failure(self):
        with patch("backend.app.main.analyze_policy", return_value={"available": False, "policy_status": "POLICY_COMPLIANT", "flags": [], "retrieved_policy": []}):
            response = self.client.post("/api/v1/analyze", json=self.legit_payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["decision"], "Auto-approve")
            self.assertFalse(data["policy_analysis"]["available"])

    # 11. Backward Compatibility with Phase 4 Payloads
    def test_pipeline_backward_compatibility_contract(self):
        response = self.client.post("/api/v1/analyze", json=self.legit_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        # Ensure every single Phase 4 field exists
        phase4_fields = [
            "case_id", "risk_score", "deterministic_risk", "ml_risk",
            "decision_confidence", "decision", "signals", "ml_prediction",
            "ml_label", "ml_confidence", "class_probabilities",
            "risk_components", "feature_contract", "model_status", "status"
        ]
        for field in phase4_fields:
            self.assertIn(field, data, f"Missing required Phase 4 field: {field}")


if __name__ == "__main__":
    unittest.main()
