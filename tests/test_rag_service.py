import unittest
from unittest.mock import patch

from backend.app.services.rag_service import analyze_policy


class TestRAGService(unittest.TestCase):

    def setUp(self):
        self.sample_case = {
            "case_id": "CASE-RAG-001",
            "days_to_return": 10,
            "product_category": "electronics",
            "return_reason": "changed_mind",
            "multiple_accounts_flag": 0,
            "is_high_value_item": 0,
        }

    def test_rag_service_standard_compliant_case(self):
        result = analyze_policy(self.sample_case, top_k=3)
        self.assertTrue(result["available"])
        self.assertEqual(result["policy_status"], "POLICY_COMPLIANT")
        self.assertIsInstance(result["flags"], list)
        self.assertIsInstance(result["retrieved_policy"], list)
        self.assertTrue(len(result["retrieved_policy"]) > 0)

    def test_rag_service_escalation_case_late_return(self):
        late_case = dict(self.sample_case)
        late_case["days_to_return"] = 45
        result = analyze_policy(late_case, top_k=3)
        self.assertTrue(result["available"])
        self.assertEqual(result["policy_status"], "HUMAN_ESCALATION")
        self.assertTrue(any(f["rule"] == "return_window" for f in result["flags"]))

    def test_rag_service_escalation_case_multiple_accounts(self):
        multi_case = dict(self.sample_case)
        multi_case["multiple_accounts_flag"] = 1
        result = analyze_policy(multi_case, top_k=3)
        self.assertTrue(result["available"])
        self.assertEqual(result["policy_status"], "HUMAN_ESCALATION")
        self.assertTrue(any(f["rule"] == "multiple_accounts" for f in result["flags"]))

    def test_rag_service_handles_missing_index_gracefully(self):
        with patch("rag.policy_agent.evaluate_policy", side_effect=FileNotFoundError("Mock index missing")):
            result = analyze_policy(self.sample_case)
            self.assertFalse(result["available"])
            # Principle: missing policy index != policy violation
            self.assertEqual(result["policy_status"], "POLICY_COMPLIANT")
            self.assertIn("unavailable", result["reason"])

    def test_rag_service_handles_unexpected_exception_gracefully(self):
        with patch("rag.policy_agent.evaluate_policy", side_effect=RuntimeError("Unexpected error")):
            result = analyze_policy(self.sample_case)
            self.assertFalse(result["available"])
            self.assertEqual(result["policy_status"], "POLICY_COMPLIANT")


if __name__ == "__main__":
    unittest.main()
