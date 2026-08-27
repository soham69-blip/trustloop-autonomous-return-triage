"""Contract tests for the live multi-agent investigation pipeline."""
import unittest

from backend.app.services.agent_orchestrator import execute_autonomous_investigation
from backend.app.services.investigation_service import get_demo_case_by_id


class TestLiveInvestigationPipeline(unittest.TestCase):
    def investigate(self, case_id):
        case = get_demo_case_by_id(case_id)
        if case is None:
            raise AssertionError(f"Missing demo case: {case_id}")
        return execute_autonomous_investigation(case["payload"])

    def test_all_agents_and_contract(self):
        result = self.investigate("CASE-001")
        names = [a["name"] for a in result["agents"]]
        self.assertTrue(names[0] == "Intake Agent")
        self.assertTrue(any(n == "Evidence Agent" for n in names))
        self.assertTrue(any(n == "Policy Agent" for n in names))
        self.assertTrue(any(n == "Risk Scoring Agent" for n in names))
        self.assertTrue(any(n == "Responsibility Agent" for n in names))
        self.assertTrue(any(n == "Investigation Agent" for n in names))
        self.assertTrue(any(n == "Decision Agent" for n in names))
        self.assertEqual(sum(result["responsibility"][k] for k in ("customer", "seller", "courier", "unknown")), 100)
        self.assertIn(result["decision"]["decision"], {"AUTO_ACCEPT", "AUTO_RETURN", "HUMAN_ESCALATION"})
        self.assertIn("components", result["score_fusion"])
        self.assertIn("options", result["expected_loss"])

    def test_real_model_and_retrieval(self):
        result = self.investigate("CASE-001")
        self.assertEqual(result["risk_analysis"]["model_source"], "LightGBM")
        self.assertTrue(result["shap_explanation"]["available"])
        self.assertTrue(result["policy_analysis"]["retrieval_available"])
        self.assertGreater(len(result["policy_analysis"]["citations"]), 0)

    def test_case_switching_changes_outcome(self):
        approve = self.investigate("CASE-001")
        reject = self.investigate("CASE-002")
        escalate = self.investigate("CASE-003")
        self.assertEqual(approve["decision"]["decision"], "AUTO_ACCEPT")
        self.assertEqual(reject["decision"]["decision"], "AUTO_RETURN")
        self.assertEqual(escalate["decision"]["decision"], "HUMAN_ESCALATION")
        self.assertEqual(approve["responsibility_analysis"]["dominant_party"], "courier")
        self.assertEqual(reject["responsibility_analysis"]["dominant_party"], "customer")

    def test_missing_image_is_explicitly_neutral(self):
        case = get_demo_case_by_id("CASE-001")
        if case is None:
            raise AssertionError("Missing CASE-001")
        payload = dict(case["payload"])
        payload.pop("image_path", None)
        result = execute_autonomous_investigation(payload)
        evidence = result["vision_analysis"]
        self.assertFalse(evidence["vision_available"])
        self.assertEqual(evidence["vision_status"], "VISION UNAVAILABLE")
        self.assertIsNone(evidence["damage_detected"])

    def test_communication_and_second_pass_are_real(self):
        case = get_demo_case_by_id("CASE-001")
        if case is None:
            raise AssertionError("Missing CASE-001")
        payload = dict(case["payload"])
        payload.pop("weight_discrepancy_grams", None)
        result = execute_autonomous_investigation(payload)
        self.assertGreaterEqual(result["investigation"]["iterations"], 2)
        self.assertTrue(result["communications"])
        self.assertTrue(any(m["from_agent"] == "Risk Scoring Agent" and m["to_agent"] == "Policy Agent" for m in result["communications"]))
        self.assertTrue(any(a["iteration"] == 2 for a in result["agents"]))
        self.assertAlmostEqual(result["score_fusion"]["final_score"], sum(c["contribution"] for c in result["score_fusion"]["components"]), places=1)

    def test_graph_integrity_and_runtime_provenance(self):
        result = self.investigate("CASE-001")
        graph = result["evidence_graph"]
        node_ids = [node["id"] for node in graph["nodes"]]
        self.assertEqual(len(node_ids), len(set(node_ids)))
        self.assertTrue(all(edge["source"] in node_ids and edge["target"] in node_ids for edge in graph["edges"]))
        self.assertTrue(all(edge.get("label") for edge in graph["edges"]))
        self.assertTrue(any(node.get("type") == "agent" for node in graph["nodes"]))
        self.assertTrue(all(message["source_outputs"] for message in result["communications"]))

    def test_disagreement_changes_investigation_request(self):
        from backend.app.services.agent_orchestrator import run_investigation_agent
        output = run_investigation_agent(
            {"product_value_usd": 1200},
            {"vision_available": True, "damage_detected": True, "evidence_score": 80, "contradictions": []},
            {"fraud_probability": 80, "claim_verification_score": 20},
            {"customer": 10, "seller": 10, "courier": 70, "unknown": 10, "dominant_party": "courier"},
            {"policy_score": 50, "compliant": False},
        )
        self.assertTrue(output["conflicting_evidence"])
        self.assertTrue(output["investigate_further"])


if __name__ == "__main__":
    unittest.main()
