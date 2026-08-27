"""
Tests for TrustLoop Case Room, End-to-End Investigation, Responsibility Attribution,
Counterfactual Challenge Decision, and Fraud Network Graph.
"""

import unittest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.responsibility_service import calculate_responsibility
from backend.app.services.investigation_service import reconstruct_timeline, DEMO_CASES


class TestInvestigationAndCaseRoom(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_demo_cases_list(self):
        resp = self.client.get("/api/v1/cases/demo")
        self.assertEqual(resp.status_code, 200)
        cases = resp.json()
        self.assertEqual(len(cases), 6)
        case_ids = [c["case_id"] for c in cases]
        self.assertIn("CASE-001", case_ids)
        self.assertIn("CASE-002", case_ids)
        self.assertIn("CASE-003", case_ids)
        self.assertIn("CASE-004", case_ids)
        self.assertIn("CASE-005", case_ids)
        self.assertIn("CASE-006", case_ids)

    def test_case_detail_and_timeline(self):
        resp = self.client.get("/api/v1/cases/CASE-001")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("case", data)
        self.assertIn("timeline", data)
        self.assertIn("responsibility", data)
        self.assertIn("evidence_graph", data)

        # Invariant: responsibility must sum to 100
        resp_dict = data["responsibility"]
        total = sum(resp_dict.values())
        self.assertEqual(total, 100)
        self.assertEqual(data["dominant_party"], "courier")

        # Invariant: timeline must have 8 chronological stages
        timeline = data["timeline"]
        self.assertEqual(len(timeline), 8)
        self.assertEqual(timeline[0]["stage"], "Seller Packed Item")
        self.assertEqual(timeline[-1]["stage"], "Decision Recommendation")

    def test_investigate_endpoint(self):
        demo_payload = DEMO_CASES[0]["payload"]
        resp = self.client.post("/api/v1/investigate", json=demo_payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("case_id"), "CASE-001")
        self.assertEqual(data.get("dominant_party"), "courier")
        self.assertEqual(data.get("recommended_action"), "AUTO_ACCEPT")

        resp_dict = data["responsibility"]
        self.assertEqual(sum(resp_dict.values()), 100)

    def test_counterfactual_challenge_decision(self):
        demo_payload = DEMO_CASES[0]["payload"]
        # Challenge by disabling courier incident history & packaging damage
        challenge_req = {
            "case_payload": demo_payload,
            "disabled_signals": ["courier_incident_history", "packaging_damage"],
        }
        resp = self.client.post("/api/v1/challenge", json=challenge_req)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("baseline", data)
        self.assertIn("counterfactual", data)
        self.assertIn("deltas", data)
        self.assertIn("explanation", data)

        # Baseline courier responsibility > Counterfactual courier responsibility
        base_cour = data["baseline"]["responsibility"]["courier"]
        cf_cour = data["counterfactual"]["responsibility"]["courier"]
        self.assertGreater(base_cour, cf_cour)
        self.assertLess(data["deltas"]["courier"]["delta"], 0)

    def test_fraud_network_graph_endpoint(self):
        resp = self.client.get("/api/v1/network/graph")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIn("active_clusters", data)
        self.assertGreater(len(data["nodes"]), 5)
        self.assertGreater(len(data["edges"]), 5)

    def test_feedback_history_endpoint(self):
        resp = self.client.get("/api/v1/feedback/history?limit=10")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)


if __name__ == "__main__":
    unittest.main()
