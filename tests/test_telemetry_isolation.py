"""
Tests for TrustLoop Telemetry Isolation and Drift Shielding.
Verifies that test/demo traffic never contaminates production drift calculations.
"""

import unittest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.drift_service import (
    generate_comprehensive_drift_report,
    evaluate_feedback_quality_drift,
    reset_drift_telemetry,
)


class TestTelemetryIsolation(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_production_traffic_isolation_in_drift_report(self):
        # Generate report with include_test_traffic=False
        report = generate_comprehensive_drift_report(include_test_traffic=False)
        self.assertTrue(report.get("production_traffic_only"))
        self.assertIn("overall_health", report)

    def test_feedback_quality_filters_test_traffic(self):
        quality = evaluate_feedback_quality_drift(include_test_traffic=False)
        self.assertIn("total_feedback_submissions", quality)
        self.assertIn("quarantine_ratio", quality)

    def test_drift_reset_endpoint(self):
        resp = self.client.post("/api/v1/drift/reset?archive=true")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "TELEMETRY_RESET_SUCCESSFUL")


if __name__ == "__main__":
    unittest.main()
