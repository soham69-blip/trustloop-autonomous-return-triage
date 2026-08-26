"""
Tests for TrustLoop Readiness Probe, Structured Errors, and Tracing Headers.
"""

import unittest
from fastapi.testclient import TestClient

from backend.app.main import app


class TestReadinessAndErrors(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_ready_probe_success(self):
        resp = self.client.get("/ready")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "READY")
        self.assertTrue(data["checks"]["production_model"]["ready"])
        self.assertTrue(data["checks"]["production_model"]["sha256_verified"])
        self.assertTrue(data["checks"]["categorical_mappings"]["ready"])
        self.assertTrue(data["checks"]["persistence"]["ready"])

    def test_tracing_headers_injected_on_response(self):
        resp = self.client.get("/api/v1/system")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("x-request-id", resp.headers)
        self.assertIn("x-response-time-ms", resp.headers)

    def test_custom_request_id_preserved(self):
        custom_id = "test-req-uuid-12345"
        resp = self.client.get("/api/v1/version", headers={"X-Request-ID": custom_id})
        self.assertEqual(resp.headers.get("x-request-id"), custom_id)

    def test_structured_error_on_missing_snapshot(self):
        resp = self.client.get("/api/v1/snapshots/nonexistent-snapshot-id-xyz")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("detail", resp.json())


if __name__ == "__main__":
    unittest.main()
