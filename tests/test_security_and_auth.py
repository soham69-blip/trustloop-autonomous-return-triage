"""
Tests for TrustLoop Security, Authentication, Authorization, and Input Protection.
"""

import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from fastapi import HTTPException

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.core.security import verify_api_key, verify_admin_key, sanitize_secure_path


class TestSecurityAndAuth(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.original_auth_enabled = settings.AUTH_ENABLED

    def tearDown(self):
        settings.AUTH_ENABLED = self.original_auth_enabled

    def test_public_endpoints_accessible_without_auth(self):
        settings.AUTH_ENABLED = True
        resp_health = self.client.get("/health")
        self.assertEqual(resp_health.status_code, 200)

        resp_ready = self.client.get("/ready")
        self.assertEqual(resp_ready.status_code, 200)

        resp_root = self.client.get("/")
        self.assertEqual(resp_root.status_code, 200)

    def test_admin_endpoints_require_admin_key_when_auth_enabled(self):
        settings.AUTH_ENABLED = True

        # Attempt rollback without header
        resp = self.client.post("/api/v1/models/rollback")
        self.assertEqual(resp.status_code, 403)
        self.assertIn("detail", resp.json())

        # Attempt rollback with invalid key
        resp_bad = self.client.post(
            "/api/v1/models/rollback",
            headers={"X-Admin-API-Key": "wrong-secret-key"}
        )
        self.assertEqual(resp_bad.status_code, 403)

        # Attempt toggle shadow mode without key
        resp_toggle = self.client.post("/api/v1/shadow/toggle?enabled=true")
        self.assertEqual(resp_toggle.status_code, 403)

    def test_request_size_limit_middleware(self):
        # Exceeds max payload size limit
        huge_headers = {"Content-Length": str(settings.MAX_REQUEST_SIZE_BYTES + 5000)}
        resp = self.client.post("/api/v1/analyze", headers=huge_headers, json={})
        self.assertEqual(resp.status_code, 413)
        data = resp.json()
        self.assertEqual(data.get("error"), "PAYLOAD_TOO_LARGE")
        self.assertIn("request_id", data)

    def test_path_traversal_sanitization(self):
        base_dir = settings.SNAPSHOTS_DIR
        safe_name = "snapshot_v1.0.csv"
        resolved = sanitize_secure_path(base_dir, safe_name)
        self.assertEqual(resolved, (base_dir / safe_name).resolve())

        # Malicious traversal attempt
        with self.assertRaises(HTTPException) as ctx:
            sanitize_secure_path(base_dir, "../../../etc/passwd")
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
