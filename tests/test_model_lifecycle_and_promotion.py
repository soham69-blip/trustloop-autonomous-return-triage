"""
Tests for TrustLoop Model Lifecycle, Promotion Safety Gates, In-Memory Cache Invalidation, and Retraining.
"""

import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app, load_model, reset_model_cache
import backend.app.main as main_mod
from backend.app.services.model_registry_service import (
    promote_candidate_to_production,
    rollback_production_model,
    validate_model_artifact,
    reset_in_memory_model_cache,
    PROD_MODEL_PATH,
    CANDIDATE_MODEL_PATH,
)


class TestModelLifecycleAndPromotion(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_in_memory_cache_invalidation_on_rollback(self):
        # Load model into RAM
        model_before = load_model()
        self.assertIsNotNone(model_before)
        self.assertIsNotNone(main_mod.MODEL)

        # Trigger cache reset helper
        reset_in_memory_model_cache()
        self.assertIsNone(main_mod.MODEL)

    def test_candidate_promotion_rejected_on_schema_mismatch(self):
        # The current candidate model has 39 features, but production inference expects 33 features.
        # Safe promotion gate MUST reject this promotion attempt.
        if CANDIDATE_MODEL_PATH.exists():
            success, msg, details = promote_candidate_to_production(enforce_gates=True)
            self.assertFalse(success)
            self.assertIn("Candidate feature schema mismatch", msg)

    def test_api_promotion_endpoint_enforces_safety_and_rejects_incompatible_candidate(self):
        resp = self.client.post("/api/v1/models/promote")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Candidate feature schema mismatch", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
