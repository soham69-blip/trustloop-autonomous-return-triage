import json
import pickle
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from backend.app.main import app, load_model, get_model_status, LABELS
from backend.app.ml_feature_builder import (
    MODEL_FEATURES,
    CANDIDATE_MODEL_FEATURES,
    build_model_dataframe,
    build_model_features,
    validate_feature_contract,
)


class TestProductionHardening(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        with open("tests/payload_legitimate.json") as f:
            self.legit_payload = json.load(f)
        with open("tests/payload_policy_abuser.json") as f:
            self.policy_payload = json.load(f)

    # 1. API valid production payload
    def test_api_valid_production_payload(self):
        response = self.client.post("/api/v1/analyze", json=self.legit_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["case_id"], self.legit_payload["case_id"])
        self.assertIn("risk_score", data)
        self.assertIn("decision", data)
        self.assertIn("feature_contract", data)
        self.assertEqual(data["feature_contract"]["target_feature_set"], "production")
        self.assertTrue(data["feature_contract"]["contract_valid"])
        self.assertEqual(data["model_status"]["feature_count"], 33)
        self.assertEqual(data["model_status"]["model_role"], "production")

    # 2. API valid candidate-complete payload
    def test_api_valid_candidate_complete_payload(self):
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
        self.assertEqual(data["decision"], "Manual review")
        self.assertEqual(data["ml_label"], "Policy Abuser")

    # 3. Missing optional profile/numeric fields default safely
    def test_api_missing_optional_fields_defaults_safely(self):
        minimal_payload = {
            "case_id": "TEST-MIN-001",
            "age": 30,
            "account_age_days": 100,
            "customer_segment": "Silver",
            "country": "US",
            "platform": "Mobile App",
            "device_type": "iPhone",
            "payment_method": "Credit Card",
            "product_category": "Clothing",
            "return_reason": "Changed Mind",
            "shipping_carrier": "UPS",
            "avg_order_value_usd": 65.0,
            "order_date": "2026-05-01",
            "return_date": "2026-05-10",
            # All optional numeric/profile fields (returns history, disputes, support contacts, multiple accounts) omitted
        }
        response = self.client.post("/api/v1/analyze", json=minimal_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["case_id"], "TEST-MIN-001")
        self.assertEqual(data["ml_prediction"], 0) # Legitimate first-time returner


    # 4. Missing required field (case_id) fails validation (422)
    def test_api_missing_required_field_fails(self):
        invalid_payload = dict(self.legit_payload)
        del invalid_payload["case_id"]
        response = self.client.post("/api/v1/analyze", json=invalid_payload)
        self.assertEqual(response.status_code, 422)

    # 5. Malformed numeric field returns 422
    def test_api_malformed_numeric_field(self):
        malformed = dict(self.legit_payload)
        malformed["age"] = "not-a-number"
        response = self.client.post("/api/v1/analyze", json=malformed)
        self.assertEqual(response.status_code, 422)

    # 6. Invalid unseen categorical string returns 400
    def test_api_invalid_unseen_category(self):
        invalid_cat = dict(self.legit_payload)
        invalid_cat["country"] = "NON_EXISTENT_COUNTRY_XYZ"
        response = self.client.post("/api/v1/analyze", json=invalid_cat)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unseen category value", response.json()["detail"])

    # 7. Contradictory return-rate data detected in audit
    def test_contradictory_return_rate_audit(self):
        payload = dict(self.policy_payload)
        payload.update({
            "total_orders_lifetime": 20,
            "total_returns_lifetime": 7,
            "return_rate_pct": 10.0, # Math is 35.0%
        })
        contract = validate_feature_contract(payload, feature_names=CANDIDATE_MODEL_FEATURES)
        self.assertGreater(len(contract["inconsistencies"]), 0)
        self.assertIn("CONTRADICTORY_RETURN_RATE", contract["inconsistencies"][0])

    # 8. Missing customer profile history flagged in candidate contract
    def test_missing_customer_profile_history_flagged(self):
        contract = validate_feature_contract(self.policy_payload, feature_names=CANDIDATE_MODEL_FEATURES)
        self.assertFalse(contract["valid"])
        self.assertIn("total_orders_lifetime", contract["missing_candidate_fields"])
        self.assertIn("customer_support_contacts", contract["missing_candidate_fields"])

    # 9. Unknown feature in payload ignored safely
    def test_unknown_feature_ignored_safely(self):
        payload = dict(self.legit_payload)
        payload["completely_unknown_custom_field"] = "arbitrary_value_123"
        response = self.client.post("/api/v1/analyze", json=payload)
        self.assertEqual(response.status_code, 200)

    # 10. Empty request fails with 422
    def test_empty_request_fails(self):
        response = self.client.post("/api/v1/analyze", json={})
        self.assertEqual(response.status_code, 422)

    # 11. Safe model metadata endpoint
    def test_model_metadata_endpoint(self):
        response = self.client.get("/api/v1/model/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["model_name"], "lightgbm_model.pkl")
        self.assertEqual(data["model_role"], "production")
        self.assertEqual(data["feature_count"], 33)
        self.assertEqual(len(data["features"]), 33)
        self.assertEqual(len(data["classes"]), 4)
        # Verify no sensitive filesystem path is exposed
        self.assertNotIn("C:\\", json.dumps(data))
        self.assertNotIn("/home/", json.dumps(data))

    # 12. Feature ordering and count safety check
    def test_feature_ordering_safety_check(self):
        with open("models/lightgbm_model.pkl", "rb") as f:
            base_model = pickle.load(f)
        expected_cols = list(base_model.feature_name_)
        df = build_model_dataframe(self.legit_payload, feature_names=expected_cols)
        self.assertEqual(list(df.columns), expected_cols)
        self.assertEqual(len(df.columns), len(expected_cols))

    # 13. Production and Candidate isolation
    def test_production_candidate_isolation(self):
        with open("models/lightgbm_model.pkl", "rb") as f:
            prod = pickle.load(f)
        with open("models/lightgbm_candidate.pkl", "rb") as f:
            cand = pickle.load(f)

        self.assertEqual(len(prod.feature_name_), 33)
        self.assertEqual(len(cand.feature_name_), 39)
        self.assertNotIn("return_rate_pct", prod.feature_name_)
        self.assertIn("return_rate_pct", cand.feature_name_)


if __name__ == "__main__":
    unittest.main()
