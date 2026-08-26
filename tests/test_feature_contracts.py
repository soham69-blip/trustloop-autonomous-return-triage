import json
import pickle
import unittest
from backend.app.ml_feature_builder import (
    MODEL_FEATURES,
    CANDIDATE_MODEL_FEATURES,
    build_model_features,
    build_model_dataframe,
    validate_feature_contract,
)


class TestFeatureContracts(unittest.TestCase):

    def setUp(self):
        with open("models/lightgbm_model.pkl", "rb") as f:
            self.base_model = pickle.load(f)
        with open("models/lightgbm_candidate.pkl", "rb") as f:
            self.candidate_model = pickle.load(f)

    # 1. Complete 39-feature payload
    def test_complete_39_feature_payload(self):
        """Complete candidate payload with all 6 features satisfies candidate contract."""
        with open("tests/payload_policy_abuser.json") as f:
            p = json.load(f)

        p.update({
            "total_orders_lifetime": 12,
            "total_returns_lifetime": 7,
            "return_rate_pct": 58.33,
            "customer_support_contacts": 3,
            "previous_dispute_count": 2,
            "refund_amount_requested_usd": 45.0,
        })

        contract = validate_feature_contract(p, feature_names=CANDIDATE_MODEL_FEATURES)
        self.assertTrue(contract["valid"])
        self.assertEqual(len(contract["inconsistencies"]), 0)
        self.assertEqual(len(contract["missing_candidate_fields"]), 0)

        df = build_model_dataframe(p, feature_names=CANDIDATE_MODEL_FEATURES)
        self.assertEqual(len(df.columns), 39)
        self.assertEqual(df["return_rate_pct"].values[0], 58.33)

    # 2. Legacy 33-feature payload
    def test_legacy_33_feature_payload(self):
        """Legacy payload must be 100% valid for production 33-feature model."""
        with open("tests/payload_policy_abuser.json") as f:
            p = json.load(f)

        contract = validate_feature_contract(p, feature_names=MODEL_FEATURES)
        self.assertTrue(contract["valid"])
        self.assertEqual(contract["target_feature_set"], "production")
        self.assertEqual(len(contract["inconsistencies"]), 0)

        df = build_model_dataframe(p, feature_names=MODEL_FEATURES)
        self.assertEqual(len(df.columns), 33)

    # 3. Policy-abuser payload with missing denominator
    def test_policy_abuser_missing_denominator(self):
        """When legacy payload has returns > 0 but orders is missing, candidate contract flags it."""
        with open("tests/payload_policy_abuser.json") as f:
            p = json.load(f)

        contract = validate_feature_contract(p, feature_names=CANDIDATE_MODEL_FEATURES)
        self.assertFalse(contract["valid"])
        self.assertGreater(len(contract["inconsistencies"]), 0)
        self.assertIn("INCONSISTENT_DENOMINATOR", contract["inconsistencies"][0])

        # Feature builder should not silently set return_rate_pct = 0.0
        features = build_model_features(p, feature_names=CANDIDATE_MODEL_FEATURES)
        self.assertEqual(features["total_returns_lifetime"], 7)
        self.assertGreater(features["return_rate_pct"], 0.0)

    # 4. Missing customer_support_contacts
    def test_missing_customer_support_contacts(self):
        """Missing customer_support_contacts is flagged in missing candidate fields."""
        with open("tests/payload_policy_abuser.json") as f:
            p = json.load(f)

        contract = validate_feature_contract(p, feature_names=CANDIDATE_MODEL_FEATURES)
        self.assertIn("customer_support_contacts", contract["missing_candidate_fields"])

    # 5. Missing previous_dispute_count
    def test_missing_previous_dispute_count(self):
        """Missing previous_dispute_count is flagged in missing candidate fields."""
        with open("tests/payload_policy_abuser.json") as f:
            p = json.load(f)

        contract = validate_feature_contract(p, feature_names=CANDIDATE_MODEL_FEATURES)
        self.assertIn("previous_dispute_count", contract["missing_candidate_fields"])

    # 6. Missing total_orders_lifetime
    def test_missing_total_orders_lifetime(self):
        """Missing total_orders_lifetime is flagged when returns > 0."""
        with open("tests/payload_policy_abuser.json") as f:
            p = json.load(f)

        contract = validate_feature_contract(p, feature_names=CANDIDATE_MODEL_FEATURES)
        self.assertIn("total_orders_lifetime", contract["missing_candidate_fields"])

    # 7. Missing refund_amount_requested_usd with refund_amount available
    def test_missing_refund_amount_requested_usd_fallback(self):
        """refund_amount_requested_usd safely falls back to refund_amount."""
        with open("tests/payload_policy_abuser.json") as f:
            p = json.load(f)

        self.assertNotIn("refund_amount_requested_usd", p)
        self.assertIn("refund_amount", p)

        features = build_model_features(p, feature_names=CANDIDATE_MODEL_FEATURES)
        self.assertEqual(features["refund_amount_requested_usd"], 45.0)

    # 8. Contradictory return rate
    def test_contradictory_return_rate(self):
        """Strict contract raises violation on contradictory return rate (returns > 0 with missing orders)."""
        with open("tests/payload_policy_abuser.json") as f:
            p = json.load(f)

        with self.assertRaises(ValueError) as ctx:
            build_model_dataframe(p, feature_names=CANDIDATE_MODEL_FEATURES, validate_contract=True)
        self.assertIn("Feature contract violation", str(ctx.exception))

    # 9. Explicit return rate provided
    def test_explicit_return_rate(self):
        """Explicit return_rate_pct is respected directly."""
        with open("tests/payload_policy_abuser.json") as f:
            p = json.load(f)

        p["return_rate_pct"] = 42.5
        p["total_orders_lifetime"] = 20
        p["total_returns_lifetime"] = 8

        features = build_model_features(p, feature_names=CANDIDATE_MODEL_FEATURES)
        self.assertEqual(features["return_rate_pct"], 42.5)

    # 10. First-time customer with zero returns
    def test_first_time_customer_zero_returns(self):
        """First time returner with 0 prior returns has honestly 0.0% return rate."""
        with open("tests/payload_legitimate.json") as f:
            p = json.load(f)

        features = build_model_features(p, feature_names=CANDIDATE_MODEL_FEATURES)
        self.assertEqual(features["total_returns_lifetime"], 0)
        self.assertEqual(features["return_rate_pct"], 0.0)

    # 11. Candidate model feature ordering
    def test_candidate_model_feature_ordering(self):
        """DataFrame column ordering exactly matches candidate model feature_name_."""
        with open("tests/payload_legitimate.json") as f:
            p = json.load(f)

        cand_expected = list(self.candidate_model.feature_name_)
        df = build_model_dataframe(p, feature_names=cand_expected)
        self.assertEqual(list(df.columns), cand_expected)

    # 12. Production model feature ordering
    def test_production_model_feature_ordering(self):
        """DataFrame column ordering exactly matches production model feature_name_."""
        with open("tests/payload_legitimate.json") as f:
            p = json.load(f)

        base_expected = list(self.base_model.feature_name_)
        df = build_model_dataframe(p, feature_names=base_expected)
        self.assertEqual(list(df.columns), base_expected)


if __name__ == "__main__":
    unittest.main()
