import unittest
import json
import pickle
from unittest.mock import patch
import numpy as np

from backend.app.ml_feature_builder import build_model_dataframe
from backend.app.services.shap_service import explain_prediction, _EXPLAINER_CACHE


class TestSHAPService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open("models/lightgbm_model.pkl", "rb") as f:
            cls.model = pickle.load(f)
        cls.labels = {0: "Legitimate", 1: "Policy Abuser", 2: "Fraudulent Return", 3: "Wardrobing"}

    def setUp(self):
        with open("tests/payload_legitimate.json") as f:
            self.sample_case = json.load(f)
        feature_names = list(getattr(self.model, "feature_name_", []))
        self.df = build_model_dataframe(self.sample_case, feature_names=feature_names)
        self.df = self.df[feature_names]

    def test_production_model_local_attribution(self):
        result = explain_prediction(
            model=self.model,
            X_df=self.df,
            predicted_class_idx=0,
            class_labels=self.labels,
            top_k=5,
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["predicted_class"], "Legitimate")
        self.assertIsInstance(result["top_positive_drivers"], list)
        self.assertIsInstance(result["top_negative_drivers"], list)
        self.assertTrue(len(result["top_positive_drivers"]) <= 5)

        for driver in result["top_positive_drivers"]:
            self.assertIn("feature", driver)
            self.assertIn("value", driver)
            self.assertIn("attribution", driver)
            self.assertTrue(driver["attribution"] > 0)

        for driver in result["top_negative_drivers"]:
            self.assertIn("feature", driver)
            self.assertIn("value", driver)
            self.assertIn("attribution", driver)
            self.assertTrue(driver["attribution"] < 0)

    def test_shap_explainer_cached(self):
        initial_cache_size = len(_EXPLAINER_CACHE)
        _ = explain_prediction(model=self.model, X_df=self.df, predicted_class_idx=0)
        # Model should now be cached
        self.assertIn(id(self.model), _EXPLAINER_CACHE)

    def test_shap_graceful_degradation_on_error(self):
        with patch("backend.app.services.shap_service._get_tree_explainer", side_effect=RuntimeError("SHAP engine error")):
            result = explain_prediction(model=self.model, X_df=self.df, predicted_class_idx=0)
            self.assertFalse(result["available"])
            self.assertEqual(result["top_positive_drivers"], [])
            self.assertEqual(result["top_negative_drivers"], [])
            self.assertIn("unavailable", result["explanation_summary"])


if __name__ == "__main__":
    unittest.main()
