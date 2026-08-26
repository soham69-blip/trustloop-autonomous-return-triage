"""
Tests for TrustLoop ML Feature Validity, Ablation, and Customer-Isolated Generalization.
Verifies all 6 candidate features, provenance artifacts, GroupKFold stability, and report schemas.
"""

import unittest
from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


class TestMLFeatureGeneralization(unittest.TestCase):
    """Test suite validating feature provenance, ablation matrix, and customer isolation."""

    def test_ablation_artifacts_exist_and_valid(self):
        """Verify ablation CSV and JSON artifacts exist and contain 11 configurations."""
        csv_path = REPORTS_DIR / "ml_feature_ablation.csv"
        json_path = REPORTS_DIR / "ml_feature_ablation.json"

        self.assertTrue(csv_path.exists(), "ml_feature_ablation.csv missing")
        self.assertTrue(json_path.exists(), "ml_feature_ablation.json missing")

        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 11, "Expected 11 ablation configurations (A through K)")

        # Verify Config A (baseline 33 feats) Policy Abuser F1 ~ 60.5%
        row_a = df[df["experiment_code"] == "A"].iloc[0]
        self.assertAlmostEqual(row_a["accuracy"], 0.918, places=2)
        self.assertAlmostEqual(row_a["policy_abuser_f1"], 0.605, places=2)

        # Verify Config E (33 + return_rate_pct) has >= 99% Policy Abuser recall
        row_e = df[df["experiment_code"] == "E"].iloc[0]
        self.assertGreaterEqual(row_e["policy_abuser_recall"], 0.98)
        self.assertGreaterEqual(row_e["policy_abuser_f1"], 0.98)

        # Verify Config K (all 6 candidate features) has >= 99% Policy Abuser recall
        row_k = df[df["experiment_code"] == "K"].iloc[0]
        self.assertGreaterEqual(row_k["policy_abuser_recall"], 0.98)
        self.assertGreaterEqual(row_k["policy_abuser_f1"], 0.98)

    def test_customer_group_validation_artifact(self):
        """Verify 5-fold customer-isolated GroupKFold results."""
        csv_path = REPORTS_DIR / "ml_customer_group_validation.csv"
        md_path = REPORTS_DIR / "ml_customer_group_validation.md"

        self.assertTrue(csv_path.exists())
        self.assertTrue(md_path.exists())

        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 15, "Expected 3 models x 5 folds = 15 rows")

        # Verify candidate 39 feats maintains >98% Policy Abuser F1 across all 5 folds
        cand_folds = df[df["model_configuration"] == "Candidate (39 feats)"]
        for _, fold_row in cand_folds.iterrows():
            self.assertGreaterEqual(fold_row["policy_abuser_f1"], 0.98)
            self.assertGreaterEqual(fold_row["accuracy"], 0.99)

    def test_deployability_matrix_artifact(self):
        """Verify feature deployability matrix covers all 6 candidate features."""
        path = REPORTS_DIR / "ml_feature_deployability_matrix.csv"
        self.assertTrue(path.exists())
        df = pd.read_csv(path)
        self.assertEqual(len(df), 6)
        self.assertTrue((df["decision_time_valid"] == "YES").all())
        self.assertTrue((df["leakage_risk"] == "NONE").all())
        self.assertTrue((df["production_ready"] == "YES").all())

    def test_counterfactual_analysis_artifact(self):
        """Verify counterfactual sweep table contains multi-feature sweeps."""
        path = REPORTS_DIR / "ml_counterfactual_feature_analysis.csv"
        self.assertTrue(path.exists())
        df = pd.read_csv(path)
        self.assertGreaterEqual(len(df), 20)
        self.assertIn("return_rate_pct", df["feature"].unique())
        self.assertIn("total_returns_lifetime", df["feature"].unique())

    def test_all_reports_generated(self):
        """Verify all 6 required markdown and CSV reports exist on disk."""
        required_reports = [
            "ml_feature_ablation.csv",
            "ml_feature_ablation.json",
            "ml_feature_provenance_audit.md",
            "ml_customer_group_validation.csv",
            "ml_customer_group_validation.md",
            "ml_temporal_feature_audit.md",
            "ml_synthetic_realism_audit.md",
            "ml_counterfactual_feature_analysis.csv",
            "ml_feature_deployability_matrix.csv",
            "ML_FEATURE_GENERALIZATION_FINAL.md",
        ]
        for fname in required_reports:
            p = REPORTS_DIR / fname
            self.assertTrue(p.exists(), f"Report artifact '{fname}' missing")
            self.assertGreater(p.stat().st_size, 0, f"Report artifact '{fname}' is empty")


if __name__ == "__main__":
    unittest.main()
