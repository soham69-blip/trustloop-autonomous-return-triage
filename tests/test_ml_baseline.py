"""
Tests for TrustLoop ML Baseline and Validation Suite.
Verifies metrics, threshold logic, calibration, hard cases, model integrity, and report artifacts.
"""

import unittest
from pathlib import Path
import json
import hashlib
import pickle
import numpy as np
import pandas as pd

from backend.app.ml_feature_builder import (
    build_model_dataframe,
    MODEL_FEATURES,
    CANDIDATE_MODEL_FEATURES,
)
from scripts.ml_baseline import (
    CLASS_NAMES,
    CLASS_IDS,
    REFERENCE_PROD_SHA256,
    REFERENCE_CAND_SHA256,
    REFERENCE_CAT_SHA256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


class TestMLBaselineSuite(unittest.TestCase):
    """Test suite verifying all components of the ML baseline and validation suite."""

    def test_model_hash_integrity(self):
        """Verify production, candidate, and categorical mapping SHA256 invariants."""
        prod_path = MODELS_DIR / "lightgbm_model.pkl"
        cand_path = MODELS_DIR / "lightgbm_candidate.pkl"
        cat_path = MODELS_DIR / "categorical_mappings.pkl"

        self.assertTrue(prod_path.exists(), "Production model missing")
        self.assertTrue(cand_path.exists(), "Candidate model missing")
        self.assertTrue(cat_path.exists(), "Categorical mappings missing")

        prod_sha = hashlib.sha256(prod_path.read_bytes()).hexdigest()
        cand_sha = hashlib.sha256(cand_path.read_bytes()).hexdigest()
        cat_sha = hashlib.sha256(cat_path.read_bytes()).hexdigest()

        self.assertEqual(prod_sha.lower(), REFERENCE_PROD_SHA256.lower(), "Production model hash mismatch")
        self.assertEqual(cand_sha.lower(), REFERENCE_CAND_SHA256.lower(), "Candidate model hash mismatch")
        self.assertEqual(cat_sha.lower(), REFERENCE_CAT_SHA256.lower(), "Categorical mappings hash mismatch")

    def test_feature_contracts(self):
        """Verify 33 production features and 39 candidate features contracts."""
        self.assertEqual(len(MODEL_FEATURES), 33)
        self.assertEqual(len(CANDIDATE_MODEL_FEATURES), 39)
        self.assertTrue(set(MODEL_FEATURES).issubset(set(CANDIDATE_MODEL_FEATURES)))

        with open(MODELS_DIR / "lightgbm_model.pkl", "rb") as f:
            prod_model = pickle.load(f)
        self.assertEqual(list(prod_model.feature_name_), MODEL_FEATURES)

    def test_dataset_baseline_report_artifact(self):
        """Verify dataset baseline JSON artifact schema and numbers."""
        path = REPORT_DIR / "dataset_baseline.json"
        self.assertTrue(path.exists(), "dataset_baseline.json not found")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["total_samples"], 60000)
        self.assertEqual(data["split_counts"]["train"], 42000)
        self.assertEqual(data["split_counts"]["val"], 9000)
        self.assertEqual(data["split_counts"]["test"], 9000)
        self.assertEqual(data["data_quality"]["duplicate_rows"], 0)
        self.assertEqual(data["data_quality"]["missing_values_total"], 0)
        self.assertEqual(data["data_quality"]["train_test_exact_overlap_rows"], 0)

    def test_ml_baseline_metrics_artifact(self):
        """Verify baseline classification metrics JSON artifact."""
        path = REPORT_DIR / "ml_baseline_metrics.json"
        self.assertTrue(path.exists(), "ml_baseline_metrics.json not found")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ovr = data["overall_metrics"]
        self.assertGreaterEqual(ovr["accuracy"], 0.90)
        self.assertGreaterEqual(ovr["macro_f1"], 0.85)
        self.assertLessEqual(ovr["log_loss"], 0.35)
        self.assertLessEqual(ovr["multiclass_brier_score"], 0.15)
        self.assertGreaterEqual(ovr["macro_roc_auc"], 0.95)

        pcm = data["per_class_metrics"]
        self.assertIn("Legitimate", pcm)
        self.assertIn("Policy Abuser", pcm)
        self.assertIn("Fraudulent Return", pcm)
        self.assertIn("Wardrobing", pcm)

        # Confirm specific known benchmark expectations
        self.assertGreaterEqual(pcm["Legitimate"]["f1"], 0.90)
        self.assertGreaterEqual(pcm["Fraudulent Return"]["f1"], 0.95)
        self.assertGreaterEqual(pcm["Wardrobing"]["f1"], 0.95)

    def test_confusion_matrix_sum_invariant(self):
        """Verify confusion matrix sum equals test sample count (9,000)."""
        path = REPORT_DIR / "confusion_matrix.csv"
        self.assertTrue(path.exists())
        cm_df = pd.read_csv(path, index_col=0)
        self.assertEqual(cm_df.to_numpy().sum(), 9000)

        # Normalized confusion matrix rows sum to 1.0
        path_norm = REPORT_DIR / "confusion_matrix_normalized.csv"
        cm_norm_df = pd.read_csv(path_norm, index_col=0)
        for row_sum in cm_norm_df.sum(axis=1):
            self.assertAlmostEqual(row_sum, 1.0, places=2)

    def test_policy_abuser_threshold_analysis_artifact(self):
        """Verify threshold sweep dataset and validity."""
        path = REPORT_DIR / "policy_abuser_threshold_analysis.csv"
        self.assertTrue(path.exists())
        df = pd.read_csv(path)
        self.assertGreaterEqual(len(df), 40)
        self.assertTrue((df["threshold"] >= 0.0).all() and (df["threshold"] <= 1.0).all())
        self.assertTrue((df["precision"] >= 0.0).all() and (df["precision"] <= 1.0).all())
        self.assertTrue((df["recall"] >= 0.0).all() and (df["recall"] <= 1.0).all())

    def test_hard_case_baseline_artifact(self):
        """Verify 15-case adversarial hard-case suite."""
        path = REPORT_DIR / "hard_case_baseline.json"
        self.assertTrue(path.exists())
        with open(path, "r", encoding="utf-8") as f:
            cases = json.load(f)
        self.assertEqual(len(cases), 15)

        # Verify HC-01 (legit frequent) passes
        hc01 = next(c for c in cases if c["case_id"] == "HC-01")
        self.assertTrue(hc01["production_correct"])
        self.assertEqual(hc01["production_prediction"], "Legitimate")

        # Verify HC-15 (unseen categorical) is correctly rejected by schema contract
        hc15 = next(c for c in cases if c["case_id"] == "HC-15")
        self.assertEqual(hc15["production_prediction"], "SCHEMA_REJECTED")

    def test_robustness_report_artifact(self):
        """Verify robustness perturbations."""
        path = REPORT_DIR / "robustness_report.json"
        self.assertTrue(path.exists())
        with open(path, "r", encoding="utf-8") as f:
            rob = json.load(f)
        self.assertGreaterEqual(len(rob), 5)
        for r in rob:
            self.assertIn("status", r)
            self.assertIn("max_probability_delta", r)

    def test_production_vs_candidate_benchmark(self):
        """Verify candidate vs production comparison dataset."""
        path = REPORT_DIR / "production_vs_candidate.csv"
        self.assertTrue(path.exists())
        df = pd.read_csv(path)
        self.assertIn("production_value", df.columns)
        self.assertIn("candidate_value", df.columns)
        self.assertIn("delta (candidate - prod)", df.columns)

    def test_bootstrap_confidence_intervals(self):
        """Verify 95% bootstrap confidence interval outputs."""
        path = REPORT_DIR / "bootstrap_confidence_intervals.csv"
        self.assertTrue(path.exists())
        df = pd.read_csv(path)
        self.assertGreaterEqual(len(df), 6)
        for _, row in df.iterrows():
            self.assertLessEqual(row["ci_95_lower"], row["mean"])
            self.assertGreaterEqual(row["ci_95_upper"], row["mean"])

    def test_final_comprehensive_report_artifact(self):
        """Verify final ML baseline markdown report exists and has complete sections."""
        path = REPORT_DIR / "ML_BASELINE_FINAL_REPORT.md"
        self.assertTrue(path.exists())
        content = path.read_text(encoding="utf-8")
        self.assertIn("ML BASELINE STATUS: COMPLETE", content)
        self.assertIn("Production Model Performance Benchmark", content)
        self.assertIn("Policy Abuser Detection", content)


if __name__ == "__main__":
    unittest.main()
