import unittest
from pathlib import Path
import tempfile
import json
import shutil
import pickle
import hashlib
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.feedback import FeedbackSubmission, FeedbackRecord
from backend.app.services.feedback_service import (
    validate_feedback_record,
    record_feedback,
    list_feedback,
    get_feedback_statistics,
    export_verified_training_dataset,
)
from backend.app.services.confidence_service import (
    calculate_probability_entropy,
    calculate_calibrated_confidence,
)
from backend.app.services.drift_service import (
    calculate_psi,
    calculate_numerical_feature_psi,
    evaluate_prediction_drift,
    evaluate_confidence_drift,
    evaluate_feedback_quality_drift,
    generate_comprehensive_drift_report,
)
from backend.app.services.model_registry_service import (
    get_production_metadata,
    get_candidate_metadata,
    list_registered_models,
    validate_model_artifact,
    backup_production_model,
    rollback_production_model,
    calculate_file_sha256,
)
from backend.app.services.retraining_service import evaluate_promotion_gate
from backend.app.services.snapshot_service import (
    create_training_snapshot,
    get_snapshot_metadata,
    list_snapshots,
    verify_snapshot_integrity,
)
from backend.app.services.shadow_service import (
    evaluate_shadow_case,
    get_shadow_summary,
    get_shadow_disagreements,
    is_shadow_mode_enabled,
    set_shadow_mode_enabled,
)


class TestSelfLearningPipeline(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    # ============================================================
    # 1. FEEDBACK INGESTION & QUALITY GATES
    # ============================================================

    def test_feedback_ingestion_eligible(self):
        sub = FeedbackSubmission(
            case_id="AUDIT-2026-001",
            human_verified_label="Fraudulent Return",
            human_decision="Rejected",
            reviewer_id="lead_auditor_jane",
            feedback_source="warehouse_inspection",
            notes="Counterfeit return verified with empty serial plate",
            raw_payload={
                "avg_order_value_usd": 420.0,
                "days_to_return": 1.0,
                "multiple_accounts_flag": 1
            }
        )
        rec = record_feedback(sub)
        self.assertTrue(rec.training_eligible)
        self.assertEqual(rec.quality_status, "ELIGIBLE")
        self.assertEqual(rec.human_verified_label, "Fraudulent Return")
        self.assertEqual(len(rec.quarantine_reasons), 0)

    def test_feedback_rejection_invalid_label(self):
        rec = FeedbackRecord(
            case_id="AUDIT-2026-002",
            human_verified_label="Definitely_Not_A_Valid_Class",
            reviewer_id="lead_auditor_jane",
            raw_features={"days_to_return": 5.0}
        )
        eligible, status, reasons = validate_feedback_record(rec)
        self.assertFalse(eligible)
        self.assertEqual(status, "REJECTED")
        self.assertTrue(any("Invalid human verified label" in r for r in reasons))

    def test_feedback_quarantine_demo_reviewer(self):
        sub = FeedbackSubmission(
            case_id="AUDIT-2026-003",
            human_verified_label="Legitimate",
            reviewer_id="synthetic-test",
            raw_payload={"days_to_return": 10.0}
        )
        rec = record_feedback(sub)
        self.assertFalse(rec.training_eligible)
        self.assertEqual(rec.quality_status, "QUARANTINED")
        self.assertTrue(any("demo/test" in r for r in rec.quarantine_reasons))

    def test_feedback_quarantine_demo_case_prefix(self):
        sub = FeedbackSubmission(
            case_id="TEST-CASE-999",
            human_verified_label="Wardrobing",
            reviewer_id="lead_auditor_jane",
            raw_payload={"days_to_return": 35.0}
        )
        rec = record_feedback(sub)
        self.assertFalse(rec.training_eligible)
        self.assertEqual(rec.quality_status, "QUARANTINED")
        self.assertTrue(any("test/demo prefix" in r for r in rec.quarantine_reasons))

    def test_feedback_quarantine_missing_features(self):
        sub = FeedbackSubmission(
            case_id="AUDIT-2026-004",
            human_verified_label="Policy Abuser",
            reviewer_id="lead_auditor_jane",
            raw_payload=None
        )
        rec = record_feedback(sub)
        self.assertFalse(rec.training_eligible)
        self.assertEqual(rec.quality_status, "QUARANTINED")
        self.assertTrue(any("Missing raw features" in r for r in rec.quarantine_reasons))

    def test_feedback_export_dataset(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            temp_path = Path(tf.name)
        try:
            count = export_verified_training_dataset(temp_path)
            self.assertGreaterEqual(count, 0)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    # ============================================================
    # 2. CONFIDENCE & PROMOTION GATE
    # ============================================================

    def test_calibrated_confidence_low_entropy(self):
        probs = {"Legitimate": 0.95, "Policy Abuser": 0.03, "Fraudulent Return": 0.01, "Wardrobing": 0.01}
        conf, uncertainty, metrics = calculate_calibrated_confidence(
            probabilities=probs,
            ml_label="Legitimate",
            deterministic_signals=[],
            policy_status="POLICY_COMPLIANT"
        )
        self.assertGreaterEqual(conf, 85.0)
        self.assertEqual(uncertainty, "LOW")
        self.assertLess(metrics["normalized_entropy"], 0.35)

    def test_calibrated_confidence_high_entropy(self):
        probs = {"Legitimate": 0.30, "Policy Abuser": 0.28, "Fraudulent Return": 0.22, "Wardrobing": 0.20}
        conf, uncertainty, metrics = calculate_calibrated_confidence(
            probabilities=probs,
            ml_label="Legitimate",
            deterministic_signals=["Elevated return activity"],
            policy_status="HUMAN_ESCALATION"
        )
        self.assertLess(conf, 65.0)
        self.assertEqual(uncertainty, "HIGH")
        self.assertGreater(metrics["normalized_entropy"], 0.80)

    def test_promotion_gate_acceptance(self):
        candidate_metrics = {
            "macro_f1": 0.9987,
            "fraudulent_return_recall": 1.0000,
            "policy_abuser_recall": 0.9953,
            "wardrobing_recall": 1.0000,
            "legitimate_precision": 1.0000,
        }
        gate = evaluate_promotion_gate(candidate_metrics)
        self.assertTrue(gate.promoted)
        self.assertEqual(len(gate.rejection_reasons), 0)

    def test_promotion_gate_rejection_on_regression(self):
        candidate_metrics = {
            "macro_f1": 0.9000,
            "fraudulent_return_recall": 0.8500,  # Regressed > 2% from 0.9878
            "policy_abuser_recall": 0.5000,
            "wardrobing_recall": 0.9700,
            "legitimate_precision": 0.9200,
        }
        gate = evaluate_promotion_gate(candidate_metrics)
        self.assertFalse(gate.promoted)
        self.assertTrue(any("Fraudulent Return recall regressed" in r for r in gate.rejection_reasons))

    def test_promotion_gate_rejection_on_insufficient_f1(self):
        candidate_metrics = {
            "macro_f1": 0.8730,  # Only +0.0009 improvement, below 0.005 threshold
            "fraudulent_return_recall": 0.9900,
            "policy_abuser_recall": 0.4800,
            "wardrobing_recall": 0.9800,
            "legitimate_precision": 0.9200,
        }
        gate = evaluate_promotion_gate(candidate_metrics)
        self.assertFalse(gate.promoted)
        self.assertTrue(any("Macro F1 did not improve" in r for r in gate.rejection_reasons))

    # ============================================================
    # 3. TRAINING SNAPSHOT SUBSYSTEM
    # ============================================================

    def test_training_snapshot_creation_and_integrity(self):
        snap = create_training_snapshot(dataset_version="test_v1.0")
        self.assertTrue(snap.snapshot_id.startswith("snap_"))
        self.assertGreater(snap.row_count, 0)
        self.assertEqual(len(snap.dataset_sha256), 64)

        # Verify integrity
        valid, msg = verify_snapshot_integrity(snap.snapshot_id)
        self.assertTrue(valid, msg)

        # Retrieve metadata
        retrieved = get_snapshot_metadata(snap.snapshot_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.dataset_sha256, snap.dataset_sha256)

    def test_training_snapshot_disallow_overwrite(self):
        import uuid
        snap_id = f"test_unique_{uuid.uuid4().hex}"
        create_training_snapshot(custom_snapshot_id=snap_id)
        with self.assertRaises(ValueError):
            create_training_snapshot(custom_snapshot_id=snap_id)


    # ============================================================
    # 4. SILENT SHADOW CANDIDATE EVALUATION
    # ============================================================

    def test_shadow_evaluation_enabled_and_disagreement_logging(self):
        set_shadow_mode_enabled(True)
        self.assertTrue(is_shadow_mode_enabled())

        payload = {
            "case_id": "SHADOW-TEST-001",
            "customer_id": "C-SHADOW-1",
            "order_id": "O-SHADOW-1",
            "age": 30,
            "account_age_days": 100,
            "customer_segment": "Silver",
            "country": "US",
            "platform": "Mobile App",
            "device_type": "iPhone",
            "payment_method": "Credit Card",
            "product_category": "Clothing",
            "avg_order_value_usd": 150.0,
            "is_high_value_item": 0,
            "discount_used": 1,
            "days_to_return": 8.0,
            "return_reason": "Too Small",
            "shipping_carrier": "FedEx",
            "multiple_accounts_flag": 0,
            "wishlist_to_cart_time_hrs": 2.0,
            "customer_return_count_prior": 2,
            "returns_last_30d_prior": 1,
            "returns_last_90d_prior": 2,
            "total_returns_lifetime_prior": 3,
            "order_date": "2026-06-01",
            "return_date": "2026-06-09",
            "total_orders_lifetime": 10,
            "total_returns_lifetime": 3,
            "return_rate_pct": 30.0,
            "customer_support_contacts": 1,
            "previous_dispute_count": 0,
            "refund_amount_requested_usd": 150.0,
        }

        res = evaluate_shadow_case(
            case_payload=payload,
            production_label="Legitimate",
            production_probabilities={"Legitimate": 0.85, "Policy Abuser": 0.15, "Fraudulent Return": 0.0, "Wardrobing": 0.0},
            production_confidence=85.0,
            production_risk_score=15.0
        )
        self.assertIsNotNone(res)
        self.assertTrue(res.get("evaluated", False))
        self.assertIn("candidate_label", res)

        summary = get_shadow_summary()
        self.assertGreaterEqual(summary["total_evaluated"], 1)

    def test_shadow_evaluation_disabled(self):
        set_shadow_mode_enabled(False)
        self.assertFalse(is_shadow_mode_enabled())

        payload = {"case_id": "SHADOW-TEST-002", "days_to_return": 5.0}
        res = evaluate_shadow_case(
            case_payload=payload,
            production_label="Legitimate",
            production_probabilities={},
            production_confidence=90.0
        )
        self.assertIsNone(res)
        # Restore enabled
        set_shadow_mode_enabled(True)

    def test_shadow_candidate_exception_isolation(self):
        # Even with an empty/malformed payload, shadow evaluation must catch error and not raise
        res = evaluate_shadow_case(
            case_payload={"case_id": "BAD-PAYLOAD-SHADOW"},
            production_label="Legitimate",
            production_probabilities={},
            production_confidence=90.0
        )
        self.assertIsNotNone(res)
        self.assertFalse(res.get("evaluated", True))
        self.assertEqual(res.get("status"), "ERROR")

    # ============================================================
    # 5. MODEL BACKUP, INTEGRITY & ROLLBACK
    # ============================================================

    def test_model_backup_creation(self):
        success, msg, backup_path = backup_production_model(reason="unit_test_backup")
        self.assertTrue(success, msg)
        self.assertIsNotNone(backup_path)
        self.assertTrue(backup_path.exists())

    def test_model_integrity_validation(self):
        prod_path = Path("models/lightgbm_model.pkl")
        valid, msg, model_obj = validate_model_artifact(prod_path)
        self.assertTrue(valid, msg)
        self.assertIsNotNone(model_obj)

        # Test corrupt model rejection
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
            corrupt_path = Path(tf.name)
            corrupt_path.write_text("NOT_A_VALID_PICKLE_STRING")

        try:
            c_valid, c_msg, _ = validate_model_artifact(corrupt_path)
            self.assertFalse(c_valid)
            self.assertIn("Failed to unpickle", c_msg)
        finally:
            if corrupt_path.exists():
                corrupt_path.unlink()

    def test_rollback_production_model(self):
        # 1. Take backup
        b_ok, b_msg, backup_path = backup_production_model(reason="pre_rollback_test")
        self.assertTrue(b_ok, b_msg)

        # 2. Rollback
        r_ok, r_msg, details = rollback_production_model(backup_path=backup_path, reason="test_rollback_execution")
        self.assertTrue(r_ok, r_msg)
        self.assertIn("restored_production_sha256", details)

        # 3. Verify production model still valid and matching
        prod_path = Path("models/lightgbm_model.pkl")
        v_ok, v_msg, _ = validate_model_artifact(prod_path)
        self.assertTrue(v_ok, v_msg)
        self.assertEqual(
            calculate_file_sha256(prod_path),
            "db3a6c03149fa096c7df0d2df214043236f9cb8a8b0fe8ea198d1c24ea94d485"
        )

    # ============================================================
    # 6. MULTI-DIMENSIONAL DRIFT MONITORING
    # ============================================================

    def test_feature_psi_calculation(self):
        import numpy as np
        np.random.seed(42)
        exp = np.random.normal(50.0, 10.0, 500).tolist()
        act_stable = np.random.normal(50.2, 10.0, 500).tolist()
        psi, status = calculate_numerical_feature_psi(exp, act_stable)
        self.assertEqual(status, "STABLE")
        self.assertLess(psi, 0.10)

        # Severe feature shift
        act_shifted = np.random.normal(75.0, 10.0, 500).tolist()
        psi_shift, status_shift = calculate_numerical_feature_psi(exp, act_shifted)
        self.assertEqual(status_shift, "SIGNIFICANT_DRIFT")
        self.assertGreater(psi_shift, 0.25)


    def test_confidence_drift_detection(self):
        # Stable confidences
        conf_stable = evaluate_confidence_drift([88.0, 85.0, 92.0, 87.0])
        self.assertFalse(conf_stable["drift_detected"])
        self.assertEqual(conf_stable["status"], "STABLE")

        # Degraded confidences (e.g. 50%)
        conf_degraded = evaluate_confidence_drift([45.0, 50.0, 52.0, 48.0])
        self.assertTrue(conf_degraded["drift_detected"])
        self.assertEqual(conf_degraded["status"], "CONFIDENCE_DEGRADED")

    def test_feedback_quality_drift(self):
        rep = evaluate_feedback_quality_drift()
        self.assertIn("total_feedback_submissions", rep)
        self.assertIn("quarantine_ratio", rep)

    def test_comprehensive_drift_report(self):
        rep = generate_comprehensive_drift_report()
        self.assertIn("overall_health", rep)
        self.assertIn("prediction_drift", rep)
        self.assertIn("confidence_drift", rep)
        self.assertIn("feedback_quality", rep)

    # ============================================================
    # 7. END-TO-END API VALIDATION
    # ============================================================

    def test_full_api_endpoints_integration(self):
        # 1. Analyze endpoint with silent shadow execution
        with open("tests/payload_legitimate.json") as f:
            legit_payload = json.load(f)
        res_analyze = self.client.post("/api/v1/analyze", json=legit_payload)
        self.assertEqual(res_analyze.status_code, 200)
        self.assertEqual(res_analyze.json()["decision"], "Auto-approve")

        # 2. Snapshots API
        res_snap_create = self.client.post("/api/v1/snapshots/create?dataset_version=v1.1")
        self.assertEqual(res_snap_create.status_code, 200)
        snap_id = res_snap_create.json()["snapshot_id"]

        res_snaps_list = self.client.get("/api/v1/snapshots")
        self.assertEqual(res_snaps_list.status_code, 200)
        self.assertGreaterEqual(len(res_snaps_list.json()), 1)

        res_snap_get = self.client.get(f"/api/v1/snapshots/{snap_id}")
        self.assertEqual(res_snap_get.status_code, 200)
        self.assertEqual(res_snap_get.json()["snapshot_id"], snap_id)

        # 3. Shadow API
        res_shadow_sum = self.client.get("/api/v1/shadow/summary")
        self.assertEqual(res_shadow_sum.status_code, 200)
        self.assertIn("total_evaluated", res_shadow_sum.json())

        res_shadow_dis = self.client.get("/api/v1/shadow/disagreements")
        self.assertEqual(res_shadow_dis.status_code, 200)

        # 4. Drift Report API
        res_drift_rep = self.client.get("/api/v1/drift/report")
        self.assertEqual(res_drift_rep.status_code, 200)
        self.assertIn("overall_health", res_drift_rep.json())

        # 5. Rollback API
        res_rollback = self.client.post("/api/v1/models/rollback?reason=api_test_rollback")
        self.assertEqual(res_rollback.status_code, 200)
        self.assertEqual(res_rollback.json()["status"], "ROLLBACK_SUCCESSFUL")


if __name__ == "__main__":
    unittest.main()
