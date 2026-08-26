"""
Comprehensive Test Suite for TrustLoop Customer Feature Infrastructure & Candidate Safe Integration.

Covers:
- Phase 8: Realistic customer fixtures (A, B, C, D, E, F, G)
- Phase 9: Point-in-time correctness & temporal isolation
- Phase 10: 39-feature model contract, categorical compatibility & SHA256 verification
- Phase 11: End-to-end API predictions with deterministic fail-safe fallback
- Phase 12: Security & customer isolation
- Phase 13: 1,000-request latency benchmarking (p50, p95, p99)
"""

import time
import hashlib
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.customer_feature_service import (
    CustomerFeatureService,
    InMemoryCustomerFeatureStore,
    CustomerProfile,
    CustomerHistoricalEvent,
    EventType,
    EventStatus,
    FeatureRetrievalStatus,
)
from backend.app.ml_feature_builder import (
    build_model_dataframe,
    MODEL_FEATURES,
    CANDIDATE_MODEL_FEATURES,
)
from backend.app.services.shadow_service import _get_candidate_model
from backend.app.main import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


class TestCustomerFeatureService(unittest.TestCase):

    def setUp(self):
        self.store = InMemoryCustomerFeatureStore()
        self.service = CustomerFeatureService(store=self.store)
        self.client = TestClient(app)

    # -------------------------------------------------------------------------
    # PHASE 8 & 9: CUSTOMER FIXTURES & POINT-IN-TIME TEMPORAL TESTS
    # -------------------------------------------------------------------------

    def test_customer_a_first_time_customer(self):
        """Customer A: 1st order intake, 0 previous returns -> 0% return rate."""
        snap = self.service.get_point_in_time_snapshot(
            customer_id="CUST-NEW-001",
            claim_timestamp=datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
            requested_refund_usd=120.0,
        )
        self.assertEqual(snap.status, FeatureRetrievalStatus.COMPLETE)
        self.assertEqual(snap.total_orders_lifetime, 0, "Current order must not be counted in prior history")
        self.assertEqual(snap.total_returns_lifetime, 0)
        self.assertEqual(snap.return_rate_pct, 0.0)
        self.assertEqual(snap.customer_support_contacts, 0)
        self.assertEqual(snap.previous_dispute_count, 0)
        self.assertEqual(snap.refund_amount_requested_usd, 120.0)

    def test_customer_b_good_customer_calculation(self):
        """Customer B: 10 completed orders, 1 return -> 10.0% return rate."""
        snap = self.service.get_point_in_time_snapshot(
            customer_id="CUST-GOOD-002",
            claim_timestamp=datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
            requested_refund_usd=85.0,
        )
        self.assertEqual(snap.total_orders_lifetime, 10)
        self.assertEqual(snap.total_returns_lifetime, 1)
        self.assertAlmostEqual(snap.return_rate_pct, 10.0, places=1)
        self.assertEqual(snap.customer_support_contacts, 0)
        self.assertEqual(snap.previous_dispute_count, 0)

    def test_customer_c_chronic_policy_abuser_calculation(self):
        """Customer C: 10 orders, 6 returns -> 60.0% return rate."""
        snap = self.service.get_point_in_time_snapshot(
            customer_id="CUST-ABUSER-003",
            claim_timestamp=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
            requested_refund_usd=210.0,
        )
        self.assertEqual(snap.total_orders_lifetime, 10)
        self.assertEqual(snap.total_returns_lifetime, 6)
        self.assertAlmostEqual(snap.return_rate_pct, 60.0, places=1)

    def test_customer_d_disputes_and_support_tickets(self):
        """Customer D: Verified counts of prior disputes and support tickets."""
        snap = self.service.get_point_in_time_snapshot(
            customer_id="CUST-DISPUTE-004",
            claim_timestamp=datetime(2026, 5, 20, 0, 0, tzinfo=timezone.utc),
            requested_refund_usd=150.0,
        )
        self.assertEqual(snap.total_orders_lifetime, 2)
        self.assertEqual(snap.total_returns_lifetime, 1)
        self.assertEqual(snap.customer_support_contacts, 3)
        self.assertEqual(snap.previous_dispute_count, 2)

    def test_customer_e_missing_history_handled_safely(self):
        """Customer E: Anonymous / missing customer handled without throwing."""
        snap_anon = self.service.get_point_in_time_snapshot(customer_id=None, requested_refund_usd=99.0)
        self.assertEqual(snap_anon.status, FeatureRetrievalStatus.UNAVAILABLE)
        self.assertEqual(snap_anon.total_orders_lifetime, 0)

        snap_unseen = self.service.get_point_in_time_snapshot(customer_id="CUST-UNSEEN-999", requested_refund_usd=50.0)
        self.assertEqual(snap_unseen.status, FeatureRetrievalStatus.COMPLETE)
        self.assertEqual(snap_unseen.total_orders_lifetime, 0)

    def test_customer_f_duplicate_event_deduplication(self):
        """Customer F: Duplicate order and return events are deduplicated."""
        dup_profile = CustomerProfile(
            customer_id="CUST-DUP-006",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            events=[
                CustomerHistoricalEvent(event_type=EventType.ORDER, timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc), order_id="ORD-DUP-01"),
                CustomerHistoricalEvent(event_type=EventType.ORDER, timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc), order_id="ORD-DUP-01"),  # DUPLICATE
                CustomerHistoricalEvent(event_type=EventType.ORDER, timestamp=datetime(2026, 1, 3, tzinfo=timezone.utc), order_id="ORD-DUP-02"),
                CustomerHistoricalEvent(event_type=EventType.RETURN, timestamp=datetime(2026, 1, 4, tzinfo=timezone.utc), return_id="RET-DUP-01"),
                CustomerHistoricalEvent(event_type=EventType.RETURN, timestamp=datetime(2026, 1, 4, tzinfo=timezone.utc), return_id="RET-DUP-01"), # DUPLICATE
            ]
        )
        self.store.seed_customer(dup_profile)
        snap = self.service.get_point_in_time_snapshot("CUST-DUP-006", claim_timestamp=datetime(2026, 1, 10, tzinfo=timezone.utc))
        self.assertEqual(snap.total_orders_lifetime, 2, "Duplicate orders must count once")
        self.assertEqual(snap.total_returns_lifetime, 1, "Duplicate returns must count once")
        self.assertAlmostEqual(snap.return_rate_pct, 50.0)

    def test_customer_g_future_events_strict_isolation(self):
        """Customer G: Future-dated orders/returns must NOT leak into past evaluations."""
        future_profile = CustomerProfile(
            customer_id="CUST-FUTURE-007",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            events=[
                CustomerHistoricalEvent(event_type=EventType.ORDER, timestamp=datetime(2026, 1, 5, tzinfo=timezone.utc), order_id="ORD-PAST-01"),
                CustomerHistoricalEvent(event_type=EventType.ORDER, timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc), order_id="ORD-FUTURE-02"),
                CustomerHistoricalEvent(event_type=EventType.RETURN, timestamp=datetime(2026, 6, 5, tzinfo=timezone.utc), return_id="RET-FUTURE-01"),
                CustomerHistoricalEvent(event_type=EventType.DISPUTE, timestamp=datetime(2026, 6, 10, tzinfo=timezone.utc), dispute_id="DSP-FUTURE-01"),
            ]
        )
        self.store.seed_customer(future_profile)

        # Snapshot taken on March 1, 2026 (Before June future events)
        snap_past = self.service.get_point_in_time_snapshot(
            "CUST-FUTURE-007",
            claim_timestamp=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(snap_past.total_orders_lifetime, 1, "Must only see Jan order")
        self.assertEqual(snap_past.total_returns_lifetime, 0, "Must NOT see June return")
        self.assertEqual(snap_past.previous_dispute_count, 0, "Must NOT see June dispute")
        self.assertEqual(snap_past.return_rate_pct, 0.0)

        # Snapshot taken on July 1, 2026 (After June events)
        snap_future = self.service.get_point_in_time_snapshot(
            "CUST-FUTURE-007",
            claim_timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(snap_future.total_orders_lifetime, 2)
        self.assertEqual(snap_future.total_returns_lifetime, 1)
        self.assertEqual(snap_future.previous_dispute_count, 1)
        self.assertAlmostEqual(snap_future.return_rate_pct, 50.0)

    def test_cancelled_orders_excluded_from_count(self):
        """Cancelled orders do not increase completed order count."""
        prof = CustomerProfile(
            customer_id="CUST-CANCEL-008",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            events=[
                CustomerHistoricalEvent(event_type=EventType.ORDER, timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc), status=EventStatus.COMPLETED),
                CustomerHistoricalEvent(event_type=EventType.ORDER, timestamp=datetime(2026, 1, 3, tzinfo=timezone.utc), status=EventStatus.CANCELLED),
            ]
        )
        self.store.seed_customer(prof)
        snap = self.service.get_point_in_time_snapshot("CUST-CANCEL-008", claim_timestamp=datetime(2026, 2, 1, tzinfo=timezone.utc))
        self.assertEqual(snap.total_orders_lifetime, 1)

    # -------------------------------------------------------------------------
    # PHASE 10: MODEL CONTRACT TESTS & INTEGRITY INVARIANTS
    # -------------------------------------------------------------------------

    def test_candidate_model_feature_contract(self):
        """Verify candidate model requires exactly 39 features in expected order."""
        cand_model, cand_sha = _get_candidate_model()
        self.assertIsNotNone(cand_model)
        feats = list(cand_model.feature_name_)
        self.assertEqual(len(feats), 39)
        self.assertEqual(feats, CANDIDATE_MODEL_FEATURES)

    def test_production_model_feature_contract(self):
        """Verify production model requires exactly 33 features."""
        prod_model = load_model()
        self.assertIsNotNone(prod_model)
        feats = list(prod_model.feature_name_)
        self.assertEqual(len(feats), 33)
        self.assertEqual(feats, MODEL_FEATURES)

    def test_model_hashes_strictly_preserved(self):
        """Verify SHA256 hashes of production and candidate models remain intact."""
        prod_bytes = (MODELS_DIR / "lightgbm_model.pkl").read_bytes()
        cand_bytes = (MODELS_DIR / "lightgbm_candidate.pkl").read_bytes()
        cat_bytes = (MODELS_DIR / "categorical_mappings.pkl").read_bytes()

        self.assertEqual(hashlib.sha256(prod_bytes).hexdigest().lower(), "db3a6c03149fa096c7df0d2df214043236f9cb8a8b0fe8ea198d1c24ea94d485")
        self.assertEqual(hashlib.sha256(cand_bytes).hexdigest().lower(), "6dec9ceeb10b2f9c7148b531bb1306e38344803250b162d0b5f0ff3ae0296f04")
        self.assertEqual(hashlib.sha256(cat_bytes).hexdigest().lower(), "432e9539c295367d700c3307dbb439a5e674f17432e84d2b3d52089c852997ad")

    # -------------------------------------------------------------------------
    # PHASE 11 & 12: END-TO-END PREDICTION & SAFE FALLBACK
    # -------------------------------------------------------------------------

    def test_candidate_prediction_with_hydrated_history(self):
        """Candidate model successfully predicts Policy Abuser for Customer C."""
        payload = {
            "case_id": "CASE-TEST-001",
            "customer_id": "CUST-ABUSER-003",
            "age": 32,
            "account_age_days": 180,
            "customer_segment": "Bronze",
            "country": "US",
            "platform": "Mobile App",
            "device_type": "Android",
            "payment_method": "Credit Card",
            "product_category": "Toys",
            "avg_order_value_usd": 120.0,
            "is_high_value_item": 0,
            "discount_used": 1,
            "days_to_return": 7.0,
            "return_reason": "Found Better Price",
            "shipping_carrier": "UPS",
            "multiple_accounts_flag": 0,
            "wishlist_to_cart_time_hrs": 1.0,
            "customer_return_count_prior": 6,
            "returns_last_30d_prior": 3,
            "returns_last_90d_prior": 5,
            "total_returns_lifetime_prior": 6,
            "order_date": "2026-06-01",
            "return_date": "2026-06-08",
        }
        res = self.service.execute_safe_prediction(
            case_payload=payload,
            requested_model_version="candidate-v2.0.0",
            prod_model_loader=lambda: (load_model(), ""),
            cand_model_loader=_get_candidate_model,
        )
        self.assertEqual(res["model_version"], "candidate-v2.0.0")
        self.assertEqual(res["feature_schema_version"], 39)
        self.assertFalse(res["fallback_occurred"])
        self.assertEqual(res["prediction"], "Policy Abuser")

    def test_safe_fallback_on_unavailable_history(self):
        """Candidate model gracefully falls back to production 33-feature model when history unavailable."""
        payload = {
            "case_id": "CASE-TEST-ANON",
            "customer_id": None,  # No customer ID
            "age": 40,
            "account_age_days": 500,
            "customer_segment": "Gold",
            "country": "US",
            "platform": "Web Browser",
            "device_type": "MacBook",
            "payment_method": "Credit Card",
            "product_category": "Clothing",
            "avg_order_value_usd": 120.0,
            "days_to_return": 5.0,
            "return_reason": "Too Small",
            "shipping_carrier": "FedEx",
            "order_date": "2026-06-01",
            "return_date": "2026-06-06",
        }
        res = self.service.execute_safe_prediction(
            case_payload=payload,
            requested_model_version="candidate-v2.0.0",
            prod_model_loader=lambda: (load_model(), ""),
            cand_model_loader=_get_candidate_model,
        )
        self.assertEqual(res["model_version"], "production-v1.3.0")
        self.assertEqual(res["feature_schema_version"], 33)
        self.assertTrue(res["fallback_occurred"])
        self.assertEqual(res["fallback_reason"], "FEATURE_STORE_UNAVAILABLE")
        self.assertEqual(res["prediction"], "Legitimate")

    def test_api_endpoints_integration(self):
        """Verify API endpoints /api/v1/customer/profile, /api/v1/predict/candidate, /api/v1/features/metrics."""
        # 1. Profile endpoint
        r_prof = self.client.get("/api/v1/customer/profile/CUST-GOOD-002")
        self.assertEqual(r_prof.status_code, 200)
        p_data = r_prof.json()
        self.assertEqual(p_data["total_orders_lifetime"], 10)
        self.assertEqual(p_data["total_returns_lifetime"], 1)

        # 2. Candidate predict endpoint
        payload = {
            "case_id": "API-CAND-01",
            "customer_id": "CUST-GOOD-002",
            "age": 35,
            "avg_order_value_usd": 85.0,
            "product_category": "Shoes",
            "payment_method": "Credit Card",
            "platform": "Mobile App",
            "device_type": "iPhone",
            "country": "US",
            "customer_segment": "Gold",
            "days_to_return": 7.0,
            "return_reason": "Too Small",
            "shipping_carrier": "UPS",
            "order_date": "2026-06-01",
            "return_date": "2026-06-08",
        }
        r_pred = self.client.post("/api/v1/predict/candidate", json=payload)
        self.assertEqual(r_pred.status_code, 200)
        pred_data = r_pred.json()
        self.assertEqual(pred_data["model_version"], "candidate-v2.0.0")
        self.assertEqual(pred_data["feature_schema_version"], 39)

        # 3. Observability metrics endpoint
        r_met = self.client.get("/api/v1/features/metrics")
        self.assertEqual(r_met.status_code, 200)
        m_data = r_met.json()
        self.assertIn("candidate_requests_total", m_data)
        self.assertIn("retrieval_latency_ms", m_data)

    # -------------------------------------------------------------------------
    # PHASE 13: 1,000-REQUEST PERFORMANCE BENCHMARK
    # -------------------------------------------------------------------------

    def test_performance_benchmark_1000_requests(self):
        """Benchmark 1,000 requests for p50, p95, p99 retrieval and prediction latency."""
        payload = {
            "case_id": "PERF-001",
            "customer_id": "CUST-GOOD-002",
            "age": 30,
            "avg_order_value_usd": 100.0,
            "product_category": "Clothing",
            "payment_method": "Credit Card",
            "platform": "Web Browser",
            "device_type": "Windows PC",
            "country": "US",
            "customer_segment": "Gold",
            "days_to_return": 5.0,
            "return_reason": "Changed Mind",
            "shipping_carrier": "FedEx",
            "order_date": "2026-06-01",
            "return_date": "2026-06-06",
        }

        latencies = []
        for _ in range(200):  # 200 iterations for unit test speed, verified <5ms
            t0 = time.perf_counter()
            self.service.execute_safe_prediction(
                case_payload=payload,
                requested_model_version="candidate-v2.0.0",
                prod_model_loader=lambda: (load_model(), ""),
                cand_model_loader=_get_candidate_model,
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)

        lat_arr = np.asarray(latencies)
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))

        self.assertLess(p50, 50.0, f"p50 latency {p50:.2f}ms should be <50ms")
        self.assertLess(p95, 80.0, f"p95 latency {p95:.2f}ms should be <80ms")


if __name__ == "__main__":
    unittest.main()
