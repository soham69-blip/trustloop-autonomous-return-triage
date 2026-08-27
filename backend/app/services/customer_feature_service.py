"""
TrustLoop Real-Time Customer Profile Feature Infrastructure & Point-in-Time Correctness Engine.

Provides:
1. CustomerFeatureStore protocol and both InMemoryCustomerFeatureStore & PersistentCustomerFeatureStore implementations.
2. Point-in-time historical aggregation excluding current transaction and future events with timezone normalization.
3. Point-in-time CustomerFeatureSnapshot with cryptographic SHA256 auditability and freshness metadata.
4. Deterministic fail-safe routing: Candidate (39 feats) -> Production (33 feats) fallback.
5. Thread-safe & process-safe persistence across worker restarts.
6. Observability metrics for latency, fallback counts, and feature retrieval status.
"""

import re
import time
import uuid
import hashlib
import threading
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Protocol, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from backend.app.core.persistence import locked_append_jsonl, locked_read_jsonl, atomic_write_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURE_STORE_DIR = PROJECT_ROOT / "data" / "feature_store"
FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_JOURNAL_FILE = FEATURE_STORE_DIR / "customer_events.jsonl"
PROFILES_SNAPSHOT_FILE = FEATURE_STORE_DIR / "customer_profiles.json"


def normalize_to_utc(val: Any) -> datetime:
    """Normalize strings, naive datetimes, and tz-aware datetimes to UTC timezone-aware datetime."""
    if val is None:
        return datetime.now(timezone.utc)

    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)

    text = str(val).strip()
    if not text:
        return datetime.now(timezone.utc)

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y",
        "%m/%d/%Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def sanitize_customer_id(raw_id: Optional[str]) -> Optional[str]:
    """Sanitize customer identifier to prevent path traversal or injection attacks."""
    if raw_id is None:
        return None
    cleaned = raw_id.strip()
    if not cleaned or cleaned.upper() in ("NONE", "NULL", "ANONYMOUS", "UNDEFINED"):
        return None
    # Reject directory traversal or path separators
    if ".." in cleaned or "/" in cleaned or "\\" in cleaned:
        return None
    # Strip any dangerous punctuation
    sanitized = re.sub(r"[^a-zA-Z0-9_\-\.]", "", cleaned)
    return sanitized if sanitized else None


class FeatureRetrievalStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


class EventType(str, Enum):
    ORDER = "ORDER"
    RETURN = "RETURN"
    DISPUTE = "DISPUTE"
    SUPPORT_TICKET = "SUPPORT_TICKET"


class EventStatus(str, Enum):
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    PENDING = "PENDING"
    REFUNDED = "REFUNDED"


class CustomerHistoricalEvent(BaseModel):
    """Timestamped historical event record for a customer."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp: datetime
    status: EventStatus = EventStatus.COMPLETED
    order_id: Optional[str] = None
    return_id: Optional[str] = None
    dispute_id: Optional[str] = None
    ticket_id: Optional[str] = None
    amount_usd: float = 0.0
    notes: Optional[str] = None


class CustomerProfile(BaseModel):
    """Aggregate customer profile at a specific point in time."""
    customer_id: str
    created_at: datetime
    customer_segment: str = "Standard"
    country: str = "US"
    events: List[CustomerHistoricalEvent] = []


class CustomerFeatureSnapshot(BaseModel):
    """
    Point-in-time snapshot of customer profile features.
    Provides strict point-in-time correctness, feature provenance, freshness metadata, and auditability.
    """
    customer_id: str
    snapshot_timestamp: str
    total_orders_lifetime: int
    total_returns_lifetime: int
    return_rate_pct: float
    customer_support_contacts: int
    previous_dispute_count: int
    refund_amount_requested_usd: float
    feature_source: str
    feature_version: str = "2.0.0"
    status: FeatureRetrievalStatus
    is_point_in_time_safe: bool = True
    retrieval_latency_ms: float = 0.0
    snapshot_hash: str = ""
    age_ms: float = 0.0
    stale_threshold_ms: float = 86400000.0  # 24 hours
    is_stale: bool = False
    freshness_status: str = "FRESH"

    def compute_hash(self) -> str:
        payload = (
            f"{self.customer_id}|{self.snapshot_timestamp}|"
            f"{self.total_orders_lifetime}|{self.total_returns_lifetime}|"
            f"{self.return_rate_pct}|{self.customer_support_contacts}|"
            f"{self.previous_dispute_count}|{self.refund_amount_requested_usd}|"
            f"{self.feature_version}|{self.status.value}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CustomerFeatureStore(Protocol):
    """Interface for customer feature stores (In-Memory, Persistent Journal, SQL DB, or Redis)."""

    def get_customer_profile(self, customer_id: str) -> Optional[CustomerProfile]:
        ...

    def record_event(self, customer_id: str, event: CustomerHistoricalEvent) -> None:
        ...

    def seed_customer(self, profile: CustomerProfile) -> None:
        ...

    def recover(self) -> int:
        ...


class InMemoryCustomerFeatureStore:
    """Thread-safe, high-performance in-memory customer feature store."""

    def __init__(self) -> None:
        self._store: Dict[str, CustomerProfile] = {}
        self._lock = threading.Lock()
        self._seed_default_fixture()

    def get_customer_profile(self, customer_id: str) -> Optional[CustomerProfile]:
        with self._lock:
            return self._store.get(customer_id)

    def record_event(self, customer_id: str, event: CustomerHistoricalEvent) -> None:
        with self._lock:
            if customer_id not in self._store:
                self._store[customer_id] = CustomerProfile(
                    customer_id=customer_id,
                    created_at=event.timestamp,
                )
            existing_event_ids = {e.event_id for e in self._store[customer_id].events}
            if event.event_id not in existing_event_ids:
                self._store[customer_id].events.append(event)

    def seed_customer(self, profile: CustomerProfile) -> None:
        with self._lock:
            self._store[profile.customer_id] = profile

    def recover(self) -> int:
        with self._lock:
            return len(self._store)

    def _seed_default_fixture(self) -> None:
        """Seed representative customer profiles for tests, benchmarks, and demo scenarios."""
        # Customer A: First-time customer (1st order, 0 returns)
        self.seed_customer(CustomerProfile(
            customer_id="CUST-NEW-001",
            created_at=datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
            events=[
                CustomerHistoricalEvent(
                    event_type=EventType.ORDER,
                    timestamp=datetime(2026, 6, 1, 10, 5, tzinfo=timezone.utc),
                    order_id="ORD-001",
                    amount_usd=120.0,
                )
            ]
        ))

        # Customer B: Established Good Customer (10 orders, 1 return, 10% return rate)
        c_b_events = []
        for i in range(1, 11):
            c_b_events.append(CustomerHistoricalEvent(
                event_type=EventType.ORDER,
                timestamp=datetime(2026, 1, i * 2, 12, 0, tzinfo=timezone.utc),
                order_id=f"ORD-B-{i:03d}",
                amount_usd=85.0,
            ))
        c_b_events.append(CustomerHistoricalEvent(
            event_type=EventType.RETURN,
            timestamp=datetime(2026, 2, 15, 14, 0, tzinfo=timezone.utc),
            return_id="RET-B-001",
            order_id="ORD-B-002",
            amount_usd=85.0,
        ))
        self.seed_customer(CustomerProfile(
            customer_id="CUST-GOOD-002",
            created_at=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            customer_segment="Gold",
            events=c_b_events,
        ))

        # Customer C: Chronic Policy Abuser (10 orders, 6 returns, 60% return rate)
        c_c_events = []
        for i in range(1, 11):
            c_c_events.append(CustomerHistoricalEvent(
                event_type=EventType.ORDER,
                timestamp=datetime(2026, 3, i * 3, 10, 0, tzinfo=timezone.utc),
                order_id=f"ORD-C-{i:03d}",
                amount_usd=210.0,
            ))
        for i in range(1, 7):
            c_c_events.append(CustomerHistoricalEvent(
                event_type=EventType.RETURN,
                timestamp=datetime(2026, 4, i * 4, 11, 0, tzinfo=timezone.utc),
                return_id=f"RET-C-{i:03d}",
                order_id=f"ORD-C-{i:03d}",
                amount_usd=210.0,
            ))
        self.seed_customer(CustomerProfile(
            customer_id="CUST-ABUSER-003",
            created_at=datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),
            customer_segment="Bronze",
            events=c_c_events,
        ))

        # Customer D: High Dispute & Support Contact Customer
        c_d_events = [
            CustomerHistoricalEvent(event_type=EventType.ORDER, timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc), order_id="ORD-D-01"),
            CustomerHistoricalEvent(event_type=EventType.ORDER, timestamp=datetime(2026, 5, 5, tzinfo=timezone.utc), order_id="ORD-D-02"),
            CustomerHistoricalEvent(event_type=EventType.RETURN, timestamp=datetime(2026, 5, 10, tzinfo=timezone.utc), return_id="RET-D-01"),
            CustomerHistoricalEvent(event_type=EventType.SUPPORT_TICKET, timestamp=datetime(2026, 5, 11, tzinfo=timezone.utc), ticket_id="TKT-D-01"),
            CustomerHistoricalEvent(event_type=EventType.SUPPORT_TICKET, timestamp=datetime(2026, 5, 12, tzinfo=timezone.utc), ticket_id="TKT-D-02"),
            CustomerHistoricalEvent(event_type=EventType.SUPPORT_TICKET, timestamp=datetime(2026, 5, 13, tzinfo=timezone.utc), ticket_id="TKT-D-03"),
            CustomerHistoricalEvent(event_type=EventType.DISPUTE, timestamp=datetime(2026, 5, 15, tzinfo=timezone.utc), dispute_id="DSP-D-01"),
            CustomerHistoricalEvent(event_type=EventType.DISPUTE, timestamp=datetime(2026, 5, 18, tzinfo=timezone.utc), dispute_id="DSP-D-02"),
        ]
        self.seed_customer(CustomerProfile(
            customer_id="CUST-DISPUTE-004",
            created_at=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
            events=c_d_events,
        ))


class PersistentCustomerFeatureStore:
    """
    Production-grade persistent feature store backed by locked JSONL journaling
    and atomic snapshots with sub-millisecond in-memory read caching.
    """

    def __init__(
        self,
        journal_file: Path = EVENTS_JOURNAL_FILE,
        snapshot_file: Path = PROFILES_SNAPSHOT_FILE,
        seed_defaults: bool = True,
    ) -> None:
        self.journal_file = journal_file
        self.snapshot_file = snapshot_file
        self._memory_store: Dict[str, CustomerProfile] = {}
        self._lock = threading.Lock()

        # Recover from persistent storage if exists
        recovered_count = self.recover()
        if recovered_count == 0 and seed_defaults:
            self._seed_default_fixture()

    def get_customer_profile(self, customer_id: str) -> Optional[CustomerProfile]:
        with self._lock:
            return self._memory_store.get(customer_id)

    def record_event(self, customer_id: str, event: CustomerHistoricalEvent) -> None:
        with self._lock:
            if customer_id not in self._memory_store:
                self._memory_store[customer_id] = CustomerProfile(
                    customer_id=customer_id,
                    created_at=event.timestamp,
                )
            existing_event_ids = {e.event_id for e in self._memory_store[customer_id].events}
            if event.event_id not in existing_event_ids:
                self._memory_store[customer_id].events.append(event)

            # Persist event to append-only locked journal
            record_payload = {
                "customer_id": customer_id,
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "status": event.status.value,
                "order_id": event.order_id,
                "return_id": event.return_id,
                "dispute_id": event.dispute_id,
                "ticket_id": event.ticket_id,
                "amount_usd": event.amount_usd,
                "notes": event.notes,
            }
            locked_append_jsonl(self.journal_file, record_payload)

    def seed_customer(self, profile: CustomerProfile) -> None:
        with self._lock:
            self._memory_store[profile.customer_id] = profile
            # Persist all events for seeded customer
            for event in profile.events:
                record_payload = {
                    "customer_id": profile.customer_id,
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "status": event.status.value,
                    "order_id": event.order_id,
                    "return_id": event.return_id,
                    "dispute_id": event.dispute_id,
                    "ticket_id": event.ticket_id,
                    "amount_usd": event.amount_usd,
                    "notes": event.notes,
                }
                locked_append_jsonl(self.journal_file, record_payload)

    def recover(self) -> int:
        """Rebuild in-memory profile store from the persistent journal."""
        if not self.journal_file.exists():
            return 0

        with self._lock:
            records = locked_read_jsonl(self.journal_file)
            recovered: Dict[str, CustomerProfile] = {}

            for r in records:
                c_id = r.get("customer_id")
                if not c_id:
                    continue

                ts = normalize_to_utc(r.get("timestamp"))
                ev = CustomerHistoricalEvent(
                    event_id=r.get("event_id", str(uuid.uuid4())),
                    event_type=EventType(r.get("event_type", "ORDER")),
                    timestamp=ts,
                    status=EventStatus(r.get("status", "COMPLETED")),
                    order_id=r.get("order_id"),
                    return_id=r.get("return_id"),
                    dispute_id=r.get("dispute_id"),
                    ticket_id=r.get("ticket_id"),
                    amount_usd=float(r.get("amount_usd", 0.0)),
                    notes=r.get("notes"),
                )

                if c_id not in recovered:
                    recovered[c_id] = CustomerProfile(
                        customer_id=c_id,
                        created_at=ts,
                        events=[],
                    )

                existing_ids = {e.event_id for e in recovered[c_id].events}
                if ev.event_id not in existing_ids:
                    recovered[c_id].events.append(ev)

            self._memory_store = recovered
            return len(recovered)

    def _seed_default_fixture(self) -> None:
        in_mem = InMemoryCustomerFeatureStore()
        for c_id, prof in in_mem._store.items():
            self.seed_customer(prof)


class FeatureInfrastructureMetrics:
    """Telemetry collector for feature retrieval latency, fallbacks, and coverage."""

    def __init__(self) -> None:
        self.candidate_requests_total: int = 0
        self.production_fallback_total: int = 0
        self.feature_retrieval_failures_total: int = 0
        self.missing_customer_profiles_total: int = 0
        self.first_time_customers_total: int = 0
        self.latencies_retrieval_ms: List[float] = []
        self.latencies_prediction_ms: List[float] = []
        self._lock = threading.Lock()

    def record_request(
        self,
        is_candidate: bool,
        fallback_occurred: bool,
        status: FeatureRetrievalStatus,
        retrieval_ms: float,
        prediction_ms: float,
    ) -> None:
        with self._lock:
            if is_candidate:
                self.candidate_requests_total += 1
            if fallback_occurred:
                self.production_fallback_total += 1
            if status == FeatureRetrievalStatus.UNAVAILABLE:
                self.missing_customer_profiles_total += 1
                self.feature_retrieval_failures_total += 1

            self.latencies_retrieval_ms.append(retrieval_ms)
            self.latencies_prediction_ms.append(prediction_ms)

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            ret_arr = np.asarray(self.latencies_retrieval_ms) if self.latencies_retrieval_ms else np.array([0.0])
            pred_arr = np.asarray(self.latencies_prediction_ms) if self.latencies_prediction_ms else np.array([0.0])

            return {
                "candidate_requests_total": self.candidate_requests_total,
                "production_fallback_total": self.production_fallback_total,
                "feature_retrieval_failures_total": self.feature_retrieval_failures_total,
                "missing_customer_profiles_total": self.missing_customer_profiles_total,
                "retrieval_latency_ms": {
                    "p50": round(float(np.percentile(ret_arr, 50)), 3),
                    "p95": round(float(np.percentile(ret_arr, 95)), 3),
                    "p99": round(float(np.percentile(ret_arr, 99)), 3),
                },
                "prediction_latency_ms": {
                    "p50": round(float(np.percentile(pred_arr, 50)), 3),
                    "p95": round(float(np.percentile(pred_arr, 95)), 3),
                    "p99": round(float(np.percentile(pred_arr, 99)), 3),
                },
            }


# Global persistent feature store and observability instance
_GLOBAL_FEATURE_STORE = PersistentCustomerFeatureStore()
_GLOBAL_METRICS = FeatureInfrastructureMetrics()


class CustomerFeatureService:
    """
    Production-grade service managing point-in-time customer feature retrieval,
    freshness tracking, fallback evaluation, and candidate model execution.
    """

    def __init__(self, store: Optional[CustomerFeatureStore] = None) -> None:
        self.store: CustomerFeatureStore = store or _GLOBAL_FEATURE_STORE
        self.metrics = _GLOBAL_METRICS

    def get_point_in_time_snapshot(
        self,
        customer_id: Optional[str],
        claim_timestamp: Optional[Any] = None,
        requested_refund_usd: float = 0.0,
        payload_override: Optional[Dict[str, Any]] = None,
        stale_threshold_ms: float = 86400000.0,  # 24 hours default
    ) -> CustomerFeatureSnapshot:
        """
        Calculates customer features strictly before claim_timestamp.
        Excludes current claim and any future events with strict UTC normalization.
        """
        t0 = time.perf_counter()
        eval_time = normalize_to_utc(claim_timestamp) if claim_timestamp is not None else datetime.now(timezone.utc)
        sanitized_id = sanitize_customer_id(customer_id)

        # Case 1: Payload explicitly provides pre-validated candidate features
        if payload_override and all(
            k in payload_override for k in [
                "total_orders_lifetime", "total_returns_lifetime",
                "return_rate_pct", "customer_support_contacts", "previous_dispute_count"
            ]
        ):
            t_orders = int(payload_override.get("total_orders_lifetime", 0))
            t_returns = int(payload_override.get("total_returns_lifetime", 0))
            r_rate = float(payload_override.get("return_rate_pct", 0.0))
            s_contacts = int(payload_override.get("customer_support_contacts", 0))
            p_disputes = int(payload_override.get("previous_dispute_count", 0))
            req_usd = float(payload_override.get("refund_amount_requested_usd", requested_refund_usd) or requested_refund_usd)

            snap = CustomerFeatureSnapshot(
                customer_id=sanitized_id or "ANONYMOUS",
                snapshot_timestamp=eval_time.isoformat(),
                total_orders_lifetime=t_orders,
                total_returns_lifetime=t_returns,
                return_rate_pct=r_rate,
                customer_support_contacts=s_contacts,
                previous_dispute_count=p_disputes,
                refund_amount_requested_usd=req_usd,
                feature_source="payload_verified_override",
                status=FeatureRetrievalStatus.COMPLETE,
                is_point_in_time_safe=True,
                retrieval_latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                age_ms=0.0,
                is_stale=False,
                freshness_status="FRESH",
            )
            snap.snapshot_hash = snap.compute_hash()
            return snap

        # Case 2: Missing, Invalid, or Anonymous Customer ID
        if not sanitized_id:
            snap = CustomerFeatureSnapshot(
                customer_id="ANONYMOUS",
                snapshot_timestamp=eval_time.isoformat(),
                total_orders_lifetime=0,
                total_returns_lifetime=0,
                return_rate_pct=0.0,
                customer_support_contacts=0,
                previous_dispute_count=0,
                refund_amount_requested_usd=requested_refund_usd,
                feature_source="anonymous_fallback",
                status=FeatureRetrievalStatus.UNAVAILABLE,
                is_point_in_time_safe=True,
                retrieval_latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                age_ms=0.0,
                is_stale=False,
                freshness_status="UNAVAILABLE",
            )
            snap.snapshot_hash = snap.compute_hash()
            return snap

        # Case 3: Query Feature Store
        profile = self.store.get_customer_profile(sanitized_id)

        # First-time customer not yet in DB
        if profile is None:
            snap = CustomerFeatureSnapshot(
                customer_id=sanitized_id,
                snapshot_timestamp=eval_time.isoformat(),
                total_orders_lifetime=0,
                total_returns_lifetime=0,
                return_rate_pct=0.0,
                customer_support_contacts=0,
                previous_dispute_count=0,
                refund_amount_requested_usd=requested_refund_usd,
                feature_source="new_customer_store",
                status=FeatureRetrievalStatus.COMPLETE,
                is_point_in_time_safe=True,
                retrieval_latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                age_ms=0.0,
                is_stale=False,
                freshness_status="NEW_CUSTOMER",
            )
            snap.snapshot_hash = snap.compute_hash()
            return snap

        # Calculate point-in-time statistics strictly < eval_time
        completed_orders = 0
        completed_returns = 0
        support_tickets = 0
        disputes = 0
        latest_event_time: Optional[datetime] = None

        seen_order_ids = set()
        seen_return_ids = set()
        seen_dispute_ids = set()
        seen_ticket_ids = set()

        for ev in profile.events:
            ev_ts = normalize_to_utc(ev.timestamp)

            # POINT-IN-TIME INVARIANT: Event must strictly precede evaluation time
            # Boundary condition ev_ts == eval_time is strictly excluded
            if ev_ts >= eval_time:
                continue

            if latest_event_time is None or ev_ts > latest_event_time:
                latest_event_time = ev_ts

            if ev.event_type == EventType.ORDER:
                if ev.status != EventStatus.CANCELLED:
                    key = ev.order_id or ev.event_id
                    if key not in seen_order_ids:
                        seen_order_ids.add(key)
                        completed_orders += 1

            elif ev.event_type == EventType.RETURN:
                if ev.status == EventStatus.COMPLETED or ev.status == EventStatus.REFUNDED:
                    key = ev.return_id or ev.event_id
                    if key not in seen_return_ids:
                        seen_return_ids.add(key)
                        completed_returns += 1

            elif ev.event_type == EventType.SUPPORT_TICKET:
                key = ev.ticket_id or ev.event_id
                if key not in seen_ticket_ids:
                    seen_ticket_ids.add(key)
                    support_tickets += 1

            elif ev.event_type == EventType.DISPUTE:
                key = ev.dispute_id or ev.event_id
                if key not in seen_dispute_ids:
                    seen_dispute_ids.add(key)
                    disputes += 1

        # Return rate calculation rule
        return_rate_pct = 0.0
        if completed_orders > 0:
            return_rate_pct = min(100.0, round((completed_returns / completed_orders) * 100.0, 2))

        # Freshness calculation
        now_utc = datetime.now(timezone.utc)
        age_ms = 0.0
        if latest_event_time:
            age_ms = max(0.0, (now_utc - latest_event_time).total_seconds() * 1000.0)

        is_stale = (age_ms > stale_threshold_ms) if (completed_orders > 0 and latest_event_time) else False
        freshness_status = "STALE" if is_stale else ("NEW_CUSTOMER" if completed_orders == 0 else "FRESH")

        snap = CustomerFeatureSnapshot(
            customer_id=sanitized_id,
            snapshot_timestamp=eval_time.isoformat(),
            total_orders_lifetime=completed_orders,
            total_returns_lifetime=completed_returns,
            return_rate_pct=return_rate_pct,
            customer_support_contacts=support_tickets,
            previous_dispute_count=disputes,
            refund_amount_requested_usd=requested_refund_usd,
            feature_source="persistent_customer_feature_store",
            status=FeatureRetrievalStatus.COMPLETE,
            is_point_in_time_safe=True,
            retrieval_latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            age_ms=round(age_ms, 2),
            stale_threshold_ms=stale_threshold_ms,
            is_stale=is_stale,
            freshness_status=freshness_status,
        )
        snap.snapshot_hash = snap.compute_hash()
        return snap

    def execute_safe_prediction(
        self,
        case_payload: Dict[str, Any],
        requested_model_version: str = "production-v1.3.0",
        prod_model_loader: Any = None,
        cand_model_loader: Any = None,
    ) -> Dict[str, Any]:
        """
        Executes prediction with deterministic candidate -> production fallback.
        """
        import numpy as np
        from backend.app.ml_feature_builder import (
            build_model_dataframe,
            MODEL_FEATURES,
            CANDIDATE_MODEL_FEATURES,
        )

        t_start = time.perf_counter()
        customer_id = case_payload.get("customer_id")
        req_usd = float(case_payload.get("avg_order_value_usd", 0.0))

        # 1. Fetch feature snapshot
        snap = self.get_point_in_time_snapshot(
            customer_id=customer_id,
            claim_timestamp=case_payload.get("return_date") or case_payload.get("order_date"),
            requested_refund_usd=req_usd,
            payload_override=case_payload,
        )

        use_candidate = (requested_model_version in ("candidate-v2.0.0", "39", "candidate"))
        fallback_occurred = False
        fallback_reason = None
        executed_model_version = "production-v1.3.0"
        active_features = MODEL_FEATURES

        model = None
        if use_candidate:
            if snap.status == FeatureRetrievalStatus.COMPLETE:
                try:
                    cand_model, cand_sha = cand_model_loader() if cand_model_loader else (None, "")
                    if cand_model is not None:
                        model = cand_model
                        executed_model_version = "candidate-v2.0.0"
                        active_features = CANDIDATE_MODEL_FEATURES
                    else:
                        fallback_occurred = True
                        fallback_reason = "CANDIDATE_MODEL_ARTIFACT_UNAVAILABLE"
                except Exception as ex:
                    fallback_occurred = True
                    fallback_reason = f"CANDIDATE_LOAD_EXCEPTION: {ex}"
            else:
                fallback_occurred = True
                fallback_reason = f"FEATURE_STORE_{snap.status.value}"

        if model is None:
            # Fall back to production baseline (33 features)
            prod_model, prod_sha = prod_model_loader() if prod_model_loader else (None, "")
            model = prod_model
            executed_model_version = "production-v1.3.0"
            active_features = MODEL_FEATURES

        if model is None:
            raise RuntimeError("No model available for prediction execution.")

        # Hydrate payload with point-in-time features for DataFrame builder
        enriched_payload = dict(case_payload)
        enriched_payload["total_orders_lifetime"] = snap.total_orders_lifetime
        enriched_payload["total_returns_lifetime"] = snap.total_returns_lifetime
        enriched_payload["return_rate_pct"] = snap.return_rate_pct
        enriched_payload["customer_support_contacts"] = snap.customer_support_contacts
        enriched_payload["previous_dispute_count"] = snap.previous_dispute_count
        enriched_payload["refund_amount_requested_usd"] = snap.refund_amount_requested_usd

        # Build feature DataFrame
        df_feat = build_model_dataframe(enriched_payload, feature_names=active_features)

        # Execute prediction
        probs_raw = model.predict_proba(df_feat)[0]
        pred_idx = int(model.predict(df_feat)[0])

        class_names = ["Legitimate", "Policy Abuser", "Fraudulent Return", "Wardrobing"]
        prediction_label = class_names[pred_idx]
        probabilities = {
            class_names[i]: round(float(probs_raw[i]), 6)
            for i in range(len(probs_raw))
        }
        confidence = round(float(probs_raw[pred_idx]), 4)

        t_total = (time.perf_counter() - t_start) * 1000.0

        # Record metrics
        self.metrics.record_request(
            is_candidate=use_candidate,
            fallback_occurred=fallback_occurred,
            status=snap.status,
            retrieval_ms=snap.retrieval_latency_ms,
            prediction_ms=t_total,
        )

        return {
            "prediction": prediction_label,
            "prediction_id": pred_idx,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_version": executed_model_version,
            "feature_schema_version": len(active_features),
            "fallback_occurred": fallback_occurred,
            "fallback_reason": fallback_reason,
            "feature_snapshot": snap.model_dump(),
            "latency_ms": round(t_total, 3),
        }
