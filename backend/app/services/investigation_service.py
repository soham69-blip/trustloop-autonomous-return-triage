"""
TrustLoop Investigation Engine & Case Reconstructor.

Reconstructs the 8-stage case timeline, builds interactive multi-entity evidence graphs,
and provides 6 standardized judge-ready demonstration scenarios across E-Commerce and Q-Commerce.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta

from backend.app.services.responsibility_service import calculate_responsibility


DEMO_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "CASE-001",
        "title": "Severe Courier Transit Damage",
        "platform_mode": "e_commerce",
        "category": "Electronics",
        "product_name": "Sony WH-1000XM5 Wireless Headphones",
        "product_value_usd": 399.99,
        "refund_amount_requested_usd": 399.99,
        "claim_type": "Damaged in Transit",
        "customer": {
            "customer_id": "CUST-88219",
            "name": "Priya Sharma",
            "tier": "Gold Customer",
            "account_age_days": 620,
            "return_rate_pct": 4.2,
            "total_orders": 48,
            "total_returns": 2,
            "previous_disputes": 0,
        },
        "seller": {
            "seller_id": "SELL-1049",
            "name": "Apex Electronics Ltd",
            "rating": 4.8,
            "defect_rate_pct": 1.2,
            "packaging_audit_passed": True,
        },
        "courier": {
            "courier_id": "COUR-FAST-44",
            "name": "SwiftExpress Logistics",
            "hub_location": "Central Distribution Hub 4B",
            "incident_rate_pct": 8.4,
            "transit_delay_hours": 36.0,
            "damage_reported_at_handover": True,
        },
        "payload": {
            "case_id": "CASE-001",
            "order_value": 399.99,
            "refund_amount_requested_usd": 399.99,
            "return_rate_pct": 4.2,
            "days_to_return": 1.0,
            "previous_dispute_count": 0,
            "package_damage_reported": True,
            "courier_incident_rate": 8.4,
            "transit_delay_hours": 36.0,
            "weight_discrepancy_grams": 45.0,
            "return_reason": "damaged",
            "has_multiple_accounts": False,
            "is_q_commerce": False,
            "platform_mode": "e_commerce",
        },
        "expected_dominant_party": "courier",
        "expected_decision": "REFUND_AND_COURIER_INVESTIGATION",
    },
    {
        "case_id": "CASE-002",
        "title": "Coordinated Customer Return Abuse",
        "platform_mode": "e_commerce",
        "category": "Apparel & Luxury",
        "product_name": "Designer Cashmere Overcoat",
        "product_value_usd": 850.00,
        "refund_amount_requested_usd": 850.00,
        "claim_type": "Defective / Missing Parts",
        "customer": {
            "customer_id": "CUST-99014",
            "name": "Vikram Malhotra",
            "tier": "New Account",
            "account_age_days": 18,
            "return_rate_pct": 68.5,
            "total_orders": 6,
            "total_returns": 4,
            "previous_disputes": 3,
        },
        "seller": {
            "seller_id": "SELL-5510",
            "name": "Haute Couture Direct",
            "rating": 4.9,
            "defect_rate_pct": 0.8,
            "packaging_audit_passed": True,
        },
        "courier": {
            "courier_id": "COUR-FED-12",
            "name": "Metro Logistics",
            "hub_location": "North Transit Center",
            "incident_rate_pct": 1.5,
            "transit_delay_hours": 4.0,
            "damage_reported_at_handover": False,
        },
        "payload": {
            "case_id": "CASE-002",
            "order_value": 850.00,
            "refund_amount_requested_usd": 850.00,
            "return_rate_pct": 68.5,
            "days_to_return": 14.0,
            "previous_dispute_count": 3,
            "tags_removed": True,
            "has_multiple_accounts": True,
            "wardrobing_risk": True,
            "package_damage_reported": False,
            "courier_incident_rate": 1.5,
            "return_reason": "defective",
            "is_q_commerce": False,
            "platform_mode": "e_commerce",
        },
        "expected_dominant_party": "customer",
        "expected_decision": "AUTO_REJECT",
    },
    {
        "case_id": "CASE-003",
        "title": "Merchant Wrong SKU / Counterfeit Item",
        "platform_mode": "e_commerce",
        "category": "Luxury Cosmetics",
        "product_name": "Premium Anti-Aging Serum 50ml",
        "product_value_usd": 220.00,
        "refund_amount_requested_usd": 220.00,
        "claim_type": "Wrong Item Dispatched",
        "customer": {
            "customer_id": "CUST-33102",
            "name": "Ananya Desai",
            "tier": "Platinum Customer",
            "account_age_days": 1100,
            "return_rate_pct": 2.1,
            "total_orders": 94,
            "total_returns": 2,
            "previous_disputes": 0,
        },
        "seller": {
            "seller_id": "SELL-7788",
            "name": "Discount Glow Store",
            "rating": 3.4,
            "defect_rate_pct": 14.6,
            "packaging_audit_passed": False,
        },
        "courier": {
            "courier_id": "COUR-EXP-09",
            "name": "Express Couriers",
            "hub_location": "South Hub",
            "incident_rate_pct": 1.2,
            "transit_delay_hours": 6.0,
            "damage_reported_at_handover": False,
        },
        "payload": {
            "case_id": "CASE-003",
            "order_value": 220.00,
            "refund_amount_requested_usd": 220.00,
            "return_rate_pct": 2.1,
            "days_to_return": 2.0,
            "previous_dispute_count": 0,
            "incorrect_sku_dispatched": True,
            "seller_defect_rate": 14.6,
            "counterfeit_risk": True,
            "package_damage_reported": False,
            "return_reason": "wrong_item",
            "is_q_commerce": False,
            "platform_mode": "e_commerce",
        },
        "expected_dominant_party": "seller",
        "expected_decision": "REFUND_AND_SELLER_INVESTIGATION",
    },
    {
        "case_id": "CASE-004",
        "title": "Genuine Low-Risk Customer Return",
        "platform_mode": "e_commerce",
        "category": "Home & Kitchen",
        "product_name": "Stainless Steel Espresso Maker",
        "product_value_usd": 65.00,
        "refund_amount_requested_usd": 65.00,
        "claim_type": "Unopened Return within Policy",
        "customer": {
            "customer_id": "CUST-10492",
            "name": "Rahul Verma",
            "tier": "Regular Customer",
            "account_age_days": 450,
            "return_rate_pct": 5.0,
            "total_orders": 20,
            "total_returns": 1,
            "previous_disputes": 0,
        },
        "seller": {
            "seller_id": "SELL-2001",
            "name": "KitchenCraft Official",
            "rating": 4.9,
            "defect_rate_pct": 0.5,
            "packaging_audit_passed": True,
        },
        "courier": {
            "courier_id": "COUR-STD-01",
            "name": "Standard Post Logistics",
            "hub_location": "West Distribution",
            "incident_rate_pct": 1.1,
            "transit_delay_hours": 2.0,
            "damage_reported_at_handover": False,
        },
        "payload": {
            "case_id": "CASE-004",
            "order_value": 65.00,
            "refund_amount_requested_usd": 65.00,
            "return_rate_pct": 5.0,
            "days_to_return": 3.0,
            "previous_dispute_count": 0,
            "package_damage_reported": False,
            "return_reason": "unwanted",
            "is_q_commerce": False,
            "platform_mode": "e_commerce",
        },
        "expected_dominant_party": "unknown",
        "expected_decision": "AUTO_APPROVE",
    },
    {
        "case_id": "CASE-005",
        "title": "Conflicting Evidence / Human Escalation",
        "platform_mode": "e_commerce",
        "category": "Smartphones",
        "product_name": "Apple iPhone 15 Pro 256GB",
        "product_value_usd": 1199.00,
        "refund_amount_requested_usd": 1199.00,
        "claim_type": "Empty Box Received",
        "customer": {
            "customer_id": "CUST-66291",
            "name": "Sneha Sen",
            "tier": "Silver Customer",
            "account_age_days": 320,
            "return_rate_pct": 14.0,
            "total_orders": 14,
            "total_returns": 2,
            "previous_disputes": 1,
        },
        "seller": {
            "seller_id": "SELL-4402",
            "name": "SmartGadgets Hub",
            "rating": 4.6,
            "defect_rate_pct": 3.2,
            "packaging_audit_passed": True,
        },
        "courier": {
            "courier_id": "COUR-EXP-88",
            "name": "Velocity Air Logistics",
            "hub_location": "Airport Transit Hub",
            "incident_rate_pct": 4.5,
            "transit_delay_hours": 18.0,
            "damage_reported_at_handover": False,
        },
        "payload": {
            "case_id": "CASE-005",
            "order_value": 1199.00,
            "refund_amount_requested_usd": 1199.00,
            "return_rate_pct": 14.0,
            "days_to_return": 1.0,
            "previous_dispute_count": 1,
            "weight_discrepancy_grams": 240.0,
            "package_damage_reported": False,
            "return_reason": "missing_item",
            "is_q_commerce": False,
            "platform_mode": "e_commerce",
        },
        "expected_dominant_party": "unknown",
        "expected_decision": "ESCALATE",
    },
    {
        "case_id": "CASE-006",
        "title": "Quick-Commerce Broken Seal & Spoiled Cargo",
        "platform_mode": "q_commerce",
        "category": "Fresh Groceries & Ice Cream",
        "product_name": "Artisanal Gelato & Organic Produce Kit",
        "product_value_usd": 48.50,
        "refund_amount_requested_usd": 48.50,
        "claim_type": "Rider Delivery Seal Broken & Melted",
        "customer": {
            "customer_id": "CUST-QC-9182",
            "name": "Aditya Rao",
            "tier": "Frequent 10-Min Shopper",
            "account_age_days": 190,
            "return_rate_pct": 3.5,
            "total_orders": 82,
            "total_returns": 3,
            "previous_disputes": 0,
        },
        "seller": {
            "seller_id": "DARK-STORE-12",
            "name": "Zepto/Blinkit Dark Store 12",
            "rating": 4.8,
            "defect_rate_pct": 0.9,
            "packaging_audit_passed": True,
        },
        "courier": {
            "courier_id": "RIDER-9921",
            "name": "Rider Fleet ID #9921",
            "hub_location": "Sector 4 Micro-warehouse",
            "incident_rate_pct": 9.2,
            "transit_delay_hours": 1.5,
            "damage_reported_at_handover": True,
        },
        "payload": {
            "case_id": "CASE-006",
            "order_value": 48.50,
            "refund_amount_requested_usd": 48.50,
            "return_rate_pct": 3.5,
            "days_to_return": 0.1,
            "previous_dispute_count": 0,
            "seal_tampered_at_handover": True,
            "transit_delay_hours": 1.5,
            "courier_incident_rate": 9.2,
            "package_damage_reported": True,
            "return_reason": "spoilage",
            "is_q_commerce": True,
            "platform_mode": "q_commerce",
        },
        "expected_dominant_party": "courier",
        "expected_decision": "REFUND_AND_COURIER_INVESTIGATION",
    },
]


def reconstruct_timeline(case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Reconstruct the chronological 8-stage case timeline.
    """
    now = datetime.now(timezone.utc)
    base_time = now - timedelta(days=float(case_data.get("days_to_return", 3.0)) + 2.0)

    t0 = base_time
    t1 = t0 + timedelta(hours=4)
    t2 = t1 + timedelta(hours=12)
    t3 = t2 + timedelta(hours=float(case_data.get("transit_delay_hours", 6.0)))
    t4 = t3 + timedelta(hours=float(case_data.get("days_to_return", 1.0)) * 24)
    t5 = t4 + timedelta(hours=1)
    t6 = t5 + timedelta(minutes=15)
    t7 = t6 + timedelta(seconds=45)

    has_damage = bool(case_data.get("package_damage_reported", False))
    has_sku_err = bool(case_data.get("incorrect_sku_dispatched", False))
    q_mode = bool(case_data.get("is_q_commerce", False)) or str(case_data.get("platform_mode", "")) == "q_commerce"

    return [
        {
            "step": 1,
            "stage": "Seller Packed Item",
            "timestamp": t0.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "actor": "Merchant Warehouse",
            "status": "FLAGGED" if has_sku_err else "VERIFIED",
            "detail": "SKU mismatch flagged in dispatch barcode scan" if has_sku_err else "Order picked, packed and tamper seal applied under CCTV inspection",
            "icon": "box",
        },
        {
            "step": 2,
            "stage": "Logistics Handover",
            "timestamp": t1.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "actor": "Carrier Manifest",
            "status": "VERIFIED",
            "detail": "Package inducted at logistics hub. Outbound weight: verified matching manifest",
            "icon": "truck",
        },
        {
            "step": 3,
            "stage": "In-Transit Route",
            "timestamp": t2.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "actor": "Logistics Hub 4B",
            "status": "INCIDENT" if has_damage else "NORMAL",
            "detail": "Logistics sortation conveyer reported belt jam and package compression" if has_damage else "Transit milestone reached within normal SLA schedule",
            "icon": "map-pin",
        },
        {
            "step": 4,
            "stage": "Customer Delivery",
            "timestamp": t3.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "actor": "Doorstep Handover" if q_mode else "Courier Delivery Agent",
            "status": "COMPLETED",
            "detail": "Geo-verified delivery OTP scan confirmed recipient possession",
            "icon": "home",
        },
        {
            "step": 5,
            "stage": "Return Claim Filed",
            "timestamp": t4.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "actor": "Customer Portal",
            "status": "FILED",
            "detail": f"Claim reason: '{case_data.get('return_reason', 'damaged')}' submitted by customer",
            "icon": "alert-circle",
        },
        {
            "step": 6,
            "stage": "Evidence Ingestion",
            "timestamp": t5.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "actor": "Vision & Claim Ingestion",
            "status": "ANALYZED",
            "detail": "High-resolution photos, metadata, and carrier logs ingested into TrustLoop engine",
            "icon": "camera",
        },
        {
            "step": 7,
            "stage": "Autonomous Investigation",
            "timestamp": t6.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "actor": "TrustLoop Intelligence Fusion",
            "status": "SYNTHESIZED",
            "detail": "Cross-referenced Tabular ML risk, RAG policy clauses, and carrier telematics",
            "icon": "cpu",
        },
        {
            "step": 8,
            "stage": "Decision Recommendation",
            "timestamp": t7.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "actor": "Decision Engine",
            "status": "FINALIZED",
            "detail": "Responsibility attributed with multi-party evidence audit trail",
            "icon": "check-circle",
        },
    ]


def build_evidence_graph(case_data: Dict[str, Any], responsibility_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build multi-entity interactive evidence graph for Case Room visualization.
    """
    case_id = str(case_data.get("case_id", "CASE-001"))
    cust_dict = case_data.get("customer") or {}
    seller_dict = case_data.get("seller") or {}
    courier_dict = case_data.get("courier") or {}

    cust_id = str(cust_dict.get("customer_id", f"CUST-{case_id[-4:]}"))
    seller_id = str(seller_dict.get("seller_id", "SELL-1049"))
    courier_id = str(courier_dict.get("courier_id", "COUR-FAST-44"))
    product_name = str(case_data.get("product_name", "Apparel & Electronics Item"))

    val = case_data.get("product_value_usd") or case_data.get("order_value") or 150.0

    resp = responsibility_data.get("responsibility", {"customer": 25, "seller": 25, "courier": 25, "unknown": 25})

    nodes = [
        {"id": "case_root", "label": f"Return Case: {case_id}", "type": "case", "status": "active", "risk": "evaluated"},
        {"id": "node_customer", "label": f"Customer: {cust_id}", "type": "customer", "score": f"{resp.get('customer', 0)}% responsibility", "return_rate": f"{case_data.get('return_rate_pct', 5.0)}%"},
        {"id": "node_seller", "label": f"Seller: {seller_id}", "type": "seller", "score": f"{resp.get('seller', 0)}% responsibility", "defect_rate": f"{case_data.get('seller_defect_rate', 1.2)}%"},
        {"id": "node_courier", "label": f"Courier: {courier_id}", "type": "courier", "score": f"{resp.get('courier', 0)}% responsibility", "transit_delay": f"{case_data.get('transit_delay_hours', 0)}h"},
        {"id": "node_product", "label": f"Product: {product_name[:24]}", "type": "product", "value": f"${float(val):.2f}"},
        {"id": "node_evidence_photo", "label": "Evidence: Doorstep Photos", "type": "evidence", "verified": True},
        {"id": "node_telemetry", "label": "Carrier Hub Telematics", "type": "telemetry", "damage_flag": bool(case_data.get("package_damage_reported"))},
        {"id": "node_policy", "label": "Policy: Section 4.2 Transit Loss", "type": "policy", "relevance": "94%"},
    ]

    edges = [
        {"source": "node_customer", "target": "case_root", "label": "filed claim for", "weight": resp["customer"]},
        {"source": "case_root", "target": "node_product", "label": "pertains to", "weight": 50},
        {"source": "node_seller", "target": "node_product", "label": "dispatched", "weight": resp["seller"]},
        {"source": "node_courier", "target": "case_root", "label": "transported", "weight": resp["courier"]},
        {"source": "node_evidence_photo", "target": "case_root", "label": "substantiates", "weight": 80},
        {"source": "node_telemetry", "target": "node_courier", "label": "logged at hub", "weight": 70},
        {"source": "node_policy", "target": "case_root", "label": "governs decision", "weight": 90},
    ]

    return {
        "nodes": nodes,
        "edges": edges,
    }


def get_demo_cases() -> List[Dict[str, Any]]:
    """
    List all pre-configured judge-ready demonstration scenarios.
    """
    return DEMO_CASES


def get_demo_case_by_id(case_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve full configuration for a specific demo case ID.
    """
    for c in DEMO_CASES:
        if c["case_id"].upper() == case_id.upper():
            return c
    return None
