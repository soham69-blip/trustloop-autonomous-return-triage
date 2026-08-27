"""
TrustLoop Fraud Network & Identity Linkage Engine.

Identifies coordinated abuse clusters, shared device fingerprints, payment token collisions,
and cross-account linkage graphs.
"""

from typing import Dict, Any, List


def get_fraud_network_graph() -> Dict[str, Any]:
    """
    Generate graph nodes and edges representing detected identity linkage and coordinated return fraud rings.
    """
    nodes = [
        {"id": "c_99014", "label": "Vikram Malhotra (CUST-99014)", "type": "customer", "risk_level": "HIGH", "return_rate": "68.5%"},
        {"id": "c_99015", "label": "V. M. Retail (CUST-99015)", "type": "customer", "risk_level": "HIGH", "return_rate": "72.0%"},
        {"id": "c_99018", "label": "Aarav K. (CUST-99018)", "type": "customer", "risk_level": "HIGH", "return_rate": "81.4%"},
        {"id": "dev_d8f9", "label": "Device Hash: #D8F9E2A0", "type": "device", "risk_level": "CRITICAL", "shared_accounts": 3},
        {"id": "ip_vpn", "label": "Exit Node IP: 185.220.101.4", "type": "ip", "risk_level": "SUSPICIOUS", "provider": "Commercial Proxy"},
        {"id": "card_tok_77", "label": "Card Token: *************4481", "type": "payment", "risk_level": "FLAGGED", "issuing_bank": "Neobank Intl"},
        {"id": "addr_ind", "label": "Address: 44 Industrial Way Ste 12", "type": "address", "risk_level": "CRITICAL", "total_claims": 14},
        {"id": "c_88219", "label": "Priya Sharma (CUST-88219)", "type": "customer", "risk_level": "LOW", "return_rate": "4.2%"},
    ]

    edges = [
        {"source": "c_99014", "target": "dev_d8f9", "label": "shared device fingerprint", "weight": 95},
        {"source": "c_99015", "target": "dev_d8f9", "label": "shared device fingerprint", "weight": 95},
        {"source": "c_99018", "target": "dev_d8f9", "label": "shared device fingerprint", "weight": 95},
        {"source": "c_99014", "target": "card_tok_77", "label": "shared payment token", "weight": 90},
        {"source": "c_99015", "target": "card_tok_77", "label": "shared payment token", "weight": 90},
        {"source": "c_99014", "target": "addr_ind", "label": "common delivery drop", "weight": 85},
        {"source": "c_99018", "target": "addr_ind", "label": "common delivery drop", "weight": 85},
        {"source": "c_99015", "target": "ip_vpn", "label": "proxy connection", "weight": 70},
        {"source": "c_99018", "target": "ip_vpn", "label": "proxy connection", "weight": 70},
    ]

    clusters = [
        {
            "cluster_id": "CLUSTER-RING-4B",
            "cluster_name": "Metro Device Collision Ring",
            "severity": "CRITICAL",
            "member_count": 3,
            "total_disputed_value_usd": 4250.00,
            "connected_customers": ["CUST-99014", "CUST-99015", "CUST-99018"],
            "primary_link": "Device Hash #D8F9E2A0 + Shared Virtual Card Token",
            "summary": "3 recently registered customer accounts sharing identical browser hardware canvas fingerprints and cycling luxury apparel return claims.",
        }
    ]

    return {
        "status": "FRAUD_NETWORK_ANALYZED",
        "data_classification": "SYNTHETIC_DEMO_FIXTURE",
        "source": "Development fixture; not production fraud intelligence",
        "total_entities": len(nodes),
        "total_linkages": len(edges),
        "active_clusters": clusters,
        "nodes": nodes,
        "edges": edges,
    }
