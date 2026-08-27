"""
TrustLoop Responsibility Attribution Engine.

Calculates normalized 4-party responsibility distribution:
Customer, Seller, Courier, and Unknown.
Guaranteed invariant: customer + seller + courier + unknown == 100.
"""

from typing import Dict, Any, List, Optional
import math


def calculate_responsibility(
    case_data: Dict[str, Any],
    ml_probabilities: Optional[Dict[str, float]] = None,
    claim_truth_score: Optional[float] = None,
    vision_result: Optional[Dict[str, Any]] = None,
    rag_result: Optional[Dict[str, Any]] = None,
    disabled_signals: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate normalized responsibility attribution across Customer, Seller, Courier, and Unknown.
    Supports counterfactual calculation by filtering out disabled_signals.
    """
    disabled = set(disabled_signals or [])

    # Base raw weights
    raw_customer = 10.0
    raw_seller = 10.0
    raw_courier = 10.0
    raw_unknown = 10.0

    drivers: Dict[str, List[str]] = {
        "customer": [],
        "seller": [],
        "courier": [],
        "unknown": [],
    }

    # 1. Customer Signals
    return_rate = float(case_data.get("return_rate_pct", 10.0))
    days_to_return = float(case_data.get("days_to_return", 5.0))
    dispute_count = int(case_data.get("previous_dispute_count", 0))
    has_multiple_accounts = bool(case_data.get("has_multiple_accounts", False))
    tags_removed = bool(case_data.get("tags_removed", False))
    wardrobing_flag = bool(case_data.get("wardrobing_risk", False)) or tags_removed

    if "customer_history" not in disabled:
        if return_rate > 35.0:
            boost = min(40.0, (return_rate - 20.0) * 1.2)
            raw_customer += boost
            drivers["customer"].append(f"Elevated customer return rate ({return_rate:.1f}%)")
        if dispute_count >= 2:
            raw_customer += 15.0 * min(dispute_count, 3)
            drivers["customer"].append(f"Customer has {dispute_count} previous dispute records")

    if "identity_linkage" not in disabled and has_multiple_accounts:
        raw_customer += 35.0
        drivers["customer"].append("Customer identity linked to multiple return abuse accounts")

    if "wardrobing_signals" not in disabled and wardrobing_flag:
        raw_customer += 30.0
        drivers["customer"].append("Item condition indicates wear/use prior to return (wardrobing)")

    # 2. Courier Signals
    transit_delay_hours = float(case_data.get("transit_delay_hours", 0.0))
    package_damage_reported = bool(case_data.get("package_damage_reported", False))
    courier_incident_rate = float(case_data.get("courier_incident_rate", 2.0))
    weight_discrepancy_grams = float(case_data.get("weight_discrepancy_grams", 0.0))
    q_commerce = bool(case_data.get("is_q_commerce", False)) or str(case_data.get("platform_mode", "")).lower() == "q_commerce"

    if "courier_incident_history" not in disabled:
        if courier_incident_rate > 5.0:
            raw_courier += 25.0
            drivers["courier"].append(f"Courier hub has elevated transit damage rate ({courier_incident_rate:.1f}%)")

    if "packaging_damage" not in disabled and package_damage_reported:
        raw_courier += 45.0
        drivers["courier"].append("Severe outer transit box puncture/crush documented at delivery handover")

    if "transit_delay" not in disabled and transit_delay_hours > 48.0:
        raw_courier += 20.0
        drivers["courier"].append(f"Excessive transit delay ({transit_delay_hours:.0f}h) exceeding SLA window")

    if "weight_discrepancy" not in disabled and weight_discrepancy_grams > 150.0:
        raw_courier += 30.0
        drivers["courier"].append(f"Package weight loss during transit ({weight_discrepancy_grams:.0f}g discrepancy)")

    if q_commerce and "rider_handover" not in disabled:
        if case_data.get("seal_tampered_at_handover"):
            raw_courier += 40.0
            drivers["courier"].append("Tamper seal broken prior to rider doorstep handover")

    # 3. Seller Signals
    seller_defect_rate = float(case_data.get("seller_defect_rate", 1.5))
    incorrect_sku_dispatched = bool(case_data.get("incorrect_sku_dispatched", False))
    counterfeit_risk = bool(case_data.get("counterfeit_risk", False))
    seller_packaging_complaint = bool(case_data.get("seller_packaging_complaint", False))

    if "seller_defect_history" not in disabled:
        if seller_defect_rate > 8.0:
            raw_seller += 30.0
            drivers["seller"].append(f"Seller has high dispatch defect rate ({seller_defect_rate:.1f}%)")

    if "sku_mismatch" not in disabled and incorrect_sku_dispatched:
        raw_seller += 55.0
        drivers["seller"].append("Warehouse dispatch logs confirm wrong SKU/model packed by seller")

    if "counterfeit_evidence" not in disabled and counterfeit_risk:
        raw_seller += 50.0
        drivers["seller"].append("Brand authentication check flagged seller inventory as unverified/counterfeit")

    if "seller_packaging" not in disabled and seller_packaging_complaint:
        raw_seller += 25.0
        drivers["seller"].append("Inadequate inner bubble packaging by merchant failed transport guidelines")

    # 4. ML Tabular Adjustments
    if ml_probabilities and "ml_model_prediction" not in disabled:
        fraud_prob = ml_probabilities.get("Fraudulent Return", 0.0)
        policy_prob = ml_probabilities.get("Policy Abuser", 0.0)
        ward_prob = ml_probabilities.get("Wardrobing", 0.0)

        if fraud_prob > 0.60 or policy_prob > 0.60 or ward_prob > 0.60:
            raw_customer += (fraud_prob + policy_prob + ward_prob) * 35.0
            drivers["customer"].append("Statistical risk model assigned high probability of return abuse")

    # 5. RAG Policy & Vision Evidence adjustments
    if vision_result and "vision_evidence" not in disabled:
        if vision_result.get("is_counterfeit_suspect"):
            raw_seller += 30.0
            drivers["seller"].append("Computer vision detected labeling irregularities consistent with counterfeit goods")
        elif vision_result.get("is_transit_damage"):
            raw_courier += 25.0
            drivers["courier"].append("Visual inspection confirmed impact damage occurred during logistics handling")

    # Normalize to exactly 100%
    total_raw = raw_customer + raw_seller + raw_courier + raw_unknown
    if total_raw <= 0:
        total_raw = 1.0

    cust_pct = round((raw_customer / total_raw) * 100.0)
    sell_pct = round((raw_seller / total_raw) * 100.0)
    cour_pct = round((raw_courier / total_raw) * 100.0)
    unk_pct = 100 - (cust_pct + sell_pct + cour_pct)

    # In case rounding gave negative unknown, re-balance safely
    if unk_pct < 0:
        overage = abs(unk_pct)
        unk_pct = 0
        if cour_pct >= cust_pct and cour_pct >= sell_pct:
            cour_pct = max(0, cour_pct - overage)
        elif cust_pct >= sell_pct:
            cust_pct = max(0, cust_pct - overage)
        else:
            sell_pct = max(0, sell_pct - overage)

    responsibility = {
        "customer": cust_pct,
        "seller": sell_pct,
        "courier": cour_pct,
        "unknown": unk_pct,
    }

    # Determine dominant party
    dominant_party = max(list(responsibility.keys()), key=lambda k: responsibility.get(k, 0))
    if responsibility[dominant_party] < 35:
        dominant_party = "unknown"

    return {
        "responsibility": responsibility,
        "dominant_party": dominant_party,
        "drivers": {k: v for k, v in drivers.items() if v},
        "disabled_signals": list(disabled),
    }
