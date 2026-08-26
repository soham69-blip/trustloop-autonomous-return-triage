"""
TrustLoop Expanded 21-Case Adversarial & Hard-Case Evaluation Benchmark.
Covers multi-class business edge cases, multimodal dependencies, and schema defenses.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np

from backend.app.ml_feature_builder import (
    build_model_dataframe,
    MODEL_FEATURES,
    CANDIDATE_MODEL_FEATURES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLASS_NAMES = ["Legitimate", "Policy Abuser", "Fraudulent Return", "Wardrobing"]

HARD_CASES: List[Dict[str, Any]] = [
    {
        "id": "HC-01",
        "name": "1. Legitimate high-frequency shopper (4 returns, 20% return rate)",
        "payload": {
            "age": 35, "account_age_days": 800, "customer_segment": "Gold", "country": "US",
            "platform": "Web Browser", "device_type": "Windows PC", "payment_method": "Credit Card",
            "product_category": "Clothing", "avg_order_value_usd": 120.0, "is_high_value_item": 0,
            "discount_used": 0, "days_to_return": 7.0, "return_reason": "Too Small", "shipping_carrier": "FedEx",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 24.0, "customer_return_count_prior": 4,
            "returns_last_30d_prior": 2, "returns_last_90d_prior": 3, "total_returns_lifetime_prior": 4,
            "order_date": "2026-06-01", "return_date": "2026-06-08", "total_orders_lifetime": 20,
            "total_returns_lifetime": 4, "return_rate_pct": 20.0, "customer_support_contacts": 1,
            "previous_dispute_count": 0, "refund_amount_requested_usd": 120.0
        },
        "expected_business_outcome": "APPROVE_REFUND",
        "expected_ml_behavior": "Legitimate",
        "responsible_subsystem": "ML (Tabular)",
        "rationale": "High lifetime order volume with normal apparel sizing return rate."
    },
    {
        "id": "HC-02",
        "name": "2. Borderline Policy Abuser with 22% return rate & prior dispute",
        "payload": {
            "age": 28, "account_age_days": 200, "customer_segment": "Silver", "country": "US",
            "platform": "Mobile App", "device_type": "iPhone", "payment_method": "Credit Card",
            "product_category": "Shoes", "avg_order_value_usd": 90.0, "is_high_value_item": 0,
            "discount_used": 1, "days_to_return": 18.0, "return_reason": "Changed Mind", "shipping_carrier": "UPS",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 1.0, "customer_return_count_prior": 3,
            "returns_last_30d_prior": 2, "returns_last_90d_prior": 3, "total_returns_lifetime_prior": 5,
            "order_date": "2026-06-01", "return_date": "2026-06-19", "total_orders_lifetime": 10,
            "total_returns_lifetime": 3, "return_rate_pct": 22.0, "customer_support_contacts": 2,
            "previous_dispute_count": 1, "refund_amount_requested_usd": 90.0
        },
        "expected_business_outcome": "FLAG_POLICY_ABUSE",
        "expected_ml_behavior": "Policy Abuser",
        "responsible_subsystem": "ML + Candidate Lifetime Features",
        "rationale": "Elevated return rate with repeat dispute history requires policy abuser classification."
    },
    {
        "id": "HC-03",
        "name": "3. Fraudulent return with multi-account collision & instant return",
        "payload": {
            "age": 22, "account_age_days": 15, "customer_segment": "New", "country": "US",
            "platform": "Web Browser", "device_type": "Windows PC", "payment_method": "Buy Now Pay Later",
            "product_category": "Electronics", "avg_order_value_usd": 150.0, "is_high_value_item": 0,
            "discount_used": 1, "days_to_return": 1.0, "return_reason": "Defective/Broken", "shipping_carrier": "DHL",
            "multiple_accounts_flag": 1, "wishlist_to_cart_time_hrs": 0.1, "customer_return_count_prior": 0,
            "returns_last_30d_prior": 0, "returns_last_90d_prior": 0, "total_returns_lifetime_prior": 0,
            "order_date": "2026-06-01", "return_date": "2026-06-02", "total_orders_lifetime": 1,
            "total_returns_lifetime": 1, "return_rate_pct": 100.0, "customer_support_contacts": 3,
            "previous_dispute_count": 0, "refund_amount_requested_usd": 150.0
        },
        "expected_business_outcome": "REJECT_AND_INVESTIGATE_FRAUD",
        "expected_ml_behavior": "Fraudulent Return",
        "responsible_subsystem": "ML (Tabular) + Fraud Graph",
        "rationale": "Synthetic account creation and immediate refund extraction."
    },
    {
        "id": "HC-04",
        "name": "4. Wardrobing with 12-day return window on apparel",
        "payload": {
            "age": 40, "account_age_days": 600, "customer_segment": "Gold", "country": "US",
            "platform": "Mobile App", "device_type": "Android", "payment_method": "Buy Now Pay Later",
            "product_category": "Clothing", "avg_order_value_usd": 320.0, "is_high_value_item": 1,
            "discount_used": 1, "days_to_return": 12.0, "return_reason": "Changed Mind", "shipping_carrier": "OnTrac",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 1.5, "customer_return_count_prior": 3,
            "returns_last_30d_prior": 1, "returns_last_90d_prior": 2, "total_returns_lifetime_prior": 4,
            "order_date": "2026-06-01", "return_date": "2026-06-13", "total_orders_lifetime": 15,
            "total_returns_lifetime": 4, "return_rate_pct": 26.6, "customer_support_contacts": 2,
            "previous_dispute_count": 1, "refund_amount_requested_usd": 320.0
        },
        "expected_business_outcome": "FLAG_WARDROBING_INSPECTION",
        "expected_ml_behavior": "Wardrobing",
        "responsible_subsystem": "ML + Vision Inspection",
        "rationale": "High-value apparel returned right after event weekend."
    },
    {
        "id": "HC-05",
        "name": "5. Legitimate luxury item ($600 high value, returned in 5 days)",
        "payload": {
            "age": 48, "account_age_days": 1500, "customer_segment": "Platinum", "country": "US",
            "platform": "Web Browser", "device_type": "MacBook", "payment_method": "Credit Card",
            "product_category": "Jewelry", "avg_order_value_usd": 600.0, "is_high_value_item": 1,
            "discount_used": 0, "days_to_return": 5.0, "return_reason": "Not As Described", "shipping_carrier": "FedEx",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 48.0, "customer_return_count_prior": 1,
            "returns_last_30d_prior": 0, "returns_last_90d_prior": 1, "total_returns_lifetime_prior": 1,
            "order_date": "2026-06-01", "return_date": "2026-06-06", "total_orders_lifetime": 30,
            "total_returns_lifetime": 1, "return_rate_pct": 3.3, "customer_support_contacts": 0,
            "previous_dispute_count": 0, "refund_amount_requested_usd": 600.0
        },
        "expected_business_outcome": "APPROVE_VIP_REFUND",
        "expected_ml_behavior": "Legitimate",
        "responsible_subsystem": "ML (Tabular)",
        "rationale": "Platinum customer with 3.3% lifetime return rate."
    },
    {
        "id": "HC-06",
        "name": "6. Strong Policy Abuser (70% return rate, 8 prior disputes)",
        "payload": {
            "age": 31, "account_age_days": 180, "customer_segment": "Bronze", "country": "US",
            "platform": "Mobile App", "device_type": "Android", "payment_method": "Credit Card",
            "product_category": "Electronics", "avg_order_value_usd": 220.0, "is_high_value_item": 0,
            "discount_used": 1, "days_to_return": 25.0, "return_reason": "Changed Mind", "shipping_carrier": "UPS",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 0.5, "customer_return_count_prior": 8,
            "returns_last_30d_prior": 4, "returns_last_90d_prior": 6, "total_returns_lifetime_prior": 10,
            "order_date": "2026-06-01", "return_date": "2026-06-26", "total_orders_lifetime": 12,
            "total_returns_lifetime": 9, "return_rate_pct": 75.0, "customer_support_contacts": 6,
            "previous_dispute_count": 8, "refund_amount_requested_usd": 220.0
        },
        "expected_business_outcome": "ENFORCE_RESTOCKING_FEE",
        "expected_ml_behavior": "Policy Abuser",
        "responsible_subsystem": "ML (Tabular)",
        "rationale": "Chronic serial returner with 8 dispute escalations."
    },
    {
        "id": "HC-07",
        "name": "7. Rapid turnaround return (Delivered & returned in 0.5 days)",
        "payload": {
            "age": 29, "account_age_days": 400, "customer_segment": "Silver", "country": "CA",
            "platform": "Web Browser", "device_type": "Windows PC", "payment_method": "PayPal",
            "product_category": "Books", "avg_order_value_usd": 45.0, "is_high_value_item": 0,
            "discount_used": 0, "days_to_return": 0.5, "return_reason": "Wrong Item Sent", "shipping_carrier": "FedEx",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 12.0, "customer_return_count_prior": 1,
            "returns_last_30d_prior": 0, "returns_last_90d_prior": 0, "total_returns_lifetime_prior": 1,
            "order_date": "2026-06-01", "return_date": "2026-06-01", "total_orders_lifetime": 8,
            "total_returns_lifetime": 1, "return_rate_pct": 12.5, "customer_support_contacts": 1,
            "previous_dispute_count": 0, "refund_amount_requested_usd": 45.0
        },
        "expected_business_outcome": "APPROVE_REFUND_SELLER_ERROR",
        "expected_ml_behavior": "Legitimate",
        "responsible_subsystem": "ML (Tabular)",
        "rationale": "Merchant wrong SKU error handled immediately by legitimate customer."
    },
    {
        "id": "HC-08",
        "name": "8. Out-of-policy late return window (48 days to return)",
        "payload": {
            "age": 42, "account_age_days": 350, "customer_segment": "Silver", "country": "GB",
            "platform": "Mobile App", "device_type": "iPhone", "payment_method": "Debit Card",
            "product_category": "Home & Kitchen", "avg_order_value_usd": 180.0, "is_high_value_item": 0,
            "discount_used": 0, "days_to_return": 48.0, "return_reason": "Changed Mind", "shipping_carrier": "USPS",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 3.0, "customer_return_count_prior": 2,
            "returns_last_30d_prior": 0, "returns_last_90d_prior": 1, "total_returns_lifetime_prior": 3,
            "order_date": "2026-04-01", "return_date": "2026-05-19", "total_orders_lifetime": 6,
            "total_returns_lifetime": 3, "return_rate_pct": 50.0, "customer_support_contacts": 3,
            "previous_dispute_count": 2, "refund_amount_requested_usd": 180.0
        },
        "expected_business_outcome": "REJECT_LATE_POLICY_RETURN",
        "expected_ml_behavior": "Policy Abuser",
        "responsible_subsystem": "RAG Policy Layer + ML",
        "rationale": "Return window exceeded standard 30-day policy."
    },
    {
        "id": "HC-09",
        "name": "9. Chronic Chargeback / Dispute Ring (6 prior chargebacks)",
        "payload": {
            "age": 26, "account_age_days": 90, "customer_segment": "New", "country": "US",
            "platform": "Mobile App", "device_type": "Android", "payment_method": "Credit Card",
            "product_category": "Electronics", "avg_order_value_usd": 380.0, "is_high_value_item": 1,
            "discount_used": 1, "days_to_return": 3.0, "return_reason": "Item Not Received", "shipping_carrier": "FedEx",
            "multiple_accounts_flag": 1, "wishlist_to_cart_time_hrs": 0.2, "customer_return_count_prior": 5,
            "returns_last_30d_prior": 3, "returns_last_90d_prior": 5, "total_returns_lifetime_prior": 6,
            "order_date": "2026-06-01", "return_date": "2026-06-04", "total_orders_lifetime": 6,
            "total_returns_lifetime": 6, "return_rate_pct": 100.0, "customer_support_contacts": 5,
            "previous_dispute_count": 6, "refund_amount_requested_usd": 380.0
        },
        "expected_business_outcome": "BLOCK_ACCOUNT_FRAUD_RING",
        "expected_ml_behavior": "Fraudulent Return",
        "responsible_subsystem": "ML + Fraud Graph",
        "rationale": "Dispute abuse across synthetic identity ring."
    },
    {
        "id": "HC-10",
        "name": "10. Aggressive escalation abuser (14 support tickets, 4 returns in 30d)",
        "payload": {
            "age": 33, "account_age_days": 210, "customer_segment": "Silver", "country": "US",
            "platform": "Web Browser", "device_type": "Windows PC", "payment_method": "PayPal",
            "product_category": "Electronics", "avg_order_value_usd": 210.0, "is_high_value_item": 0,
            "discount_used": 1, "days_to_return": 20.0, "return_reason": "Not As Described", "shipping_carrier": "UPS",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 1.0, "customer_return_count_prior": 6,
            "returns_last_30d_prior": 4, "returns_last_90d_prior": 5, "total_returns_lifetime_prior": 7,
            "order_date": "2026-06-01", "return_date": "2026-06-21", "total_orders_lifetime": 10,
            "total_returns_lifetime": 7, "return_rate_pct": 70.0, "customer_support_contacts": 14,
            "previous_dispute_count": 3, "refund_amount_requested_usd": 210.0
        },
        "expected_business_outcome": "REQUIRE_MANAGER_ESCALATION",
        "expected_ml_behavior": "Policy Abuser",
        "responsible_subsystem": "ML (Tabular)",
        "rationale": "High contact frequency combined with continuous return velocity."
    },
    {
        "id": "HC-11",
        "name": "11. High return-rate habitual returner (90% return rate on 15 items)",
        "payload": {
            "age": 38, "account_age_days": 500, "customer_segment": "Bronze", "country": "US",
            "platform": "Mobile App", "device_type": "iPhone", "payment_method": "Credit Card",
            "product_category": "Clothing", "avg_order_value_usd": 130.0, "is_high_value_item": 0,
            "discount_used": 1, "days_to_return": 16.0, "return_reason": "Too Small", "shipping_carrier": "USPS",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 2.0, "customer_return_count_prior": 9,
            "returns_last_30d_prior": 4, "returns_last_90d_prior": 7, "total_returns_lifetime_prior": 13,
            "order_date": "2026-06-01", "return_date": "2026-06-17", "total_orders_lifetime": 15,
            "total_returns_lifetime": 13, "return_rate_pct": 86.7, "customer_support_contacts": 4,
            "previous_dispute_count": 2, "refund_amount_requested_usd": 130.0
        },
        "expected_business_outcome": "REQUIRE_RETURN_FEE",
        "expected_ml_behavior": "Policy Abuser",
        "responsible_subsystem": "Candidate ML (39 feats) / Threshold Opt",
        "rationale": "Habitual over-purchaser abusing return policy."
    },
    {
        "id": "HC-12",
        "name": "12. Conflicting Signals (High value $800, verified merchant packaging)",
        "payload": {
            "age": 52, "account_age_days": 1200, "customer_segment": "Gold", "country": "US",
            "platform": "Web Browser", "device_type": "MacBook", "payment_method": "Credit Card",
            "product_category": "Electronics", "avg_order_value_usd": 800.0, "is_high_value_item": 1,
            "discount_used": 0, "days_to_return": 4.0, "return_reason": "Quality Issue", "shipping_carrier": "FedEx",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 18.0, "customer_return_count_prior": 2,
            "returns_last_30d_prior": 0, "returns_last_90d_prior": 1, "total_returns_lifetime_prior": 2,
            "order_date": "2026-06-01", "return_date": "2026-06-05", "total_orders_lifetime": 25,
            "total_returns_lifetime": 2, "return_rate_pct": 8.0, "customer_support_contacts": 1,
            "previous_dispute_count": 0, "refund_amount_requested_usd": 800.0
        },
        "expected_business_outcome": "APPROVE_REFUND_COURIER_CHECK",
        "expected_ml_behavior": "Legitimate",
        "responsible_subsystem": "ML + Vision",
        "rationale": "High value with low return rate indicates genuine issue."
    },
    {
        "id": "HC-13",
        "name": "13. Missing optional telemetry fields (Sparse intake)",
        "payload": {
            "age": 30, "account_age_days": 100, "customer_segment": "New", "country": "US",
            "platform": "Web Browser", "device_type": "Windows PC", "payment_method": "Credit Card",
            "product_category": "Clothing", "avg_order_value_usd": 75.0, "is_high_value_item": 0,
            "discount_used": 0, "days_to_return": 7.0, "return_reason": "Changed Mind", "shipping_carrier": "UPS",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 0.0, "customer_return_count_prior": 0,
            "returns_last_30d_prior": 0, "returns_last_90d_prior": 0, "total_returns_lifetime_prior": 0,
            "order_date": "2026-06-01", "return_date": "2026-06-08",
        },
        "expected_business_outcome": "APPROVE_STANDARD_REFUND",
        "expected_ml_behavior": "Legitimate",
        "responsible_subsystem": "ML (Robustness)",
        "rationale": "Robust handling of missing optional fields."
    },
    {
        "id": "HC-14",
        "name": "14. Extreme Outlier Values (Order $12,500, Account age 1,500 days)",
        "payload": {
            "age": 65, "account_age_days": 1500, "customer_segment": "Platinum", "country": "US",
            "platform": "Web Browser", "device_type": "MacBook", "payment_method": "Credit Card",
            "product_category": "Jewelry", "avg_order_value_usd": 12500.0, "is_high_value_item": 1,
            "discount_used": 0, "days_to_return": 3.0, "return_reason": "Not As Described", "shipping_carrier": "FedEx",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 72.0, "customer_return_count_prior": 0,
            "returns_last_30d_prior": 0, "returns_last_90d_prior": 0, "total_returns_lifetime_prior": 0,
            "order_date": "2026-06-01", "return_date": "2026-06-04",
        },
        "expected_business_outcome": "CONCIERGE_HANDLING",
        "expected_ml_behavior": "Legitimate",
        "responsible_subsystem": "ML (Robustness)",
        "rationale": "High value without runaway risk scores."
    },
    {
        "id": "HC-15",
        "name": "15. Novel / Unseen Categorical Encodings (Platform: VR_Headset, Carrier: DroneX)",
        "payload": {
            "age": 27, "account_age_days": 400, "customer_segment": "Silver", "country": "US",
            "platform": "VR_Headset", "device_type": "Windows PC", "payment_method": "Crypto",
            "product_category": "Electronics", "avg_order_value_usd": 299.0, "is_high_value_item": 0,
            "discount_used": 1, "days_to_return": 6.0, "return_reason": "Changed Mind", "shipping_carrier": "DroneX",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 4.0, "customer_return_count_prior": 1,
            "returns_last_30d_prior": 0, "returns_last_90d_prior": 1, "total_returns_lifetime_prior": 1,
            "order_date": "2026-06-01", "return_date": "2026-06-07",
        },
        "expected_business_outcome": "REJECT_UNRECOGNIZED_SCHEMA",
        "expected_ml_behavior": "SCHEMA_CONTRACT_REJECTION",
        "responsible_subsystem": "Contract Validator",
        "rationale": "Safe schema defense rejecting out-of-vocabulary categories."
    },
    {
        "id": "HC-16",
        "name": "16. Short-Window Return with Genuine Defect (0.2 days, Defective/Broken)",
        "payload": {
            "age": 31, "account_age_days": 550, "customer_segment": "Gold", "country": "US",
            "platform": "Mobile App", "device_type": "iPhone", "payment_method": "Credit Card",
            "product_category": "Electronics", "avg_order_value_usd": 180.0, "is_high_value_item": 0,
            "discount_used": 0, "days_to_return": 0.2, "return_reason": "Defective/Broken", "shipping_carrier": "UPS",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 8.0, "customer_return_count_prior": 1,
            "returns_last_30d_prior": 0, "returns_last_90d_prior": 1, "total_returns_lifetime_prior": 1,
            "order_date": "2026-06-01", "return_date": "2026-06-01",
        },
        "expected_business_outcome": "INSTANT_REPLACEMENT",
        "expected_ml_behavior": "Legitimate",
        "responsible_subsystem": "ML (Tabular)",
        "rationale": "Fast reporting of out-of-box hardware failure."
    },
    {
        "id": "HC-17",
        "name": "17. Repeated Returns Across Diverse Categories (Category Surfing Abuser)",
        "payload": {
            "age": 29, "account_age_days": 320, "customer_segment": "Bronze", "country": "US",
            "platform": "Web Browser", "device_type": "Windows PC", "payment_method": "Credit Card",
            "product_category": "Electronics", "avg_order_value_usd": 240.0, "is_high_value_item": 0,
            "discount_used": 1, "days_to_return": 22.0, "return_reason": "Changed Mind", "shipping_carrier": "FedEx",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 0.8, "customer_return_count_prior": 7,
            "returns_last_30d_prior": 3, "returns_last_90d_prior": 5, "total_returns_lifetime_prior": 9,
            "order_date": "2026-06-01", "return_date": "2026-06-23",
        },
        "expected_business_outcome": "RESTRICT_RETURN_PRIVILEGES",
        "expected_ml_behavior": "Policy Abuser",
        "responsible_subsystem": "ML (Tabular)",
        "rationale": "High velocity cross-category return habit."
    },
    {
        "id": "HC-18",
        "name": "18. Multiple Accounts Collision on Shared Android Device",
        "payload": {
            "age": 24, "account_age_days": 30, "customer_segment": "New", "country": "US",
            "platform": "Mobile App", "device_type": "Android", "payment_method": "Buy Now Pay Later",
            "product_category": "Shoes", "avg_order_value_usd": 160.0, "is_high_value_item": 0,
            "discount_used": 1, "days_to_return": 2.0, "return_reason": "Item Not Received", "shipping_carrier": "DHL",
            "multiple_accounts_flag": 1, "wishlist_to_cart_time_hrs": 0.3, "customer_return_count_prior": 2,
            "returns_last_30d_prior": 2, "returns_last_90d_prior": 2, "total_returns_lifetime_prior": 2,
            "order_date": "2026-06-01", "return_date": "2026-06-03",
        },
        "expected_business_outcome": "BLOCK_DEVICE_HASH_RING",
        "expected_ml_behavior": "Fraudulent Return",
        "responsible_subsystem": "ML + Fraud Network Graph",
        "rationale": "Colliding device identifier on multi-account network."
    },
    {
        "id": "HC-19",
        "name": "19. Cross-Border Luxury Purchase with Legitimate Inspection",
        "payload": {
            "age": 45, "account_age_days": 900, "customer_segment": "Gold", "country": "AU",
            "platform": "Web Browser", "device_type": "MacBook", "payment_method": "Credit Card",
            "product_category": "Jewelry", "avg_order_value_usd": 1800.0, "is_high_value_item": 1,
            "discount_used": 0, "days_to_return": 8.0, "return_reason": "Quality Issue", "shipping_carrier": "DHL",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 36.0, "customer_return_count_prior": 1,
            "returns_last_30d_prior": 0, "returns_last_90d_prior": 1, "total_returns_lifetime_prior": 1,
            "order_date": "2026-06-01", "return_date": "2026-06-09",
        },
        "expected_business_outcome": "APPROVE_CUSTOMS_RETURN",
        "expected_ml_behavior": "Legitimate",
        "responsible_subsystem": "ML (Tabular)",
        "rationale": "International customer with solid account tenure."
    },
    {
        "id": "HC-20",
        "name": "20. Inconsistent Timestamps (Return Date Preceding Delivery)",
        "payload": {
            "age": 32, "account_age_days": 300, "customer_segment": "Silver", "country": "US",
            "platform": "Web Browser", "device_type": "Windows PC", "payment_method": "Credit Card",
            "product_category": "Clothing", "avg_order_value_usd": 85.0, "is_high_value_item": 0,
            "discount_used": 0, "days_to_return": -2.0, "return_reason": "Accidental Order", "shipping_carrier": "UPS",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 1.0, "customer_return_count_prior": 0,
            "returns_last_30d_prior": 0, "returns_last_90d_prior": 0, "total_returns_lifetime_prior": 0,
            "order_date": "2026-06-05", "return_date": "2026-06-03",
        },
        "expected_business_outcome": "REJECT_INVALID_TIMESTAMPS",
        "expected_ml_behavior": "Legitimate",
        "responsible_subsystem": "Intake Validation Pipeline",
        "rationale": "Temporal anomaly handled by intake validator before ML inference."
    },
    {
        "id": "HC-21",
        "name": "21. Out-of-Distribution Payment & Category (CryptoToken, SmartGlasses)",
        "payload": {
            "age": 28, "account_age_days": 400, "customer_segment": "Silver", "country": "US",
            "platform": "Web Browser", "device_type": "Windows PC", "payment_method": "CryptoToken",
            "product_category": "SmartGlasses", "avg_order_value_usd": 399.0, "is_high_value_item": 1,
            "discount_used": 0, "days_to_return": 5.0, "return_reason": "Quality Issue", "shipping_carrier": "FedEx",
            "multiple_accounts_flag": 0, "wishlist_to_cart_time_hrs": 12.0, "customer_return_count_prior": 0,
            "returns_last_30d_prior": 0, "returns_last_90d_prior": 0, "total_returns_lifetime_prior": 0,
            "order_date": "2026-06-01", "return_date": "2026-06-06",
        },
        "expected_business_outcome": "REJECT_UNRECOGNIZED_SCHEMA",
        "expected_ml_behavior": "SCHEMA_CONTRACT_REJECTION",
        "responsible_subsystem": "Contract Validator",
        "rationale": "Safe schema rejection on out-of-vocabulary payment method."
    },
]


def evaluate_hard_cases():
    print("=" * 80)
    print("TRUSTLOOP 21-CASE ADVERSARIAL & EDGE-CASE BENCHMARK")
    print("=" * 80)

    with open(MODELS_DIR / "lightgbm_model.pkl", "rb") as f:
        prod_m = pickle.load(f)
    prod_feats = list(prod_m.feature_name_)

    with open(MODELS_DIR / "lightgbm_candidate.pkl", "rb") as f:
        cand_m = pickle.load(f)
    cand_feats = list(cand_m.feature_name_)

    results = []
    for c in HARD_CASES:
        p = c["payload"]
        try:
            df_p = build_model_dataframe(p, feature_names=prod_feats)
            pred_p = int(prod_m.predict(df_p)[0])
            prob_p = prod_m.predict_proba(df_p)[0]
            pred_p_name = CLASS_NAMES[pred_p]
            correct_p = (pred_p_name == c["expected_ml_behavior"])
            conf_p = round(float(prob_p[pred_p]), 4)
            probs_dict = {CLASS_NAMES[i]: round(float(prob_p[i]), 4) for i in range(4)}
        except ValueError:
            pred_p_name = "SCHEMA_REJECTED"
            correct_p = (c["expected_ml_behavior"] == "SCHEMA_CONTRACT_REJECTION")
            conf_p = 1.0
            probs_dict = {"REJECTED": 1.0}

        try:
            df_c = build_model_dataframe(p, feature_names=cand_feats)
            pred_c = int(cand_m.predict(df_c)[0])
            cand_pred_name = CLASS_NAMES[pred_c]
            correct_c = (cand_pred_name == c["expected_ml_behavior"])
        except ValueError:
            cand_pred_name = "SCHEMA_REJECTED"
            correct_c = (c["expected_ml_behavior"] == "SCHEMA_CONTRACT_REJECTION")

        results.append({
            "case_id": c["id"],
            "case_name": c["name"],
            "expected_business_outcome": c["expected_business_outcome"],
            "expected_ml_behavior": c["expected_ml_behavior"],
            "responsible_subsystem": c["responsible_subsystem"],
            "production_prediction": pred_p_name,
            "production_confidence": conf_p,
            "production_correct": correct_p,
            "candidate_prediction": cand_pred_name,
            "candidate_correct": correct_c,
            "rationale": c["rationale"],
        })

    with open(REPORTS_DIR / "hard_case_expanded_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(results).to_csv(REPORTS_DIR / "hard_case_expanded_benchmark.csv", index=False)

    p_corr = sum(1 for r in results if r["production_correct"])
    c_corr = sum(1 for r in results if r["candidate_correct"])
    print(f"\n[OK] Hard-case benchmark complete:")
    print(f"  Production Model Correct: {p_corr}/{len(results)} ({p_corr/len(results):.1%})")
    print(f"  Candidate Model Correct:  {c_corr}/{len(results)} ({c_corr/len(results):.1%})")
    print(f"  Saved -> {REPORTS_DIR / 'hard_case_expanded_benchmark.json'}")

    return results


if __name__ == "__main__":
    evaluate_hard_cases()
