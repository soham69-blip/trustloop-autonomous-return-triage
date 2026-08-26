from rag.trustloop_decision import analyze_return


CASES = [

    {
        "case_id": "SCENARIO-LEGIT-001",

        "age": 28,
        "account_age_days": 700,
        "customer_segment": "regular",
        "country": "India",
        "platform": "web",
        "device_type": "mobile",
        "payment_method": "credit_card",
        "product_category": "electronics",
        "avg_order_value_usd": 120.0,
        "is_high_value_item": 0,
        "discount_used": 0,

        "order_date": "2026-07-01",
        "return_date": "2026-07-08",
        "days_to_return": 7,

        "return_reason": "Changed mind",
        "shipping_carrier": "standard",
        "multiple_accounts_flag": 0,
        "wishlist_to_cart_time_hrs": 18.0,

        "customer_return_count_prior": 0,
        "returns_last_30d_prior": 0,
        "returns_last_90d_prior": 0,
        "total_returns_lifetime_prior": 1,
    },

    {
        "case_id": "SCENARIO-POLICY-001",

        "age": 24,
        "account_age_days": 90,
        "customer_segment": "high_risk",
        "country": "India",
        "platform": "web",
        "device_type": "mobile",
        "payment_method": "credit_card",
        "product_category": "electronics",
        "avg_order_value_usd": 450.0,
        "is_high_value_item": 1,
        "discount_used": 1,

        "order_date": "2026-06-01",
        "return_date": "2026-07-20",
        "days_to_return": 49,

        "return_reason": "Changed mind",
        "shipping_carrier": "standard",
        "multiple_accounts_flag": 1,
        "wishlist_to_cart_time_hrs": 1.0,

        "customer_return_count_prior": 8,
        "returns_last_30d_prior": 4,
        "returns_last_90d_prior": 7,
        "total_returns_lifetime_prior": 10,
    },

    {
        "case_id": "SCENARIO-FRAUD-001",

        "age": 22,
        "account_age_days": 40,
        "customer_segment": "high_risk",
        "country": "India",
        "platform": "web",
        "device_type": "mobile",
        "payment_method": "cash_on_delivery",
        "product_category": "electronics",
        "avg_order_value_usd": 800.0,
        "is_high_value_item": 1,
        "discount_used": 1,

        "order_date": "2026-07-01",
        "return_date": "2026-07-05",
        "days_to_return": 4,

        "return_reason": "Item not received",
        "shipping_carrier": "standard",
        "multiple_accounts_flag": 1,
        "wishlist_to_cart_time_hrs": 0.5,

        "customer_return_count_prior": 15,
        "returns_last_30d_prior": 8,
        "returns_last_90d_prior": 14,
        "total_returns_lifetime_prior": 18,
    },

    {
        "case_id": "SCENARIO-WARDROBE-001",

        "age": 26,
        "account_age_days": 120,
        "customer_segment": "regular",
        "country": "India",
        "platform": "app",
        "device_type": "mobile",
        "payment_method": "credit_card",
        "product_category": "fashion",
        "avg_order_value_usd": 220.0,
        "is_high_value_item": 0,
        "discount_used": 1,

        "order_date": "2026-07-01",
        "return_date": "2026-07-10",
        "days_to_return": 9,

        "return_reason": "Worn and returned",
        "shipping_carrier": "standard",
        "multiple_accounts_flag": 0,
        "wishlist_to_cart_time_hrs": 2.0,

        "customer_return_count_prior": 7,
        "returns_last_30d_prior": 5,
        "returns_last_90d_prior": 7,
        "total_returns_lifetime_prior": 9,
    },

]


print("=" * 75)
print("TRUSTLOOP — FOUR-CLASS LEARNING SCENARIO TEST")
print("=" * 75)

for case in CASES:

    print("\n" + "-" * 75)
    print("CASE:", case["case_id"])

    try:

        result = analyze_return(
            case,
            image_path=None
        )

        ml = result["ml_result"]
        final = result["final_decision"]

        print(
            "ML Prediction:",
            ml["predicted_label"]
        )

        print(
            "ML Confidence:",
            f"{ml['confidence']:.2%}"
        )

        print(
            "Final Decision:",
            final["decision"]
        )

        raw_conf = final.get("decision_confidence")
        if raw_conf is None:
            raw_conf = final.get("confidence", 0.0)
        conf_val = float(raw_conf) if isinstance(raw_conf, (int, float, str)) else 0.0
        print(
            "Decision Confidence:",
            f"{conf_val:.2f}%"
        )

        print(
            "Reason:",
            final["reason"]
        )

        print(
            "Pending Case:",
            result["case_id"]
        )

    except Exception as exc:

        print(
            "ERROR:",
            repr(exc)
        )

print("\n" + "=" * 75)
print("SCENARIO GENERATION COMPLETED")
print("=" * 75)
