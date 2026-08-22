from pathlib import Path
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag.trustloop_decision import analyze_return


# ============================================================
# TEST IMAGE
# ============================================================

TEST_IMAGE = (
    PROJECT_ROOT
    / "test_images"
    / "return_test.jpg.png"
)


# ============================================================
# BASE CASE
# ============================================================

BASE_CASE = {
    "age": 32,
    "account_age_days": 900,
    "customer_segment": "regular",
    "country": "India",
    "platform": "web",
    "device_type": "mobile",
    "payment_method": "credit_card",
    "product_category": "electronics",
    "avg_order_value_usd": 180.0,
    "is_high_value_item": 0,
    "discount_used": 0,
    "order_date": "2026-06-01",
    "return_date": "2026-06-10",
    "days_to_return": 9,
    "return_reason": "Product arrived damaged",
    "shipping_carrier": "standard",
    "multiple_accounts_flag": 0,
    "wishlist_to_cart_time_hrs": 30.0,
    "customer_return_count_prior": 0,
    "returns_last_30d_prior": 0,
    "returns_last_90d_prior": 0,
    "total_returns_lifetime_prior": 0,
}


# ============================================================
# CASE BUILDER
# ============================================================

def make_case(**changes):

    case = BASE_CASE.copy()
    case.update(changes)

    return case


# ============================================================
# RUN TEST
# ============================================================

def run_test(
    name,
    case,
    image_path=None,
    expected_decision=None,
):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    try:

        result = analyze_return(
            case,
            image_path
        )

        decision = (
            result["final_decision"]["decision"]
        )

        ml_label = (
            result["ml_result"]["predicted_label"]
        )

        ml_confidence = (
            result["ml_result"]["confidence"]
        )

        policy_status = (
            result["policy_result"]["policy_status"]
        )

        print(
            f"ML: {ml_label} "
            f"({ml_confidence:.4f})"
        )

        print(
            f"Policy: {policy_status}"
        )

        if result["vision_result"]:

            vision = result[
                "vision_result"
            ]

            print(
                "Vision:",
                vision["product_condition"],
                "| Damage:",
                vision["damage_detected"],
                "| Confidence:",
                f"{vision['confidence']:.4f}"
            )

        print(
            f"FINAL DECISION: {decision}"
        )

        print(
            f"REASON: "
            f"{result['final_decision']['reason']}"
        )

        if expected_decision:

            if decision == expected_decision:

                print(
                    f"TEST: PASS "
                    f"(expected {expected_decision})"
                )

                return True

            print(
                f"TEST: FAIL "
                f"(expected {expected_decision})"
            )

            return False

        print("TEST: COMPLETED")

        return True

    except Exception as exc:

        print(
            "TEST: ERROR"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return False


# ============================================================
# TEST SUITE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TRUSTLOOP — DECISION SCENARIO TEST SUITE")
    print("=" * 70)

    results = []


    # --------------------------------------------------------
    # 1. NORMAL LEGITIMATE RETURN
    # --------------------------------------------------------

    results.append(
        run_test(
            "CASE 1 — NORMAL RETURN",
            make_case(
                return_reason="Changed my mind"
            ),
            image_path=None,
            expected_decision="AUTO_APPROVE",
        )
    )


    # --------------------------------------------------------
    # 2. DAMAGED PRODUCT + IMAGE
    # --------------------------------------------------------

    results.append(
        run_test(
            "CASE 2 — DAMAGED PRODUCT WITH IMAGE",
            make_case(
                return_reason="Product arrived damaged"
            ),
            image_path=str(TEST_IMAGE),
            expected_decision="HUMAN_ESCALATION",
        )
    )


    # --------------------------------------------------------
    # 3. LATE RETURN
    # --------------------------------------------------------

    results.append(
        run_test(
            "CASE 3 — LATE RETURN",
            make_case(
                return_date="2026-07-15",
                days_to_return=44,
                return_reason="Changed my mind",
            ),
            image_path=None,
            expected_decision="HUMAN_ESCALATION",
        )
    )


    # --------------------------------------------------------
    # 4. MULTIPLE ACCOUNTS
    # --------------------------------------------------------

    results.append(
        run_test(
            "CASE 4 — MULTIPLE ACCOUNTS",
            make_case(
                multiple_accounts_flag=1,
                return_reason="Changed my mind",
            ),
            image_path=None,
            expected_decision="HUMAN_ESCALATION",
        )
    )


    # --------------------------------------------------------
    # 5. HIGH VALUE PRODUCT
    # --------------------------------------------------------

    results.append(
        run_test(
            "CASE 5 — HIGH VALUE PRODUCT",
            make_case(
                is_high_value_item=1,
                avg_order_value_usd=500.0,
                return_reason="Product arrived damaged",
            ),
            image_path=str(TEST_IMAGE),
            expected_decision="HUMAN_ESCALATION",
        )
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 70)
    print("SCENARIO TEST SUMMARY")
    print("=" * 70)

    print(
        f"Passed: {passed}/{total}"
    )

    if passed == total:

        print(
            "STATUS: ALL TESTS PASSED"
        )

    else:

        print(
            "STATUS: SOME TESTS FAILED"
        )

    print(
        "=" * 70
    )