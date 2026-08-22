from datetime import date, timedelta

from rag.trustloop_decision import analyze_return


# ============================================================
# TRUSTLOOP SYNTHETIC TRAINING SCENARIO GENERATOR V2
# ============================================================
#
# IMPORTANT:
# These are synthetic validation/demo cases.
# They are NOT real customer feedback.
#
# Ground-truth labels are stored separately and are never
# passed to the model.
# ============================================================


BASE_ORDER_DATE = date(2026, 7, 1)


def make_case(
    case_id,
    label,
    age,
    account_age_days,
    segment,
    product,
    aov,
    days_to_return,
    reason,
    multiple_accounts,
    returns_prior,
    returns_30,
    returns_90,
    lifetime_returns,
    high_value,
    discount,
    platform="web",
    device="mobile",
    payment="credit_card",
    shipping="standard",
    wishlist_hours=24.0,
):
    """
    Create one internally consistent synthetic return case.

    IMPORTANT:
    return_date is derived from:
        order_date + days_to_return
    """

    order_date = BASE_ORDER_DATE

    return_date = (
        order_date
        + timedelta(
            days=days_to_return
        )
    )

    return {
        "case_id": case_id,

        "age": age,

        "account_age_days":
            account_age_days,

        "customer_segment":
            segment,

        "country":
            "India",

        "platform":
            platform,

        "device_type":
            device,

        "payment_method":
            payment,

        "product_category":
            product,

        "avg_order_value_usd":
            aov,

        "is_high_value_item":
            high_value,

        "discount_used":
            discount,

        "order_date":
            order_date.isoformat(),

        "return_date":
            return_date.isoformat(),

        "days_to_return":
            days_to_return,

        "return_reason":
            reason,

        "shipping_carrier":
            shipping,

        "multiple_accounts_flag":
            multiple_accounts,

        "wishlist_to_cart_time_hrs":
            wishlist_hours,

        "customer_return_count_prior":
            returns_prior,

        "returns_last_30d_prior":
            returns_30,

        "returns_last_90d_prior":
            returns_90,

        "total_returns_lifetime_prior":
            lifetime_returns,

        # Synthetic label only.
        # This is NOT passed to TrustLoop.
        "_ground_truth":
            label,
    }


# ============================================================
# SCENARIOS
# ============================================================

CASES = []


# ============================================================
# LEGITIMATE
# ============================================================

CASES += [

    make_case(
        "TRAIN-LEGIT-001",
        "Legitimate",
        29,
        800,
        "regular",
        "electronics",
        110,
        7,
        "Changed mind",
        0,
        0,
        0,
        0,
        1,
        0,
        0,
    ),

    make_case(
        "TRAIN-LEGIT-002",
        "Legitimate",
        34,
        1200,
        "regular",
        "fashion",
        90,
        10,
        "Size issue",
        0,
        1,
        0,
        1,
        2,
        0,
        0,
    ),

    make_case(
        "TRAIN-LEGIT-003",
        "Legitimate",
        41,
        1500,
        "loyal",
        "electronics",
        180,
        5,
        "Defective product",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ),

    make_case(
        "TRAIN-LEGIT-004",
        "Legitimate",
        25,
        600,
        "regular",
        "home",
        75,
        6,
        "Wrong size",
        0,
        0,
        0,
        0,
        0,
        0,
        1,
    ),

    make_case(
        "TRAIN-LEGIT-005",
        "Legitimate",
        38,
        1000,
        "loyal",
        "fashion",
        130,
        8,
        "Changed mind",
        0,
        1,
        0,
        1,
        2,
        0,
        0,
    ),

    make_case(
        "TRAIN-LEGIT-006",
        "Legitimate",
        31,
        500,
        "regular",
        "electronics",
        150,
        9,
        "Product arrived damaged",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ),
]


# ============================================================
# POLICY ABUSER
# ============================================================
#
# Characteristics intentionally emphasize:
# - repeated returns
# - multiple accounts
# - unusually late returns
# - repeated discretionary returns
# - high-value / discount-heavy activity
#
# This is distinct from Wardrobing, which focuses on
# repeated use-and-return behavior for fashion items.
# ============================================================

CASES += [

    make_case(
        "TRAIN-POLICY-001",
        "Policy Abuser",
        24,
        90,
        "high_risk",
        "electronics",
        450,
        49,
        "Changed mind",
        1,
        8,
        4,
        7,
        10,
        1,
        1,
        wishlist_hours=2.0,
    ),

    make_case(
        "TRAIN-POLICY-002",
        "Policy Abuser",
        35,
        150,
        "high_risk",
        "home",
        320,
        44,
        "Changed mind",
        1,
        7,
        3,
        6,
        9,
        1,
        1,
        wishlist_hours=3.0,
    ),

    make_case(
        "TRAIN-POLICY-003",
        "Policy Abuser",
        22,
        60,
        "high_risk",
        "electronics",
        520,
        55,
        "Changed mind",
        1,
        10,
        5,
        9,
        12,
        1,
        1,
        wishlist_hours=1.0,
    ),

    make_case(
        "TRAIN-POLICY-004",
        "Policy Abuser",
        30,
        150,
        "high_risk",
        "home",
        390,
        38,
        "Changed mind",
        1,
        6,
        3,
        5,
        8,
        1,
        0,
        wishlist_hours=4.0,
    ),

    make_case(
        "TRAIN-POLICY-005",
        "Policy Abuser",
        26,
        200,
        "high_risk",
        "electronics",
        280,
        41,
        "Changed mind",
        1,
        9,
        4,
        8,
        11,
        1,
        1,
        wishlist_hours=2.5,
    ),
]


# ============================================================
# FRAUDULENT RETURN
# ============================================================
#
# Characteristics emphasize:
# - new/young accounts
# - extremely high return activity
# - multiple accounts
# - high-value electronics
# - suspicious non-receipt / missing-item claims
# ============================================================

CASES += [

    make_case(
        "TRAIN-FRAUD-001",
        "Fraudulent Return",
        21,
        40,
        "high_risk",
        "electronics",
        800,
        4,
        "Item not received",
        1,
        15,
        8,
        14,
        18,
        1,
        1,
        wishlist_hours=0.5,
    ),

    make_case(
        "TRAIN-FRAUD-002",
        "Fraudulent Return",
        23,
        55,
        "high_risk",
        "electronics",
        950,
        3,
        "Refund requested",
        1,
        14,
        7,
        13,
        16,
        1,
        1,
        wishlist_hours=0.25,
    ),

    make_case(
        "TRAIN-FRAUD-003",
        "Fraudulent Return",
        20,
        30,
        "high_risk",
        "electronics",
        700,
        2,
        "Item missing",
        1,
        17,
        9,
        15,
        20,
        1,
        1,
        wishlist_hours=0.2,
    ),

    make_case(
        "TRAIN-FRAUD-004",
        "Fraudulent Return",
        28,
        80,
        "high_risk",
        "electronics",
        1000,
        5,
        "Wrong item received",
        1,
        12,
        6,
        11,
        14,
        1,
        1,
        wishlist_hours=0.75,
    ),

    make_case(
        "TRAIN-FRAUD-005",
        "Fraudulent Return",
        19,
        25,
        "high_risk",
        "electronics",
        650,
        3,
        "Item not received",
        1,
        16,
        8,
        14,
        19,
        1,
        1,
        wishlist_hours=0.15,
    ),
]


# ============================================================
# WARDROBING
# ============================================================
#
# Characteristics emphasize:
# - fashion
# - repeated returns
# - explicit use/wear pattern
# - relatively normal account identity
# - no multiple-account signal
#
# This is deliberately separated from Policy Abuser.
# ============================================================

CASES += [

    make_case(
        "TRAIN-WARDROBE-001",
        "Wardrobing",
        26,
        120,
        "regular",
        "fashion",
        220,
        9,
        "Worn and returned",
        0,
        7,
        5,
        7,
        9,
        0,
        1,
        wishlist_hours=6.0,
    ),

    make_case(
        "TRAIN-WARDROBE-002",
        "Wardrobing",
        29,
        180,
        "regular",
        "fashion",
        260,
        8,
        "Used and returned",
        0,
        8,
        5,
        8,
        10,
        0,
        1,
        wishlist_hours=5.0,
    ),

    make_case(
        "TRAIN-WARDROBE-003",
        "Wardrobing",
        24,
        100,
        "regular",
        "fashion",
        190,
        7,
        "Worn and returned",
        0,
        9,
        6,
        8,
        11,
        0,
        1,
        wishlist_hours=4.0,
    ),

    make_case(
        "TRAIN-WARDROBE-004",
        "Wardrobing",
        32,
        240,
        "regular",
        "fashion",
        310,
        10,
        "Used and returned",
        0,
        6,
        4,
        6,
        8,
        1,
        1,
        wishlist_hours=7.0,
    ),
]


# ============================================================
# RUN
# ============================================================

print("=" * 75)
print(
    "TRUSTLOOP — TRAINING SCENARIO GENERATOR V2"
)
print("=" * 75)

created = 0

for case in CASES:

    # Ground truth is kept ONLY for human review.
    ground_truth = case.pop(
        "_ground_truth"
    )

    try:

        result = analyze_return(
            case,
            image_path=None,
        )

        ml = result[
            "ml_result"
        ]

        decision = result[
            "final_decision"
        ]

        print(
            f"\n{case['case_id']}"
        )

        print(
            f"Ground truth : "
            f"{ground_truth}"
        )

        print(
            f"Prediction   : "
            f"{ml['predicted_label']}"
        )

        print(
            f"ML confidence: "
            f"{ml['confidence']:.2%}"
        )

        print(
            f"Decision     : "
            f"{decision['decision']}"
        )

        created += 1

    except Exception as exc:

        print(
            f"\nERROR {case['case_id']}: "
            f"{exc}"
        )


print(
    "\nGenerated pending cases:",
    created,
)

print(
    "\nIMPORTANT:"
)

print(
    "These are SYNTHETIC scenario cases."
)

print(
    "They are NOT real customer feedback."
)

print(
    "Do not automatically verify them "
    "without reviewing the intended label."
)

print(
    "\n" + "=" * 75
)
