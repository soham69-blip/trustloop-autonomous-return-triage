
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


MODEL_FEATURES = [
    "age",
    "account_age_days",
    "customer_segment",
    "country",
    "platform",
    "device_type",
    "payment_method",
    "product_category",
    "avg_order_value_usd",
    "is_high_value_item",
    "discount_used",
    "days_to_return",
    "return_reason",
    "shipping_carrier",
    "multiple_accounts_flag",
    "wishlist_to_cart_time_hrs",
    "customer_return_count_prior",
    "returns_last_30d_prior",
    "returns_last_90d_prior",
    "total_returns_lifetime_prior",
    "order_date_year",
    "order_date_month",
    "order_date_day",
    "order_date_dayofweek",
    "order_date_dayofyear",
    "order_date_is_weekend",
    "return_date_year",
    "return_date_month",
    "return_date_day",
    "return_date_dayofweek",
    "return_date_dayofyear",
    "return_date_is_weekend",
    "calculated_days_to_return",
]

CANDIDATE_MODEL_FEATURES = [
    "age",
    "account_age_days",
    "customer_segment",
    "country",
    "platform",
    "device_type",
    "payment_method",
    "product_category",
    "avg_order_value_usd",
    "is_high_value_item",
    "discount_used",
    "days_to_return",
    "return_reason",
    "shipping_carrier",
    "multiple_accounts_flag",
    "wishlist_to_cart_time_hrs",
    "total_returns_lifetime",
    "total_orders_lifetime",
    "return_rate_pct",
    "customer_support_contacts",
    "previous_dispute_count",
    "refund_amount_requested_usd",
    "customer_return_count_prior",
    "returns_last_30d_prior",
    "returns_last_90d_prior",
    "total_returns_lifetime_prior",
    "order_date_year",
    "order_date_month",
    "order_date_day",
    "order_date_dayofweek",
    "order_date_dayofyear",
    "order_date_is_weekend",
    "return_date_year",
    "return_date_month",
    "return_date_day",
    "return_date_dayofweek",
    "return_date_dayofyear",
    "return_date_is_weekend",
    "calculated_days_to_return",
]


def _parse_datetime(value: Any):
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    text = str(value).strip()

    if not text:
        return None

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
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _date_features(prefix: str, value: Any) -> Dict[str, Any]:
    dt = _parse_datetime(value)

    if dt is None:
        return {
            f"{prefix}_year": 0,
            f"{prefix}_month": 0,
            f"{prefix}_day": 0,
            f"{prefix}_dayofweek": 0,
            f"{prefix}_dayofyear": 0,
            f"{prefix}_is_weekend": 0,
        }

    return {
        f"{prefix}_year": dt.year,
        f"{prefix}_month": dt.month,
        f"{prefix}_day": dt.day,
        f"{prefix}_dayofweek": dt.weekday(),
        f"{prefix}_dayofyear": dt.timetuple().tm_yday,
        f"{prefix}_is_weekend": int(dt.weekday() >= 5),
    }


import pickle
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_READY = ROOT_DIR / "data" / "processed" / "trustloop" / "model_ready.csv"
MAPPINGS_PATH = ROOT_DIR / "models" / "categorical_mappings.pkl"

_CACHED_MAPPINGS: Dict[str, Any] | None = None


def _load_or_build_category_mappings() -> Dict[str, Any]:
    """Load saved category mappings or build them deterministically from
    data/processed/trustloop/model_ready.csv using the same chronological
    split logic as training. Returns a dict: {col: [categories...]}
    """
    global _CACHED_MAPPINGS
    if _CACHED_MAPPINGS is not None:
        return _CACHED_MAPPINGS

    if MAPPINGS_PATH.exists():
        try:
            with open(MAPPINGS_PATH, "rb") as f:
                _CACHED_MAPPINGS = pickle.load(f)
                return _CACHED_MAPPINGS
        except Exception:
            # fallback to rebuilding
            pass

    # Build mappings from the canonical model_ready.csv
    mappings = {}

    if not MODEL_READY.exists():
        return mappings

    import pandas as pd
    import numpy as np

    df = pd.read_csv(MODEL_READY)

    # replicate training chronological sort
    sort_dates = pd.to_datetime(df["return_date"], errors="coerce")
    order_dates = pd.to_datetime(df["order_date"], errors="coerce")

    # convert to int64 (nan -> NaT -> becomes NaN when astype)
    sort_order = np.lexsort((  # pyrefly: ignore[bad-argument-type]
        order_dates.astype("int64").to_numpy(),
        sort_dates.astype("int64").to_numpy(),
    ))

    df_sorted = df.iloc[sort_order].reset_index(drop=True)  # pyrefly: ignore[bad-index]


    n = len(df_sorted)
    train_end = int(n * 0.70)
    train_df = df_sorted.iloc[:train_end]

    categorical_columns = [
        "country",
        "customer_segment",
        "device_type",
        "payment_method",
        "platform",
        "product_category",
        "return_reason",
        "shipping_carrier",
    ]

    for col in categorical_columns:
        if col in train_df.columns:
            # preserve order of first appearance
            vals = train_df[col].fillna("unknown").astype(str).tolist()
            seen = {}
            uniq = []
            for v in vals:
                if v not in seen:
                    seen[v] = True
                    uniq.append(v)
            mappings[col] = uniq

    # persist mappings for future use
    try:
        MAPPINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MAPPINGS_PATH, "wb") as f:
            pickle.dump(mappings, f)
    except Exception:
        # non-fatal
        pass

    return mappings


def validate_feature_contract(
    case: Dict[str, Any],
    feature_names: list[str] | None = None,
) -> Dict[str, Any]:
    """Validate whether an input payload satisfies the contract for the target feature set.

    Returns a dict with:
      - valid (bool): True if the payload contains sufficient information for reliable inference.
      - target_feature_set (str): 'production' (33 features) or 'candidate' (39 features).
      - feature_count (int): Count of expected features.
      - provenance (dict): Detailed provenance category for each candidate feature.
      - missing_candidate_fields (list): Candidate-specific fields missing from payload.
      - derived_fields (list): Fields that were successfully derived from legacy/fallback fields.
      - inconsistencies (list): Inconsistencies detected (e.g. positive returns with missing orders or contradictory rates).
      - warnings (list): Explanatory audit warnings.
    """
    target_cols = feature_names if feature_names is not None else MODEL_FEATURES
    is_candidate = "return_rate_pct" in target_cols or len(target_cols) == 39

    missing_fields = []
    derived_fields = []
    inconsistencies = []
    warnings = []
    provenance = {}

    if is_candidate:
        candidate_fields = [
            "total_returns_lifetime",
            "total_orders_lifetime",
            "return_rate_pct",
            "customer_support_contacts",
            "previous_dispute_count",
            "refund_amount_requested_usd",
        ]

        for f in candidate_fields:
            if f not in case or case.get(f) is None:
                missing_fields.append(f)

        # 1. Total returns provenance
        if case.get("total_returns_lifetime") is not None:
            provenance["total_returns_lifetime"] = "EXPLICITLY_SUPPLIED"
        elif case.get("total_returns_lifetime_prior") is not None:
            provenance["total_returns_lifetime"] = "SAFELY_DERIVED (from total_returns_lifetime_prior)"
            derived_fields.append("total_returns_lifetime")
        elif case.get("customer_return_count_prior") is not None:
            provenance["total_returns_lifetime"] = "SAFELY_DERIVED (from customer_return_count_prior)"
            derived_fields.append("total_returns_lifetime")
        else:
            provenance["total_returns_lifetime"] = "DEFAULTED_NEUTRAL (0)"

        # 2. Refund amount provenance
        if case.get("refund_amount_requested_usd") is not None:
            provenance["refund_amount_requested_usd"] = "EXPLICITLY_SUPPLIED"
        elif case.get("refund_amount") is not None:
            provenance["refund_amount_requested_usd"] = "SAFELY_DERIVED (from refund_amount)"
            derived_fields.append("refund_amount_requested_usd")
        elif case.get("avg_order_value_usd") is not None:
            provenance["refund_amount_requested_usd"] = "SAFELY_DERIVED (from avg_order_value_usd)"
            derived_fields.append("refund_amount_requested_usd")
        else:
            provenance["refund_amount_requested_usd"] = "DEFAULTED_NEUTRAL (0.0)"

        # 3. Total orders provenance
        raw_orders = case.get("total_orders_lifetime")
        total_returns = int(case.get("total_returns_lifetime", case.get("total_returns_lifetime_prior", case.get("customer_return_count_prior", 0))) or 0)

        if raw_orders is not None and int(raw_orders) > 0:
            provenance["total_orders_lifetime"] = "EXPLICITLY_SUPPLIED"
        elif total_returns > 0:
            provenance["total_orders_lifetime"] = "ASSUMED_LOWER_BOUND (orders >= returns)"
        else:
            provenance["total_orders_lifetime"] = "DEFAULTED_NEUTRAL (0)"

        # 4. Return rate provenance & contradiction check
        raw_rr = case.get("return_rate_pct")
        if raw_rr is not None and float(raw_rr) > 0.0:
            provenance["return_rate_pct"] = "EXPLICITLY_SUPPLIED"
            # Check for mathematical contradiction if both orders and returns are provided
            if raw_orders is not None and int(raw_orders) > 0 and total_returns >= 0:
                math_rate = round((total_returns / int(raw_orders)) * 100.0, 2)
                if abs(float(raw_rr) - math_rate) > 1.0:
                    inconsistencies.append(
                        f"CONTRADICTORY_RETURN_RATE: Provided return_rate_pct ({float(raw_rr)}%) does not match "
                        f"mathematical ratio of total_returns_lifetime ({total_returns}) / total_orders_lifetime ({int(raw_orders)}) = {math_rate}%."
                    )
                    warnings.append(
                        f"Mathematical return rate ({math_rate}%) differs from supplied return_rate_pct ({float(raw_rr)}%)."
                    )
        elif raw_orders is not None and int(raw_orders) > 0:
            provenance["return_rate_pct"] = "SAFELY_DERIVED (from returns / orders)"
            derived_fields.append("return_rate_pct")
        elif total_returns == 0:
            provenance["return_rate_pct"] = "SAFELY_DERIVED (0 returns -> 0.0%)"
        else:
            provenance["return_rate_pct"] = "ASSUMED_LOWER_BOUND (100.0% due to missing orders denominator)"

        # Inconsistent denominator check
        if total_returns > 0 and (raw_orders is None or int(raw_orders) <= 0) and (raw_rr is None or float(raw_rr) <= 0.0):
            inconsistencies.append(
                f"INCONSISTENT_DENOMINATOR: total_returns_lifetime is positive ({total_returns}) "
                "but total_orders_lifetime is missing/zero and return_rate_pct was not provided."
            )
            warnings.append(
                "Legacy payload does not provide customer order volume. "
                "Fallback lower-bound order count was used to prevent false zero-return-rate assignment."
            )

        # 5. Customer support contacts
        if case.get("customer_support_contacts") is not None:
            provenance["customer_support_contacts"] = "EXPLICITLY_SUPPLIED"
        else:
            provenance["customer_support_contacts"] = "DEFAULTED_NEUTRAL (0)"
            warnings.append("customer_support_contacts was omitted; defaulting to 0.")

        # 6. Previous dispute count
        if case.get("previous_dispute_count") is not None:
            provenance["previous_dispute_count"] = "EXPLICITLY_SUPPLIED"
        else:
            provenance["previous_dispute_count"] = "DEFAULTED_NEUTRAL (0)"
            warnings.append("previous_dispute_count was omitted; defaulting to 0.")

        is_valid = len(inconsistencies) == 0 and len(missing_fields) <= 3

        return {
            "valid": is_valid,
            "target_feature_set": "candidate",
            "feature_count": len(target_cols),
            "provenance": provenance,
            "missing_candidate_fields": missing_fields,
            "derived_fields": derived_fields,
            "inconsistencies": inconsistencies,
            "warnings": warnings,
        }

    else:
        # Production model (33 features)
        legacy_required = [
            "age", "account_age_days", "customer_segment", "country",
            "platform", "device_type", "payment_method", "product_category",
            "avg_order_value_usd", "order_date", "return_date"
        ]
        missing_legacy = [f for f in legacy_required if f not in case or case.get(f) is None]

        return {
            "valid": len(missing_legacy) == 0,
            "target_feature_set": "production",
            "feature_count": len(target_cols),
            "provenance": {f: "EXPLICITLY_SUPPLIED" if f in case else "DEFAULTED_NEUTRAL" for f in target_cols},
            "missing_candidate_fields": [],
            "derived_fields": [],
            "inconsistencies": [],
            "warnings": [f"Missing field '{f}'" for f in missing_legacy],
        }


def build_model_features(
    case: Dict[str, Any],
    feature_names: list[str] | None = None,
    validate_contract: bool = False,
) -> Dict[str, Any]:

    target_cols = feature_names if feature_names is not None else MODEL_FEATURES

    if validate_contract:
        contract_info = validate_feature_contract(case, feature_names=target_cols)
        if not contract_info["valid"] and contract_info["inconsistencies"]:
            raise ValueError(f"Feature contract violation: {contract_info['inconsistencies'][0]}")

    features = {}

    # ---------------------------------------------------------
    # Base features & Candidate Profile features
    # ---------------------------------------------------------

    defaults = {
        "age": 0,
        "account_age_days": 0,
        "customer_segment": "unknown",
        "country": "unknown",
        "platform": "unknown",
        "device_type": "unknown",
        "payment_method": "unknown",
        "product_category": "unknown",
        "avg_order_value_usd": 0.0,
        "is_high_value_item": 0,
        "discount_used": 0,
        "days_to_return": 0.0,
        "return_reason": "unknown",
        "shipping_carrier": "unknown",
        "multiple_accounts_flag": 0,
        "wishlist_to_cart_time_hrs": 0.0,
        "customer_return_count_prior": 0,
        "returns_last_30d_prior": 0,
        "returns_last_90d_prior": 0,
        "total_returns_lifetime_prior": 0,
        # Safe decision-time profile & claim features (Experiment A compatible)
        "total_returns_lifetime": 0,
        "total_orders_lifetime": 0,
        "return_rate_pct": 0.0,
        "customer_support_contacts": 0,
        "previous_dispute_count": 0,
        "refund_amount_requested_usd": 0.0,
    }

    for feature, default in defaults.items():
        value = case.get(feature, default)

        if value is None:
            value = default

        features[feature] = value

    # ---------------------------------------------------------
    # Honest Fallbacks & Derivations
    # ---------------------------------------------------------

    # 1. Total returns lifetime fallback
    if case.get("total_returns_lifetime") is not None:
        total_returns = int(case["total_returns_lifetime"])
    elif case.get("total_returns_lifetime_prior") is not None:
        total_returns = int(case["total_returns_lifetime_prior"])
    elif case.get("customer_return_count_prior") is not None:
        total_returns = int(case["customer_return_count_prior"])
    else:
        total_returns = int(features.get("total_returns_lifetime", 0))

    features["total_returns_lifetime"] = total_returns

    # 2. Refund amount requested fallback
    if case.get("refund_amount_requested_usd") is not None:
        features["refund_amount_requested_usd"] = float(case["refund_amount_requested_usd"])
    elif case.get("refund_amount") is not None:
        features["refund_amount_requested_usd"] = float(case["refund_amount"])
    elif case.get("avg_order_value_usd") is not None:
        features["refund_amount_requested_usd"] = float(case["avg_order_value_usd"])

    # 3. Total orders & Return rate derivation
    raw_orders = case.get("total_orders_lifetime")
    if raw_orders is not None and int(raw_orders) > 0:
        total_orders = int(raw_orders)
    else:
        # If orders not provided: if returns > 0, lower bound total orders to total returns
        # (an account with R returns has completed at least R orders).
        total_orders = total_returns if total_returns > 0 else 0

    features["total_orders_lifetime"] = total_orders

    raw_return_rate = case.get("return_rate_pct")
    if raw_return_rate is not None and float(raw_return_rate) > 0.0:
        features["return_rate_pct"] = float(raw_return_rate)
    elif total_orders > 0 and total_returns > 0:
        features["return_rate_pct"] = round((total_returns / total_orders) * 100.0, 2)
    elif total_returns == 0:
        features["return_rate_pct"] = 0.0
    else:
        # Fallback when total_returns > 0 and orders is lower-bounded
        features["return_rate_pct"] = 100.0

    # ---------------------------------------------------------
    # Dates
    # ---------------------------------------------------------

    order_date = case.get("order_date")
    return_date = case.get("return_date")

    features.update(
        _date_features(
            "order_date",
            order_date
        )
    )

    features.update(
        _date_features(
            "return_date",
            return_date
        )
    )

    # ---------------------------------------------------------
    # Calculated return duration
    # ---------------------------------------------------------

    order_dt = _parse_datetime(order_date)
    return_dt = _parse_datetime(return_date)

    if order_dt is not None and return_dt is not None:
        features["calculated_days_to_return"] = (
            return_dt - order_dt
        ).total_seconds() / 86400.0

    else:
        features["calculated_days_to_return"] = float(
            features.get("days_to_return", 0.0)
        )

    # ---------------------------------------------------------
    # Final strict feature alignment
    # ---------------------------------------------------------

    return {
        feature: features.get(feature, 0)
        for feature in target_cols
    }


def build_model_dataframe(
    case: Dict[str, Any],
    feature_names: list[str] | None = None,
    validate_contract: bool = False,
):
    """Return a single-row pandas DataFrame matching the training dtypes.

    - Applies the category mappings (built from model_ready.csv) to
      convert string values to pandas.Categorical with the same categories
      used during training.
    - Validates unseen categorical values and raises ValueError with a
      clear message rather than silently mapping them.
    """
    import pandas as pd

    target_cols = feature_names if feature_names is not None else MODEL_FEATURES
    features = build_model_features(case, feature_names=target_cols, validate_contract=validate_contract)
    df = pd.DataFrame([features])[target_cols]

    mappings = _load_or_build_category_mappings()

    categorical_columns = [
        "country",
        "customer_segment",
        "device_type",
        "payment_method",
        "platform",
        "product_category",
        "return_reason",
        "shipping_carrier",
    ]

    # Apply categorical dtypes and validate unseen values
    for col in categorical_columns:
        if col not in df.columns:
            continue

        col_val = df.at[0, col]
        # normalize missing
        if pd.isna(col_val):
            col_val = "unknown"
            df.at[0, col] = col_val

        categories = mappings.get(col)

        if categories is None:
            # No mapping available; fail fast to avoid silent mapping
            raise ValueError(
                f"No category mapping available for column '{col}'."
                " Ensure training artifacts exist."
            )

        if str(col_val) not in categories:
            raise ValueError(
                f"Unseen category value for '{col}': '{col_val}'."
                " This value was not present in training data."
            )

        df[col] = pd.Categorical(df[col].astype(str), categories=categories)

    # Coerce numeric columns to numeric types explicitly
    numeric_cols = [
        c for c in df.columns if c not in categorical_columns
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

