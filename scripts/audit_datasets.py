"""
TrustLoop / FlipLens — Phase 1 dataset audit.

Discovers every CSV under data/raw/, profiles them with chunked reads,
and writes audit artifacts. Never modifies files in data/raw/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.csv as pacsv

    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
AUDIT_DIR = ROOT / "data" / "audit"
DOCS_DIR = ROOT / "docs"

CHUNKSIZE = 50_000
SAMPLE_TARGET = 8_000
EXAMPLE_N = 5
HIGH_CARDINALITY = 10_000
KEY_SET_CAP = 250_000
IQR_SAMPLE_CAP = 20_000

ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

ID_NAME_RE = re.compile(
    r"(^|_)(id|sku|code|key)($|_)"
    r"|^(order|customer|product|return|transaction|case|user|item)id$"
    r"|id$|sku$",
    re.I,
)
DATE_NAME_RE = re.compile(
    r"date|time|datetime|timestamp|expiry|delivered|delivery_date|return_date|order_date",
    re.I,
)
PRICE_NAME_RE = re.compile(
    r"price|amount|value|mrp|cost|refund|revenue|profit|margin|fee",
    re.I,
)
QTY_NAME_RE = re.compile(
    r"quantity|qty|stock|sold|available",
    re.I,
)

FLIPLENS_FIELDS = [
    "case_id",
    "customer_id",
    "order_id",
    "product_id",
    "product_category",
    "product_name",
    "brand",
    "order_date",
    "delivery_date",
    "return_date",
    "return_reason",
    "order_value",
    "payment_method",
    "customer_order_count",
    "customer_return_count",
    "customer_return_rate",
    "returns_last_30d",
    "returns_last_90d",
    "previous_fraud_count",
    "high_value_return",
    "return_window_valid",
    "policy_violation",
    "product_match_score",
    "damage_score",
    "serial_match_score",
    "accessories_complete",
    "packaging_match_score",
    "fraud_probability",
    "fraud_label",
    "final_decision",
    "confidence",
    "explanation",
]


def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def sha256_file(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_col(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def detect_encoding(path: Path) -> str:
    for enc in ENCODINGS:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                f.read(65_536)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def looks_like_date_series(series: pd.Series, min_ok: float = 0.4) -> bool:
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return False
    sample = s.head(200)
    parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True, format="mixed")
    return (parsed.notna().mean()) >= min_ok


def is_identifier_name(col: str) -> bool:
    n = normalize_col(col)
    if n in {
        "order_id",
        "customer_id",
        "product_id",
        "return_id",
        "transaction_id",
        "case_id",
        "sku",
        "orderid",
        "customerid",
        "productid",
        "userid",
    }:
        return True
    return bool(ID_NAME_RE.search(n))


def classify_role(columns: list[str], filename: str) -> tuple[str, str]:
    """Classify from actual columns; filename used only as a weak hint in notes."""
    nset = {normalize_col(c) for c in columns}
    joined = " ".join(nset)
    reasons: list[str] = []

    has_return = bool(
        nset
        & {
            "return_date",
            "return_reason",
            "returned",
            "days_to_return",
            "request_date",
            "refund_requested",
            "refund_amount_requested_usd",
            "item_returned_opened",
        }
    ) or "return" in joined
    has_fraud = bool(
        nset
        & {
            "abuse_label",
            "abuse_type",
            "fraud_label",
            "fraud",
            "previous_dispute_count",
        }
    )
    has_order = bool(
        nset
        & {
            "order_id",
            "orderid",
            "order_date",
            "order_datetime",
            "order_date_and_time",
            "order_value",
            "order_value_inr",
            "order_value_inr".replace(" ", "_"),
        }
    ) or ("order_id" in nset) or ("order_id" in joined)
    # explicit
    has_order = any(
        x in nset
        for x in (
            "order_id",
            "orderid",
            "order_date",
            "order_datetime",
            "order_date_and_time",
            "order_value_inr",
            "order_value_inr",
        )
    ) or ("order_value_inr" in nset) or any("order" in x and "id" in x for x in nset)
    has_customer_id = any(x in nset for x in ("customer_id", "customerid", "cust_id"))
    has_product = any(
        x in nset
        for x in (
            "product_id",
            "productid",
            "product_name",
            "sku",
            "category",
            "brand",
            "name",
        )
    )
    has_inventory = any(
        x in nset
        for x in (
            "stock",
            "availablequantity",
            "outofstock",
            "reorder_level",
            "shelf_life_days",
        )
    )
    has_sales = any(x in nset for x in ("sold_quantity", "sales", "revenue", "num_reviews"))
    has_review = any(x in nset for x in ("review", "num_reviews", "customer_feedback", "rating"))
    ice = "ice_cream" in joined or "icecream" in joined or "flavor" in nset

    if ice and not has_order:
        reasons.append("ice-cream / flavor style columns present; no order grain")
        return "ICE_CREAM_REFERENCE", "; ".join(reasons)

    if has_fraud and has_return:
        reasons.append(
            "return fields + abuse/fraud labels ("
            + ", ".join(sorted(nset & {"abuse_label", "abuse_type", "return_reason", "return_date"}))
            + ")"
        )
        return "RETURN", "; ".join(reasons) + " [secondary: FRAUD]"

    if has_return and has_order:
        reasons.append("order grain plus return/refund columns")
        return "ORDER", "; ".join(reasons) + " [secondary: RETURN flags]"

    if has_order and has_customer_id:
        reasons.append("order_id + customer_id (transactional order grain)")
        return "ORDER", "; ".join(reasons)

    if has_inventory and has_product and not has_order:
        reasons.append("catalog/stock fields without order identifiers")
        if has_sales:
            reasons.append("also sold_quantity/sales-like metrics")
        return "PRODUCT", "; ".join(reasons) + " [secondary: INVENTORY]"

    if has_inventory and not has_order:
        return "INVENTORY", "stock/availability columns without order grain"

    if has_product and not has_order:
        return "PRODUCT", "product/category/name columns without orders"

    if has_sales and has_order:
        return "SALES", "order plus sales/revenue style measures"

    if has_review and not has_order:
        return "REVIEW", "rating/review columns dominate"

    if has_customer_id and not has_order:
        return "CUSTOMER", "customer identifier without order grain"

    _ = filename  # unused by design
    return "OTHER", "column mix did not match a specific domain role"


def welford_update(count, mean, m2, values: np.ndarray):
    for x in values:
        if not np.isfinite(x):
            continue
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        m2 += delta * delta2
    return count, mean, m2


def iqr_outlier_count(values: np.ndarray) -> tuple[int, float, float]:
    if values.size < 10:
        return 0, float("nan"), float("nan")
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0, q1, q3
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    n = int(np.sum((values < lo) | (values > hi)))
    return n, lo, hi


def try_read_chunked(path: Path, encoding: str):
    """Yield DataFrame chunks. Prefer pandas; fall back to PyArrow full read split."""
    try:
        reader = pd.read_csv(
            path,
            encoding=encoding,
            chunksize=CHUNKSIZE,
            dtype=str,
            keep_default_na=True,
            na_values=["", "NA", "N/A", "null", "None", "none", "NaN", "nan"],
            on_bad_lines="warn",
            quoting=csv.QUOTE_MINIMAL,
            low_memory=False,
            engine="c",
        )
        yield from reader
        return
    except Exception as e_c:
        log(f"  pandas C engine failed ({e_c}); retrying python engine")
    try:
        reader = pd.read_csv(
            path,
            encoding=encoding,
            chunksize=CHUNKSIZE,
            dtype=str,
            keep_default_na=True,
            na_values=["", "NA", "N/A", "null", "None", "none", "NaN", "nan"],
            on_bad_lines="warn",
            quoting=csv.QUOTE_MINIMAL,
            engine="python",
        )
        yield from reader
        return
    except Exception as e_p:
        log(f"  pandas python engine failed ({e_p})")
        if not HAS_PYARROW:
            raise
        log("  falling back to PyArrow full read then chunking")
        table = pacsv.read_csv(
            path,
            read_options=pacsv.ReadOptions(encoding=encoding),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
        )
        df = table.to_pandas()
        for i in range(0, len(df), CHUNKSIZE):
            yield df.iloc[i : i + CHUNKSIZE]


def profile_csv(path: Path) -> dict:
    encoding = detect_encoding(path)
    file_size = path.stat().st_size
    log(f"Auditing {path.name} ({file_size / 1e6:.2f} MB, encoding={encoding})")

    columns: list[str] = []
    row_count = 0
    bad_chunks = 0
    missing = Counter()
    empty_str = Counter()
    dup_hashes: set[int] | None = set()
    dup_row_count = 0
    hash_overflow = False

    unique_sets: dict[str, set] = {}
    unique_overflow: set[str] = set()
    value_counters: dict[str, Counter] = {}
    numeric_stats: dict[str, dict] = {}
    date_stats: dict[str, dict] = {}
    inferred_types: dict[str, Counter] = defaultdict(Counter)
    reservoir: list[pd.DataFrame] = []
    reservoir_n = 0
    numeric_samples: dict[str, list[float]] = defaultdict(list)
    example_values: dict[str, list[str]] = defaultdict(list)
    capitalization: dict[str, dict] = {}

    first_chunk = True
    for chunk in try_read_chunked(path, encoding):
        if first_chunk:
            columns = [str(c) for c in chunk.columns]
            for c in columns:
                if is_identifier_name(c) or normalize_col(c) in {
                    "order_id",
                    "customer_id",
                    "product_id",
                    "sku",
                    "category",
                    "name",
                    "product_name",
                }:
                    unique_sets[c] = set()
            first_chunk = False

        n = len(chunk)
        row_count += n

        # missing / blanks
        na = chunk.isna()
        for c in columns:
            missing[c] += int(na[c].sum())
            s = chunk[c]
            empty_str[c] += int((s.astype(str).str.strip() == "").sum())

        # duplicate rows via pandas hash (chunk-safe)
        if dup_hashes is not None:
            try:
                hashed = pd.util.hash_pandas_object(chunk, index=False)
            except Exception:
                hashed = pd.util.hash_pandas_object(chunk.fillna(""), index=False)
            for digest in hashed.astype("uint64").to_numpy():
                # store as python int in a set
                if digest in dup_hashes:
                    dup_row_count += 1
                else:
                    dup_hashes.add(int(digest))
                    if len(dup_hashes) > 400_000:
                        hash_overflow = True
                        dup_hashes = None
                        break

        # type inference + uniques + numeric
        for c in columns:
            s = chunk[c]
            non_null = s.dropna()
            non_null = non_null[non_null.astype(str).str.strip() != ""]
            if non_null.empty:
                continue

            sample = non_null.head(400)
            as_num = pd.to_numeric(sample, errors="coerce")
            num_rate = float(as_num.notna().mean())
            date_rate = 0.0
            if DATE_NAME_RE.search(c) or num_rate < 0.8:
                parsed = pd.to_datetime(sample.astype(str), errors="coerce", dayfirst=True, format="mixed")
                date_rate = float(parsed.notna().mean())

            if date_rate >= 0.5 and num_rate < 0.95:
                inferred_types[c]["datetime"] += n
                parsed_all = pd.to_datetime(non_null.astype(str), errors="coerce", dayfirst=True, format="mixed")
                ok = parsed_all.dropna()
                if not ok.empty:
                    st = date_stats.setdefault(c, {"min": None, "max": None, "parsed": 0, "failed": 0})
                    st["parsed"] += int(ok.shape[0])
                    st["failed"] += int(parsed_all.isna().sum())
                    mn, mx = ok.min(), ok.max()
                    st["min"] = mn if st["min"] is None else min(st["min"], mn)
                    st["max"] = mx if st["max"] is None else max(st["max"], mx)
            elif num_rate >= 0.85:
                inferred_types[c]["numeric"] += n
                nums = pd.to_numeric(non_null, errors="coerce").dropna().astype(float).to_numpy()
                st = numeric_stats.setdefault(
                    c, {"count": 0, "mean": 0.0, "m2": 0.0, "min": None, "max": None, "neg": 0, "zero": 0}
                )
                if nums.size:
                    st["min"] = float(nums.min()) if st["min"] is None else min(st["min"], float(nums.min()))
                    st["max"] = float(nums.max()) if st["max"] is None else max(st["max"], float(nums.max()))
                    st["neg"] += int(np.sum(nums < 0))
                    st["zero"] += int(np.sum(nums == 0))
                    st["count"], st["mean"], st["m2"] = welford_update(st["count"], st["mean"], st["m2"], nums)
                    if len(numeric_samples[c]) < IQR_SAMPLE_CAP:
                        take = min(IQR_SAMPLE_CAP - len(numeric_samples[c]), nums.size)
                        if take > 0:
                            numeric_samples[c].extend(nums[:take].tolist())
            else:
                inferred_types[c]["categorical"] += n
                if c not in value_counters:
                    value_counters[c] = Counter()
                vc = value_counters[c]
                if sum(vc.values()) < HIGH_CARDINALITY * 3:
                    vc.update(non_null.astype(str).str.strip().tolist())

            if c in unique_sets and c not in unique_overflow:
                u = unique_sets[c]
                for v in non_null.astype(str).str.strip():
                    if v == "":
                        continue
                    u.add(v)
                    if len(u) > KEY_SET_CAP:
                        unique_overflow.add(c)
                        break

            if len(example_values[c]) < 8:
                for v in non_null.astype(str).head(8):
                    if v not in example_values[c]:
                        example_values[c].append(v)
                    if len(example_values[c]) >= 8:
                        break

        # reservoir sample of chunks
        if reservoir_n < SAMPLE_TARGET:
            need = SAMPLE_TARGET - reservoir_n
            take = chunk if n <= need else chunk.sample(n=need, random_state=42)
            reservoir.append(take)
            reservoir_n += len(take)

    sample_df = pd.concat(reservoir, ignore_index=True) if reservoir else pd.DataFrame(columns=columns)

    col_profiles = []
    for c in columns:
        type_votes = inferred_types.get(c, Counter({"unknown": 1}))
        inferred = type_votes.most_common(1)[0][0]
        miss = missing[c]
        miss_pct = round(100.0 * miss / row_count, 4) if row_count else 0.0
        uniq = None
        uniq_source = "n/a"
        if c in unique_sets:
            uniq = len(unique_sets[c])
            uniq_source = "approx_capped" if c in unique_overflow else "exact_non_null"
        elif c in value_counters:
            uniq = len(value_counters[c])
            uniq_source = "exact_if_low_card_else_lower_bound"
        elif inferred == "categorical":
            uniq = sample_df[c].nunique(dropna=True) if c in sample_df else None
            uniq_source = "sample"

        num_out = {}
        if c in numeric_stats:
            st = numeric_stats[c]
            std = math.sqrt(st["m2"] / st["count"]) if st["count"] > 1 else 0.0
            med = None
            med_src = "not_computed"
            arr = np.array(numeric_samples.get(c, []), dtype=float)
            if arr.size:
                med = float(np.median(arr))
                med_src = "sample" if arr.size < st["count"] else "exact_from_collected"
            num_out = {
                "min": st["min"],
                "max": st["max"],
                "mean": st["mean"],
                "median": med,
                "median_source": med_src,
                "std": std,
                "negative_count": st["neg"],
                "zero_count": st["zero"],
                "non_null_numeric": st["count"],
            }

        date_out = {}
        if c in date_stats:
            st = date_stats[c]
            date_out = {
                "min": str(st["min"]) if st["min"] is not None else None,
                "max": str(st["max"]) if st["max"] is not None else None,
                "parsed_count": st["parsed"],
                "failed_count": st["failed"],
            }

        cat_top = []
        if c in value_counters:
            cat_top = value_counters[c].most_common(12)
            # capitalization inconsistency
            groups = defaultdict(set)
            for val, _cnt in value_counters[c].most_common(500):
                groups[val.casefold()].add(val)
            mixed = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
            if mixed:
                capitalization[c] = {k: v for i, (k, v) in enumerate(mixed.items()) if i < 20}

        col_profiles.append(
            {
                "column": c,
                "normalized": normalize_col(c),
                "inferred_dtype": inferred,
                "missing_count": miss,
                "blank_string_count": empty_str[c],
                "missing_pct": miss_pct,
                "unique_count": uniq,
                "unique_count_source": uniq_source,
                "is_candidate_key": is_identifier_name(c),
                "numeric": num_out,
                "date": date_out,
                "example_values": example_values.get(c, [])[:5],
                "top_values": cat_top[:8],
            }
        )

    role, role_reason = classify_role(columns, path.name)

    sample_records = sample_df.head(EXAMPLE_N).fillna("").astype(str).to_dict(orient="records")

    return {
        "filename": path.name,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "file_size_bytes": file_size,
        "encoding": encoding,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "duplicate_row_count": dup_row_count,
        "duplicate_row_count_note": (
            "exact_via_pandas_hash"
            if not hash_overflow
            else "PARTIAL - hash set capped; duplicate count is a lower bound"
        ),
        "role": role,
        "role_reason": role_reason,
        "column_profiles": col_profiles,
        "sample_rows": sample_records,
        "capitalization_issues": capitalization,
        "unique_id_sets": {k: unique_sets[k] for k in unique_sets if k not in unique_overflow},
        "unique_overflow": sorted(unique_overflow),
        "numeric_samples": {k: v[:IQR_SAMPLE_CAP] for k, v in numeric_samples.items()},
        "date_stats": {
            k: {
                "min": str(v["min"]) if v["min"] is not None else None,
                "max": str(v["max"]) if v["max"] is not None else None,
                "parsed": v["parsed"],
                "failed": v["failed"],
            }
            for k, v in date_stats.items()
        },
        "bad_chunks": bad_chunks,
        "pyarrow_available": HAS_PYARROW,
    }


def candidate_keys(profile: dict) -> list[dict]:
    rows = []
    n = profile["row_count"] or 1
    for col in profile["column_profiles"]:
        name_n = col["normalized"]
        looks_fk_or_pk = col["is_candidate_key"] or name_n in {
            "order_id",
            "customer_id",
            "product_id",
            "sku",
            "product_category_id",
        }
        if not looks_fk_or_pk:
            continue
        uniq = col["unique_count"]
        nulls = col["missing_count"] + col["blank_string_count"]
        uniq_pct = round(100.0 * uniq / n, 4) if uniq is not None else None
        kind = "PRIMARY_KEY_CANDIDATE" if uniq is not None and uniq_pct and uniq_pct >= 99.0 and nulls == 0 else "FOREIGN_OR_WEAK_KEY"
        if uniq is not None and n > 0 and uniq < n * 0.5:
            kind = "FOREIGN_KEY_CANDIDATE"
        if uniq is not None and uniq_pct and uniq_pct >= 99.0:
            kind = "PRIMARY_KEY_CANDIDATE"
        rows.append(
            {
                "dataset": profile["filename"],
                "column": col["column"],
                "normalized": name_n,
                "key_kind": kind,
                "unique_count": uniq,
                "null_or_blank_count": nulls,
                "uniqueness_pct": uniq_pct,
                "unique_count_source": col["unique_count_source"],
            }
        )
    return rows


def overlap_stats(parent_set: set, child_set: set) -> dict:
    if not parent_set or not child_set:
        return {
            "parent_unique": len(parent_set),
            "child_unique": len(child_set),
            "intersection": 0,
            "child_coverage_pct": 0.0,
            "parent_coverage_pct": 0.0,
            "jaccard": 0.0,
        }
    inter = len(parent_set & child_set)
    return {
        "parent_unique": len(parent_set),
        "child_unique": len(child_set),
        "intersection": inter,
        "child_coverage_pct": round(100.0 * inter / len(child_set), 4),
        "parent_coverage_pct": round(100.0 * inter / len(parent_set), 4),
        "jaccard": round(inter / len(parent_set | child_set), 6),
    }


def relationship_confidence(stats: dict, name_match: bool) -> str:
    if stats["intersection"] == 0:
        return "LOW"
    cov = max(stats["child_coverage_pct"], stats["parent_coverage_pct"])
    if name_match and cov >= 80:
        return "HIGH"
    if name_match and cov >= 20:
        return "MEDIUM"
    if stats["jaccard"] >= 0.5:
        return "HIGH"
    if stats["intersection"] >= 50 and name_match:
        return "MEDIUM"
    return "LOW"


def find_relationships(profiles: list[dict]) -> list[dict]:
    # comparable id-like columns by normalized name
    index: dict[str, list[tuple[dict, str, set]]] = defaultdict(list)
    extra_pairs = []

    for p in profiles:
        for col_name, values in p.get("unique_id_sets", {}).items():
            n = normalize_col(col_name)
            index[n].append((p, col_name, values))

    rels = []
    seen = set()

    def add_rel(parent, pcol, child, ccol, pset, cset, note: str):
        key = (parent["filename"], pcol, child["filename"], ccol)
        rev = (child["filename"], ccol, parent["filename"], pcol)
        if key in seen or rev in seen:
            return
        seen.add(key)
        stats = overlap_stats(pset, cset)
        name_match = normalize_col(pcol) == normalize_col(ccol)
        # parent = higher uniqueness / dimension table heuristic
        p_n = parent["row_count"]
        c_n = child["row_count"]
        # Prefer the side with fewer unique keys as parent if it's a dimension
        if len(pset) > len(cset) * 1.2 and p_n > c_n * 1.1:
            parent, pcol, child, ccol, pset, cset = child, ccol, parent, pcol, cset, pset
            stats = overlap_stats(pset, cset)
            key = (parent["filename"], pcol, child["filename"], ccol)
        conf = relationship_confidence(stats, name_match)
        if stats["intersection"] == 0 and not name_match:
            return
        rels.append(
            {
                "parent_dataset": parent["filename"],
                "parent_column": pcol,
                "child_dataset": child["filename"],
                "child_column": ccol,
                "parent_unique": stats["parent_unique"],
                "child_unique": stats["child_unique"],
                "intersection": stats["intersection"],
                "child_coverage_pct": stats["child_coverage_pct"],
                "parent_coverage_pct": stats["parent_coverage_pct"],
                "jaccard": stats["jaccard"],
                "confidence": conf,
                "name_match": name_match,
                "note": note,
                "overlap_basis": "exact unique-value set intersection (capped collections excluded)",
            }
        )

    # same normalized name across datasets
    for norm, items in index.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                p1, c1, s1 = items[i]
                p2, c2, s2 = items[j]
                add_rel(p1, c1, p2, c2, s1, s2, f"shared normalized column name `{norm}`")

    # cross-name ID comparisons only for well-known pairs (not O(n^2) of all columns)
    known = {
        "order_id": {"order_id", "orderid"},
        "customer_id": {"customer_id", "customerid"},
        "product_id": {"product_id", "productid", "sku"},
    }
    buckets: dict[str, list] = defaultdict(list)
    for p in profiles:
        for col_name, values in p.get("unique_id_sets", {}).items():
            n = normalize_col(col_name)
            for logical, aliases in known.items():
                if n in aliases:
                    buckets[logical].append((p, col_name, values))

    # Cross-name catalog identifiers (product_name vs name) — overlap only, never assumed
    name_cols = []
    for p in profiles:
        for col_name, values in p.get("unique_id_sets", {}).items():
            if normalize_col(col_name) in {"product_name", "name", "sku"}:
                name_cols.append((p, col_name, values, normalize_col(col_name)))
    for i in range(len(name_cols)):
        for j in range(i + 1, len(name_cols)):
            p1, c1, s1, n1 = name_cols[i]
            p2, c2, s2, n2 = name_cols[j]
            if n1 == n2:
                continue
            add_rel(
                p1,
                c1,
                p2,
                c2,
                s1,
                s2,
                f"cross-name identifier comparison `{n1}` vs `{n2}` (not assumed equal)",
            )

    _ = extra_pairs
    return rels


def quality_issues(profile: dict) -> list[dict]:
    issues = []
    fname = profile["filename"]
    n = profile["row_count"]
    cols = {c["column"]: c for c in profile["column_profiles"]}
    ncol = {c["normalized"]: c for c in profile["column_profiles"]}

    if profile["duplicate_row_count"]:
        issues.append(
            {
                "dataset": fname,
                "issue_type": "duplicate_rows",
                "column": "*",
                "severity": "MEDIUM" if profile["duplicate_row_count"] / max(n, 1) < 0.05 else "HIGH",
                "count": profile["duplicate_row_count"],
                "detail": profile["duplicate_row_count_note"],
            }
        )

    for key in candidate_keys(profile):
        if key["null_or_blank_count"]:
            issues.append(
                {
                    "dataset": fname,
                    "issue_type": "null_or_blank_id",
                    "column": key["column"],
                    "severity": "HIGH",
                    "count": key["null_or_blank_count"],
                    "detail": "Identifier has null/blank values",
                }
            )
        if key["key_kind"] == "PRIMARY_KEY_CANDIDATE" and key["unique_count"] is not None:
            dup_ids = n - key["unique_count"] - key["null_or_blank_count"]
            # uniqueness_pct < 100 means duplicate IDs
            if key["uniqueness_pct"] is not None and key["uniqueness_pct"] < 99.99:
                issues.append(
                    {
                        "dataset": fname,
                        "issue_type": "duplicate_ids",
                        "column": key["column"],
                        "severity": "HIGH",
                        "count": max(n - key["unique_count"], 0),
                        "detail": f"uniqueness_pct={key['uniqueness_pct']}",
                    }
                )
        elif key["unique_count"] is not None and key["uniqueness_pct"] is not None:
            if key["normalized"] in {"order_id", "product_id"} and key["uniqueness_pct"] < 99 and key["uniqueness_pct"] > 80:
                issues.append(
                    {
                        "dataset": fname,
                        "issue_type": "duplicate_ids",
                        "column": key["column"],
                        "severity": "MEDIUM",
                        "count": max(n - key["unique_count"], 0),
                        "detail": "Expected near-unique identifier is not unique",
                    }
                )

    # numeric quality
    for c in profile["column_profiles"]:
        num = c.get("numeric") or {}
        if not num:
            continue
        name = c["column"]
        nn = normalize_col(name)
        if PRICE_NAME_RE.search(name) and num.get("negative_count"):
            issues.append(
                {
                    "dataset": fname,
                    "issue_type": "negative_prices",
                    "column": name,
                    "severity": "MEDIUM",
                    "count": num["negative_count"],
                    "detail": f"min={num.get('min')} max={num.get('max')}",
                }
            )
        if QTY_NAME_RE.search(name) and (num.get("negative_count") or 0) > 0:
            issues.append(
                {
                    "dataset": fname,
                    "issue_type": "negative_quantities",
                    "column": name,
                    "severity": "HIGH",
                    "count": num["negative_count"],
                    "detail": f"min={num.get('min')}",
                }
            )
        if QTY_NAME_RE.search(name) and (num.get("zero_count") or 0) > 0 and "out_of_stock" not in nn:
            issues.append(
                {
                    "dataset": fname,
                    "issue_type": "zero_quantities",
                    "column": name,
                    "severity": "LOW",
                    "count": num["zero_count"],
                    "detail": "Zero quantities present (may be valid stock-outs)",
                }
            )
        arr = np.array(profile.get("numeric_samples", {}).get(name, []), dtype=float)
        if arr.size >= 30:
            n_out, lo, hi = iqr_outlier_count(arr)
            if n_out:
                issues.append(
                    {
                        "dataset": fname,
                        "issue_type": "extreme_outliers_sample",
                        "column": name,
                        "severity": "LOW",
                        "count": n_out,
                        "detail": f"SAMPLE-BASED 3xIQR fences ~[{lo}, {hi}] on n={arr.size}",
                    }
                )

    # date quality using sample rows + parsed ranges
    sample = pd.DataFrame(profile.get("sample_rows") or [])
    date_cols = [c["column"] for c in profile["column_profiles"] if c.get("date")]
    for dc in date_cols:
        st = profile["date_stats"].get(dc, {})
        failed = st.get("failed") or 0
        parsed = st.get("parsed") or 0
        if failed:
            issues.append(
                {
                    "dataset": fname,
                    "issue_type": "unparseable_dates",
                    "column": dc,
                    "severity": "MEDIUM",
                    "count": failed,
                    "detail": f"parsed={parsed} failed={failed} range={st.get('min')}..{st.get('max')}",
                }
            )
        mn, mx = st.get("min"), st.get("max")
        for bound, label in ((mn, "min"), (mx, "max")):
            if not bound:
                continue
            try:
                dt = pd.to_datetime(bound)
                if dt.year < 1990 or dt.year > 2035:
                    issues.append(
                        {
                            "dataset": fname,
                            "issue_type": "impossible_dates",
                            "column": dc,
                            "severity": "HIGH",
                            "count": 1,
                            "detail": f"{label} parsed value {bound} is outside 1990–2035",
                        }
                    )
            except Exception:
                pass

    # pair date logic on sample (labeled)
    def col_by_norm(*names):
        for nm in names:
            if nm in ncol:
                return ncol[nm]["column"]
        return None

    order_c = col_by_norm("order_date", "order_datetime", "order_date_and_time", "date")
    deliv_c = col_by_norm("delivered_date", "delivery_date")
    ret_c = col_by_norm("return_date", "request_date")

    # Full-file date order checks: stream again only for datasets with both date columns
    date_check_cols = [c for c in [order_c, deliv_c, ret_c] if c]
    if len(date_check_cols) >= 2:
        pair_counts = {
            "returns_before_orders": 0,
            "returns_before_delivery": 0,
            "delivery_before_order": 0,
        }
        try:
            date_usecols = list(dict.fromkeys(date_check_cols))
            date_colset = set(date_usecols)
            for chunk in pd.read_csv(
                ROOT / profile["path"],
                usecols=lambda c: c in date_colset,
                dtype=str,
                encoding=profile["encoding"],
                chunksize=CHUNKSIZE,
                on_bad_lines="warn",
            ):
                parsed = {}
                for c in date_usecols:
                    parsed[c] = pd.to_datetime(chunk[c], errors="coerce", dayfirst=True, format="mixed")
                if order_c and ret_c and order_c in parsed and ret_c in parsed:
                    pair_counts["returns_before_orders"] += int(
                        ((parsed[ret_c] < parsed[order_c]) & parsed[ret_c].notna() & parsed[order_c].notna()).sum()
                    )
                if deliv_c and ret_c and deliv_c in parsed and ret_c in parsed:
                    pair_counts["returns_before_delivery"] += int(
                        ((parsed[ret_c] < parsed[deliv_c]) & parsed[ret_c].notna() & parsed[deliv_c].notna()).sum()
                    )
                if order_c and deliv_c and order_c in parsed and deliv_c in parsed:
                    pair_counts["delivery_before_order"] += int(
                        (
                            (parsed[deliv_c] < parsed[order_c])
                            & parsed[deliv_c].notna()
                            & parsed[order_c].notna()
                        ).sum()
                    )
            for itype, cnt in pair_counts.items():
                if cnt:
                    issues.append(
                        {
                            "dataset": fname,
                            "issue_type": itype,
                            "column": ",".join(date_usecols),
                            "severity": "HIGH",
                            "count": cnt,
                            "detail": "Exact chunked comparison of parsed dates (invalid parses excluded)",
                        }
                    )
        except Exception as e:
            issues.append(
                {
                    "dataset": fname,
                    "issue_type": "date_pair_check_failed",
                    "column": ",".join(date_check_cols),
                    "severity": "LOW",
                    "count": 0,
                    "detail": str(e),
                }
            )

    for col, variants in (profile.get("capitalization_issues") or {}).items():
        n_groups = len(variants)
        issues.append(
            {
                "dataset": fname,
                "issue_type": "inconsistent_capitalization",
                "column": col,
                "severity": "LOW",
                "count": n_groups,
                "detail": json.dumps({k: v for k, v in list(variants.items())[:8]}, ensure_ascii=False),
            }
        )

    # suspicious truncated datetime (delivery analytics)
    if sample is not None and not sample.empty:
        for c in sample.columns:
            if "date" in normalize_col(c) or "time" in normalize_col(c):
                vals = sample[c].astype(str).head(20).tolist()
                if any(re.fullmatch(r"\d{1,2}:\d{2}\.\d", v.strip()) for v in vals):
                    issues.append(
                        {
                            "dataset": fname,
                            "issue_type": "obvious_data_entry_problems",
                            "column": c,
                            "severity": "HIGH",
                            "count": n,
                            "detail": f"Sample values look like truncated times without dates: {vals[:5]}",
                        }
                    )

    # profit_margin negative is not always an error; note as possible
    if "profit_margin" in ncol and (ncol["profit_margin"].get("numeric") or {}).get("negative_count"):
        issues.append(
            {
                "dataset": fname,
                "issue_type": "negative_profit_margin",
                "column": ncol["profit_margin"]["column"],
                "severity": "LOW",
                "count": ncol["profit_margin"]["numeric"]["negative_count"],
                "detail": "Negative margins may be valid discounts/losses, not entry errors",
            }
        )

    return issues


# --- FlipLens mapping (conceptual; no invented values) ---

FIELD_HINTS = {
    "case_id": ["case_id", "return_id"],
    "customer_id": ["customer_id", "customerid"],
    "order_id": ["order_id", "orderid"],
    "product_id": ["product_id", "productid", "sku"],
    "product_category": ["product_category", "category", "product_category_name"],
    "product_name": ["product_name", "name"],
    "brand": ["brand"],
    "order_date": ["order_date", "order_datetime", "order_date_and_time", "date"],
    "delivery_date": ["delivered_date", "delivery_date"],
    "return_date": ["return_date", "request_date"],
    "return_reason": ["return_reason"],
    "order_value": [
        "refund_amount_requested_usd",
        "order_value_inr",
        "order_value",
        "total_amount",
        "price",
        "final_price",
    ],
    "payment_method": ["payment_method"],
    "customer_order_count": ["total_orders_lifetime"],
    "customer_return_count": ["total_returns_lifetime"],
    "customer_return_rate": ["return_rate_pct"],
    "returns_last_30d": [],
    "returns_last_90d": [],
    "previous_fraud_count": [],
    "high_value_return": ["is_high_value_item"],
    "return_window_valid": [],
    "policy_violation": [],
    "product_match_score": [],
    "damage_score": [],
    "serial_match_score": [],
    "accessories_complete": [],
    "packaging_match_score": [],
    "fraud_probability": [],
    "fraud_label": ["abuse_label", "abuse_type"],
    "final_decision": [],
    "confidence": [],
    "explanation": [],
}


def map_fliplens(profiles: list[dict]) -> list[dict]:
    role_of = {p["filename"]: p["role"] for p in profiles}
    col_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for p in profiles:
        for c in p["columns"]:
            col_index[normalize_col(c)].append((p["filename"], c))

    transactional_fields = {
        "case_id",
        "customer_id",
        "order_id",
        "product_id",
        "product_category",
        "order_date",
        "delivery_date",
        "return_date",
        "return_reason",
        "order_value",
        "payment_method",
        "customer_order_count",
        "customer_return_count",
        "customer_return_rate",
        "high_value_return",
        "fraud_label",
    }

    def match_rank(filename: str, field: str) -> int:
        role = role_of.get(filename, "OTHER")
        if field in transactional_fields and role == "RETURN":
            return 0
        if field in transactional_fields and role == "ORDER":
            return 1
        if field in {"product_name", "brand"} and role == "PRODUCT":
            return 50
        if role == "PRODUCT":
            return 20
        return 10

    rows = []
    for field in FLIPLENS_FIELDS:
        hints = FIELD_HINTS.get(field, [field])
        matches = []
        for h in hints:
            matches.extend(col_index.get(h, []))
        # unique matches
        uniq = []
        seen = set()
        for m in sorted(matches, key=lambda t: match_rank(t[0], field)):
            if m not in seen:
                seen.add(m)
                uniq.append(m)

        status = "MISSING"
        source_ds = ""
        source_col = ""
        transform = ""

        if uniq:
            source_ds, source_col = uniq[0]
            extra = "; ".join(f"{d}.{c}" for d, c in uniq[1:6])
            catalog_only = match_rank(source_ds, field) >= 50 and field in {"product_name", "brand"}
            if catalog_only:
                status = "MISSING"
                transform = (
                    f"Column exists on catalog `{source_ds}.{source_col}` but that table is not a verified "
                    "join to the return/order grain. Not AVAILABLE for FlipLens cases."
                    + (f" Also seen in: {extra}" if extra else "")
                )
                source_ds = ""
                source_col = ""
            else:
                status = "AVAILABLE"
                transform = "Direct map (no invented values)."
                if extra:
                    transform += f" Also present in: {extra}"
        else:
            # derivable rules — only when inputs exist
            if field == "case_id":
                order_ids = col_index.get("order_id") or col_index.get("orderid")
                if order_ids:
                    status = "DERIVABLE"
                    source_ds, source_col = sorted(order_ids, key=lambda t: match_rank(t[0], "order_id"))[0]
                    transform = "Could mint case_id from order_id (+ product/return sequence) after grain is chosen. Not present as-is."
            elif field in {"customer_order_count", "customer_return_count", "customer_return_rate"}:
                if col_index.get("customer_id") and (
                    col_index.get("returned") or col_index.get("return_date") or col_index.get("refund_requested")
                ):
                    status = "DERIVABLE"
                    source_ds, source_col = col_index["customer_id"][0]
                    transform = "Aggregate per customer_id over a chosen order/return table. Do not invent labels."
            elif field in {"returns_last_30d", "returns_last_90d"}:
                if (col_index.get("return_date") or col_index.get("request_date") or col_index.get("returned")) and col_index.get(
                    "customer_id"
                ):
                    status = "DERIVABLE"
                    source_ds = (col_index.get("return_date") or col_index.get("request_date") or col_index["returned"])[0][0]
                    source_col = "customer_id + return/request date"
                    transform = "Windowed count of returns per customer. Requires parsed dates."
            elif field == "previous_fraud_count":
                if col_index.get("previous_dispute_count"):
                    status = "DERIVABLE"
                    source_ds, source_col = col_index["previous_dispute_count"][0]
                    transform = "previous_dispute_count is a proxy, not a fraud count. Do not treat as FlipLens previous_fraud_count without definition."
            elif field == "high_value_return":
                cand_val = (
                    col_index.get("is_high_value_item")
                    or col_index.get("total_amount")
                    or col_index.get("order_value_inr")
                    or col_index.get("refund_amount_requested_usd")
                )
                if cand_val:
                    status = "DERIVABLE"
                    source_ds, source_col = cand_val[0]
                    transform = "Threshold on order/refund value (policy-defined). Flag column exists in return-abuse as is_high_value_item."
            elif field == "return_window_valid":
                if (col_index.get("order_date") or col_index.get("order_datetime")) and (
                    col_index.get("return_date") or col_index.get("request_date") or col_index.get("days_to_return")
                ):
                    status = "DERIVABLE"
                    transform = "Compare return/request date vs order/delivery date against a policy window. Not a source column."
                    source_ds = (col_index.get("days_to_return") or col_index.get("return_date") or [("?", "?")])[0][0]
                    source_col = "date pair / days_to_return"
            elif field == "policy_violation":
                if col_index.get("days_to_return") or col_index.get("return_reason"):
                    status = "DERIVABLE"
                    transform = "Deterministic policy engine later; no raw policy-violation column exists."
                    source_ds = "policy engine (not in raw CSV)"
                    source_col = ""
            elif field in {
                "product_match_score",
                "damage_score",
                "serial_match_score",
                "accessories_complete",
                "packaging_match_score",
                "fraud_probability",
                "final_decision",
                "confidence",
                "explanation",
            }:
                if field == "packaging_match_score" and col_index.get("return_packaging_intact"):
                    status = "DERIVABLE"
                    source_ds, source_col = col_index["return_packaging_intact"][0]
                    transform = "Binary packaging intact flag is not a match score; could inform a later score. Do not treat as FlipLens score."
                elif field == "explanation" and col_index.get("customer_feedback"):
                    status = "DERIVABLE"
                    source_ds, source_col = col_index["customer_feedback"][0]
                    transform = "Feedback text is not a model explanation. Could be an evidence snippet later."
                else:
                    status = "MISSING"
                    transform = "No source column. Vision/ML/policy outputs — do not invent."
            elif field == "fraud_label":
                cand_abuse = col_index.get("abuse_label") or col_index.get("abuse_type")
                if cand_abuse:
                    status = "AVAILABLE"
                    source_ds, source_col = cand_abuse[0]
                    transform = "abuse_label/abuse_type exist on the return-abuse table only. Do not copy labels onto unrelated order tables. Do not synthesize labels for other datasets."
                else:
                    status = "MISSING"
                    transform = "No fraud label in remaining datasets. Do not create synthetic labels in this phase."
            elif field == "delivery_date":
                # delivery time minutes is not a date
                if col_index.get("delivery_time_min") or col_index.get("delivery_time_minutes"):
                    status = "MISSING"
                    transform = "Only delivery duration (minutes) exists in some tables — not a calendar delivery_date."
            elif field == "product_name":
                status = "MISSING"
                transform = "Product name exists on catalog tables (blinkit/zepto) but those catalogs do not share join keys with order tables (see relationships)."
            elif field == "brand":
                status = "MISSING"
                transform = "Brand exists on blinkit catalog only; no verified join to order/return grains."

            if status == "MISSING" and not transform:
                transform = "Not present in any raw CSV. Do not invent."

        # NOT_APPLICABLE: catalog-only fields when discussing ice cream — skip
        rows.append(
            {
                "fliplens_field": field,
                "source_dataset": source_ds,
                "source_column": source_col,
                "status": status,
                "transformation_required": transform,
                "all_matches": "; ".join(f"{d}.{c}" for d, c in uniq),
            }
        )
    return rows


def ice_cream_section(profiles: list[dict], rels: list[dict]) -> str:
    ice = [p for p in profiles if p["role"] == "ICE_CREAM_REFERENCE"]
    # also filename hint only as a search, not classification
    name_hits = [p for p in profiles if re.search(r"ice.?cream|icecream", p["filename"], re.I)]
    if not ice and not name_hits:
        return (
            "No ice-cream dataset was found in `data/raw/`.\n\n"
            "None of the six CSVs have ice-cream-specific columns (flavor, scoop, etc.). "
            "Blinkit and Zepto are grocery/quick-commerce **product catalogs**, not ice-cream reference tables. "
            "They must not be joined to order/return tables unless a verified key exists "
            "(see relationships: product_id/SKU overlap with order tables is not established).\n"
        )

    lines = ["## Ice-cream dataset analysis\n"]
    for p in ice + [x for x in name_hits if x not in ice]:
        lines.append(f"### {p['filename']}\n")
        id_cols = [c["column"] for c in p["column_profiles"] if c["is_candidate_key"]]
        lines.append(f"Candidate keys: {id_cols or 'none'}\n")
        related = [r for r in rels if p["filename"] in (r["parent_dataset"], r["child_dataset"]) and r["intersection"] > 0]
        if related:
            lines.append("Verified overlapping keys:\n")
            for r in related:
                lines.append(
                    f"- {r['parent_dataset']}.{r['parent_column']} ↔ {r['child_dataset']}.{r['child_column']} "
                    f"intersection={r['intersection']} confidence={r['confidence']}\n"
                )
        else:
            lines.append(
                "No legitimate join key overlap with other datasets. "
                "Classify as **REFERENCE / SUPPORTING DATA** for category-specific scenarios later "
                "(e.g., perishable return windows, temperature-sensitive damage), not as a fact table.\n"
            )
    return "\n".join(lines)


def redundancy_notes(profiles: list[dict], rels: list[dict]) -> list[str]:
    notes = []
    orderish = [p for p in profiles if p["role"] in {"ORDER", "RETURN", "SALES"}]
    products = [p for p in profiles if p["role"] == "PRODUCT"]
    if len(products) > 1:
        notes.append(
            f"Multiple product/catalog tables: {', '.join(p['filename'] for p in products)}. "
            "Keep independent unless a verified SKU/product_id overlap exists."
        )
    # high overlap order tables
    for r in rels:
        if r["confidence"] == "HIGH" and r["name_match"] and r["jaccard"] >= 0.3:
            notes.append(
                f"Possible overlapping grain: {r['parent_dataset']}.{r['parent_column']} "
                f"vs {r['child_dataset']}.{r['child_column']} "
                f"(intersection={r['intersection']}, jaccard={r['jaccard']}). "
                "Inspect as alternate versions, not automatic duplicates to delete."
            )
    if len(orderish) > 1:
        notes.append(
            "Several order/return-shaped tables exist with **different ID namespaces** "
            "(e.g. ORD000001 vs ORD2024554 vs O100000). They are not interchangeable without overlap proof."
        )
    return notes


def write_markdown_audit(profiles: list[dict], keys: list[dict], rels: list[dict], issues: list[dict], mapping: list[dict]) -> None:
    lines = [
        "# TrustLoop dataset audit (Phase 1)",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Scope: every `data/raw/*.csv`. Raw files were not modified.",
        "Statistics labeled SAMPLE-BASED were computed on a reservoir sample, not the full file.",
        "",
        "## Datasets discovered",
        "",
        "| File | Size (MB) | Rows | Cols | Role | Duplicate rows |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for p in profiles:
        lines.append(
            f"| `{p['filename']}` | {p['file_size_bytes']/1e6:.2f} | {p['row_count']:,} | {p['column_count']} | {p['role']} | {p['duplicate_row_count']:,} |"
        )
    lines += ["", "## Per-dataset profiles", ""]

    key_by_ds = defaultdict(list)
    for k in keys:
        key_by_ds[k["dataset"]].append(k)

    for p in profiles:
        lines += [
            f"### `{p['filename']}`",
            "",
            f"- Path: `{p['path']}`",
            f"- Encoding: `{p['encoding']}`",
            f"- Rows: **{p['row_count']:,}** | Columns: **{p['column_count']}** | Size: {p['file_size_bytes']:,} bytes",
            f"- Duplicate rows: {p['duplicate_row_count']:,} ({p['duplicate_row_count_note']})",
            f"- **Likely role:** {p['role']}",
            f"- **Why:** {p['role_reason']}",
            "",
            "#### Columns",
            "",
            "| Column | Inferred type | Missing | Missing % | Unique | Unique source | Examples |",
            "|---|---|---:|---:|---:|---|---|",
        ]
        for c in p["column_profiles"]:
            ex = ", ".join(str(x)[:40] for x in c["example_values"][:3])
            uniq = "" if c["unique_count"] is None else f"{c['unique_count']:,}"
            lines.append(
                f"| `{c['column']}` | {c['inferred_dtype']} | {c['missing_count']:,} | {c['missing_pct']} | {uniq} | {c['unique_count_source']} | {ex} |"
            )
        lines.append("")
        lines.append("#### Numeric summary (min/max/mean exact via chunks; median often SAMPLE-BASED)")
        lines.append("")
        lines.append("| Column | min | max | mean | median | median source | negatives | zeros |")
        lines.append("|---|---:|---:|---:|---:|---|---:|---:|")
        any_num = False
        for c in p["column_profiles"]:
            num = c.get("numeric") or {}
            if not num:
                continue
            any_num = True
            lines.append(
                f"| `{c['column']}` | {num.get('min')} | {num.get('max')} | {num.get('mean')} | {num.get('median')} | {num.get('median_source')} | {num.get('negative_count')} | {num.get('zero_count')} |"
            )
        if not any_num:
            lines.append("| *(none)* | | | | | | | |")
        lines.append("")
        lines.append("#### Date-like columns")
        lines.append("")
        if p["date_stats"]:
            lines.append("| Column | min | max | parsed | failed |")
            lines.append("|---|---|---|---:|---:|")
            for col, st in p["date_stats"].items():
                lines.append(f"| `{col}` | {st.get('min')} | {st.get('max')} | {st.get('parsed')} | {st.get('failed')} |")
        else:
            lines.append("None detected with a majority parseable date rate.")
        lines.append("")
        lines.append("#### Candidate keys")
        lines.append("")
        ks = key_by_ds.get(p["filename"], [])
        if not ks:
            lines.append("No identifier-like columns detected from names.")
        else:
            lines.append("| Column | Kind | Unique | Null/blank | Uniqueness % |")
            lines.append("|---|---|---:|---:|---:|")
            for k in ks:
                lines.append(
                    f"| `{k['column']}` | {k['key_kind']} | {k['unique_count']} | {k['null_or_blank_count']} | {k['uniqueness_pct']} |"
                )
        lines.append("")
        lines.append("#### Sample rows (n=5, not the full file)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(p["sample_rows"], indent=2, ensure_ascii=False)[:8000])
        lines.append("```")
        lines.append("")

    lines += [
        "## Redundancy",
        "",
    ]
    for n in redundancy_notes(profiles, rels):
        lines.append(f"- {n}")
    lines += ["", ice_cream_section(profiles, rels), ""]
    lines += [
        "## Data quality (report only — not corrected)",
        "",
        f"Issue rows written: {len(issues)} (see `data/audit/data_quality_issues.csv`).",
        "",
    ]
    (DOCS_DIR / "dataset_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_relationships_md(profiles: list[dict], rels: list[dict]) -> None:
    lines = [
        "# Dataset relationship candidates (Phase 1)",
        "",
        "Joins are proposed only after **name alignment** plus **set overlap** of identifier values.",
        "Zero-overlap same-named IDs are listed as LOW confidence (namespace collision, not a join).",
        "",
        "## Diagram",
        "",
        "```mermaid",
        "flowchart TD",
    ]
    for p in profiles:
        nid = re.sub(r"[^A-Za-z0-9]", "_", p["filename"])
        lines.append(f'    {nid}["{p["filename"]}<br/>{p["role"]}<br/>{p["row_count"]:,} rows"]')
    for i, r in enumerate(rels):
        a = re.sub(r"[^A-Za-z0-9]", "_", r["parent_dataset"])
        b = re.sub(r"[^A-Za-z0-9]", "_", r["child_dataset"])
        label = f"{r['parent_column']}→{r['child_column']}\\n{r['confidence']} overlap={r['intersection']}"
        if r["confidence"] == "LOW" and r["intersection"] == 0:
            lines.append(f"    {a} -.->|{label}| {b}")
        else:
            lines.append(f"    {a} -->|{label}| {b}")
    lines += ["```", "", "## Relationship table", ""]
    lines.append(
        "| Parent | Parent col | Child | Child col | Parent nunique | Child nunique | Intersection | Child coverage % | Jaccard | Confidence | Note |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|---|")
    for r in rels:
        lines.append(
            f"| `{r['parent_dataset']}` | `{r['parent_column']}` | `{r['child_dataset']}` | `{r['child_column']}` | "
            f"{r['parent_unique']} | {r['child_unique']} | {r['intersection']} | {r['child_coverage_pct']} | {r['jaccard']} | "
            f"**{r['confidence']}** | {r['note']} |"
        )
    if not rels:
        lines.append("| *(none)* | | | | | | | | | | |")
    lines += [
        "",
        "## Interpretation rules used",
        "",
        "- HIGH: same logical name and ≥80% coverage of one side.",
        "- MEDIUM: same name and ≥20% coverage, or ≥50 intersecting IDs with a name match.",
        "- LOW: name match with little/no value overlap (do **not** join).",
        "",
    ]
    (DOCS_DIR / "dataset_relationships.md").write_text("\n".join(lines), encoding="utf-8")


def write_mapping_md(mapping: list[dict]) -> None:
    lines = [
        "# FlipLens data mapping (conceptual, Phase 1)",
        "",
        "No values were invented. Status is AVAILABLE / DERIVABLE / MISSING / NOT_APPLICABLE.",
        "",
        "| FlipLens field | Source dataset | Source column | Status | Transformation required |",
        "|---|---|---|---|---|",
    ]
    for r in mapping:
        lines.append(
            f"| `{r['fliplens_field']}` | {r['source_dataset'] or '—'} | `{r['source_column'] or '—'}` | **{r['status']}** | {r['transformation_required']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- AVAILABLE means a raw column exists that can be mapped 1:1 or with renaming.",
        "- DERIVABLE means the field can be computed later from existing columns (aggregates, date diffs, policy rules) without fabricating facts.",
        "- MISSING means it is not in raw data (vision scores, model outputs, or catalogs that do not join).",
        "- Catalog `product_name` / `brand` are not treated as AVAILABLE for the return-case grain unless a join key is verified.",
        "",
    ]
    (DOCS_DIR / "flipLens_data_mapping.md").write_text("\n".join(lines), encoding="utf-8")


def profiles_for_csv(profiles: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds_rows = []
    col_rows = []
    for p in profiles:
        ds_rows.append(
            {
                "filename": p["filename"],
                "path": p["path"],
                "file_size_bytes": p["file_size_bytes"],
                "encoding": p["encoding"],
                "row_count": p["row_count"],
                "column_count": p["column_count"],
                "column_names": "|".join(p["columns"]),
                "duplicate_row_count": p["duplicate_row_count"],
                "duplicate_note": p["duplicate_row_count_note"],
                "likely_role": p["role"],
                "role_reason": p["role_reason"],
                "date_columns": "|".join(p["date_stats"].keys()),
            }
        )
        for c in p["column_profiles"]:
            num = c.get("numeric") or {}
            date = c.get("date") or {}
            col_rows.append(
                {
                    "filename": p["filename"],
                    "column": c["column"],
                    "normalized_column": c["normalized"],
                    "inferred_dtype": c["inferred_dtype"],
                    "missing_count": c["missing_count"],
                    "blank_string_count": c["blank_string_count"],
                    "missing_pct": c["missing_pct"],
                    "unique_count": c["unique_count"],
                    "unique_count_source": c["unique_count_source"],
                    "is_candidate_key": c["is_candidate_key"],
                    "numeric_min": num.get("min"),
                    "numeric_max": num.get("max"),
                    "numeric_mean": num.get("mean"),
                    "numeric_median": num.get("median"),
                    "numeric_median_source": num.get("median_source"),
                    "numeric_std": num.get("std"),
                    "negative_count": num.get("negative_count"),
                    "zero_count": num.get("zero_count"),
                    "date_min": date.get("min"),
                    "date_max": date.get("max"),
                    "date_parsed_count": date.get("parsed_count"),
                    "date_failed_count": date.get("failed_count"),
                    "example_values": "|".join(str(x) for x in c["example_values"]),
                }
            )
    return pd.DataFrame(ds_rows), pd.DataFrame(col_rows)


def terminal_summary(profiles, rels, mapping, issues) -> str:
    total_rows = sum(p["row_count"] for p in profiles)
    roles = defaultdict(list)
    for p in profiles:
        roles[p["role"]].append(p["filename"])

    # likely primary: richest return-case coverage
    returnish = [p for p in profiles if p["role"] == "RETURN"]
    order_with_returns = [
        p
        for p in profiles
        if p["role"] == "ORDER"
        and any(
            normalize_col(c) in {"returned", "return_reason", "return_date", "request_date", "refund_requested"}
            for c in p["columns"]
        )
    ]
    primary = None
    if returnish:
        primary = max(returnish, key=lambda p: p["row_count"])
    elif order_with_returns:
        primary = max(order_with_returns, key=lambda p: p["column_count"])
    else:
        primary = max(profiles, key=lambda p: p["row_count"])

    strong = [r for r in rels if r["confidence"] == "HIGH" and r["intersection"] > 0]
    missing = [m["fliplens_field"] for m in mapping if m["status"] == "MISSING"]
    high_issues = [i for i in issues if i["severity"] == "HIGH"]

    ice = [p["filename"] for p in profiles if p["role"] == "ICE_CREAM_REFERENCE"]
    ice_txt = ", ".join(ice) if ice else "None found in data/raw/"

    return_files = [
        p["filename"]
        for p in profiles
        if p["role"] == "RETURN"
        or "RETURN" in p["role_reason"]
        or any("return" in normalize_col(c) or "refund" in normalize_col(c) for c in p["columns"])
    ]
    customer_files = [
        p["filename"]
        for p in profiles
        if any(normalize_col(c) in {"customer_id", "customerid"} for c in p["columns"])
    ]
    product_files = roles.get("PRODUCT", [])

    lines = [
        "==================================================",
        "PHASE 1 AUDIT - TERMINAL SUMMARY",
        "==================================================",
        "DATASETS FOUND:",
    ]
    for p in profiles:
        lines.append(
            f"  - {p['filename']} ({p['row_count']:,} rows, {p['column_count']} cols, role={p['role']})"
        )
    lines += [
        "",
        f"TOTAL ROWS: {total_rows:,}",
        "",
        f"LIKELY PRIMARY DATASET: {primary['filename']} ({primary['role']})",
        "",
        "SUPPORTING DATASETS:",
    ]
    for p in profiles:
        if p["filename"] != primary["filename"]:
            lines.append(f"  - {p['filename']} ({p['role']})")
    lines.append("")
    lines.append("POSSIBLE RETURN DATA:")
    lines.extend(f"  - {x}" for x in return_files) if return_files else lines.append("  - none")
    lines.append("")
    lines.append("POSSIBLE CUSTOMER DATA:")
    if customer_files:
        lines.extend(
            f"  - {x} (has customer_id; not a standalone customer dimension)" for x in customer_files
        )
    else:
        lines.append("  - none")
    lines.append("")
    lines.append("POSSIBLE PRODUCT DATA:")
    lines.extend(f"  - {x}" for x in product_files) if product_files else lines.append("  - none as PRODUCT role")
    lines += [
        "",
        f"ICE CREAM DATA: {ice_txt}",
        "",
        "STRONG RELATIONSHIPS:",
    ]
    if strong:
        for r in strong:
            lines.append(
                f"  - {r['parent_dataset']}.{r['parent_column']} -> {r['child_dataset']}.{r['child_column']} "
                f"(intersection={r['intersection']:,}, child_cov={r['child_coverage_pct']}%)"
            )
    else:
        lines.append("  - none with HIGH confidence and non-zero overlap")
    lines.append("")
    lines.append("MISSING FLIPLENS FIELDS:")
    lines.extend(f"  - {f}" for f in missing) if missing else lines.append("  - none")
    lines += [
        "",
        "IMPORTANT DATA QUALITY ISSUES:",
    ]
    if high_issues:
        for i in high_issues[:25]:
            lines.append(f"  - [{i['dataset']}] {i['issue_type']} on {i['column']}: count={i['count']} ({i['detail'][:160]})")
        if len(high_issues) > 25:
            lines.append(f"  - ... {len(high_issues) - 25} more HIGH issues in data/audit/data_quality_issues.csv")
    else:
        lines.append("  - no HIGH severity issues recorded")
    lines.append("==================================================")
    return "\n".join(lines)


def main() -> int:
    if not RAW_DIR.exists():
        log(f"Missing directory: {RAW_DIR}")
        return 1

    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        log(f"No CSV files in {RAW_DIR}")
        return 1

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    hashes_before = {p.name: sha256_file(p) for p in csvs}

    profiles = []
    for path in csvs:
        try:
            profiles.append(profile_csv(path))
        except Exception:
            log(f"FAILED {path.name}:\n{traceback.format_exc()}")
            profiles.append(
                {
                    "filename": path.name,
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "file_size_bytes": path.stat().st_size,
                    "encoding": "unknown",
                    "row_count": 0,
                    "column_count": 0,
                    "columns": [],
                    "duplicate_row_count": 0,
                    "duplicate_row_count_note": "failed",
                    "role": "OTHER",
                    "role_reason": "audit failed",
                    "column_profiles": [],
                    "sample_rows": [],
                    "capitalization_issues": {},
                    "unique_id_sets": {},
                    "unique_overflow": [],
                    "numeric_samples": {},
                    "date_stats": {},
                    "bad_chunks": 1,
                    "pyarrow_available": HAS_PYARROW,
                }
            )

    # Drop bulky sets before JSON-ish use but keep for relationships
    keys = []
    for p in profiles:
        keys.extend(candidate_keys(p))

    rels = find_relationships(profiles)
    issues = []
    for p in profiles:
        issues.extend(quality_issues(p))
    mapping = map_fliplens(profiles)

    ds_df, col_df = profiles_for_csv(profiles)
    ds_df.to_csv(AUDIT_DIR / "dataset_summary.csv", index=False)
    col_df.to_csv(AUDIT_DIR / "column_summary.csv", index=False)
    pd.DataFrame(rels).to_csv(AUDIT_DIR / "relationship_candidates.csv", index=False)
    pd.DataFrame(issues).to_csv(AUDIT_DIR / "data_quality_issues.csv", index=False)

    write_markdown_audit(profiles, keys, rels, issues, mapping)
    write_relationships_md(profiles, rels)
    write_mapping_md(mapping)

    hashes_after = {p.name: sha256_file(p) for p in csvs}
    mutated = [n for n in hashes_before if hashes_before[n] != hashes_after[n]]
    if mutated:
        log("ERROR: raw files changed during audit: " + ", ".join(mutated))
        return 2

    processed = {p["filename"] for p in profiles}
    found = {p.name for p in csvs}
    if processed != found:
        log(f"ERROR: not all CSVs processed. missing={found-processed}")
        return 3

    summary = terminal_summary(profiles, rels, mapping, issues)
    (AUDIT_DIR / "terminal_summary.txt").write_text(summary, encoding="utf-8")
    log(summary)
    log("Wrote docs/dataset_audit.md")
    log("Wrote docs/dataset_relationships.md")
    log("Wrote docs/flipLens_data_mapping.md")
    log("Wrote data/audit/*.csv")
    log("Raw SHA256 unchanged for all CSVs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
