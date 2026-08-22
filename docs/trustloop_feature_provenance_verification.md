# TrustLoop Feature Provenance Verification
Generated: 2026-08-20T15:50:00+05:30

Objective: verify whether conditionally excluded fields are legitimately available at prediction time (return submission). Use only data/raw/ecommerce_return_abuse_dataset.csv. This report inspects the raw values and the Stage-1 feature pipeline implementation.

Summary verdict (final section contains concise verdict). Full details below.

---

## 1) total_orders_lifetime
- Observed behavior:
  - Values are present in the raw CSV; sampling multiple customers shows inconsistent behavior across records for the same customer: values sometimes decrease and sometimes increase over time.
  - In a sample of 200 customers with multiple records, 105 showed non-monotonic total_orders_lifetime sequences (examples below).
- Likely semantics: UNKNOWN (cannot be proven to be a pre-submission historical snapshot).
- Decision: EXCLUDE / UNKNOWN — do NOT rely on raw total_orders_lifetime as a submission-time field.
- Confidence: LOW
- Reason: Non-monotonic sequences for many customers indicate the field is not a reliable cumulative prior-order counter. The Stage-1 script used a plausibility check and derived customer_order_count_prior by subtracting 1 from total_orders_lifetime when the check passed; that is only a heuristic and not definitive proof of pre-submission availability.

Examples (customer_id, sorted return_date, total_orders_lifetime sample):
- CUST101028: [(2022-02-07, 48), (2022-07-23, 21)]
- CUST102902: [(2022-05-15, 68), (2022-05-15, 62)]
- CUST103611: [(2023-09-12, 100), (2023-09-27, 55)]

Required action to verify:
- Provide system-level definition: does total_orders_lifetime reflect the number of orders up to submission time, or was it computed later? If not available, recompute customer_order_count_prior from authoritative order logs.

---

## 2) photo_evidence_provided
- Observed behavior:
  - Many customers show changes across different returns (0→1 and 1→0), indicating the value is not constant per customer.
  - Sample shows multiple customers with toggles.
- Likely semantics: CONDITIONAL
- Decision: CONDITIONAL (exclude by default until proven submission-time)
- Confidence: MEDIUM
- Reason: It could represent submission-time uploads in many cases, but it could also reflect evidence added later during investigation. The raw CSV lacks an upload timestamp.

Required action to verify:
- Confirm whether photo_evidence_provided indicates an upload at submission time (has timestamp) or whether investigators can attach photos later. If the former, include as SAFE; if the latter, exclude for Stage 1.

---

## 3) tracking_number_valid
- Observed behavior:
  - Values vary by customer and across returns (sample shows toggles 1↔0).
- Likely semantics: CONDITIONAL
- Decision: CONDITIONAL (exclude until verified)
- Confidence: MEDIUM
- Reason: The flag could be set by an automated check at submission (safe) or by subsequent carrier verification (post-submission). The CSV lacks provenance.

Required action:
- Confirm whether an automated validation occurs immediately at submission and whether the stored flag is taken at submission time.

---

## 4) refund_amount_requested_usd
- Observed behavior:
  - Ratio refund_amount_requested_usd / avg_order_value_usd has mean ≈ 0.92 (std ≈ 0.06), min ≈ 0.80, max ≈ 1.05. About 3,013 rows have refund_amount_requested_usd > avg_order_value_usd.
- Likely semantics: CONDITIONAL
- Decision: CONDITIONAL (exclude until verified)
- Confidence: MEDIUM
- Reason: Values are plausible as requested amounts, but cannot confirm whether column records 'requested' amount at submission or a later approved/settled/refunded amount. Presence of refunds greater than avg_order_value suggests settlements/adjustments may be possible.

Required action:
- Confirm column definition: is it 'requested' at claim time, or 'settled' after processing? If 'requested' at submission, include; otherwise exclude.

---

## 5) address_change_before_delivery
- Observed behavior:
  - Binary flag present; sample inspection cannot conclusively show timing.
- Likely semantics: CONDITIONAL / UNKNOWN
- Decision: CONDITIONAL (exclude until verified)
- Confidence: LOW
- Reason: Address changes may be logged before delivery or during investigation; without timestamps, cannot prove pre-submission availability.

Required action:
- Provide event-level logs or timestamped address-change records to confirm whether the flag denotes a pre-submission change.

---

## 6) customer_support_contacts
- Observed behavior:
  - Counts per customer are non-monotonic in many samples (counts sometimes decrease), suggesting the supplied column may not be a simple prior-only cumulative snapshot.
- Likely semantics: UNKNOWN / CONDITIONAL
- Decision: CONDITIONAL / EXCLUDE raw field until recomputed
- Confidence: LOW
- Reason: Non-monotonicity suggests the column may be a transient snapshot or include current-case interactions. Using the raw value risks leakage (if it includes current-case contact count) or data inconsistency.

Required action:
- Recompute prior-only customer_support_contacts_prior (count of contacts with contact_date < return_date) from event logs, or confirm that the column is a historical snapshot excluding the current case.

---

## 7) abuse_type
- Observed behavior:
  - abuse_type maps one-to-one with abuse_label across the dataset. Groups: Fraudulent Return -> 2; Legitimate -> 0; Policy Abuser -> 1; Wardrobing -> 3.
- Likely semantics: TARGET_MAPPING
- Decision: EXCLUDE (DIRECT_LEAKAGE)
- Confidence: HIGH
- Reason: The mapping is exact; abuse_type is a human-readable label of the target and is therefore a post-decision field. It must be excluded for any predictive modeling.

---

## 8) HISTORICAL FEATURES verification
We inspected the Stage-1 implementation (scripts/build_trustloop_features.py) to confirm correctness of historical feature calculations.

Implementation summary (verified in code):
- Deterministic ordering: rows are sorted by (customer_id, return_date, order_date, order_id, __row_id) using a stable mergesort. __row_id is the original file row index and acts as final tie-breaker.
- customer_return_count_prior:
  - Implementation: grouped.cumcount() over the deterministic sorted data.
  - Cutoff: events counted are strictly prior in the sorted order; mathematically excludes current row.
  - Source: same raw CSV (returns); uses return_date ordering.
  - current_row_excluded: YES
  - future_rows_excluded: YES (only prior rows included)
  - status: VERIFIED (HIGH confidence)

- returns_last_30d_prior and returns_last_90d_prior:
  - Implementation: per-customer sorted arrays of return_date; for each row at position i, compute left index = searchsorted(dates, t - K) and count i - left (excludes current row)
  - Cutoff: window [T - K, T) — strictly excludes current event.
  - Source: raw return_date field in CSV.
  - current_row_excluded: YES
  - future_rows_excluded: YES
  - status: VERIFIED (HIGH confidence)

- total_returns_lifetime_prior:
  - Implementation: explicitly set equal to customer_return_count_prior
  - Cutoff: same as customer_return_count_prior (< return_date)
  - status: VERIFIED (HIGH confidence)

- customer_order_count_prior:
  - Implementation: derived from raw total_orders_lifetime using heuristic: if raw total_orders_lifetime >= total_returns_lifetime_prior + 1 for >=90% rows, the script assumes total_orders_lifetime includes the current order and computes prior = total_orders_lifetime - 1.
  - Cutoff: NOT independently verified by the dataset; relies on the assumption that raw column includes current order.
  - current_row_excluded: NOT guaranteed by raw semantics — the derived prior subtracts 1, but this is only correct if the raw column includes the current order. We observed many non-monotonic total_orders_lifetime examples, undermining that assumption.
  - status: NOT VERIFIED (LOW confidence) — treat as CONDITIONAL and prefer recomputation from authoritative order logs or exclusion until verified.

Notes on deterministic ordering sufficiency:
- The ordering key (return_date, order_date, order_id) with file row fallback is deterministic and sufficient to produce reproducible prior aggregates.
- Potential residual leakage: if multiple events for the same customer share identical return_date and order_date and order_id (rare), __row_id tie-break ensures deterministic order. However if the true event-time ordering (submission timestamps) differs at sub-second resolution, small ordering differences could cause marginal differences in prior counts for simultaneous events. This is a small risk; ideally use a higher-resolution submission timestamp if available.

---

## 9) Final recommendations & actions required
- Permanently EXCLUDE abuse_type (DIRECT_LEAKAGE). Done in Stage 1 manifest.
- Do NOT use raw total_orders_lifetime as a source for customer_order_count_prior unless you can provide documentation proving it was captured at submission time and is a prior-only snapshot. Current data shows non-monotonic behavior and is unreliable.
- Remove or mark customer_order_count_prior as CONDITIONAL in the manifest until total_orders_lifetime semantics are proven. Prefer recomputing customer_order_count_prior from authoritative order records (orders with order_date < T).
- Keep customer_return_count_prior, returns_last_30d_prior, returns_last_90d_prior, total_returns_lifetime_prior as VERIFIED historical features (they were computed without lookahead and mathematically exclude the current row).
- Conditional fields (photo_evidence_provided, tracking_number_valid, refund_amount_requested_usd, address_change_before_delivery, customer_support_contacts) remain EXCLUDED for Stage 1 until the data owner provides timestamped provenance showing these values reflect submission-time state.

---

(See attached CSV summary for machine-readable per-column decisions.)

