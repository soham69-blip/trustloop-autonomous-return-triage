# TrustLoop Stage 1 Feature Pipeline

This document describes the conservative, decision-time feature engineering performed for TrustLoop Stage 1.

Source: data/raw/ecommerce_return_abuse_dataset.csv
Source rows: 60000

Key decisions:
- Excluded direct leakage columns: abuse_type, item_returned_opened, return_packaging_intact, review_left_after_return, refund_to_different_account
- Excluded identifiers from model input: order_id, customer_id
- Conditional features excluded unless semantics verified; see feature_manifest.csv for details.

Historical feature computation:
- customer_return_count_prior: computed as cumulative prior returns per customer (group.cumcount) using deterministic ordering.
- returns_last_30d_prior / returns_last_90d_prior: computed per-customer using searchsorted on sorted return_date arrays.
- total_returns_lifetime_prior: set equal to customer_return_count_prior (recomputed).
- customer_order_count_prior: NOT reliably derivable from dataset; excluded.
- previous_dispute_count_prior: excluded (not verifiably historical-only).

Outputs:
- C:\Users\khura\Downloads\TrustLoop_VSCode_Starter\data\processed\trustloop\model_ready.csv
- C:\Users\khura\Downloads\TrustLoop_VSCode_Starter\data\processed\trustloop\feature_manifest.csv
- C:\Users\khura\Downloads\TrustLoop_VSCode_Starter\data\processed\trustloop\processing_summary.json
