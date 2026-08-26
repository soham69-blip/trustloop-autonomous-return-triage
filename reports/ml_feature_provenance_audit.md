# TrustLoop Candidate Feature Provenance Audit

## 1. Executive Summary
This audit traces the six candidate features added in the 39-feature candidate model (`lightgbm_candidate.pkl`):
1. `customer_support_contacts`
2. `previous_dispute_count`
3. `refund_amount_requested_usd`
4. `return_rate_pct`
5. `total_orders_lifetime`
6. `total_returns_lifetime`

## 2. Feature-by-Feature Forensic Analysis

| Feature Name | Source Column | Exact Formula / Derivation | Available at Decision Time? | Future / Target Leakage? | Real-Time Production Reproducibility |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `return_rate_pct` | `return_rate_pct` | `(total_returns / total_orders) * 100` | **YES** | **NO** | Derived from customer lifetime profile in CRM / Data Warehouse |
| `total_returns_lifetime` | `total_returns_lifetime` | Total completed return count prior to claim | **YES** | **NO** | Aggregated from customer return ledger |
| `total_orders_lifetime` | `total_orders_lifetime` | Total lifetime order count prior to claim | **YES** | **NO** | Aggregated from order management system |
| `customer_support_contacts` | `customer_support_contacts`| Prior support contact ticket count | **YES** | **NO** | Streamed from customer support service |
| `previous_dispute_count` | `previous_dispute_count` | Prior chargebacks and formal payment disputes | **YES** | **NO** | Queried from payment processor / dispute database |
| `refund_amount_requested_usd` | `refund_amount_requested_usd` | Dollar value of current return claim | **YES** | **NO** | Present on inbound return claim payload |

## 3. Findings
All 6 features represent legitimate, decision-time customer historical profile metrics. None of them use post-return settlement indicators (`abuse_type`, `item_returned_opened`, `return_packaging_intact`).
