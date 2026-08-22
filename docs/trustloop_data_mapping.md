# FlipLens data mapping (conceptual, Phase 1)

No values were invented. Status is AVAILABLE / DERIVABLE / MISSING / NOT_APPLICABLE.

| FlipLens field | Source dataset | Source column | Status | Transformation required |
|---|---|---|---|---|
| `case_id` | ecommerce_return_abuse_dataset.csv | `order_id` | **DERIVABLE** | Could mint case_id from order_id (+ product/return sequence) after grain is chosen. Not present as-is. |
| `customer_id` | ecommerce_return_abuse_dataset.csv | `customer_id` | **AVAILABLE** | Direct map (no invented values). Also present in: ecommerce_delivery_analytics.csv.Customer ID; final_new_cleaned_orders.csv.customer_id; Kaggle_Ecommerce Data.csv.customer_id |
| `order_id` | ecommerce_return_abuse_dataset.csv | `order_id` | **AVAILABLE** | Direct map (no invented values). Also present in: ecommerce_delivery_analytics.csv.Order ID; final_new_cleaned_orders.csv.order_id; Kaggle_Ecommerce Data.csv.order_id |
| `product_id` | Kaggle_Ecommerce Data.csv | `product_id` | **AVAILABLE** | Direct map (no invented values). Also present in: blinkit_dataset.csv.product_id |
| `product_category` | ecommerce_return_abuse_dataset.csv | `product_category` | **AVAILABLE** | Direct map (no invented values). Also present in: ecommerce_delivery_analytics.csv.Product Category; Kaggle_Ecommerce Data.csv.category; final_new_cleaned_orders.csv.product_category_name; blinkit_dataset.csv.category; zepto_v2.csv.Category |
| `product_name` | — | `—` | **MISSING** | Column exists on catalog `blinkit_dataset.csv.product_name` but that table is not a verified join to the return/order grain. Not AVAILABLE for FlipLens cases. Also seen in: zepto_v2.csv.name |
| `brand` | — | `—` | **MISSING** | Column exists on catalog `blinkit_dataset.csv.brand` but that table is not a verified join to the return/order grain. Not AVAILABLE for FlipLens cases. |
| `order_date` | ecommerce_return_abuse_dataset.csv | `order_date` | **AVAILABLE** | Direct map (no invented values). Also present in: Kaggle_Ecommerce Data.csv.order_date; final_new_cleaned_orders.csv.order_datetime; ecommerce_delivery_analytics.csv.Order Date & Time; final_new_cleaned_orders.csv.date |
| `delivery_date` | Kaggle_Ecommerce Data.csv | `delivered_date` | **AVAILABLE** | Direct map (no invented values). |
| `return_date` | ecommerce_return_abuse_dataset.csv | `return_date` | **AVAILABLE** | Direct map (no invented values). Also present in: Kaggle_Ecommerce Data.csv.request_date |
| `return_reason` | ecommerce_return_abuse_dataset.csv | `return_reason` | **AVAILABLE** | Direct map (no invented values). Also present in: Kaggle_Ecommerce Data.csv.return_reason |
| `order_value` | ecommerce_return_abuse_dataset.csv | `refund_amount_requested_usd` | **AVAILABLE** | Direct map (no invented values). Also present in: ecommerce_delivery_analytics.csv.Order Value (INR); final_new_cleaned_orders.csv.order_value_inr; Kaggle_Ecommerce Data.csv.total_amount; Kaggle_Ecommerce Data.csv.price; blinkit_dataset.csv.price |
| `payment_method` | ecommerce_return_abuse_dataset.csv | `payment_method` | **AVAILABLE** | Direct map (no invented values). Also present in: Kaggle_Ecommerce Data.csv.payment_method |
| `customer_order_count` | ecommerce_return_abuse_dataset.csv | `total_orders_lifetime` | **AVAILABLE** | Direct map (no invented values). |
| `customer_return_count` | ecommerce_return_abuse_dataset.csv | `total_returns_lifetime` | **AVAILABLE** | Direct map (no invented values). |
| `customer_return_rate` | ecommerce_return_abuse_dataset.csv | `return_rate_pct` | **AVAILABLE** | Direct map (no invented values). |
| `returns_last_30d` | ecommerce_return_abuse_dataset.csv | `customer_id + return/request date` | **DERIVABLE** | Windowed count of returns per customer. Requires parsed dates. |
| `returns_last_90d` | ecommerce_return_abuse_dataset.csv | `customer_id + return/request date` | **DERIVABLE** | Windowed count of returns per customer. Requires parsed dates. |
| `previous_fraud_count` | ecommerce_return_abuse_dataset.csv | `previous_dispute_count` | **DERIVABLE** | previous_dispute_count is a proxy, not a fraud count. Do not treat as FlipLens previous_fraud_count without definition. |
| `high_value_return` | ecommerce_return_abuse_dataset.csv | `is_high_value_item` | **AVAILABLE** | Direct map (no invented values). |
| `return_window_valid` | ecommerce_return_abuse_dataset.csv | `date pair / days_to_return` | **DERIVABLE** | Compare return/request date vs order/delivery date against a policy window. Not a source column. |
| `policy_violation` | policy engine (not in raw CSV) | `—` | **DERIVABLE** | Deterministic policy engine later; no raw policy-violation column exists. |
| `product_match_score` | — | `—` | **MISSING** | No source column. Vision/ML/policy outputs — do not invent. |
| `damage_score` | — | `—` | **MISSING** | No source column. Vision/ML/policy outputs — do not invent. |
| `serial_match_score` | — | `—` | **MISSING** | No source column. Vision/ML/policy outputs — do not invent. |
| `accessories_complete` | — | `—` | **MISSING** | No source column. Vision/ML/policy outputs — do not invent. |
| `packaging_match_score` | ecommerce_return_abuse_dataset.csv | `return_packaging_intact` | **DERIVABLE** | Binary packaging intact flag is not a match score; could inform a later score. Do not treat as FlipLens score. |
| `fraud_probability` | — | `—` | **MISSING** | No source column. Vision/ML/policy outputs — do not invent. |
| `fraud_label` | ecommerce_return_abuse_dataset.csv | `abuse_label` | **AVAILABLE** | Direct map (no invented values). Also present in: ecommerce_return_abuse_dataset.csv.abuse_type |
| `final_decision` | — | `—` | **MISSING** | No source column. Vision/ML/policy outputs — do not invent. |
| `confidence` | — | `—` | **MISSING** | No source column. Vision/ML/policy outputs — do not invent. |
| `explanation` | ecommerce_delivery_analytics.csv | `Customer Feedback` | **DERIVABLE** | Feedback text is not a model explanation. Could be an evidence snippet later. |

## Notes

- AVAILABLE means a raw column exists that can be mapped 1:1 or with renaming.
- DERIVABLE means the field can be computed later from existing columns (aggregates, date diffs, policy rules) without fabricating facts.
- MISSING means it is not in raw data (vision scores, model outputs, or catalogs that do not join).
- Catalog `product_name` / `brand` are not treated as AVAILABLE for the return-case grain unless a join key is verified.
