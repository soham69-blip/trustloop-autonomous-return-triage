# TrustLoop Data Architecture Validation
Generated: 2026-08-20T15:30:44.997496

## Dataset inventory and grain
- blinkit_dataset.csv: grain=PRODUCT role=PRODUCT suitability=15 candidate_key=product_id notes=has product id; has order value
- ecommerce_delivery_analytics.csv: grain=DELIVERY role=DELIVERY suitability=35 candidate_key=Order ID notes=has order id; has customer id; has order value
- ecommerce_return_abuse_dataset.csv: grain=RETURN role=RETURN suitability=90 candidate_key=order_id notes=has return date; has return reason; has order id; has order date; has customer id; has fraud/abuse labels
- final_new_cleaned_orders.csv: grain=DELIVERY role=DELIVERY suitability=45 candidate_key=order_id notes=has order id; has order date; has customer id; has order value
- Kaggle_Ecommerce Data.csv: grain=ORDER role=ORDER suitability=85 candidate_key=order_id notes=has return date; has return reason; has order id; has order date; has customer id; has product id; has order value
- zepto_v2.csv: grain=OTHER role=OTHER suitability=0 candidate_key= notes=

## Validated relationships (id-based)
- blinkit_dataset.csv.product_id <-> ecommerce_delivery_analytics.csv.Order ID: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- blinkit_dataset.csv.product_id <-> ecommerce_delivery_analytics.csv.Customer ID: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- blinkit_dataset.csv.product_id <-> ecommerce_return_abuse_dataset.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- blinkit_dataset.csv.product_id <-> ecommerce_return_abuse_dataset.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- blinkit_dataset.csv.product_id <-> final_new_cleaned_orders.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- blinkit_dataset.csv.product_id <-> final_new_cleaned_orders.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- blinkit_dataset.csv.product_id <-> final_new_cleaned_orders.csv.product_category_id: intersection=6 child_cov=100.0% parent_cov=0.0462% type=STRONG_ENTITY_RELATIONSHIP recommended=True reason=high coverage
- blinkit_dataset.csv.product_id <-> Kaggle_Ecommerce Data.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- blinkit_dataset.csv.product_id <-> Kaggle_Ecommerce Data.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- blinkit_dataset.csv.product_id <-> Kaggle_Ecommerce Data.csv.product_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Order ID <-> ecommerce_delivery_analytics.csv.Customer ID: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Order ID <-> ecommerce_return_abuse_dataset.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Order ID <-> ecommerce_return_abuse_dataset.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Order ID <-> final_new_cleaned_orders.csv.order_id: intersection=100000 child_cov=100.0% parent_cov=100.0% type=STRONG_ENTITY_RELATIONSHIP recommended=True reason=high coverage
- ecommerce_delivery_analytics.csv.Order ID <-> final_new_cleaned_orders.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Order ID <-> final_new_cleaned_orders.csv.product_category_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Order ID <-> Kaggle_Ecommerce Data.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Order ID <-> Kaggle_Ecommerce Data.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Order ID <-> Kaggle_Ecommerce Data.csv.product_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Customer ID <-> ecommerce_return_abuse_dataset.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Customer ID <-> ecommerce_return_abuse_dataset.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Customer ID <-> final_new_cleaned_orders.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Customer ID <-> final_new_cleaned_orders.csv.customer_id: intersection=9000 child_cov=100.0% parent_cov=100.0% type=STRONG_ENTITY_RELATIONSHIP recommended=True reason=high coverage
- ecommerce_delivery_analytics.csv.Customer ID <-> final_new_cleaned_orders.csv.product_category_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Customer ID <-> Kaggle_Ecommerce Data.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Customer ID <-> Kaggle_Ecommerce Data.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_delivery_analytics.csv.Customer ID <-> Kaggle_Ecommerce Data.csv.product_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_return_abuse_dataset.csv.order_id <-> ecommerce_return_abuse_dataset.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_return_abuse_dataset.csv.order_id <-> final_new_cleaned_orders.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_return_abuse_dataset.csv.order_id <-> final_new_cleaned_orders.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_return_abuse_dataset.csv.order_id <-> final_new_cleaned_orders.csv.product_category_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_return_abuse_dataset.csv.order_id <-> Kaggle_Ecommerce Data.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_return_abuse_dataset.csv.order_id <-> Kaggle_Ecommerce Data.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_return_abuse_dataset.csv.order_id <-> Kaggle_Ecommerce Data.csv.product_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_return_abuse_dataset.csv.customer_id <-> final_new_cleaned_orders.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_return_abuse_dataset.csv.customer_id <-> final_new_cleaned_orders.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_return_abuse_dataset.csv.customer_id <-> final_new_cleaned_orders.csv.product_category_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_return_abuse_dataset.csv.customer_id <-> Kaggle_Ecommerce Data.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_return_abuse_dataset.csv.customer_id <-> Kaggle_Ecommerce Data.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- ecommerce_return_abuse_dataset.csv.customer_id <-> Kaggle_Ecommerce Data.csv.product_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- final_new_cleaned_orders.csv.order_id <-> final_new_cleaned_orders.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- final_new_cleaned_orders.csv.order_id <-> final_new_cleaned_orders.csv.product_category_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- final_new_cleaned_orders.csv.order_id <-> Kaggle_Ecommerce Data.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- final_new_cleaned_orders.csv.order_id <-> Kaggle_Ecommerce Data.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- final_new_cleaned_orders.csv.order_id <-> Kaggle_Ecommerce Data.csv.product_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- final_new_cleaned_orders.csv.customer_id <-> final_new_cleaned_orders.csv.product_category_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- final_new_cleaned_orders.csv.customer_id <-> Kaggle_Ecommerce Data.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- final_new_cleaned_orders.csv.customer_id <-> Kaggle_Ecommerce Data.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- final_new_cleaned_orders.csv.customer_id <-> Kaggle_Ecommerce Data.csv.product_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- final_new_cleaned_orders.csv.product_category_id <-> Kaggle_Ecommerce Data.csv.order_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- final_new_cleaned_orders.csv.product_category_id <-> Kaggle_Ecommerce Data.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- final_new_cleaned_orders.csv.product_category_id <-> Kaggle_Ecommerce Data.csv.product_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- Kaggle_Ecommerce Data.csv.order_id <-> Kaggle_Ecommerce Data.csv.customer_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- Kaggle_Ecommerce Data.csv.order_id <-> Kaggle_Ecommerce Data.csv.product_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap
- Kaggle_Ecommerce Data.csv.customer_id <-> Kaggle_Ecommerce Data.csv.product_id: intersection=0 child_cov=0.0% parent_cov=0.0% type=INVALID recommended=False reason=no overlap

## Proposed architecture
# Proposed TrustLoop Data Architecture

blinkit_dataset.csv.product_id --> final_new_cleaned_orders.csv.product_category_id  (intersection=6)
ecommerce_delivery_analytics.csv.Order ID --> final_new_cleaned_orders.csv.order_id  (intersection=100000)
ecommerce_delivery_analytics.csv.Customer ID --> final_new_cleaned_orders.csv.customer_id  (intersection=9000)

Recommendation: Use a RETURN table as canonical return-case grain. Join ORDER source where STRONG relationships exist to map order metadata.

## Canonical return-case schema availability
- case_id: MISSING (source: .)
- customer_id: AVAILABLE (source: ecommerce_delivery_analytics.csv.Customer ID)
- order_id: AVAILABLE (source: ecommerce_delivery_analytics.csv.Order ID)
- product_id: AVAILABLE (source: Kaggle_Ecommerce Data.csv.product_id)
- product_category: AVAILABLE (source: ecommerce_delivery_analytics.csv.Product Category)
- product_name: AVAILABLE (source: blinkit_dataset.csv.product_name)
- brand: AVAILABLE (source: blinkit_dataset.csv.brand)
- order_date: AVAILABLE (source: ecommerce_return_abuse_dataset.csv.order_date)
- delivery_date: MISSING (source: .)
- return_date: AVAILABLE (source: ecommerce_return_abuse_dataset.csv.return_date)
- return_reason: AVAILABLE (source: ecommerce_return_abuse_dataset.csv.return_reason)
- order_value: MISSING (source: .)
- payment_method: AVAILABLE (source: ecommerce_return_abuse_dataset.csv.payment_method)
- customer_order_count: DERIVABLE (source: ecommerce_delivery_analytics.csv.customer_id + return dates)
- customer_return_count: DERIVABLE (source: ecommerce_delivery_analytics.csv.customer_id + return dates)
- customer_return_rate: DERIVABLE (source: ecommerce_delivery_analytics.csv.customer_id + return dates)
- returns_last_30d: DERIVABLE (source: ecommerce_delivery_analytics.csv.customer_id + return dates)
- returns_last_90d: DERIVABLE (source: ecommerce_delivery_analytics.csv.customer_id + return dates)
- previous_fraud_count: MISSING (source: .)
- high_value_return_count: MISSING (source: .)
- return_window_valid: MISSING (source: .)
- policy_violation: MISSING (source: .)
- fraud_probability: MISSING (source: .)
- fraud_label: MISSING (source: .)
- final_decision: MISSING (source: .)
- confidence: MISSING (source: .)
- explanation: MISSING (source: .)

## Ice-cream datasets: NO STANDALONE ICE-CREAM DATASET FOUND

## Next step
- Choose canonical RETURN dataset (based on suitability and presence of return identifiers).