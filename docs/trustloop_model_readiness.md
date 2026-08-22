# TrustLoop Return-Model Readiness Audit
Generated: 2026-08-20T10:06:17.992851Z

## Dataset level metrics
- file: ecommerce_return_abuse_dataset.csv
- total_rows: 60000
- order_id: nulls=0, unique=CAPPED, duplicates=None
- customer_id: nulls=0, unique=CAPPED, duplicates=None
- target (abuse_label) distribution: {"0": 42060, "1": 7192, "2": 6112, "3": 4636}
- target positive count: 17940
- target negative count: 42060
- target positive percentage: 29.9

## Target audit (abuse_label and abuse_type)
- abuse_label present: yes
- abuse_label unique values and counts: {"0": 42060, "1": 7192, "2": 6112, "3": 4636}
- abuse_label nulls: 0
- abuse_type top values: [["Legitimate", 42060], ["Policy Abuser", 7192], ["Fraudulent Return", 6112], ["Wardrobing", 4636]]

## Target leakage audit (heuristic classifications)
Classification rules applied: IDENTIFIER (endswith _id or known), TARGET (abuse_label), DIRECT_LEAKAGE (label-like keywords), POST_DECISION_INFORMATION (resolution/decision keywords), POTENTIAL_LEAKAGE (suspicious numeric outcomes)

- order_id: IDENTIFIER — Identifier-like column (null%=0.0, unique=CAPPED)
- customer_id: IDENTIFIER — Identifier-like column (null%=0.0, unique=CAPPED)
- age: SAFE —  (null%=0.0, unique=CAPPED)
- account_age_days: SAFE —  (null%=0.0, unique=CAPPED)
- customer_segment: SAFE —  (null%=0.0, unique=CAPPED)
- country: SAFE —  (null%=0.0, unique=CAPPED)
- platform: SAFE —  (null%=0.0, unique=CAPPED)
- device_type: SAFE —  (null%=0.0, unique=CAPPED)
- payment_method: SAFE —  (null%=0.0, unique=CAPPED)
- product_category: SAFE —  (null%=0.0, unique=CAPPED)
- avg_order_value_usd: SAFE —  (null%=0.0, unique=CAPPED)
- refund_amount_requested_usd: SAFE —  (null%=0.0, unique=CAPPED)
- is_high_value_item: SAFE —  (null%=0.0, unique=CAPPED)
- discount_used: SAFE —  (null%=0.0, unique=CAPPED)
- order_date: SAFE —  (null%=0.0, unique=CAPPED)
- return_date: SAFE —  (null%=0.0, unique=CAPPED)
- days_to_return: SAFE —  (null%=0.0, unique=CAPPED)
- return_reason: SAFE —  (null%=0.0, unique=CAPPED)
- total_orders_lifetime: SAFE —  (null%=0.0, unique=CAPPED)
- total_returns_lifetime: SAFE —  (null%=0.0, unique=CAPPED)
- return_rate_pct: SAFE —  (null%=0.0, unique=CAPPED)
- item_returned_opened: SAFE —  (null%=0.0, unique=CAPPED)
- return_packaging_intact: SAFE —  (null%=0.0, unique=CAPPED)
- photo_evidence_provided: SAFE —  (null%=0.0, unique=CAPPED)
- tracking_number_valid: SAFE —  (null%=0.0, unique=CAPPED)
- shipping_carrier: SAFE —  (null%=0.0, unique=CAPPED)
- address_change_before_delivery: SAFE —  (null%=0.0, unique=CAPPED)
- refund_to_different_account: SAFE —  (null%=0.0, unique=CAPPED)
- multiple_accounts_flag: SAFE —  (null%=0.0, unique=CAPPED)
- customer_support_contacts: SAFE —  (null%=0.0, unique=CAPPED)
- previous_dispute_count: SAFE —  (null%=0.0, unique=CAPPED)
- wishlist_to_cart_time_hrs: SAFE —  (null%=0.0, unique=CAPPED)
- review_left_after_return: SAFE —  (null%=0.0, unique=CAPPED)
- abuse_type: TARGET_TYPE — Supplementary target type (null%=0.0, unique=CAPPED)
- abuse_label: TARGET — Defined model target (null%=0.0, unique=CAPPED)

## Feature inventory (sample)
Column | datatype_sample | null% | unique_count | role | ml_status | reason | top_10
--- | --- | --- | --- | --- | --- | --- | ---
order_id | str | 0.0 | CAPPED | identifier | IDENTIFIER | Identifier-like column | [["ORD2024554", 1], ["ORD2019797", 1], ["ORD2058733", 1], ["ORD2015301", 1], ["ORD2014206", 1], ["ORD2012147", 1], ["ORD2041328", 1], ["ORD2039097", 1], ["ORD2052018", 1], ["ORD2014522", 1]]
customer_id | str | 0.0 | CAPPED | identifier | IDENTIFIER | Identifier-like column | [["CUST322252", 4], ["CUST791491", 3], ["CUST869741", 3], ["CUST855010", 3], ["CUST178753", 3], ["CUST815242", 3], ["CUST634045", 3], ["CUST275958", 3], ["CUST907893", 3], ["CUST747001", 3]]
age | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["23", 1208], ["59", 1188], ["55", 1181], ["32", 1180], ["33", 1174], ["43", 1173], ["50", 1165], ["37", 1164], ["34", 1163], ["45", 1163]]
account_age_days | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["2378", 41], ["2176", 38], ["407", 37], ["1904", 35], ["240", 35], ["17", 34], ["639", 34], ["1638", 34], ["521", 34], ["892", 33]]
customer_segment | str | 0.0 | CAPPED | categorical | SAFE |  | [["Bronze", 16180], ["Silver", 15928], ["New", 12469], ["Gold", 10705], ["Platinum", 4718]]
country | str | 0.0 | CAPPED | categorical | SAFE |  | [["US", 25050], ["AU", 5078], ["CA", 5031], ["FR", 5021], ["IN", 5016], ["BR", 4987], ["GB", 4958], ["DE", 4859]]
platform | str | 0.0 | CAPPED | categorical | SAFE |  | [["Tablet App", 20052], ["Mobile App", 19990], ["Web Browser", 19958]]
device_type | str | 0.0 | CAPPED | categorical | SAFE |  | [["MacBook", 12105], ["Windows PC", 12069], ["Android", 12034], ["iPhone", 11996], ["iPad", 11796]]
payment_method | str | 0.0 | CAPPED | categorical | SAFE |  | [["Credit Card", 19389], ["Debit Card", 13880], ["PayPal", 11893], ["Buy Now Pay Later", 7763], ["Gift Card", 5100], ["Crypto", 1975]]
product_category | str | 0.0 | CAPPED | categorical | SAFE |  | [["Clothing", 12475], ["Electronics", 8502], ["Shoes", 7436], ["Home & Kitchen", 5835], ["Beauty", 5147], ["Sports", 4744], ["Books", 3361], ["Toys", 2960], ["Jewelry", 2607], ["Furniture", 2329]]
avg_order_value_usd | float64 | 0.0 | CAPPED | numeric | SAFE |  | [["154.56", 8], ["93.36", 8], ["92.88", 8], ["52.96", 8], ["223.24", 8], ["118.26", 8], ["132.87", 8], ["56.96", 7], ["88.78", 7], ["55.39", 7]]
refund_amount_requested_usd | float64 | 0.0 | CAPPED | numeric | SAFE |  | [["37.95", 11], ["112.16", 10], ["70.53", 9], ["68.88", 8], ["183.19", 8], ["88.69", 8], ["54.87", 8], ["162.37", 8], ["217.14", 8], ["111.78", 8]]
is_high_value_item | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["0", 37242], ["1", 22758]]
discount_used | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["0", 45681], ["1", 14319]]
order_date | str | 0.0 | CAPPED | date | SAFE |  | [["2022-06-29", 123], ["2022-04-09", 117], ["2022-02-04", 109], ["2023-07-12", 108], ["2022-07-07", 106], ["2023-04-30", 104], ["2023-07-19", 96], ["2023-05-04", 94], ["2023-10-03", 93], ["2023-05-11", 93]]
return_date | str | 0.0 | CAPPED | date | SAFE |  | [["2023-02-22", 116], ["2023-03-04", 110], ["2022-03-11", 109], ["2022-06-22", 106], ["2022-04-22", 98], ["2023-04-28", 97], ["2023-05-26", 97], ["2023-02-08", 97], ["2022-10-20", 96], ["2022-12-21", 93]]
days_to_return | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["4", 2898], ["2", 2886], ["3", 2877], ["1", 2863], ["5", 2829], ["30", 1817], ["29", 1816], ["25", 1804], ["28", 1778], ["27", 1778]]
return_reason | str | 0.0 | CAPPED | categorical | SAFE |  | [["Not As Described", 8320], ["Defective/Broken", 7095], ["Changed Mind", 6850], ["Wrong Item Sent", 6138], ["Too Large", 5870], ["Too Small", 5817], ["Quality Issue", 4354], ["Found Better Price", 3931], ["Item Not Received", 3845], ["Arrived Late", 3294]]
total_orders_lifetime | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["27", 859], ["11", 849], ["12", 844], ["50", 834], ["25", 830], ["23", 820], ["19", 819], ["41", 817], ["18", 816], ["42", 815]]
total_returns_lifetime | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["0", 12026], ["1", 7708], ["2", 5915], ["3", 4921], ["4", 3952], ["5", 3282], ["6", 2698], ["7", 2242], ["8", 1794], ["9", 1424]]
return_rate_pct | float64 | 0.0 | CAPPED | numeric | SAFE |  | [["0.0", 12026], ["50.0", 935], ["11.1", 798], ["10.0", 781], ["9.1", 715], ["8.3", 679], ["12.5", 676], ["7.1", 592], ["7.7", 577], ["33.3", 576]]
item_returned_opened | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["0", 34984], ["1", 25016]]
return_packaging_intact | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["1", 44199], ["0", 15801]]
photo_evidence_provided | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["0", 37235], ["1", 22765]]
tracking_number_valid | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["1", 55710], ["0", 4290]]
shipping_carrier | str | 0.0 | CAPPED | categorical | SAFE |  | [["FedEx", 12064], ["UPS", 12049], ["OnTrac", 12038], ["DHL", 12021], ["USPS", 11828]]
address_change_before_delivery | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["0", 54393], ["1", 5607]]
refund_to_different_account | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["0", 57548], ["1", 2452]]
multiple_accounts_flag | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["0", 55299], ["1", 4701]]
customer_support_contacts | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["1", 23997], ["0", 21056], ["2", 3031], ["4", 3031], ["3", 3018], ["6", 2948], ["5", 2919]]
previous_dispute_count | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["0", 24111], ["1", 23888], ["3", 3056], ["5", 3015], ["2", 2984], ["4", 2946]]
wishlist_to_cart_time_hrs | float64 | 0.0 | CAPPED | numeric | SAFE |  | [["4.8", 328], ["4.5", 326], ["1.1", 324], ["4.0", 317], ["4.1", 315], ["0.7", 314], ["3.7", 314], ["2.0", 302], ["0.8", 302], ["1.2", 300]]
review_left_after_return | int64 | 0.0 | CAPPED | numeric | SAFE |  | [["0", 45005], ["1", 14995]]
abuse_type | str | 0.0 | CAPPED | target_type | TARGET_TYPE | Supplementary target type | [["Legitimate", 42060], ["Policy Abuser", 7192], ["Fraudulent Return", 6112], ["Wardrobing", 4636]]
abuse_label | int64 | 0.0 | CAPPED | target | TARGET | Defined model target | [["0", 42060], ["1", 7192], ["2", 6112], ["3", 4636]]

## Temporal features
- order_date present: True
- return_date present: True
- days_to_return derivable (requires both order_date & return_date): True

## Customer behavior availability
- customer_order_count: MISSING
- customer_return_count: DERIVABLE
- customer_return_rate: DERIVABLE
- returns_last_30d / 90d: DERIVABLE
- average_order_value: AVAILABLE

## Numeric / categorical suspicious columns
- negative-valued numeric columns: []
- constant columns: []
- high-cardinality columns: []

## Model readiness recommendations
- target: abuse_label
- safe features (heuristic): ['age', 'account_age_days', 'customer_segment', 'country', 'platform', 'device_type', 'payment_method', 'product_category', 'avg_order_value_usd', 'refund_amount_requested_usd', 'is_high_value_item', 'discount_used', 'order_date', 'return_date', 'days_to_return', 'return_reason', 'total_orders_lifetime', 'total_returns_lifetime', 'return_rate_pct', 'item_returned_opened', 'return_packaging_intact', 'photo_evidence_provided', 'tracking_number_valid', 'shipping_carrier', 'address_change_before_delivery', 'refund_to_different_account', 'multiple_accounts_flag', 'customer_support_contacts', 'previous_dispute_count', 'wishlist_to_cart_time_hrs', 'review_left_after_return']
- leakage features (heuristic): []
- identifier features: ['order_id', 'customer_id']
- missing important features: case_id (MISSING), delivery_date (MISSING)
- class imbalance (positive %): 29.9
- recommended split: time-based (by return_date)

## Final verdict
MODEL READINESS:
READY

TARGET:
abuse_label

SAFE FEATURES:
age, account_age_days, customer_segment, country, platform, device_type, payment_method, product_category, avg_order_value_usd, refund_amount_requested_usd, is_high_value_item, discount_used, order_date, return_date, days_to_return, return_reason, total_orders_lifetime, total_returns_lifetime, return_rate_pct, item_returned_opened, return_packaging_intact, photo_evidence_provided, tracking_number_valid, shipping_carrier, address_change_before_delivery, refund_to_different_account, multiple_accounts_flag, customer_support_contacts, previous_dispute_count, wishlist_to_cart_time_hrs, review_left_after_return

LEAKAGE FEATURES:


IDENTIFIERS:
order_id, customer_id

MISSING IMPORTANT FEATURES:
case_id, delivery_date, previous_fraud_count, high_value_return_count

CLASS IMBALANCE:
positive_percentage=29.9

RECOMMENDED SPLIT:
time-based (by return_date)

EXACT NEXT STEP:
Human review of leakage-tagged columns and confirmation of the target semantics (mapping numeric codes to textual meanings). If acceptable, proceed to a controlled feature engineering stage that does not alter raw files.
