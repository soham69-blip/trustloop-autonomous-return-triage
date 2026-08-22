# TrustLoop dataset audit (Phase 1)

Generated: 2026-08-20T14:55:34

Scope: every `data/raw/*.csv`. Raw files were not modified.
Statistics labeled SAMPLE-BASED were computed on a reservoir sample, not the full file.

## Datasets discovered

| File | Size (MB) | Rows | Cols | Role | Duplicate rows |
|---|---:|---:|---:|---|---:|
| `blinkit_dataset.csv` | 2.23 | 13,000 | 25 | PRODUCT | 0 |
| `ecommerce_delivery_analytics.csv` | 9.34 | 100,000 | 11 | ORDER | 0 |
| `ecommerce_return_abuse_dataset.csv` | 11.35 | 60,000 | 35 | RETURN | 0 |
| `final_new_cleaned_orders.csv` | 14.08 | 100,000 | 18 | ORDER | 0 |
| `Kaggle_Ecommerce Data.csv` | 3.79 | 34,500 | 19 | ORDER | 0 |
| `zepto_v2.csv` | 0.30 | 3,732 | 9 | PRODUCT | 2 |

## Per-dataset profiles

### `blinkit_dataset.csv`

- Path: `data/raw/blinkit_dataset.csv`
- Encoding: `utf-8-sig`
- Rows: **13,000** | Columns: **25** | Size: 2,228,855 bytes
- Duplicate rows: 0 (exact_via_pandas_hash)
- **Likely role:** PRODUCT
- **Why:** catalog/stock fields without order identifiers; also sold_quantity/sales-like metrics [secondary: INVENTORY]

#### Columns

| Column | Inferred type | Missing | Missing % | Unique | Unique source | Examples |
|---|---|---:|---:|---:|---|---|
| `product_id` | numeric | 0 | 0.0 | 13,000 | exact_non_null | 1, 2, 3 |
| `product_name` | categorical | 0 | 0.0 | 12,617 | exact_non_null | Tata Organic Grocery 300, Mother Dairy Lite Dairy 275, P&G Classic Personal 439 |
| `category` | categorical | 0 | 0.0 | 8 | exact_non_null | Grocery, Dairy, Personal Care |
| `brand` | categorical | 0 | 0.0 | 28 | exact_if_low_card_else_lower_bound | Tata, Mother Dairy, P&G |
| `price` | numeric | 0 | 0.0 |  | n/a | 199.78, 44.32, 501.13 |
| `discount_pct` | numeric | 0 | 0.0 |  | n/a | 25, 30, 0 |
| `final_price` | numeric | 0 | 0.0 |  | n/a | 149.84, 31.02, 501.13 |
| `rating` | numeric | 0 | 0.0 |  | n/a | 4.5, 4.0, 3.7 |
| `num_reviews` | numeric | 0 | 0.0 |  | n/a | 146, 264, 69 |
| `delivery_time_min` | numeric | 0 | 0.0 |  | n/a | 37, 36, 17 |
| `city` | categorical | 0 | 0.0 | 10 | exact_if_low_card_else_lower_bound | Bengaluru, Jaipur, Chennai |
| `seller` | categorical | 0 | 0.0 | 6 | exact_if_low_card_else_lower_bound | UrbanSeller, QuickStores, LocalMart |
| `stock` | numeric | 0 | 0.0 |  | n/a | 76, 122, 126 |
| `sold_quantity` | numeric | 0 | 0.0 |  | n/a | 241, 28, 583 |
| `profit_margin_pct` | numeric | 0 | 0.0 |  | n/a | 29.8, 15.2, 6.6 |
| `is_organic` | categorical | 0 | 0.0 | 2 | exact_if_low_card_else_lower_bound | True, False |
| `packaging_type` | categorical | 0 | 0.0 | 6 | exact_if_low_card_else_lower_bound | Can, Jar, Bottle |
| `weight_g` | numeric | 0 | 0.0 |  | n/a | 750, 1000, 200 |
| `shelf_life_days` | numeric | 0 | 0.0 |  | n/a | 212, 17, 1463 |
| `reorder_level` | numeric | 0 | 0.0 |  | n/a | 15, 24, 25 |
| `demand_index` | numeric | 0 | 0.0 |  | n/a | 73, 25, 100 |
| `date_added` | datetime | 0 | 0.0 |  | n/a | 2023-11-27, 2024-08-07, 2024-03-03 |
| `expiry_date` | datetime | 0 | 0.0 |  | n/a | 2024-06-26, 2024-08-24, 2028-03-05 |
| `offer_type` | categorical | 6,544 | 50.3385 | 4 | exact_if_low_card_else_lower_bound | FreeDelivery, FlatDiscount |
| `delivery_status` | categorical | 0 | 0.0 | 2 | exact_if_low_card_else_lower_bound | On-Time, Delayed |

#### Numeric summary (min/max/mean exact via chunks; median often SAMPLE-BASED)

| Column | min | max | mean | median | median source | negatives | zeros |
|---|---:|---:|---:|---:|---|---:|---:|
| `product_id` | 1.0 | 13000.0 | 6500.5 | 6500.5 | exact_from_collected | 0 | 0 |
| `price` | 10.18 | 999.93 | 267.3033561538475 | 220.24 | exact_from_collected | 0 | 0 |
| `discount_pct` | 0.0 | 30.0 | 9.959230769230736 | 10.0 | exact_from_collected | 0 | 3292 |
| `final_price` | 8.14 | 998.92 | 240.7352115384605 | 197.185 | exact_from_collected | 0 | 0 |
| `rating` | 2.5 | 5.0 | 4.196930769230771 | 4.2 | exact_from_collected | 0 | 0 |
| `num_reviews` | 1.0 | 1050.0 | 255.00030769230713 | 219.0 | exact_from_collected | 0 | 0 |
| `delivery_time_min` | 10.0 | 56.0 | 27.55876923076921 | 27.0 | exact_from_collected | 0 | 0 |
| `stock` | 52.0 | 169.0 | 110.14053846153844 | 110.0 | exact_from_collected | 0 | 0 |
| `sold_quantity` | 0.0 | 720.0 | 162.40007692307694 | 120.0 | exact_from_collected | 0 | 47 |
| `profit_margin_pct` | 5.0 | 40.0 | 22.652707692307764 | 22.9 | exact_from_collected | 0 | 0 |
| `weight_g` | 100.0 | 2000.0 | 536.1307692307693 | 400.0 | exact_from_collected | 0 | 0 |
| `shelf_life_days` | 2.0 | 1825.0 | 397.5553846153848 | 231.0 | exact_from_collected | 0 | 0 |
| `reorder_level` | 10.0 | 33.0 | 21.627769230769303 | 22.0 | exact_from_collected | 0 | 0 |
| `demand_index` | 0.0 | 100.0 | 43.38046153846158 | 38.0 | exact_from_collected | 0 | 258 |

#### Date-like columns

| Column | min | max | parsed | failed |
|---|---|---|---:|---:|
| `date_added` | 2023-01-11 00:00:00 | 2025-12-10 00:00:00 | 13000 | 0 |
| `expiry_date` | 2023-01-11 00:00:00 | 2030-12-07 00:00:00 | 13000 | 0 |

#### Candidate keys

| Column | Kind | Unique | Null/blank | Uniqueness % |
|---|---|---:|---:|---:|
| `product_id` | PRIMARY_KEY_CANDIDATE | 13000 | 0 | 100.0 |

#### Sample rows (n=5, not the full file)

```json
[
  {
    "product_id": "3616",
    "product_name": "Modern Classic Bakery 552",
    "category": "Bakery",
    "brand": "Modern",
    "price": "166.61",
    "discount_pct": "10",
    "final_price": "149.95",
    "rating": "3.4",
    "num_reviews": "71",
    "delivery_time_min": "31",
    "city": "Kolkata",
    "seller": "QuickStores",
    "stock": "133",
    "sold_quantity": "305",
    "profit_margin_pct": "10.7",
    "is_organic": "False",
    "packaging_type": "Box",
    "weight_g": "250",
    "shelf_life_days": "3",
    "reorder_level": "26",
    "demand_index": "68",
    "date_added": "2024-10-06",
    "expiry_date": "2024-10-09",
    "offer_type": "",
    "delivery_status": "On-Time"
  },
  {
    "product_id": "2537",
    "product_name": "Tata Classic Grocery 870",
    "category": "Grocery",
    "brand": "Tata",
    "price": "438.83",
    "discount_pct": "15",
    "final_price": "373.01",
    "rating": "3.8",
    "num_reviews": "561",
    "delivery_time_min": "23",
    "city": "Lucknow",
    "seller": "QuickStores",
    "stock": "114",
    "sold_quantity": "280",
    "profit_margin_pct": "26.3",
    "is_organic": "False",
    "packaging_type": "Box",
    "weight_g": "500",
    "shelf_life_days": "531",
    "reorder_level": "22",
    "demand_index": "84",
    "date_added": "2025-04-15",
    "expiry_date": "2026-09-28",
    "offer_type": "",
    "delivery_status": "On-Time"
  },
  {
    "product_id": "5398",
    "product_name": "Lizol Classic Household 505",
    "category": "Household",
    "brand": "Lizol",
    "price": "707.08",
    "discount_pct": "10",
    "final_price": "636.37",
    "rating": "4.6",
    "num_reviews": "184",
    "delivery_time_min": "21",
    "city": "Chennai",
    "seller": "QuickStores",
    "stock": "113",
    "sold_quantity": "177",
    "profit_margin_pct": "9.8",
    "is_organic": "False",
    "packaging_type": "Can",
    "weight_g": "100",
    "shelf_life_days": "611",
    "reorder_level": "22",
    "demand_index": "47",
    "date_added": "2025-04-12",
    "expiry_date": "2026-12-14",
    "offer_type": "",
    "delivery_status": "On-Time"
  },
  {
    "product_id": "9983",
    "product_name": "Mother Dairy Premium Dairy 987",
    "category": "Dairy",
    "brand": "Mother Dairy",
    "price": "244.43",
    "discount_pct": "20",
    "final_price": "195.54",
    "rating": "4.9",
    "num_reviews": "210",
    "delivery_time_min": "25",
    "city": "Hyderabad",
    "seller": "LocalMart",
    "stock": "98",
    "sold_quantity": "118",
    "profit_margin_pct": "16.2",
    "is_organic": "False",
    "packaging_type": "Box",
    "weight_g": "750",
    "shelf_life_days": "29",
    "reorder_level": "19",
    "demand_index": "53",
    "date_added": "2025-09-29",
    "expiry_date": "2025-10-28",
    "offer_type": "",
    "delivery_status": "On-Time"
  },
  {
    "product_id": "1499",
    "product_name": "Nivea Classic Personal 552",
    "category": "Personal Care",
    "brand": "Nivea",
    "price": "419.44",
    "discount_pct": "10",
    "final_price": "377.5",
    "rating": "3.9",
    "num_reviews": "237",
    "delivery_time_min": "41",
    "city": "Lucknow",
    "seller": "SellerB",
    "stock": "93",
    "sold_quantity": "62",
    "profit_margin_pct": "27.2",
    "is_organic": "True",
    "packaging_type": "Jar",
    "weight_g": "250",
    "shelf_life_days": "1304",
    "reorder_level": "18",
    "demand_index": "3",
    "date_added": "2025-09-11",
    "expiry_date": "2029-04-07",
    "offer_type": "FreeDelivery",
    "delivery_status": "On-Time"
  }
]
```

### `ecommerce_delivery_analytics.csv`

- Path: `data/raw/ecommerce_delivery_analytics.csv`
- Encoding: `utf-8-sig`
- Rows: **100,000** | Columns: **11** | Size: 9,340,073 bytes
- Duplicate rows: 0 (exact_via_pandas_hash)
- **Likely role:** ORDER
- **Why:** order grain plus return/refund columns [secondary: RETURN flags]

#### Columns

| Column | Inferred type | Missing | Missing % | Unique | Unique source | Examples |
|---|---|---:|---:|---:|---|---|
| `Order ID` | categorical | 0 | 0.0 | 100,000 | exact_non_null | ORD000001, ORD000002, ORD000003 |
| `Customer ID` | categorical | 0 | 0.0 | 9,000 | exact_non_null | CUST2824, CUST1409, CUST5506 |
| `Platform` | categorical | 0 | 0.0 | 3 | exact_if_low_card_else_lower_bound | JioMart, Blinkit, Swiggy Instamart |
| `Order Date & Time` | categorical | 0 | 0.0 | 60 | exact_if_low_card_else_lower_bound | 19:29.5, 54:29.5, 21:29.5 |
| `Delivery Time (Minutes)` | numeric | 0 | 0.0 |  | n/a | 30, 16, 25 |
| `Product Category` | categorical | 0 | 0.0 | 6 | exact_if_low_card_else_lower_bound | Fruits & Vegetables, Dairy, Beverages |
| `Order Value (INR)` | numeric | 0 | 0.0 |  | n/a | 382, 279, 599 |
| `Customer Feedback` | categorical | 0 | 0.0 | 13 | exact_if_low_card_else_lower_bound | Fast delivery, great service!, Quick and reliable!, Items missing from order. |
| `Service Rating` | numeric | 0 | 0.0 |  | n/a | 5, 2, 1 |
| `Delivery Delay` | categorical | 0 | 0.0 | 2 | exact_if_low_card_else_lower_bound | No, Yes |
| `Refund Requested` | categorical | 0 | 0.0 | 2 | exact_if_low_card_else_lower_bound | No, Yes |

#### Numeric summary (min/max/mean exact via chunks; median often SAMPLE-BASED)

| Column | min | max | mean | median | median source | negatives | zeros |
|---|---:|---:|---:|---:|---|---:|---:|
| `Delivery Time (Minutes)` | 5.0 | 76.0 | 29.536139999999936 | 30.0 | sample | 0 | 0 |
| `Order Value (INR)` | 50.0 | 2000.0 | 590.9944000000086 | 482.0 | sample | 0 | 0 |
| `Service Rating` | 1.0 | 5.0 | 3.2407900000000027 | 3.0 | sample | 0 | 0 |

#### Date-like columns

None detected with a majority parseable date rate.

#### Candidate keys

| Column | Kind | Unique | Null/blank | Uniqueness % |
|---|---|---:|---:|---:|
| `Order ID` | PRIMARY_KEY_CANDIDATE | 100000 | 0 | 100.0 |
| `Customer ID` | FOREIGN_KEY_CANDIDATE | 9000 | 0 | 9.0 |

#### Sample rows (n=5, not the full file)

```json
[
  {
    "Order ID": "ORD033554",
    "Customer ID": "CUST5606",
    "Platform": "Swiggy Instamart",
    "Order Date & Time": "30:29.5",
    "Delivery Time (Minutes)": "25",
    "Product Category": "Snacks",
    "Order Value (INR)": "211",
    "Customer Feedback": "Wrong item delivered.",
    "Service Rating": "1",
    "Delivery Delay": "No",
    "Refund Requested": "Yes"
  },
  {
    "Order ID": "ORD009428",
    "Customer ID": "CUST5939",
    "Platform": "JioMart",
    "Order Date & Time": "28:29.5",
    "Delivery Time (Minutes)": "48",
    "Product Category": "Dairy",
    "Order Value (INR)": "417",
    "Customer Feedback": "Very late delivery, not happy.",
    "Service Rating": "2",
    "Delivery Delay": "Yes",
    "Refund Requested": "Yes"
  },
  {
    "Order ID": "ORD000200",
    "Customer ID": "CUST7735",
    "Platform": "Blinkit",
    "Order Date & Time": "27:29.5",
    "Delivery Time (Minutes)": "20",
    "Product Category": "Beverages",
    "Order Value (INR)": "786",
    "Customer Feedback": "Excellent experience!",
    "Service Rating": "5",
    "Delivery Delay": "No",
    "Refund Requested": "No"
  },
  {
    "Order ID": "ORD012448",
    "Customer ID": "CUST4734",
    "Platform": "Blinkit",
    "Order Date & Time": "52:29.5",
    "Delivery Time (Minutes)": "19",
    "Product Category": "Grocery",
    "Order Value (INR)": "1117",
    "Customer Feedback": "Horrible experience, never ordering again.",
    "Service Rating": "1",
    "Delivery Delay": "No",
    "Refund Requested": "Yes"
  },
  {
    "Order ID": "ORD039490",
    "Customer ID": "CUST4569",
    "Platform": "Swiggy Instamart",
    "Order Date & Time": "22:29.5",
    "Delivery Time (Minutes)": "41",
    "Product Category": "Dairy",
    "Order Value (INR)": "441",
    "Customer Feedback": "Horrible experience, never ordering again.",
    "Service Rating": "1",
    "Delivery Delay": "Yes",
    "Refund Requested": "Yes"
  }
]
```

### `ecommerce_return_abuse_dataset.csv`

- Path: `data/raw/ecommerce_return_abuse_dataset.csv`
- Encoding: `utf-8-sig`
- Rows: **60,000** | Columns: **35** | Size: 11,345,555 bytes
- Duplicate rows: 0 (exact_via_pandas_hash)
- **Likely role:** RETURN
- **Why:** return fields + abuse/fraud labels (abuse_label, abuse_type, return_date, return_reason) [secondary: FRAUD]

#### Columns

| Column | Inferred type | Missing | Missing % | Unique | Unique source | Examples |
|---|---|---:|---:|---:|---|---|
| `order_id` | categorical | 0 | 0.0 | 60,000 | exact_non_null | ORD2024554, ORD2019797, ORD2058733 |
| `customer_id` | categorical | 0 | 0.0 | 58,006 | exact_non_null | CUST408891, CUST898762, CUST906263 |
| `age` | numeric | 0 | 0.0 |  | n/a | 68, 64, 52 |
| `account_age_days` | numeric | 0 | 0.0 |  | n/a | 1473, 1211, 1291 |
| `customer_segment` | categorical | 0 | 0.0 | 5 | exact_if_low_card_else_lower_bound | New, Silver, Bronze |
| `country` | categorical | 0 | 0.0 | 8 | exact_if_low_card_else_lower_bound | US, FR, CA |
| `platform` | categorical | 0 | 0.0 | 3 | exact_if_low_card_else_lower_bound | Web Browser, Tablet App, Mobile App |
| `device_type` | categorical | 0 | 0.0 | 5 | exact_if_low_card_else_lower_bound | iPhone, MacBook, iPad |
| `payment_method` | categorical | 0 | 0.0 | 6 | exact_if_low_card_else_lower_bound | Crypto, PayPal, Debit Card |
| `product_category` | categorical | 0 | 0.0 | 12 | exact_if_low_card_else_lower_bound | Toys, Books, Clothing |
| `avg_order_value_usd` | numeric | 0 | 0.0 |  | n/a | 258.77, 93.63, 95.66 |
| `refund_amount_requested_usd` | numeric | 0 | 0.0 |  | n/a | 254.92, 84.11, 95.58 |
| `is_high_value_item` | numeric | 0 | 0.0 |  | n/a | 1, 0 |
| `discount_used` | numeric | 0 | 0.0 |  | n/a | 0, 1 |
| `order_date` | datetime | 0 | 0.0 |  | n/a | 2022-12-25, 2022-08-30, 2022-10-29 |
| `return_date` | datetime | 0 | 0.0 |  | n/a | 2023-01-09, 2022-09-29, 2022-11-12 |
| `days_to_return` | numeric | 0 | 0.0 |  | n/a | 15, 30, 14 |
| `return_reason` | categorical | 0 | 0.0 | 12 | exact_if_low_card_else_lower_bound | Defective/Broken, Wrong Item Sent, Item Not Received |
| `total_orders_lifetime` | numeric | 0 | 0.0 |  | n/a | 80, 54, 43 |
| `total_returns_lifetime` | numeric | 0 | 0.0 |  | n/a | 1, 2, 0 |
| `return_rate_pct` | numeric | 0 | 0.0 |  | n/a | 1.2, 3.7, 0.0 |
| `item_returned_opened` | numeric | 0 | 0.0 |  | n/a | 0, 1 |
| `return_packaging_intact` | numeric | 0 | 0.0 |  | n/a | 1, 0 |
| `photo_evidence_provided` | numeric | 0 | 0.0 |  | n/a | 1, 0 |
| `tracking_number_valid` | numeric | 0 | 0.0 | 2 | exact_non_null | 1, 0 |
| `shipping_carrier` | categorical | 0 | 0.0 | 5 | exact_if_low_card_else_lower_bound | OnTrac, FedEx, USPS |
| `address_change_before_delivery` | numeric | 0 | 0.0 |  | n/a | 0, 1 |
| `refund_to_different_account` | numeric | 0 | 0.0 |  | n/a | 0, 1 |
| `multiple_accounts_flag` | numeric | 0 | 0.0 |  | n/a | 0 |
| `customer_support_contacts` | numeric | 0 | 0.0 |  | n/a | 1, 0, 3 |
| `previous_dispute_count` | numeric | 0 | 0.0 |  | n/a | 0, 1, 2 |
| `wishlist_to_cart_time_hrs` | numeric | 0 | 0.0 |  | n/a | 70.5, 25.3, 36.6 |
| `review_left_after_return` | numeric | 0 | 0.0 |  | n/a | 0, 1 |
| `abuse_type` | categorical | 0 | 0.0 | 4 | exact_if_low_card_else_lower_bound | Legitimate, Fraudulent Return, Policy Abuser |
| `abuse_label` | numeric | 0 | 0.0 |  | n/a | 0, 2, 1 |

#### Numeric summary (min/max/mean exact via chunks; median often SAMPLE-BASED)

| Column | min | max | mean | median | median source | negatives | zeros |
|---|---:|---:|---:|---:|---|---:|---:|
| `age` | 18.0 | 70.0 | 44.1053833333328 | 44.0 | sample | 0 | 0 |
| `account_age_days` | 1.0 | 2500.0 | 1250.6877000000018 | 1245.0 | sample | 0 | 0 |
| `avg_order_value_usd` | 15.0 | 799.96 | 188.98709799999767 | 165.385 | sample | 0 | 0 |
| `refund_amount_requested_usd` | 12.12 | 836.68 | 176.19737133333314 | 150.60500000000002 | sample | 0 | 0 |
| `is_high_value_item` | 0.0 | 1.0 | 0.3793000000000009 | 0.0 | sample | 0 | 37242 |
| `discount_used` | 0.0 | 1.0 | 0.23865000000000083 | 0.0 | sample | 0 | 45681 |
| `days_to_return` | 1.0 | 55.0 | 16.087550000000178 | 15.0 | sample | 0 | 0 |
| `total_orders_lifetime` | 1.0 | 120.0 | 41.30455000000048 | 39.0 | sample | 0 | 0 |
| `total_returns_lifetime` | 0.0 | 101.0 | 8.943850000000023 | 3.0 | sample | 0 | 12026 |
| `return_rate_pct` | 0.0 | 84.7 | 19.073006666666636 | 8.7 | sample | 0 | 12026 |
| `item_returned_opened` | 0.0 | 1.0 | 0.4169333333333356 | 0.0 | sample | 0 | 34984 |
| `return_packaging_intact` | 0.0 | 1.0 | 0.7366499999999944 | 1.0 | sample | 0 | 15801 |
| `photo_evidence_provided` | 0.0 | 1.0 | 0.3794166666666628 | 0.0 | sample | 0 | 37235 |
| `tracking_number_valid` | 0.0 | 1.0 | 0.9284999999999962 | 1.0 | sample | 0 | 4290 |
| `address_change_before_delivery` | 0.0 | 1.0 | 0.09345000000000088 | 0.0 | sample | 0 | 54393 |
| `refund_to_different_account` | 0.0 | 1.0 | 0.04086666666666693 | 0.0 | sample | 0 | 57548 |
| `multiple_accounts_flag` | 0.0 | 1.0 | 0.07835000000000018 | 0.0 | sample | 0 | 55299 |
| `customer_support_contacts` | 0.0 | 6.0 | 1.3920000000000066 | 1.0 | sample | 0 | 21056 |
| `previous_dispute_count` | 0.0 | 5.0 | 1.0980500000000148 | 1.0 | sample | 0 | 24111 |
| `wishlist_to_cart_time_hrs` | 0.1 | 72.0 | 30.16844499999994 | 28.3 | sample | 0 | 0 |
| `review_left_after_return` | 0.0 | 1.0 | 0.24991666666666468 | 0.0 | sample | 0 | 45005 |
| `abuse_label` | 0.0 | 3.0 | 0.5553999999999989 | 0.0 | sample | 0 | 42060 |

#### Date-like columns

| Column | min | max | parsed | failed |
|---|---|---|---:|---:|
| `order_date` | 2022-01-01 00:00:00 | 2023-12-11 00:00:00 | 60000 | 0 |
| `return_date` | 2022-01-02 00:00:00 | 2024-12-01 00:00:00 | 60000 | 0 |

#### Candidate keys

| Column | Kind | Unique | Null/blank | Uniqueness % |
|---|---|---:|---:|---:|
| `order_id` | PRIMARY_KEY_CANDIDATE | 60000 | 0 | 100.0 |
| `customer_id` | FOREIGN_OR_WEAK_KEY | 58006 | 0 | 96.6767 |
| `tracking_number_valid` | FOREIGN_KEY_CANDIDATE | 2 | 0 | 0.0033 |

#### Sample rows (n=5, not the full file)

```json
[
  {
    "order_id": "ORD2004320",
    "customer_id": "CUST811705",
    "age": "48",
    "account_age_days": "1765",
    "customer_segment": "Gold",
    "country": "BR",
    "platform": "Web Browser",
    "device_type": "iPhone",
    "payment_method": "Debit Card",
    "product_category": "Clothing",
    "avg_order_value_usd": "247.27",
    "refund_amount_requested_usd": "199.51",
    "is_high_value_item": "1",
    "discount_used": "0",
    "order_date": "2022-12-25",
    "return_date": "2023-01-19",
    "days_to_return": "25",
    "return_reason": "Changed Mind",
    "total_orders_lifetime": "27",
    "total_returns_lifetime": "2",
    "return_rate_pct": "7.4",
    "item_returned_opened": "1",
    "return_packaging_intact": "1",
    "photo_evidence_provided": "1",
    "tracking_number_valid": "1",
    "shipping_carrier": "UPS",
    "address_change_before_delivery": "0",
    "refund_to_different_account": "0",
    "multiple_accounts_flag": "0",
    "customer_support_contacts": "0",
    "previous_dispute_count": "1",
    "wishlist_to_cart_time_hrs": "19.7",
    "review_left_after_return": "0",
    "abuse_type": "Legitimate",
    "abuse_label": "0"
  },
  {
    "order_id": "ORD2019506",
    "customer_id": "CUST287922",
    "age": "50",
    "account_age_days": "2040",
    "customer_segment": "Bronze",
    "country": "FR",
    "platform": "Web Browser",
    "device_type": "iPhone",
    "payment_method": "PayPal",
    "product_category": "Sports",
    "avg_order_value_usd": "196.12",
    "refund_amount_requested_usd": "168.5",
    "is_high_value_item": "0",
    "discount_used": "0",
    "order_date": "2023-05-20",
    "return_date": "2023-05-29",
    "days_to_return": "9",
    "return_reason": "Accidental Order",
    "total_orders_lifetime": "50",
    "total_returns_lifetime": "2",
    "return_rate_pct": "4.0",
    "item_returned_opened": "1",
    "return_packaging_intact": "0",
    "photo_evidence_provided": "0",
    "tracking_number_valid": "1",
    "shipping_carrier": "FedEx",
    "address_change_before_delivery": "0",
    "refund_to_different_account": "0",
    "multiple_accounts_flag": "0",
    "customer_support_contacts": "0",
    "previous_dispute_count": "1",
    "wishlist_to_cart_time_hrs": "9.7",
    "review_left_after_return": "0",
    "abuse_type": "Legitimate",
    "abuse_label": "0"
  },
  {
    "order_id": "ORD2022581",
    "customer_id": "CUST594231",
    "age": "65",
    "account_age_days": "915",
    "customer_segment": "Silver",
    "country": "US",
    "platform": "Web Browser",
    "device_type": "MacBook",
    "payment_method": "PayPal",
    "product_category": "Clothing",
    "avg_order_value_usd": "121.59",
    "refund_amount_requested_usd": "114.23",
    "is_high_value_item": "0",
    "discount_used": "0",
    "order_date": "2023-08-08",
    "return_date": "2023-08-18",
    "days_to_return": "10",
    "return_reason": "Not As Described",
    "total_orders_lifetime": "16",
    "total_returns_lifetime": "1",
    "return_rate_pct": "6.2",
    "item_returned_opened": "0",
    "return_packaging_intact": "1",
    "photo_evidence_provided": "0",
    "tracking_number_valid": "1",
    "shipping_carrier": "FedEx",
    "address_change_before_delivery": "0",
    "refund_to_different_account": "0",
    "multiple_accounts_flag": "0",
    "customer_support_contacts": "1",
    "previous_dispute_count": "1",
    "wishlist_to_cart_time_hrs": "19.5",
    "review_left_after_return": "0",
    "abuse_type": "Legitimate",
    "abuse_label": "0"
  },
  {
    "order_id": "ORD2004152",
    "customer_id": "CUST398788",
    "age": "70",
    "account_age_days": "2204",
    "customer_segment": "Platinum",
    "country": "US",
    "platform": "Web Browser",
    "device_type": "Windows PC",
    "payment_method": "Credit Card",
    "product_category": "Sports",
    "avg_order_value_usd": "71.99",
    "refund_amount_requested_usd": "68.91",
    "is_high_value_item": "0",
    "discount_used": "1",
    "order_date": "2022-10-16",
    "return_date": "2022-10-31",
    "days_to_return": "15",
    "return_reason": "Changed Mind",
    "total_orders_lifetime": "13",
    "total_returns_lifetime": "6",
    "return_rate_pct": "46.2",
    "item_returned_opened": "1",
    "return_packaging_intact": "0",
    "photo_evidence_provided": "0",
    "tracking_number_valid": "1",
    "shipping_carrier": "OnTrac",
    "address_change_before_delivery": "0",
    "refund_to_different_account": "0",
    "multiple_accounts_flag": "0",
    "customer_support_contacts": "3",
    "previous_dispute_count": "0",
    "wishlist_to_cart_time_hrs": "58.9",
    "review_left_after_return": "0",
    "abuse_type": "Policy Abuser",
    "abuse_label": "1"
  },
  {
    "order_id": "ORD2046969",
    "customer_id": "CUST438172",
    "age": "30",
    "account_age_days": "2243",
    "customer_segment": "New",
    "country": "IN",
    "platform": "Web Browser",
    "device_type": "Windows PC",
    "payment_method": "Debit Card",
    "product_category": "Electronics",
    "avg_order_value_usd": "88.87",
    "refund_amount_requested_usd": "85.01",
    "is_high_value_item": "0",
    "discount_used": "1",
    "order_date": "2023-09-03",
    "return_date": "2023-09-24",
    "days_to_return": "21",
    "return_reason": "Gift Duplicate",
    "total_orders_lifetime": "42",
    "total_returns_lifetime": "0",
    "return_rate_pct": "0.0",
    "item_returned_opened": "0",
    "return_packaging_intact": "1",
    "photo_evidence_provided": "0",
    "tracking_number_valid": "1",
    "shipping_carrier": "OnTrac",
    "address_change_before_delivery": "0",
    "refund_to_different_account": "0",
    "multiple_accounts_flag": "0",
    "customer_support_contacts": "1",
    "previous_dispute_count": "1",
    "wishlist_to_cart_time_hrs": "14.8",
    "review_left_after_return": "0",
    "abuse_type": "Legitimate",
    "abuse_label": "0"
  }
]
```

### `final_new_cleaned_orders.csv`

- Path: `data/raw/final_new_cleaned_orders.csv`
- Encoding: `utf-8-sig`
- Rows: **100,000** | Columns: **18** | Size: 14,079,583 bytes
- Duplicate rows: 0 (exact_via_pandas_hash)
- **Likely role:** ORDER
- **Why:** order grain plus return/refund columns [secondary: RETURN flags]

#### Columns

| Column | Inferred type | Missing | Missing % | Unique | Unique source | Examples |
|---|---|---:|---:|---:|---|---|
| `order_id` | categorical | 0 | 0.0 | 100,000 | exact_non_null | ORD000001, ORD000002, ORD000003 |
| `customer_id` | categorical | 0 | 0.0 | 9,000 | exact_non_null | CUST2824, CUST1409, CUST5506 |
| `platform_name` | categorical | 0 | 0.0 | 3 | exact_if_low_card_else_lower_bound | Blinkit, Swiggy Instamart, JioMart |
| `product_category_id` | numeric | 0 | 0.0 | 6 | exact_non_null | 4, 1, 5 |
| `order_datetime` | datetime | 0 | 0.0 |  | n/a | 16-05-2025 14:31, 02-05-2025 18:45, 09-05-2025 22:28 |
| `delivery_time_min` | numeric | 0 | 0.0 |  | n/a | 15, 6, 14 |
| `order_value_inr` | numeric | 0 | 0.0 |  | n/a | 382, 90, 599 |
| `delivery_delay` | categorical | 0 | 0.0 | 2 | exact_if_low_card_else_lower_bound | No, Yes |
| `refund_requested` | numeric | 0 | 0.0 |  | n/a | 0, 1 |
| `service_rating` | numeric | 0 | 0.0 |  | n/a | 5, 4, 2 |
| `customer_feedback` | categorical | 0 | 0.0 | 13 | exact_if_low_card_else_lower_bound | Fast delivery, great service!, Quick and reliable!, Items missing from order. |
| `product_category_name` | categorical | 0 | 0.0 | 6 | exact_if_low_card_else_lower_bound | Fruits & Vegetables, Dairy , Beverages |
| `sla_delay` | categorical | 0 | 0.0 | 2 | exact_if_low_card_else_lower_bound | No, Yes |
| `Segment` | categorical | 0 | 0.0 | 4 | exact_if_low_card_else_lower_bound | Price-only, Loyalist, Promisable |
| `hour` | numeric | 0 | 0.0 |  | n/a | 8, 5, 23 |
| `weekday` | categorical | 0 | 0.0 | 7 | exact_if_low_card_else_lower_bound | Monday, Saturday, Sunday |
| `date` | datetime | 0 | 0.0 |  | n/a | 16-05-2025, 02-05-2025, 09-05-2025 |
| `order_hour` | numeric | 0 | 0.0 |  | n/a | 14, 18, 22 |

#### Numeric summary (min/max/mean exact via chunks; median often SAMPLE-BASED)

| Column | min | max | mean | median | median source | negatives | zeros |
|---|---:|---:|---:|---:|---|---:|---:|
| `product_category_id` | 1.0 | 6.0 | 3.488570000000005 | 3.0 | sample | 0 | 0 |
| `delivery_time_min` | 5.0 | 76.0 | 13.352989999999945 | 12.0 | sample | 0 | 0 |
| `order_value_inr` | 19.0 | 5445.0 | 429.9553600000074 | 346.0 | sample | 0 | 0 |
| `refund_requested` | 0.0 | 1.0 | 0.08871000000000057 | 0.0 | sample | 0 | 91129 |
| `service_rating` | 1.0 | 5.0 | 3.965889999999976 | 4.0 | sample | 0 | 0 |
| `hour` | 0.0 | 23.0 | 11.831499999999842 | 12.0 | sample | 0 | 4263 |
| `order_hour` | 0.0 | 23.0 | 12.33094999999988 | 12.0 | sample | 0 | 4506 |

#### Date-like columns

| Column | min | max | parsed | failed |
|---|---|---|---:|---:|
| `order_datetime` | 2025-05-01 00:00:00 | 2025-05-17 23:59:00 | 100000 | 0 |
| `date` | 2025-05-01 00:00:00 | 2025-05-17 00:00:00 | 100000 | 0 |

#### Candidate keys

| Column | Kind | Unique | Null/blank | Uniqueness % |
|---|---|---:|---:|---:|
| `order_id` | PRIMARY_KEY_CANDIDATE | 100000 | 0 | 100.0 |
| `customer_id` | FOREIGN_KEY_CANDIDATE | 9000 | 0 | 9.0 |
| `product_category_id` | FOREIGN_KEY_CANDIDATE | 6 | 0 | 0.006 |

#### Sample rows (n=5, not the full file)

```json
[
  {
    "order_id": "ORD033554",
    "customer_id": "CUST5606",
    "platform_name": "Swiggy Instamart",
    "product_category_id": "3",
    "order_datetime": "16-05-2025 19:41",
    "delivery_time_min": "15",
    "order_value_inr": "211",
    "delivery_delay": "No",
    "refund_requested": "0",
    "service_rating": "1",
    "customer_feedback": "Wrong item delivered.",
    "product_category_name": "Snacks",
    "sla_delay": "No",
    "Segment": "Price-only",
    "hour": "20",
    "weekday": "Friday",
    "date": "16-05-2025",
    "order_hour": "19"
  },
  {
    "order_id": "ORD009428",
    "customer_id": "CUST5939",
    "platform_name": "JioMart",
    "product_category_id": "1",
    "order_datetime": "04-05-2025 05:14",
    "delivery_time_min": "9",
    "order_value_inr": "417",
    "delivery_delay": "Yes",
    "refund_requested": "0",
    "service_rating": "5",
    "customer_feedback": "Very late delivery, not happy.",
    "product_category_name": "Dairy ",
    "sla_delay": "No",
    "Segment": "Price-only",
    "hour": "13",
    "weekday": "Monday",
    "date": "04-05-2025",
    "order_hour": "5"
  },
  {
    "order_id": "ORD000200",
    "customer_id": "CUST7735",
    "platform_name": "Blinkit",
    "product_category_id": "5",
    "order_datetime": "05-05-2025 20:23",
    "delivery_time_min": "11",
    "order_value_inr": "66",
    "delivery_delay": "No",
    "refund_requested": "0",
    "service_rating": "4",
    "customer_feedback": "Excellent experience!",
    "product_category_name": "Beverages",
    "sla_delay": "No",
    "Segment": "Loyalist",
    "hour": "19",
    "weekday": "Wednesday",
    "date": "05-05-2025",
    "order_hour": "20"
  },
  {
    "order_id": "ORD012448",
    "customer_id": "CUST4734",
    "platform_name": "Blinkit",
    "product_category_id": "2",
    "order_datetime": "04-05-2025 04:23",
    "delivery_time_min": "14",
    "order_value_inr": "150",
    "delivery_delay": "No",
    "refund_requested": "0",
    "service_rating": "4",
    "customer_feedback": "Horrible experience, never ordering again.",
    "product_category_name": "Grocery",
    "sla_delay": "No",
    "Segment": "Price-only",
    "hour": "22",
    "weekday": "Sunday",
    "date": "04-05-2025",
    "order_hour": "4"
  },
  {
    "order_id": "ORD039490",
    "customer_id": "CUST4569",
    "platform_name": "Swiggy Instamart",
    "product_category_id": "1",
    "order_datetime": "12-05-2025 10:23",
    "delivery_time_min": "8",
    "order_value_inr": "441",
    "delivery_delay": "Yes",
    "refund_requested": "0",
    "service_rating": "5",
    "customer_feedback": "Horrible experience, never ordering again.",
    "product_category_name": "Dairy ",
    "sla_delay": "No",
    "Segment": "Price-only",
    "hour": "11",
    "weekday": "Monday",
    "date": "12-05-2025",
    "order_hour": "10"
  }
]
```

### `Kaggle_Ecommerce Data.csv`

- Path: `data/raw/Kaggle_Ecommerce Data.csv`
- Encoding: `utf-8-sig`
- Rows: **34,500** | Columns: **19** | Size: 3,787,531 bytes
- Duplicate rows: 0 (exact_via_pandas_hash)
- **Likely role:** ORDER
- **Why:** order grain plus return/refund columns [secondary: RETURN flags]

#### Columns

| Column | Inferred type | Missing | Missing % | Unique | Unique source | Examples |
|---|---|---:|---:|---:|---|---|
| `order_id` | categorical | 0 | 0.0 | 34,500 | exact_non_null | O100000, O100001, O100002 |
| `customer_id` | categorical | 0 | 0.0 | 7,903 | exact_non_null | C17270, C17603, C10860 |
| `product_id` | categorical | 0 | 0.0 | 24,912 | exact_non_null | P234890, P228204, P213892 |
| `category` | categorical | 0 | 0.0 | 7 | exact_non_null | Home, Grocery, Electronics |
| `price` | numeric | 0 | 0.0 |  | n/a | 164.08, 24.73, 175.58 |
| `discount` | numeric | 0 | 0.0 |  | n/a | 0.15, 0, 0.05 |
| `quantity` | numeric | 0 | 0.0 |  | n/a | 1, 2 |
| `payment_method` | categorical | 0 | 0.0 | 6 | exact_if_low_card_else_lower_bound | Credit Card, UPI, COD |
| `order_date` | datetime | 0 | 0.0 |  | n/a | 23/12/2023, 3/4/2025, 8/10/2024 |
| `delivered_date` | datetime | 0 | 0.0 |  | n/a | 27/12/2023, 9/4/2025, 12/10/2024 |
| `region` | categorical | 0 | 0.0 | 5 | exact_if_low_card_else_lower_bound | West, South, North |
| `returned` | categorical | 0 | 0.0 | 2 | exact_if_low_card_else_lower_bound | No |
| `request_date` | datetime | 32,597 | 94.4841 |  | n/a | 16/10/2024, 15/3/2024, 12/3/2025 |
| `return_reason` | categorical | 32,597 | 94.4841 | 5 | exact_if_low_card_else_lower_bound | Missing/Wrong item, No longer needed, Not as described |
| `total_amount` | numeric | 0 | 0.0 |  | n/a | 139.47, 24.73, 166.8 |
| `shipping_cost` | numeric | 0 | 0.0 |  | n/a | 7.88, 4.6, 6.58 |
| `profit_margin` | numeric | 0 | 0.0 |  | n/a | 31.17, -2.62, 13.44 |
| `customer_age` | numeric | 0 | 0.0 |  | n/a | 60, 37, 34 |
| `customer_gender` | categorical | 0 | 0.0 | 3 | exact_if_low_card_else_lower_bound | Female, Male, Other |

#### Numeric summary (min/max/mean exact via chunks; median often SAMPLE-BASED)

| Column | min | max | mean | median | median source | negatives | zeros |
|---|---:|---:|---:|---:|---|---:|---:|
| `price` | 1.01 | 2930.47 | 119.39163246376869 | 45.8 | sample | 0 | 0 |
| `discount` | 0.0 | 0.3 | 0.04929130434782639 | 0.0 | sample | 0 | 18939 |
| `quantity` | 1.0 | 5.0 | 1.4907246376811731 | 1.0 | sample | 0 | 0 |
| `total_amount` | 0.82 | 12931.8 | 170.00849420289703 | 57.224999999999994 | sample | 0 | 0 |
| `shipping_cost` | 0.0 | 15.65 | 6.152119710144901 | 6.1 | sample | 0 | 50 |
| `profit_margin` | -6.2 | 1536.17 | 28.116504637681192 | 10.56 | sample | 6104 | 18 |
| `customer_age` | 18.0 | 69.0 | 43.474376811594034 | 43.0 | sample | 0 | 0 |

#### Date-like columns

| Column | min | max | parsed | failed |
|---|---|---|---:|---:|
| `order_date` | 2023-09-12 00:00:00 | 2025-09-11 00:00:00 | 34500 | 0 |
| `delivered_date` | 2023-09-15 00:00:00 | 2025-09-21 00:00:00 | 34500 | 0 |
| `request_date` | 2023-09-24 00:00:00 | 2025-10-17 00:00:00 | 1903 | 0 |

#### Candidate keys

| Column | Kind | Unique | Null/blank | Uniqueness % |
|---|---|---:|---:|---:|
| `order_id` | PRIMARY_KEY_CANDIDATE | 34500 | 0 | 100.0 |
| `customer_id` | FOREIGN_KEY_CANDIDATE | 7903 | 0 | 22.9072 |
| `product_id` | FOREIGN_OR_WEAK_KEY | 24912 | 0 | 72.2087 |

#### Sample rows (n=5, not the full file)

```json
[
  {
    "order_id": "O114875",
    "customer_id": "C13439",
    "product_id": "P212021",
    "category": "Fashion",
    "price": "58.82",
    "discount": "0.15",
    "quantity": "5",
    "payment_method": "Credit Card",
    "order_date": "24/9/2024",
    "delivered_date": "28/9/2024",
    "region": "West",
    "returned": "No",
    "request_date": "",
    "return_reason": "",
    "total_amount": "249.98",
    "shipping_cost": "8.7",
    "profit_margin": "78.79",
    "customer_age": "69",
    "customer_gender": "Male"
  },
  {
    "order_id": "O104749",
    "customer_id": "C15827",
    "product_id": "P246342",
    "category": "Electronics",
    "price": "433.43",
    "discount": "0.05",
    "quantity": "3",
    "payment_method": "Credit Card",
    "order_date": "30/10/2023",
    "delivered_date": "5/11/2023",
    "region": "East",
    "returned": "No",
    "request_date": "",
    "return_reason": "",
    "total_amount": "1235.28",
    "shipping_cost": "12.01",
    "profit_margin": "136.22",
    "customer_age": "68",
    "customer_gender": "Female"
  },
  {
    "order_id": "O115973",
    "customer_id": "C17096",
    "product_id": "P229928",
    "category": "Fashion",
    "price": "94.68",
    "discount": "0",
    "quantity": "2",
    "payment_method": "Credit Card",
    "order_date": "31/7/2025",
    "delivered_date": "6/8/2025",
    "region": "North",
    "returned": "Yes",
    "request_date": "16/8/2025",
    "return_reason": "Not as described",
    "total_amount": "189.36",
    "shipping_cost": "7",
    "profit_margin": "59.28",
    "customer_age": "34",
    "customer_gender": "Female"
  },
  {
    "order_id": "O102858",
    "customer_id": "C15105",
    "product_id": "P228268",
    "category": "Sports",
    "price": "130.79",
    "discount": "0",
    "quantity": "1",
    "payment_method": "Credit Card",
    "order_date": "20/3/2025",
    "delivered_date": "25/3/2025",
    "region": "North",
    "returned": "No",
    "request_date": "",
    "return_reason": "",
    "total_amount": "130.79",
    "shipping_cost": "8.48",
    "profit_margin": "30.76",
    "customer_age": "34",
    "customer_gender": "Female"
  },
  {
    "order_id": "O103928",
    "customer_id": "C10581",
    "product_id": "P201399",
    "category": "Sports",
    "price": "28.37",
    "discount": "0.05",
    "quantity": "2",
    "payment_method": "Credit Card",
    "order_date": "3/1/2025",
    "delivered_date": "7/1/2025",
    "region": "North",
    "returned": "No",
    "request_date": "",
    "return_reason": "",
    "total_amount": "53.9",
    "shipping_cost": "5.25",
    "profit_margin": "10.92",
    "customer_age": "54",
    "customer_gender": "Female"
  }
]
```

### `zepto_v2.csv`

- Path: `data/raw/zepto_v2.csv`
- Encoding: `cp1252`
- Rows: **3,732** | Columns: **9** | Size: 300,179 bytes
- Duplicate rows: 2 (exact_via_pandas_hash)
- **Likely role:** PRODUCT
- **Why:** catalog/stock fields without order identifiers [secondary: INVENTORY]

#### Columns

| Column | Inferred type | Missing | Missing % | Unique | Unique source | Examples |
|---|---|---:|---:|---:|---|---|
| `Category` | categorical | 0 | 0.0 | 14 | exact_non_null | Fruits & Vegetables |
| `name` | categorical | 0 | 0.0 | 1,674 | exact_non_null | Onion, Tomato Hybrid, Tender Coconut |
| `mrp` | numeric | 0 | 0.0 |  | n/a | 2500, 4200, 5100 |
| `discountPercent` | numeric | 0 | 0.0 |  | n/a | 16, 15, 14 |
| `availableQuantity` | numeric | 0 | 0.0 |  | n/a | 3 |
| `discountedSellingPrice` | numeric | 0 | 0.0 |  | n/a | 2100, 3500, 4300 |
| `weightInGms` | numeric | 0 | 0.0 |  | n/a | 1000, 58, 100 |
| `outOfStock` | categorical | 0 | 0.0 | 2 | exact_if_low_card_else_lower_bound | FALSE |
| `quantity` | numeric | 0 | 0.0 |  | n/a | 1, 100, 250 |

#### Numeric summary (min/max/mean exact via chunks; median often SAMPLE-BASED)

| Column | min | max | mean | median | median source | negatives | zeros |
|---|---:|---:|---:|---:|---|---:|---:|
| `mrp` | 0.0 | 260000.0 | 15680.117899249726 | 11000.0 | exact_from_collected | 0 | 1 |
| `discountPercent` | 0.0 | 51.0 | 7.617095391211134 | 6.0 | exact_from_collected | 0 | 1178 |
| `availableQuantity` | 0.0 | 6.0 | 4.008574490889599 | 5.0 | exact_from_collected | 0 | 453 |
| `discountedSellingPrice` | 0.0 | 139900.0 | 14192.83494105039 | 10400.0 | exact_from_collected | 0 | 1 |
| `weightInGms` | 0.0 | 10000.0 | 387.8437834941046 | 225.0 | exact_from_collected | 0 | 4 |
| `quantity` | 0.0 | 1500.0 | 213.27090032154328 | 186.0 | exact_from_collected | 0 | 6 |

#### Date-like columns

None detected with a majority parseable date rate.

#### Candidate keys

No identifier-like columns detected from names.

#### Sample rows (n=5, not the full file)

```json
[
  {
    "Category": "Fruits & Vegetables",
    "name": "Onion",
    "mrp": "2500",
    "discountPercent": "16",
    "availableQuantity": "3",
    "discountedSellingPrice": "2100",
    "weightInGms": "1000",
    "outOfStock": "FALSE",
    "quantity": "1"
  },
  {
    "Category": "Fruits & Vegetables",
    "name": "Tomato Hybrid",
    "mrp": "4200",
    "discountPercent": "16",
    "availableQuantity": "3",
    "discountedSellingPrice": "3500",
    "weightInGms": "1000",
    "outOfStock": "FALSE",
    "quantity": "1"
  },
  {
    "Category": "Fruits & Vegetables",
    "name": "Tender Coconut",
    "mrp": "5100",
    "discountPercent": "15",
    "availableQuantity": "3",
    "discountedSellingPrice": "4300",
    "weightInGms": "58",
    "outOfStock": "FALSE",
    "quantity": "1"
  },
  {
    "Category": "Fruits & Vegetables",
    "name": "Coriander Leaves",
    "mrp": "2000",
    "discountPercent": "15",
    "availableQuantity": "3",
    "discountedSellingPrice": "1700",
    "weightInGms": "100",
    "outOfStock": "FALSE",
    "quantity": "100"
  },
  {
    "Category": "Fruits & Vegetables",
    "name": "Ladies Finger ",
    "mrp": "1400",
    "discountPercent": "14",
    "availableQuantity": "3",
    "discountedSellingPrice": "1200",
    "weightInGms": "250",
    "outOfStock": "FALSE",
    "quantity": "250"
  }
]
```

## Redundancy

- Multiple product/catalog tables: blinkit_dataset.csv, zepto_v2.csv. Keep independent unless a verified SKU/product_id overlap exists.
- Possible overlapping grain: ecommerce_delivery_analytics.csv.Order ID vs final_new_cleaned_orders.csv.order_id (intersection=100000, jaccard=1.0). Inspect as alternate versions, not automatic duplicates to delete.
- Possible overlapping grain: ecommerce_delivery_analytics.csv.Customer ID vs final_new_cleaned_orders.csv.customer_id (intersection=9000, jaccard=1.0). Inspect as alternate versions, not automatic duplicates to delete.
- Several order/return-shaped tables exist with **different ID namespaces** (e.g. ORD000001 vs ORD2024554 vs O100000). They are not interchangeable without overlap proof.

No ice-cream dataset was found in `data/raw/`.

None of the six CSVs have ice-cream-specific columns (flavor, scoop, etc.). Blinkit and Zepto are grocery/quick-commerce **product catalogs**, not ice-cream reference tables. They must not be joined to order/return tables unless a verified key exists (see relationships: product_id/SKU overlap with order tables is not established).


## Data quality (report only — not corrected)

Issue rows written: 26 (see `data/audit/data_quality_issues.csv`).
