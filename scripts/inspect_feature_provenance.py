import pandas as pd
import numpy as np
from collections import defaultdict

RAW = r"data/raw/ecommerce_return_abuse_dataset.csv"
print('Loading columns')
usecols = ['order_id','customer_id','order_date','return_date','total_orders_lifetime','total_returns_lifetime','return_rate_pct','photo_evidence_provided','tracking_number_valid','refund_amount_requested_usd','address_change_before_delivery','customer_support_contacts','abuse_type','abuse_label']
df = pd.read_csv(RAW, usecols=[c for c in usecols if c in pd.read_csv(RAW, nrows=0).columns], low_memory=False)
# Ensure dates
for c in ['order_date','return_date']:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')

out = {}
# abuse_type mapping
if 'abuse_type' in df.columns and 'abuse_label' in df.columns:
    mapping = df.groupby('abuse_type')['abuse_label'].nunique().to_dict()
    pairs = df.groupby(['abuse_type','abuse_label']).size().to_dict()
    out['abuse_type_to_label_unique_counts'] = mapping
    out['abuse_type_label_pairs_sample'] = list(pairs.items())[:20]
else:
    out['abuse_type_to_label_unique_counts'] = None

# Customers with multiple records: check monotonicity of total_orders_lifetime
cust_groups = df.groupby('customer_id')
nonmono_customers = []
checked = 0
for cid, g in cust_groups:
    if len(g) < 2:
        continue
    checked += 1
    # sort by return_date
    g2 = g.sort_values(['return_date','order_date','order_id'])
    if 'total_orders_lifetime' in g2.columns:
        vals = g2['total_orders_lifetime'].fillna(-999).astype(int).values
        # ignore if any placeholder
        # check monotonic non-decreasing
        if not np.all(np.diff(vals) >= 0):
            nonmono_customers.append((cid, list(zip(g2['return_date'].astype(str).tolist(), vals.tolist()))[:10]))
    if checked >= 200:
        break
out['checked_customers_sample'] = checked
out['nonmono_customers_sample_count'] = len(nonmono_customers)
out['nonmono_customers_sample'] = nonmono_customers[:10]

# photo_evidence_provided behavior per customer sample
photo_changes = []
if 'photo_evidence_provided' in df.columns:
    for cid, g in cust_groups:
        if len(g) < 2:
            continue
        g2 = g.sort_values(['return_date','order_date','order_id'])
        vals = g2['photo_evidence_provided'].dropna().astype(str).tolist()
        if len(vals) >=2 and len(set(vals))>1:
            photo_changes.append((cid, vals[:10]))
        if len(photo_changes)>=10:
            break
out['photo_changes_sample'] = photo_changes[:10]

# tracking_number_valid changes
track_changes = []
if 'tracking_number_valid' in df.columns:
    for cid, g in cust_groups:
        if len(g) < 2:
            continue
        g2 = g.sort_values(['return_date','order_date','order_id'])
        vals = g2['tracking_number_valid'].dropna().astype(str).tolist()
        if len(vals) >=2 and len(set(vals))>1:
            track_changes.append((cid, vals[:10]))
        if len(track_changes)>=10:
            break
out['track_changes_sample'] = track_changes[:10]

# refund amount relation to avg_order_value_usd
# need to read avg_order_value_usd too
try:
    df2 = pd.read_csv(RAW, usecols=['avg_order_value_usd','refund_amount_requested_usd'], low_memory=False)
    df2['ratio'] = df2['refund_amount_requested_usd'] / (df2['avg_order_value_usd'].replace({0:np.nan}))
    out['refund_ratio_summary'] = df2['ratio'].describe().to_dict()
    out['refund_greater_than_avg_count'] = int((df2['refund_amount_requested_usd'] > df2['avg_order_value_usd']).sum())
except Exception as e:
    out['refund_ratio_summary'] = str(e)

# customer_support_contacts: check monotonicity sample
cust_support_nonmono = []
if 'customer_support_contacts' in df.columns:
    for cid, g in cust_groups:
        if len(g) < 2:
            continue
        g2 = g.sort_values(['return_date','order_date','order_id'])
        vals = g2['customer_support_contacts'].fillna(-999).astype(float).values
        if not all(np.diff(vals) >= 0):
            cust_support_nonmono.append((cid, list(zip(g2['return_date'].astype(str).tolist(), vals.tolist()))[:10]))
        if len(cust_support_nonmono)>=10:
            break
    out['cust_support_nonmono_sample'] = cust_support_nonmono[:10]

# Output
import json
print(json.dumps(out, indent=2, default=str))
