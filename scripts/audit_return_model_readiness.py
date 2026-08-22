"""
Audit script for verifying model target & feature readiness on the canonical
return-case source: data/raw/ecommerce_return_abuse_dataset.csv

Creates:
 - data/audit/v2/return_model_readiness.csv
 - docs/trustloop_model_readiness.md

This script does NOT modify raw data or create processed/model-ready datasets.
It only inspects the CSV (chunked) and reports counts, distributions, and
leakage-suspect fields.

Run (from repo root):
 python scripts\audit_return_model_readiness.py
"""
import os
import csv
import json
from collections import Counter, defaultdict
import math
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW = os.path.join(ROOT, 'data', 'raw', 'ecommerce_return_abuse_dataset.csv')
OUT_CSV = os.path.join(ROOT, 'data', 'audit', 'v2', 'return_model_readiness.csv')
OUT_MD = os.path.join(ROOT, 'docs', 'trustloop_model_readiness.md')

CHUNKSIZE = 50000
UNIQUE_CAP = 5_000_000  # cap to avoid OOM (very large unique sets)

# Heuristic helpers
IDENTIFIER_KEYWORDS = set(['order_id', 'customer_id', 'transaction_id', 'return_id', 'sku', 'tracking_number'])
TARGET_COL = 'abuse_label'
TARGET_TYPE_COL = 'abuse_type'

POST_DECISION_KEYWORDS = ['resolution', 'decision', 'final', 'outcome', 'investigation', 'resolved', 'verdict']
DIRECT_LEAKAGE_KEYWORDS = ['fraud_label', 'fraudulent', 'fraud', 'abuse_label', 'abuse_type']

# Storage
dataset_metrics = {}
col_nulls = defaultdict(int)
col_non_nulls = defaultdict(int)
col_unique_sets = defaultdict(set)
col_value_counts = defaultdict(Counter)
col_numeric_min = {}
col_numeric_max = {}
col_numeric_has_negative = defaultdict(bool)
col_constant = {}
col_types_sample = {}
col_top_values = {}
col_is_date = {}
col_date_min = {}
col_date_max = {}
col_date_invalid = defaultdict(int)
columns = None

total_rows = 0

# Target counters
target_counter = Counter()

def safe_add_unique(col, values):
    s = col_unique_sets[col]
    cap = UNIQUE_CAP
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if len(s) >= cap:
            # mark by setting to None to indicate capped
            col_unique_sets[col] = None
            return
        s.add(v)

# Read in chunks
for chunk in pd.read_csv(RAW, chunksize=CHUNKSIZE, low_memory=False):
    if columns is None:
        columns = list(chunk.columns)
    n = len(chunk)
    total_rows += n
    # normalize empty strings to NaN for counting
    chunk = chunk.replace({'': np.nan})

    for col in columns:
        ser = chunk[col]
        # nulls
        null_count = ser.isna().sum()
        col_nulls[col] += int(null_count)
        col_non_nulls[col] += int(n - null_count)
        # sample dtype
        if col not in col_types_sample:
            col_types_sample[col] = str(ser.dtype)
        # unique values (incremental via set)
        try:
            non_null_vals = ser.dropna().unique()
        except Exception:
            non_null_vals = ser.dropna().astype(str).unique()
        if col_unique_sets.get(col) is not None:
            safe_add_unique(col, non_null_vals)
        # value counts (top values)
        # update Counter limited to top frequent items per chunk
        vc = ser.dropna().astype(str).value_counts()
        for v, c in vc.head(50).items():
            col_value_counts[col][v] += int(c)
        # numeric checks
        if pd.api.types.is_numeric_dtype(ser):
            # compute min/max
            try:
                cmin = ser.min()
                cmax = ser.max()
                if col in col_numeric_min:
                    col_numeric_min[col] = min(col_numeric_min[col], cmin)
                    col_numeric_max[col] = max(col_numeric_max[col], cmax)
                else:
                    col_numeric_min[col] = cmin
                    col_numeric_max[col] = cmax
                if (ser < 0).any():
                    col_numeric_has_negative[col] = True
            except Exception:
                pass
        # date detection for columns with 'date' in name or look-like dates
        if 'date' in col.lower() or col.lower().endswith('_at'):
            # try parse
            parsed = pd.to_datetime(ser, errors='coerce')
            invalid = ser.notna() & parsed.isna()
            col_date_invalid[col] += int(invalid.sum())
            valid = parsed.dropna()
            if not valid.empty:
                if col in col_date_min:
                    col_date_min[col] = min(col_date_min[col], valid.min())
                    col_date_max[col] = max(col_date_max[col], valid.max())
                else:
                    col_date_min[col] = valid.min()
                    col_date_max[col] = valid.max()
                col_is_date[col] = True

    # target counters
    if TARGET_COL in chunk.columns:
        tser = chunk[TARGET_COL].astype(str).fillna('')
        for v, c in tser.value_counts().items():
            target_counter[v] += int(c)
    if TARGET_TYPE_COL in chunk.columns:
        # record types separately in col_value_counts above
        pass

# Post-processing summaries
# dataset-level metrics
dataset_metrics['file'] = os.path.basename(RAW)
dataset_metrics['total_rows'] = total_rows

# For primary ids (order_id, customer_id)
order_id = 'order_id'
customer_id = 'customer_id'

order_null = col_nulls.get(order_id, 0)
order_unique = None
order_dup = None
if col_unique_sets.get(order_id) is None:
    order_unique = 'CAPPED'
else:
    order_unique = len(col_unique_sets.get(order_id, set()))
    order_dup = max(0, (col_non_nulls.get(order_id, 0) - order_unique))

customer_null = col_nulls.get(customer_id, 0)
customer_unique = None
customer_dup = None
if col_unique_sets.get(customer_id) is None:
    customer_unique = 'CAPPED'
else:
    customer_unique = len(col_unique_sets.get(customer_id, set()))
    customer_dup = max(0, (col_non_nulls.get(customer_id, 0) - customer_unique))

# Build per-column inventory
col_inventory = []
for col in columns:
    non_null = col_non_nulls.get(col, 0)
    nulls = col_nulls.get(col, 0)
    null_pct = round(100.0 * nulls / total_rows, 3) if total_rows else None
    uniq = col_unique_sets.get(col)
    if uniq is None:
        uniq_count = 'CAPPED'
    else:
        uniq_count = len(uniq)
    # role/ML status heuristics
    lname = col.lower()
    role = 'numeric' if col in col_numeric_min else ('date' if col_is_date.get(col) else 'categorical')
    ml_status = 'REQUIRES_TRANSFORM'
    reason = ''
    if col == TARGET_COL:
        ml_status = 'TARGET'
        role = 'target'
        reason = 'Defined model target'
    elif col == TARGET_TYPE_COL:
        ml_status = 'TARGET_TYPE'
        role = 'target_type'
        reason = 'Supplementary target type'
    elif lname in IDENTIFIER_KEYWORDS or lname.endswith('_id') or lname == 'tracking_number':
        ml_status = 'IDENTIFIER'
        role = 'identifier'
        reason = 'Identifier-like column'
    else:
        # check for direct leakage keywords
        if any(k in lname for k in DIRECT_LEAKAGE_KEYWORDS):
            ml_status = 'DIRECT_LEAKAGE'
            reason = 'Contains direct label-like keywords'
        elif any(k in lname for k in POST_DECISION_KEYWORDS):
            ml_status = 'POST_DECISION_INFORMATION'
            reason = 'Likely recorded after investigation/decision'
        else:
            # further heuristics
            if role == 'numeric':
                ml_status = 'SAFE' if not col_numeric_has_negative.get(col, False) else 'POTENTIAL_LEAKAGE'
                if col_numeric_has_negative.get(col, False):
                    reason = 'Contains negative values'
            else:
                ml_status = 'SAFE'
    # top values
    top_vals = col_value_counts.get(col)
    top10 = []
    if top_vals:
        top10 = top_vals.most_common(10)
    # numeric min/max
    nmin = col_numeric_min.get(col)
    nmax = col_numeric_max.get(col)
    # date min/max
    dmin = col_date_min.get(col)
    dmax = col_date_max.get(col)

    col_inventory.append({
        'column': col,
        'datatype_sample': col_types_sample.get(col),
        'null_count': int(nulls),
        'null_percentage': null_pct,
        'unique_count': uniq_count,
        'role': role,
        'ml_status': ml_status,
        'reason': reason,
        'top_10': json.dumps(top10, ensure_ascii=False),
        'numeric_min': nmin,
        'numeric_max': nmax,
        'date_min': str(dmin) if dmin is not None else None,
        'date_max': str(dmax) if dmax is not None else None,
        'date_invalid_count': int(col_date_invalid.get(col, 0))
    })

# TARGET AUDIT
target_values = {k: v for k, v in target_counter.items()}
# Determine positive classes: assume non-zero/int>0 or labels != 'Legitimate' are positive
# But prefer numeric mapping if labels are numeric
positive_count = 0
negative_count = 0
null_target = col_nulls.get(TARGET_COL, 0)
# Try numeric mapping
numeric_vals = []
for k in list(target_values.keys()):
    try:
        numeric_vals.append(int(k))
    except Exception:
        pass
if numeric_vals and set(numeric_vals).issubset({0,1,2,3}):
    # integer-coded labels observed (example: 0 legitimate, 1 policy abuser, 2 fraudulent, 3 wardrobing)
    # treat any non-zero as positive (abuse)
    for k, v in target_values.items():
        try:
            if int(k) != 0:
                positive_count += v
            else:
                negative_count += v
        except Exception:
            # fallback
            if str(k).lower() not in ('0','legitimate','false','no','none','nan',''):
                positive_count += v
            else:
                negative_count += v
else:
    # textual labels like 'Legitimate', 'Policy Abuser', 'Fraudulent Return'
    for k, v in target_values.items():
        kl = str(k).strip().lower()
        if kl in ('legitimate','0','no','false','none','nan',''):
            negative_count += v
        else:
            positive_count += v

positive_pct = round(100.0 * positive_count / (positive_count + negative_count), 4) if (positive_count + negative_count) else None

# TEMPORAL FEATURES availability
order_date_present = 'order_date' in columns
return_date_present = 'return_date' in columns
can_compute_days_to_return = order_date_present and return_date_present

# CUSTOMER BEHAVIOR availability heuristics
customer_id_present = customer_id in columns and col_non_nulls.get(customer_id,0) > 0
multiple_orders_per_customer = False
if customer_id_present and isinstance(col_unique_sets.get(customer_id), set):
    # crude check: compare unique customer count to total rows
    cust_unique = len(col_unique_sets.get(customer_id))
    if cust_unique < total_rows:
        multiple_orders_per_customer = True

# NUMERIC/CATEGORICAL suspicious checks
suspicious = {
    'negative_columns': [col for col, neg in col_numeric_has_negative.items() if neg],
    'constant_columns': [c for c in columns if (col_unique_sets.get(c) is not None and len(col_unique_sets.get(c))<=1)],
    'high_cardinality_columns': [c for c in columns if (isinstance(col_unique_sets.get(c), set) and len(col_unique_sets.get(c))> (0.5*total_rows if total_rows else 10000))]
}

# MODEL READINESS recommendations
safe_features = [c for c in columns if next((ci for ci in col_inventory if ci['column']==c and ci['ml_status']=='SAFE'), None)]
identifier_features = [c for c in columns if next((ci for ci in col_inventory if ci['column']==c and ci['ml_status']=='IDENTIFIER'), None)]
leakage_features = [c for c in columns if next((ci for ci in col_inventory if ci['column']==c and ci['ml_status'] in ('DIRECT_LEAKAGE','POST_DECISION_INFORMATION','POTENTIAL_LEAKAGE')), None)]

# Recommended split: if return_date exists, use time-based split by return_date
recommended_split = 'time-based (by return_date)' if return_date_present else 'random'

# Final verdict logic (conservative)
# READY if target exists, positive and negative present, and no direct leakage; else READY WITH CHANGES or NOT READY
if TARGET_COL not in columns:
    final_verdict = 'NOT READY'
elif (positive_count + negative_count) == 0:
    final_verdict = 'NOT READY'
else:
    if any(ci['ml_status']=='DIRECT_LEAKAGE' for ci in col_inventory):
        final_verdict = 'READY WITH CHANGES'
    else:
        final_verdict = 'READY WITH CHANGES' if leakage_features else 'READY'

# Write CSV (structured): dataset metrics rows + column inventory
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # dataset-level
    writer.writerow(['section','metric','value'])
    writer.writerow(['dataset','file', dataset_metrics['file']])
    writer.writerow(['dataset','total_rows', dataset_metrics['total_rows']])
    writer.writerow(['dataset','order_id_nulls', order_null])
    writer.writerow(['dataset','order_id_unique', order_unique])
    writer.writerow(['dataset','order_id_duplicates', order_dup])
    writer.writerow(['dataset','customer_id_nulls', customer_null])
    writer.writerow(['dataset','customer_id_unique', customer_unique])
    writer.writerow(['dataset','customer_id_duplicates', customer_dup])
    writer.writerow(['dataset','target_values', json.dumps(target_values, ensure_ascii=False)])
    writer.writerow(['dataset','target_positive_count', positive_count])
    writer.writerow(['dataset','target_negative_count', negative_count])
    writer.writerow(['dataset','target_positive_percentage', positive_pct])
    writer.writerow(['dataset','return_date_present', return_date_present])
    writer.writerow(['dataset','order_date_present', order_date_present])
    writer.writerow(['dataset','can_compute_days_to_return', can_compute_days_to_return])
    writer.writerow([])
    # column-level header
    writer.writerow(['section','column','datatype_sample','null_count','null_percentage','unique_count','role','ml_status','reason','top_10','numeric_min','numeric_max','date_min','date_max','date_invalid_count'])
    for ci in col_inventory:
        writer.writerow(['column', ci['column'], ci['datatype_sample'], ci['null_count'], ci['null_percentage'], ci['unique_count'], ci['role'], ci['ml_status'], ci['reason'], ci['top_10'], ci['numeric_min'], ci['numeric_max'], ci['date_min'], ci['date_max'], ci['date_invalid_count']])

# Write docs markdown
os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write(f"# TrustLoop Return-Model Readiness Audit\nGenerated: {datetime.utcnow().isoformat()}Z\n\n")
    f.write("## Dataset level metrics\n")
    f.write(f"- file: {dataset_metrics['file']}\n")
    f.write(f"- total_rows: {total_rows}\n")
    f.write(f"- order_id: nulls={order_null}, unique={order_unique}, duplicates={order_dup}\n")
    f.write(f"- customer_id: nulls={customer_null}, unique={customer_unique}, duplicates={customer_dup}\n")
    f.write(f"- target ({TARGET_COL}) distribution: {json.dumps(target_values, ensure_ascii=False)}\n")
    f.write(f"- target positive count: {positive_count}\n")
    f.write(f"- target negative count: {negative_count}\n")
    f.write(f"- target positive percentage: {positive_pct}\n\n")

    f.write("## Target audit (abuse_label and abuse_type)\n")
    f.write(f"- abuse_label present: {'yes' if TARGET_COL in columns else 'no'}\n")
    if TARGET_COL in columns:
        f.write(f"- abuse_label unique values and counts: {json.dumps(target_values, ensure_ascii=False)}\n")
        f.write(f"- abuse_label nulls: {col_nulls.get(TARGET_COL,0)}\n")
    if TARGET_TYPE_COL in columns:
        tv = col_value_counts.get(TARGET_TYPE_COL).most_common(20)
        f.write(f"- abuse_type top values: {json.dumps(tv, ensure_ascii=False)}\n")
    f.write('\n')

    f.write("## Target leakage audit (heuristic classifications)\n")
    f.write("Classification rules applied: IDENTIFIER (endswith _id or known), TARGET (abuse_label), DIRECT_LEAKAGE (label-like keywords), POST_DECISION_INFORMATION (resolution/decision keywords), POTENTIAL_LEAKAGE (suspicious numeric outcomes)\n\n")
    for ci in col_inventory:
        f.write(f"- {ci['column']}: {ci['ml_status']} — {ci['reason']} (null%={ci['null_percentage']}, unique={ci['unique_count']})\n")
    f.write('\n')

    f.write("## Feature inventory (sample)\n")
    f.write("Column | datatype_sample | null% | unique_count | role | ml_status | reason | top_10\n")
    f.write("--- | --- | --- | --- | --- | --- | --- | ---\n")
    for ci in col_inventory:
        top10 = ci['top_10']
        f.write(f"{ci['column']} | {ci['datatype_sample']} | {ci['null_percentage']} | {ci['unique_count']} | {ci['role']} | {ci['ml_status']} | {ci['reason']} | {top10}\n")
    f.write('\n')

    f.write("## Temporal features\n")
    f.write(f"- order_date present: {order_date_present}\n")
    f.write(f"- return_date present: {return_date_present}\n")
    f.write(f"- days_to_return derivable (requires both order_date & return_date): {can_compute_days_to_return}\n")
    f.write('\n')

    f.write("## Customer behavior availability\n")
    def avail(v): return 'AVAILABLE' if v else 'MISSING'
    f.write(f"- customer_order_count: {'DERIVABLE' if multiple_orders_per_customer else 'MISSING'}\n")
    f.write(f"- customer_return_count: {'DERIVABLE' if customer_id_present else 'MISSING'}\n")
    f.write(f"- customer_return_rate: {'DERIVABLE' if customer_id_present else 'MISSING'}\n")
    f.write(f"- returns_last_30d / 90d: {'DERIVABLE' if return_date_present and customer_id_present else 'MISSING'}\n")
    f.write(f"- average_order_value: {'AVAILABLE' if 'avg_order_value_usd' in columns else 'MISSING'}\n")
    f.write('\n')

    f.write("## Numeric / categorical suspicious columns\n")
    f.write(f"- negative-valued numeric columns: {suspicious['negative_columns']}\n")
    f.write(f"- constant columns: {suspicious['constant_columns']}\n")
    f.write(f"- high-cardinality columns: {suspicious['high_cardinality_columns']}\n")
    f.write('\n')

    f.write("## Model readiness recommendations\n")
    f.write(f"- target: {TARGET_COL}\n")
    f.write(f"- safe features (heuristic): {safe_features}\n")
    f.write(f"- leakage features (heuristic): {leakage_features}\n")
    f.write(f"- identifier features: {identifier_features}\n")
    f.write(f"- missing important features: case_id (MISSING), delivery_date ({'AVAILABLE' if 'delivery_date' in columns else 'MISSING'})\n")
    f.write(f"- class imbalance (positive %): {positive_pct}\n")
    f.write(f"- recommended split: {recommended_split}\n")
    f.write('\n')

    f.write("## Final verdict\n")
    f.write(f"MODEL READINESS:\n{final_verdict}\n\n")
    f.write("TARGET:\n")
    f.write(f"{TARGET_COL}\n\n")
    f.write("SAFE FEATURES:\n")
    f.write(', '.join(safe_features) + '\n\n')
    f.write("LEAKAGE FEATURES:\n")
    f.write(', '.join(leakage_features) + '\n\n')
    f.write("IDENTIFIERS:\n")
    f.write(', '.join(identifier_features) + '\n\n')
    f.write("MISSING IMPORTANT FEATURES:\n")
    miss = ['case_id','delivery_date','previous_fraud_count','high_value_return_count']
    f.write(', '.join(miss) + '\n\n')
    f.write("CLASS IMBALANCE:\n")
    f.write(f"positive_percentage={positive_pct}\n\n")
    f.write("RECOMMENDED SPLIT:\n")
    f.write(f"{recommended_split}\n\n")
    f.write("EXACT NEXT STEP:\n")
    f.write("Human review of leakage-tagged columns and confirmation of the target semantics (mapping numeric codes to textual meanings). If acceptable, proceed to a controlled feature engineering stage that does not alter raw files.\n")

print('Wrote', OUT_CSV, 'and', OUT_MD)
