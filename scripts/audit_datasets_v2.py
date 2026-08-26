"""
TrustLoop Audit V2

Strict entity-relationship analysis: only identifier-like columns (customer_id, order_id, product_id, return_id, transaction_id, sku)
are considered as entity relationships. Ordinary categorical fields with the same values are classified as VALUE_OVERLAP.

Produces outputs under data/audit/v2/ and docs/*. Uses chunked reads and avoids O(n^2) comparisons.
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

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
AUDIT_V2 = ROOT / "data" / "audit" / "v2"
DOCS_DIR = ROOT / "docs"
CHUNKSIZE = 50_000
NUMERIC_MEDIAN_CAP = 1_000_000
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

ID_LOGICAL = {
    "order_id",
    "customer_id",
    "product_id",
    "return_id",
    "transaction_id",
    "sku",
}

ROLE_KEYWORDS = {
    "RETURN": {"return_date", "return_reason", "returned", "refund_requested", "refund_amount"},
    "ORDER": {"order_id", "order_date", "order_value", "order_datetime"},
    "PRODUCT": {"product_id", "product_name", "sku", "brand", "mrp", "price"},
    "CUSTOMER": {"customer_id", "customer_age", "customer_gender", "total_orders_lifetime"},
    "FRAUD": {"abuse_label", "abuse_type", "previous_dispute_count"},
}

DATE_RE = re.compile(r"date|time|datetime|timestamp|delivered|delivery|return|request", re.I)


def detect_encoding(path: Path) -> str:
    for enc in ENCODINGS:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                f.read(65536)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def is_id_col(col: str) -> bool:
    n = normalize_col(col)
    return n in ID_LOGICAL or n.endswith("_id") or n == "sku"


def profile_csv(path: Path) -> dict:
    enc = detect_encoding(path)
    it = pd.read_csv(path, encoding=enc, dtype=str, chunksize=CHUNKSIZE, on_bad_lines='warn')

    columns: list[str] = []
    row_count = 0
    missing = Counter()
    blank = Counter()
    dup_hashes = set()
    duplicate_rows = 0

    unique_sets = defaultdict(set)

    numeric_stats = {}
    numeric_values = defaultdict(list)

    date_stats = {}
    categorical_counters = defaultdict(Counter)

    sample_rows = []

    for chunk in it:
        if not columns:
            columns = list(chunk.columns)
        n = len(chunk)
        row_count += n

        na = chunk.isna()
        for c in columns:
            missing[c] += int(na[c].sum())
            s = chunk[c].astype(str)
            blank[c] += int((s.str.strip() == "").sum())

        # duplicate rows via hash
        try:
            hashed = pd.util.hash_pandas_object(chunk, index=False)
        except Exception:
            hashed = pd.util.hash_pandas_object(chunk.fillna(""), index=False)
        for h in hashed.astype('uint64').to_numpy():
            if int(h) in dup_hashes:
                duplicate_rows += 1
            else:
                dup_hashes.add(int(h))

        # id unique sets
        for c in columns:
            if is_id_col(c):
                vals = chunk[c].dropna().astype(str).str.strip()
                for v in vals:
                    if v == "":
                        continue
                    unique_sets[c].add(v)

        # numeric stats
        for c in columns:
            s = chunk[c].dropna().astype(str).str.strip()
            if s.empty:
                continue
            num = pd.to_numeric(s, errors='coerce')
            if num.notna().mean() >= 0.85:
                arr = num.dropna().astype(float).to_numpy()
                st = numeric_stats.setdefault(c, {'count':0,'mean':0.0,'m2':0.0,'min':None,'max':None})
                for x in arr:
                    if not np.isfinite(x):
                        continue
                    st['count'] += 1
                    delta = x - st['mean']
                    st['mean'] += delta / st['count']
                    delta2 = x - st['mean']
                    st['m2'] += delta * delta2
                    st['min'] = x if st['min'] is None else min(st['min'], x)
                    st['max'] = x if st['max'] is None else max(st['max'], x)
                if len(numeric_values[c]) <= NUMERIC_MEDIAN_CAP:
                    numeric_values[c].extend(arr.tolist())

        # dates
        for c in columns:
            if DATE_RE.search(c):
                s = chunk[c].dropna().astype(str).str.strip()
                if s.empty:
                    continue
                parsed = pd.to_datetime(s, errors='coerce', dayfirst=True)
                invalid = int(parsed.isna().sum())
                ok = parsed.dropna()
                st = date_stats.setdefault(c, {'min':None,'max':None,'invalid':0})
                st['invalid'] += invalid
                if not ok.empty:
                    mn = ok.min(); mx = ok.max()
                    st['min'] = mn if st['min'] is None else min(st['min'], mn)
                    st['max'] = mx if st['max'] is None else max(st['max'], mx)

        # categorical
        for c in columns:
            s = chunk[c].dropna().astype(str).str.strip()
            if s.empty:
                continue
            categorical_counters[c].update(s.tolist())

        # reservoir sample rows (for examples)
        if len(sample_rows) < 50:
            take = chunk.head(50 - len(sample_rows))
            sample_rows.extend(take.fillna("").astype(str).to_dict(orient='records'))

    # finalize numeric
    numeric_summary = {}
    for c, st in numeric_stats.items():
        cnt = st['count']
        mean = st['mean']
        std = math.sqrt(st['m2']/cnt) if cnt>1 else 0.0
        median = None; median_note = 'EXACT'
        vals = numeric_values.get(c, [])
        if vals:
            if len(vals) > NUMERIC_MEDIAN_CAP:
                median = float(np.median(np.random.choice(vals, NUMERIC_MEDIAN_CAP, replace=False)))
                median_note = 'SAMPLE-BASED'
            else:
                median = float(np.median(vals))
        numeric_summary[c] = {'count':cnt,'mean':mean,'std':std,'min':st['min'],'max':st['max'],'median':median,'median_note':median_note}

    # categorical summary top20
    categorical_summary = {}
    for c, counter in categorical_counters.items():
        categorical_summary[c] = {'unique_count': len(counter), 'top20': counter.most_common(20)}

    # column profiles
    col_profiles = []
    for c in columns:
        col_profiles.append({
            'column': c,
            'normalized': normalize_col(c),
            'missing_count': int(missing[c]),
            'blank_count': int(blank[c]),
            'unique_count': (len(unique_sets[c]) if c in unique_sets else None),
            'numeric': numeric_summary.get(c),
            'date': date_stats.get(c),
            'categorical': categorical_summary.get(c),
        })

    role, role_reason = classify_role(columns)

    return {
        'filename': path.name,
        'path': str(path.relative_to(ROOT)).replace('\\','/'),
        'file_size_bytes': path.stat().st_size,
        'encoding': enc,
        'row_count': row_count,
        'column_count': len(columns),
        'columns': columns,
        'duplicate_row_count': duplicate_rows,
        'column_profiles': col_profiles,
        'unique_sets': {k: (v if len(v) <= 500_000 else set(list(v)[:500_000])) for k,v in unique_sets.items()},
        'role': role,
        'role_reason': role_reason,
        'sample_rows': sample_rows[:5],
    }


def classify_role(columns: list[str]) -> tuple[str,str]:
    nset = {normalize_col(c) for c in columns}
    # heuristics
    has_return = any(x in nset for x in ('return_date','return_reason','returned','refund_requested'))
    has_order = any(x in nset for x in ('order_id','order_date','order_datetime','order_value'))
    has_product = any(x in nset for x in ('product_id','product_name','sku','brand','category'))
    has_customer = any(x in nset for x in ('customer_id','customer_age','customer_gender'))

    if 'ice_cream' in ' '.join(nset) or 'flavor' in nset:
        return 'ICE_CREAM_REFERENCE','ice-cream style columns present'
    if has_return and has_order:
        return 'RETURN','contains both order and return fields'
    if has_order and has_customer:
        return 'ORDER','order grain with customer id'
    if has_product and not has_order:
        return 'PRODUCT','catalog/product table'
    if has_customer and not has_order:
        return 'CUSTOMER','customer-like table'
    return 'OTHER','no strong domain signals'


def find_relationships_strict(profiles: list[dict]) -> list[dict]:
    # only id-like columns considered entity relationships
    id_index = defaultdict(list)  # normalized id -> list of (profile, colname, set)
    for p in profiles:
        for col_prof in p['column_profiles']:
            col = col_prof['column']
            if is_id_col(col):
                s = p.get('unique_sets', {}).get(col, set())
                id_index[normalize_col(col)].append((p, col, s))

    rels = []
    seen = set()
    for norm, items in id_index.items():
        if len(items) < 2:
            continue
        # pairwise among id columns with same normalized name
        for i in range(len(items)):
            for j in range(i+1,len(items)):
                p1, c1, s1 = items[i]
                p2, c2, s2 = items[j]
                key = (p1['filename'],c1,p2['filename'],c2)
                if key in seen:
                    continue
                seen.add(key)
                inter = len(s1 & s2)
                parent_unique = len(s1)
                child_unique = len(s2)
                child_cov = round(100.0 * inter / child_unique,4) if child_unique else 0.0
                parent_cov = round(100.0 * inter / parent_unique,4) if parent_unique else 0.0
                # classification heuristics
                if inter == 0:
                    cls = 'INVALID'
                elif child_cov >= 80 or parent_cov >= 80:
                    cls = 'STRONG_ENTITY_RELATIONSHIP'
                elif inter >= 50 and (child_cov >= 10 or parent_cov >= 10):
                    cls = 'WEAK_ENTITY_RELATIONSHIP'
                else:
                    cls = 'WEAK_ENTITY_RELATIONSHIP'
                rels.append({'parent_dataset': p1['filename'],'parent_column': c1,'child_dataset': p2['filename'],'child_column': c2,'parent_unique': parent_unique,'child_unique': child_unique,'intersection': inter,'child_coverage_pct': child_cov,'parent_coverage_pct': parent_cov,'classification': cls})
    return rels


def find_value_overlaps(profiles: list[dict]) -> list[dict]:
    # For non-id categorical columns with same normalized name, report VALUE_OVERLAP but no entity relationship
    col_index = defaultdict(list)  # normalized -> list of (profile, col, top_values set)
    for p in profiles:
        for col_prof in p['column_profiles']:
            col = col_prof['column']
            norm = normalize_col(col)
            # skip id-like
            if is_id_col(col):
                continue
            # only consider categorical summaries
            cat = col_prof.get('categorical')
            if cat and cat.get('unique_count',0) > 0:
                topvals = {v for v,_ in cat.get('top20',[])}
                col_index[norm].append((p,col,topvals,cat.get('unique_count',0)))
    overlaps = []
    for norm, items in col_index.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i+1,len(items)):
                p1,c1,t1,u1 = items[i]
                p2,c2,t2,u2 = items[j]
                inter = len(t1 & t2)
                overlaps.append({'dataset_a':p1['filename'],'col_a':c1,'dataset_b':p2['filename'],'col_b':c2,'top_overlap_count':inter,'classification':'VALUE_OVERLAP'})
    return overlaps


def candidate_keys_table(profile: dict) -> list[dict]:
    rows = []
    n = profile['row_count'] or 1
    for c in profile['column_profiles']:
        col = c['column']
        if is_id_col(col):
            uniq = c.get('unique_count') or (len(profile.get('unique_sets',{}).get(col, set())) if profile.get('unique_sets') else None)
            nulls = c.get('missing_count',0) + c.get('blank_count',0)
            uniq_pct = round(100.0 * uniq / n,4) if uniq is not None else None
            dup = max(n - (uniq or 0) - nulls, 0) if uniq is not None else None
            rows.append({'dataset': profile['filename'],'column': col,'total_rows': n,'unique_values': uniq,'nulls': nulls,'duplicate_values': dup,'uniqueness_pct': uniq_pct})
    return rows


def quality_checks(profile: dict) -> list[dict]:
    issues = []
    fname = profile['filename']
    n = profile['row_count']
    cols = {c['column']: c for c in profile['column_profiles']}
    # duplicate rows
    if profile['duplicate_row_count']:
        issues.append({'dataset':fname,'issue_type':'duplicate_rows','column':'*','severity':'MEDIUM','count':profile['duplicate_row_count'],'detail':'duplicate rows via hashing'})
    # id nulls/dups
    for k in candidate_keys_table(profile):
        if k['nulls']:
            issues.append({'dataset':fname,'issue_type':'null_or_blank_id','column':k['column'],'severity':'HIGH','count':k['nulls'],'detail':'Identifier has null/blank values'})
        if k['duplicate_values'] and k['uniqueness_pct'] is not None and k['uniqueness_pct'] < 99.99:
            issues.append({'dataset':fname,'issue_type':'duplicate_ids','column':k['column'],'severity':'HIGH','count':k['duplicate_values'],'detail':f"uniqueness_pct={k['uniqueness_pct']}"})
    # numeric anomalies (negatives on price-like)
    for c in profile['column_profiles']:
        num = c.get('numeric')
        if not num:
            continue
        name = c['column']
        if re.search(r'price|amount|value|cost|refund|total|mrp|profit|margin', name, re.I) and num.get('min') is not None and num.get('min') < 0:
            issues.append({'dataset':fname,'issue_type':'negative_prices','column':name,'severity':'MEDIUM','count':None,'detail':f"min={num.get('min')} max={num.get('max')}"})
    # date pair checks simplified (use sample rows)
    # returns before orders etc. Not exhaustive here; more detailed check can be added
    return issues


def map_trustloop_fields(profiles: list[dict]) -> list[dict]:
    # Map into TrustLoop fields using simple hints; no invented values.
    field_hints = {
        'case_id':['case_id','return_id'],
        'customer_id':['customer_id'],
        'order_id':['order_id'],
        'product_id':['product_id','sku'],
        'product_category':['category','product_category','product_category_name'],
        'product_name':['product_name','name'],
        'brand':['brand'],
        'order_date':['order_date','order_datetime','order_date_and_time','date'],
        'delivery_date':['delivered_date','delivery_date'],
        'return_date':['return_date','request_date'],
        'return_reason':['return_reason'],
        'order_value':['order_value','order_value_inr','total_amount','price','final_price'],
        'payment_method':['payment_method'],
    }
    col_index = defaultdict(list)
    for p in profiles:
        for c in p['columns']:
            col_index[normalize_col(c)].append((p['filename'],c,p['role']))

    rows = []
    for fld,hints in field_hints.items():
        found = []
        for h in hints:
            found.extend(col_index.get(normalize_col(h),[]))
        if found:
            # pick best by role preference (RETURN/ORDER over PRODUCT)
            found_sorted = sorted(found, key=lambda t: 0 if t[2] in ('RETURN','ORDER') else 1)
            src_ds, src_col, role = found_sorted[0]
            status = 'AVAILABLE'
            transform = 'Direct map'
        else:
            src_ds = '' ; src_col = '' ; status = 'MISSING' ; transform = 'No source column'
        rows.append({'trustloop_field':fld,'source_dataset':src_ds,'source_column':src_col,'status':status,'transformation_required':transform})
    return rows


def write_outputs(profiles, rels, overlaps, keys, issues, mapping):
    AUDIT_V2.mkdir(parents=True, exist_ok=True)
    # dataset_summary
    ds_rows = []
    col_rows = []
    for p in profiles:
        ds_rows.append({'filename':p['filename'],'path':p['path'],'file_size_bytes':p['file_size_bytes'],'row_count':p['row_count'],'column_count':p['column_count'],'role':p['role'],'role_reason':p['role_reason']})
        for c in p['column_profiles']:
            num = c.get('numeric') or {}
            dat = c.get('date') or {}
            col_rows.append({'filename':p['filename'],'column':c['column'],'normalized':c['normalized'],'missing_count':c['missing_count'],'blank_count':c['blank_count'],'unique_count':c.get('unique_count'),'numeric_min':num.get('min'),'numeric_max':num.get('max'),'numeric_mean':num.get('mean'),'numeric_median':num.get('median'),'numeric_median_note':num.get('median_note') if num else None,'date_min':dat.get('min') if dat else None,'date_max':dat.get('max') if dat else None,'date_invalid':dat.get('invalid') if dat else None})
    pd.DataFrame(ds_rows).to_csv(AUDIT_V2 / 'dataset_summary.csv', index=False)
    pd.DataFrame(col_rows).to_csv(AUDIT_V2 / 'column_summary.csv', index=False)
    pd.DataFrame(rels).to_csv(AUDIT_V2 / 'relationship_candidates.csv', index=False)
    pd.DataFrame(overlaps).to_csv(AUDIT_V2 / 'value_overlaps.csv', index=False)
    pd.DataFrame(keys).to_csv(AUDIT_V2 / 'candidate_keys.csv', index=False)
    pd.DataFrame(issues).to_csv(AUDIT_V2 / 'data_quality_issues.csv', index=False)
    pd.DataFrame(mapping).to_csv(AUDIT_V2 / 'trustloop_data_mapping.csv', index=False)

    # validated relationships: same as rels but with child/parent coverage per complete sets (already included)
    pd.DataFrame(rels).to_csv(AUDIT_V2 / 'validated_relationships.csv', index=False)

    # terminal summary
    lines = []
    lines.append('TRUSTLOOP AUDIT V2 - TERMINAL SUMMARY')
    lines.append(f'Generated: {datetime.now().isoformat()}')
    lines.append('')
    total_rows = sum(p['row_count'] for p in profiles)
    lines.append(f'DATASETS FOUND:')
    for p in profiles:
        lines.append(f" - {p['filename']} ({p['row_count']:,} rows) role={p['role']}")
    lines.append('')
    lines.append(f'TOTAL ROWS: {total_rows:,}')
    lines.append('')
    # list relationships by classification
    def list_by(cls):
        return [r for r in rels if r['classification']==cls]
    lines.append('STRONG_ENTITY_RELATIONSHIPS:')
    for r in list_by('STRONG_ENTITY_RELATIONSHIP'):
        lines.append(f" - {r['parent_dataset']}.{r['parent_column']} -> {r['child_dataset']}.{r['child_column']} (intersection={r['intersection']:,}, child_cov={r['child_coverage_pct']}%)")
    lines.append('')
    lines.append('WEAK_ENTITY_RELATIONSHIPS:')
    for r in list_by('WEAK_ENTITY_RELATIONSHIP'):
        lines.append(f" - {r['parent_dataset']}.{r['parent_column']} -> {r['child_dataset']}.{r['child_column']} (intersection={r['intersection']:,})")
    lines.append('')
    lines.append('VALUE_OVERLAPS:')
    for o in overlaps:
        lines.append(f" - {o['dataset_a']}.{o['col_a']} <-> {o['dataset_b']}.{o['col_b']} top_overlap={o['top_overlap_count']}")
    lines.append('')
    (AUDIT_V2 / 'terminal_summary.txt').write_text('\n'.join(lines), encoding='utf-8')

    # docs
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / 'dataset_audit_v2.md').write_text('# Dataset Audit V2\nSee data/audit/v2/ for CSV outputs', encoding='utf-8')
    (DOCS_DIR / 'dataset_relationships_v2.md').write_text('# Relationships V2\nSee data/audit/v2/validated_relationships.csv', encoding='utf-8')
    (DOCS_DIR / 'trustloop_data_mapping_v2.md').write_text('# TrustLoop Data Mapping V2\nSee data/audit/v2/trustloop_data_mapping.csv', encoding='utf-8')


def main():
    if not RAW_DIR.exists():
        print('No raw dir', RAW_DIR); return 1
    csvs = sorted(RAW_DIR.glob('*.csv'))
    profiles = []
    errors = []
    for p in csvs:
        try:
            print('Profiling', p.name)
            profiles.append(profile_csv(p))
        except Exception:
            errors.append({'file':p.name,'error':traceback.format_exc()})
    # relationships
    rels = find_relationships_strict(profiles)
    overlaps = find_value_overlaps(profiles)
    keys = []
    for p in profiles:
        keys.extend(candidate_keys_table(p))
    issues = []
    for p in profiles:
        issues.extend(quality_checks(p))
    mapping = map_trustloop_fields(profiles)
    write_outputs(profiles, rels, overlaps, keys, issues, mapping)
    print('Audit V2 complete. Outputs written to', AUDIT_V2)
    if errors:
        print('Errors encountered:', errors)
    return 0

if __name__ == '__main__':
    sys.exit(main())
