"""
Validate TrustLoop data architecture (final validation step).

Outputs:
- data/audit/v2/data_architecture_validation.csv
- data/audit/v2/validated_relationships_final.csv
- docs/trustloop_data_architecture.md

Strict id-based overlaps; chunked reads; no data modification.
"""
from __future__ import annotations

import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
AUDIT_V2 = ROOT / "data" / "audit" / "v2"
DOCS = ROOT / "docs"
CHUNKSIZE = 50000
ENCODINGS = ("utf-8-sig","utf-8","cp1252","latin-1")

ID_CANDIDATES = {"customer_id","order_id","product_id","return_id","transaction_id","sku"}
ICE_CREAM_KEYS = {"ice_cream","flavor","scoop"}
FRAUD_COLUMNS = {"abuse_label","abuse_type","fraud_label","previous_dispute_count","previous_fraud_count","abuse_score"}


def detect_encoding(path: Path)->str:
    for e in ENCODINGS:
        try:
            with path.open('r',encoding=e,errors='strict') as f:
                f.read(65536)
            return e
        except Exception:
            continue
    return 'latin-1'


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def collect_column_info(path: Path):
    enc = detect_encoding(path)
    it = pd.read_csv(path, encoding=enc, dtype=str, chunksize=CHUNKSIZE, on_bad_lines='warn')
    columns = None
    row_count = 0
    missing_counts = defaultdict(int)
    blank_counts = defaultdict(int)
    duplicate_hashes = set()
    duplicate_rows = 0
    id_unique_sets = defaultdict(set)
    has_ice = False
    col_samples = defaultdict(list)

    for chunk in it:
        if columns is None:
            columns = list(chunk.columns)
        n = len(chunk)
        row_count += n
        na = chunk.isna()
        for c in columns:
            missing_counts[c] += int(na[c].sum())
            s = chunk[c].astype(str)
            blank_counts[c] += int((s.str.strip()=='').sum())
            if len(col_samples[c])<5:
                col_samples[c].extend(s.dropna().astype(str).head(5-len(col_samples[c])).tolist())
        # duplicate rows via pandas hash
        try:
            hashed = pd.util.hash_pandas_object(chunk, index=False)
        except Exception:
            hashed = pd.util.hash_pandas_object(chunk.fillna(''), index=False)
        for h in hashed.astype('uint64').to_numpy():
            if int(h) in duplicate_hashes:
                duplicate_rows += 1
            else:
                duplicate_hashes.add(int(h))
        # id sets
        for c in columns:
            nrm = normalize(c)
            if nrm in ID_CANDIDATES or nrm.endswith('_id') or nrm=='sku':
                vals = chunk[c].dropna().astype(str).str.strip()
                for v in vals:
                    if v=='':
                        continue
                    id_unique_sets[c].add(v)
        # ice cream
        for c in columns:
            if any(k in normalize(c) for k in ICE_CREAM_KEYS):
                has_ice = True
    info = {
        'path': str(path.relative_to(ROOT)).replace('\\','/'),
        'filename':path.name,
        'file_size_bytes':path.stat().st_size,
        'row_count':row_count,
        'columns':columns or [],
        'missing_counts':dict(missing_counts),
        'blank_counts':dict(blank_counts),
        'duplicate_row_count':duplicate_rows,
        'id_unique_sets':{k:len(v) for k,v in id_unique_sets.items()},
        'id_sets_raw': {k:v for k,v in id_unique_sets.items()},
        'has_ice':has_ice,
        'samples':{k:v for k,v in col_samples.items()},
    }
    return info


def score_suitability(info):
    # scoring weights
    score = 0
    reasons = []
    cols = {normalize(c) for c in info['columns']}
    # return identifiers
    if any(x in cols for x in ('return_id','returnid','return')):
        score += 30; reasons.append('has return id')
    if any(x in cols for x in ('return_date','request_date')):
        score += 20; reasons.append('has return date')
    if any(x in cols for x in ('return_reason','return_reason_text')):
        score += 15; reasons.append('has return reason')
    # order identifiers
    if any(x in cols for x in ('order_id','orderid')):
        score += 15; reasons.append('has order id')
    if any(x in cols for x in ('order_date','order_datetime')):
        score += 10; reasons.append('has order date')
    # customer/product
    if any(x in cols for x in ('customer_id','customerid')):
        score += 10; reasons.append('has customer id')
    if any(x in cols for x in ('product_id','productid','sku')):
        score += 5; reasons.append('has product id')
    # fraud signals
    if any(x in cols for x in FRAUD_COLUMNS):
        score += 20; reasons.append('has fraud/abuse labels')
    # order value
    if any(x in cols for x in ('order_value','total_amount','price','final_price','order_value_inr')):
        score += 10; reasons.append('has order value')
    # normalize to 0-100
    score = min(score,100)
    return score, '; '.join(reasons)


def detect_fraud_signal(info):
    cols = {normalize(c) for c in info['columns']}
    present = cols & FRAUD_COLUMNS
    if present:
        return 'REAL_SOURCE_LABEL', ','.join(sorted(present))
    # derivable proxies
    if any(x in cols for x in ('previous_dispute_count','previous_dispute')):
        return 'DERIVABLE','previous_dispute_count'
    return 'UNAVAILABLE',''


def determine_grain(info):
    # heuristics
    cols = {normalize(c) for c in info['columns']}
    if any(x in cols for x in ('order_id','order_datetime','order_date')) and any(x in cols for x in ('product_id','product')):
        return 'ORDER','one row per ordered item/line or order'
    if any(x in cols for x in ('return_id','return_date','return_reason')):
        return 'RETURN','one row per return case'
    if any(x in cols for x in ('product_id','product_name','sku')) and not any(x in cols for x in ('order_id','order_date')):
        return 'PRODUCT','product catalog row'
    if any(x in cols for x in ('customer_id',)) and len(cols)<=6:
        return 'CUSTOMER','customer-level info'
    if 'delivery_time_min' in cols or 'delivery_delay' in cols:
        return 'DELIVERY','delivery analytics row'
    return 'OTHER','unclassified'


def compute_relationships(infos):
    # build map of file->id columns raw sets
    id_maps = {}  # (filename, col) -> set
    for info in infos:
        for col,sz in info.get('id_unique_sets',{}).items():
            # load the raw set from id_sets_raw if present
            raw = info.get('id_sets_raw',{}).get(col,set())
            id_maps[(info['filename'],col)] = raw
    rels = []
    keys = list(id_maps.keys())
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            a = keys[i]; b = keys[j]
            fname_a,col_a = a; fname_b,col_b = b
            set_a = id_maps[a]
            set_b = id_maps[b]
            inter = len(set_a & set_b)
            parent_unique = len(set_a); child_unique = len(set_b)
            child_cov = round(100.0 * inter / child_unique,4) if child_unique else 0.0
            parent_cov = round(100.0 * inter / parent_unique,4) if parent_unique else 0.0
            # classification
            if inter==0:
                typ='INVALID'
                rec=False
                reason='no overlap'
            elif child_cov>=80 or parent_cov>=80:
                typ='STRONG_ENTITY_RELATIONSHIP'; rec=True; reason='high coverage'
            elif inter>=50 and (child_cov>=10 or parent_cov>=10):
                typ='WEAK_ENTITY_RELATIONSHIP'; rec=False; reason='moderate overlap'
            else:
                typ='WEAK_ENTITY_RELATIONSHIP'; rec=False; reason='small overlap'
            rels.append({'parent_dataset':fname_a,'parent_column':col_a,'child_dataset':fname_b,'child_column':col_b,'intersection_count':inter,'child_coverage_percentage':child_cov,'parent_coverage_percentage':parent_cov,'relationship_type':typ,'recommended_for_join':rec,'reason':reason})
    return rels


def propose_architecture(valid_rels, infos):
    # Build simple architecture: choose strongest order-return relationships
    # Find strong entity relationships where one side is RETURN and other ORDER or RETURN
    by_file = {p['filename']:p for p in infos}
    strong = [r for r in valid_rels if r['relationship_type']=='STRONG_ENTITY_RELATIONSHIP']
    arch_lines = []
    arch_lines.append('# Proposed TrustLoop Data Architecture')
    arch_lines.append('')
    if not strong:
        arch_lines.append('No STRONG entity relationships found. Manual integration needed.')
        return '\n'.join(arch_lines)
    # for each strong rel, add line
    for r in strong:
        arch_lines.append(f"{r['parent_dataset']}.{r['parent_column']} --> {r['child_dataset']}.{r['child_column']}  (intersection={r['intersection_count']})")
    arch_lines.append('')
    arch_lines.append('Recommendation: Use a RETURN table as canonical return-case grain. Join ORDER source where STRONG relationships exist to map order metadata.')
    return '\n'.join(arch_lines)


def build_canonical_schema(infos):
    # For each field in canonical list, mark AVAILABLE/DERVIABLE/MISSING and source
    fields = [
        'case_id','customer_id','order_id','product_id','product_category','product_name','brand','order_date','delivery_date','return_date','return_reason','order_value','payment_method',
        'customer_order_count','customer_return_count','customer_return_rate','returns_last_30d','returns_last_90d','previous_fraud_count','high_value_return_count','return_window_valid','policy_violation','fraud_probability','fraud_label','final_decision','confidence','explanation'
    ]
    col_index = defaultdict(list)
    for p in infos:
        for c in p['columns']:
            col_index[normalize(c)].append((p['filename'],c,p['row_count']))
    rows = []
    for f in fields:
        status='MISSING'; src_ds=''; src_col=''
        normf = normalize(f)
        # direct matches
        if normf in col_index:
            src_ds,src_col,_ = sorted(col_index[normf], key=lambda t: t[2], reverse=True)[0]
            status='AVAILABLE'
        else:
            # some mappings
            if f in ('customer_order_count','customer_return_count','customer_return_rate','returns_last_30d','returns_last_90d'):
                if 'customer_id' in col_index and any(k in col_index for k in ('return_date','request_date','returned')):
                    status='DERIVABLE'; src_ds=col_index['customer_id'][0][0] if 'customer_id' in col_index else '' ; src_col='customer_id + return dates'
        rows.append({'field':f,'status':status,'source_dataset':src_ds,'source_column':src_col})
    return rows


def main():
    if not RAW_DIR.exists():
        print('No raw dir', RAW_DIR); return 1
    csvs = sorted(RAW_DIR.glob('*.csv'))
    infos = []
    for p in csvs:
        print('Analyzing',p.name)
        infos.append(collect_column_info(p))
    # Step1 grains
    arch_rows = []
    for info in infos:
        grain,reason = determine_grain(info)
        role = grain
        score, score_reasons = score_suitability(info)
        fraud, fraud_detail = detect_fraud_signal(info)
        # candidate key heuristics: choose id-like column with unique count == rows
        cand_key = ''
        for c,ct in info.get('id_unique_sets',{}).items():
            if ct==info['row_count']:
                cand_key=c; break
        arch_rows.append({'dataset':info['filename'],'grain':grain,'role':role,'return_case_suitability_score':score,'fraud_signal':fraud,'candidate_key':cand_key,'notes':score_reasons})
    # relationships
    rels = compute_relationships(infos)
    # filter to only id-based relationships between different datasets
    # write outputs
    AUDIT_V2.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(arch_rows).to_csv(AUDIT_V2 / 'data_architecture_validation.csv', index=False)
    pd.DataFrame(rels).to_csv(AUDIT_V2 / 'validated_relationships_final.csv', index=False)
    # docs
    # build docs/trustloop_data_architecture.md
    lines = []
    lines.append('# TrustLoop Data Architecture Validation')
    lines.append(f'Generated: {datetime.now().isoformat()}')
    lines.append('')
    lines.append('## Dataset inventory and grain')
    for r in arch_rows:
        lines.append(f"- {r['dataset']}: grain={r['grain']} role={r['role']} suitability={r['return_case_suitability_score']} candidate_key={r['candidate_key']} notes={r['notes']}")
    lines.append('')
    lines.append('## Validated relationships (id-based)')
    for r in rels:
        lines.append(f"- {r['parent_dataset']}.{r['parent_column']} <-> {r['child_dataset']}.{r['child_column']}: intersection={r['intersection_count']} child_cov={r['child_coverage_percentage']}% parent_cov={r['parent_coverage_percentage']}% type={r['relationship_type']} recommended={r['recommended_for_join']} reason={r['reason']}")
    lines.append('')
    lines.append('## Proposed architecture')
    lines.append(propose_architecture(rels, infos))
    lines.append('')
    lines.append('## Canonical return-case schema availability')
    schema_rows = build_canonical_schema(infos)
    for s in schema_rows:
        lines.append(f"- {s['field']}: {s['status']} (source: {s['source_dataset']}.{s['source_column']})")
    # ice-cream conclusion
    ice_datasets = [info['filename'] for info in infos if info.get('has_ice')]
    if ice_datasets:
        lines.append('\n## Ice-cream datasets found:')
        for d in ice_datasets:
            lines.append(f'- {d}')
    else:
        lines.append('\n## Ice-cream datasets: NO STANDALONE ICE-CREAM DATASET FOUND')
    lines.append('\n## Next step')
    lines.append('- Choose canonical RETURN dataset (based on suitability and presence of return identifiers).')
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / 'trustloop_data_architecture.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote data/audit/v2/data_architecture_validation.csv and validated_relationships_final.csv and docs/trustloop_data_architecture.md')
    return 0

if __name__=='__main__':
    sys.exit(main())
