"""
Verify existing dataset audit by computing exact metrics from raw CSVs.

Produces:
- data/audit/audit_verification.csv
- docs/audit_verification.md

The script uses chunked reads and avoids loading full huge files into memory when possible.

Do NOT modify files under data/raw/.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Config
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
AUDIT_DIR = ROOT / "data" / "audit"
DOCS_DIR = ROOT / "docs"
CHUNKSIZE = 50_000
NUMERIC_MEDIAN_CAP = 1_000_000  # if more values than this, median will be SAMPLE-BASED
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

ID_COLS_CANDIDATES = {"customer_id", "order_id", "product_id", "return_id", "transaction_id", "sku"}

DATE_NAME_REPL = ["date", "time", "datetime", "timestamp", "expiry", "delivered", "delivery_date", "return_date", "request_date", "order_date"]


def detect_encoding(path: Path) -> str:
    for enc in ENCODINGS:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                f.read(65_536)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def normalize_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def looks_like_date_name(col: str) -> bool:
    n = normalize_col(col)
    return any(k in n for k in ("date", "time", "datetime", "timestamp", "delivered", "return", "order", "request"))


def compute_file_metrics(path: Path) -> dict:
    encoding = detect_encoding(path)
    it = pd.read_csv(path, encoding=encoding, dtype=str, chunksize=CHUNKSIZE, on_bad_lines='warn')

    columns = None
    row_count = 0
    missing: dict[str, int] = {}
    blank_string: dict[str, int] = {}
    dup_hashes = set()
    duplicate_row_count = 0

    unique_sets = defaultdict(set)  # per column
    unique_overflow = set()

    numeric_stats = {}  # col -> {'count','mean','m2','min','max'} for Welford
    numeric_values = defaultdict(list)  # for median if practical

    date_stats = {}  # col -> {'min':..., 'max':..., 'invalid': int}
    categorical_counters = defaultdict(Counter)

    for chunk in it:
        if columns is None:
            columns = list(chunk.columns)
            missing = dict.fromkeys(columns, 0)
            blank_string = dict.fromkeys(columns, 0)

        n = len(chunk)
        row_count += n

        # missing
        na = chunk.isna()
        for c in columns:
            missing[c] += int(na[c].sum())
            s = chunk[c].astype(str)
            blank_string[c] += int((s.str.strip() == "").sum())

        # duplicate rows via pandas hash
        try:
            hashed = pd.util.hash_pandas_object(chunk, index=False)
        except Exception:
            hashed = pd.util.hash_pandas_object(chunk.fillna(""), index=False)
        for h in hashed.astype('uint64').to_numpy():
            if int(h) in dup_hashes:
                duplicate_row_count += 1
            else:
                dup_hashes.add(int(h))

        # unique sets for candidate id cols and any id-like column name
        for c in columns:
            nrm = normalize_col(c)
            if nrm in ID_COLS_CANDIDATES or nrm.endswith("_id") or nrm == "sku":
                vals = chunk[c].dropna().astype(str).str.strip()
                for v in vals:
                    if v == "":
                        continue
                    unique_sets[c].add(v)

        # numeric detection per chunk
        for c in columns:
            s = chunk[c].dropna().astype(str).str.strip()
            if s.empty:
                continue
            # try numeric
            num = pd.to_numeric(s, errors='coerce')
            if num.notna().mean() >= 0.85:
                arr = num.dropna().astype(float).to_numpy()
                st = numeric_stats.setdefault(c, {'count':0,'mean':0.0,'m2':0.0,'min':None,'max':None})
                # update Welford
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
                # collect values for median if not too many
                if len(numeric_values[c]) <= NUMERIC_MEDIAN_CAP:
                    numeric_values[c].extend(arr.tolist())

        # date-like columns
        for c in columns:
            if looks_like_date_name(c):
                s = chunk[c].dropna().astype(str).str.strip()
                if s.empty:
                    continue
                parsed = pd.to_datetime(s, errors='coerce', dayfirst=True)
                invalid = int(parsed.isna().sum())
                ok = parsed.dropna()
                st = date_stats.setdefault(c, {'min':None,'max':None,'invalid':0})
                st['invalid'] += invalid
                if not ok.empty:
                    mn = ok.min()
                    mx = ok.max()
                    st['min'] = mn if st['min'] is None else min(st['min'], mn)
                    st['max'] = mx if st['max'] is None else max(st['max'], mx)

        # categorical top values (limited)
        for c in columns:
            s = chunk[c].dropna().astype(str).str.strip()
            if s.empty:
                continue
            # update counter but limit memory by only counting top frequencies via Counter
            categorical_counters[c].update(s.tolist())

    # finalize numeric stats: compute std
    numeric_summary = {}
    for c, st in numeric_stats.items():
        count = st['count']
        mean = st['mean']
        std = math.sqrt(st['m2'] / count) if count > 1 else 0.0
        median = None
        median_note = 'EXACT'
        vals = numeric_values.get(c, [])
        if vals:
            if len(vals) > NUMERIC_MEDIAN_CAP:
                # sample-based median
                median = float(np.median(np.random.choice(vals, NUMERIC_MEDIAN_CAP, replace=False)))
                median_note = 'SAMPLE-BASED'
            else:
                median = float(np.median(vals))
        numeric_summary[c] = {
            'count': count,
            'mean': mean,
            'std': std,
            'min': st['min'],
            'max': st['max'],
            'median': median,
            'median_note': median_note,
        }

    # categorical top 20
    categorical_summary = {}
    for c, counter in categorical_counters.items():
        total = sum(counter.values())
        top20 = counter.most_common(20)
        categorical_summary[c] = {'unique_count': len(counter), 'top20': top20, 'total': total}

    # prepare id column exact unique counts
    id_col_summary = {}
    for c, s in unique_sets.items():
        id_col_summary[c] = {'unique_count': len(s), 'null_count': 0}  # null_count computed earlier if needed

    result = {
        'filename': path.name,
        'path': str(path.relative_to(ROOT)).replace('\\','/'),
        'file_size_bytes': path.stat().st_size,
        'encoding': encoding,
        'row_count': row_count,
        'column_count': len(columns) if columns else 0,
        'columns': columns or [],
        'missing_counts': missing or {},
        'blank_string_counts': blank_string or {},
        'duplicate_row_count': duplicate_row_count,
        'numeric_summary': numeric_summary,
        'date_stats': date_stats,
        'categorical_summary': categorical_summary,
        'id_col_summary': id_col_summary,
    }
    return result


def load_old_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds = pd.read_csv(AUDIT_DIR / 'dataset_summary.csv') if (AUDIT_DIR / 'dataset_summary.csv').exists() else pd.DataFrame()
    cols = pd.read_csv(AUDIT_DIR / 'column_summary.csv') if (AUDIT_DIR / 'column_summary.csv').exists() else pd.DataFrame()
    rels = pd.read_csv(AUDIT_DIR / 'relationship_candidates.csv') if (AUDIT_DIR / 'relationship_candidates.csv').exists() else pd.DataFrame()
    return ds, cols, rels


def compare_and_emit(old_ds_df: pd.DataFrame, old_col_df: pd.DataFrame, verified_results: list[dict]) -> None:
    rows = []
    # index old by filename
    old_ds = {r['filename']: r for _, r in old_ds_df.iterrows()} if not old_ds_df.empty else {}
    old_cols = {}
    if not old_col_df.empty:
        for _, r in old_col_df.iterrows():
            old_cols.setdefault(r['filename'], {})[r['column']] = r

    total_rows_verified = 0
    for res in verified_results:
        fname = res['filename']
        total_rows_verified += res['row_count']
        # dataset-level comparisons
        # row_count
        old_row = old_ds.get(fname, {}).get('row_count') if old_ds else None
        if old_row is not None:
            old_val = int(old_row)
            new_val = int(res['row_count'])
            status = 'MATCH' if old_val == new_val else 'MISMATCH'
            notes = '' if status == 'MATCH' else 'Row count differs'
            rows.append({'dataset': fname, 'metric': 'row_count', 'old_value': old_val, 'verified_value': new_val, 'status': status, 'notes': notes})
        else:
            rows.append({'dataset': fname, 'metric': 'row_count', 'old_value': None, 'verified_value': int(res['row_count']), 'status': 'MISSING_FROM_OLD_AUDIT', 'notes': ''})

        # column_count
        old_colc = old_ds.get(fname, {}).get('column_count') if old_ds else None
        if pd.notna(old_colc):
            old_val = int(old_colc)
            new_val = int(res['column_count'])
            status = 'MATCH' if old_val == new_val else 'MISMATCH'
            rows.append({'dataset': fname, 'metric': 'column_count', 'old_value': old_val, 'verified_value': new_val, 'status': status, 'notes': ''})
        else:
            rows.append({'dataset': fname, 'metric': 'column_count', 'old_value': None, 'verified_value': int(res['column_count']), 'status': 'MISSING_FROM_OLD_AUDIT', 'notes': ''})

        # column names
        old_cols_str = old_ds.get(fname, {}).get('column_names') if old_ds else None
        old_cols_list = old_cols_str.split('|') if pd.notna(old_cols_str) else None
        new_cols_list = res['columns']
        if old_cols_list is not None:
            status = 'MATCH' if [c for c in old_cols_list] == [c for c in new_cols_list] else 'MISMATCH'
            rows.append({'dataset': fname, 'metric': 'column_names', 'old_value': '|'.join(old_cols_list), 'verified_value': '|'.join(new_cols_list), 'status': status, 'notes': ''})
        else:
            rows.append({'dataset': fname, 'metric': 'column_names', 'old_value': None, 'verified_value': '|'.join(new_cols_list), 'status': 'MISSING_FROM_OLD_AUDIT', 'notes': ''})

        # duplicate rows
        old_dup = old_ds.get(fname, {}).get('duplicate_row_count') if old_ds else None
        if pd.notna(old_dup):
            old_val = int(old_dup)
            new_val = int(res['duplicate_row_count'])
            status = 'MATCH' if old_val == new_val else 'MISMATCH'
            rows.append({'dataset': fname, 'metric': 'duplicate_row_count', 'old_value': old_val, 'verified_value': new_val, 'status': status, 'notes': ''})
        else:
            rows.append({'dataset': fname, 'metric': 'duplicate_row_count', 'old_value': None, 'verified_value': int(res['duplicate_row_count']), 'status': 'MISSING_FROM_OLD_AUDIT', 'notes': ''})

        # per-column: missing counts vs old
        for col in res['columns']:
            old_col = old_cols.get(fname, {}).get(col) if old_cols else None
            new_missing = int(res['missing_counts'].get(col, 0))
            if old_col is not None and len(old_col) > 0:
                old_missing = int(old_col.get('missing_count', 0))
                status = 'MATCH' if old_missing == new_missing else 'MISMATCH'
                rows.append({'dataset': fname, 'metric': f'{col}.missing_count', 'old_value': old_missing, 'verified_value': new_missing, 'status': status, 'notes': ''})
            else:
                rows.append({'dataset': fname, 'metric': f'{col}.missing_count', 'old_value': None, 'verified_value': new_missing, 'status': 'MISSING_FROM_OLD_AUDIT', 'notes': ''})

            # unique counts if present in old audit
            if old_col is not None and pd.notna(old_col.get('unique_count')):
                try:
                    old_uniq = int(old_col.get('unique_count'))
                    # if we computed unique (id_col_summary)
                    idsum = res['id_col_summary'].get(col)
                    if idsum:
                        new_uniq = int(idsum['unique_count'])
                        status = 'MATCH' if old_uniq == new_uniq else 'MISMATCH'
                        rows.append({'dataset': fname, 'metric': f'{col}.unique_count', 'old_value': old_uniq, 'verified_value': new_uniq, 'status': status, 'notes': ''})
                except Exception:
                    pass

    out_df = pd.DataFrame(rows)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(AUDIT_DIR / 'audit_verification.csv', index=False)

    # also write a markdown summary
    md_lines = []
    md_lines.append('# Audit verification report')
    md_lines.append(f'Generated: {datetime.now().isoformat()}')
    md_lines.append('')
    md_lines.append('Summary of dataset verification and differences vs existing audit files.')
    md_lines.append('')

    # high-level counts
    total_csv = len(verified_results)
    total_rows = sum(r['row_count'] for r in verified_results)
    mismatches = out_df[out_df['status'] == 'MISMATCH']
    missing_old = out_df[out_df['status'] == 'MISSING_FROM_OLD_AUDIT']

    md_lines.append(f'- Total CSV files verified: {total_csv}')
    md_lines.append(f'- Total rows verified (sum): {total_rows:,}')
    md_lines.append(f'- Metric mismatches found: {len(mismatches)}')
    md_lines.append(f'- Metrics missing from old audit: {len(missing_old)}')
    md_lines.append('')
    md_lines.append('## Mismatches (sample)')
    md_lines.append('')
    for _, r in mismatches.head(50).iterrows():
        md_lines.append(f"- {r['dataset']} | {r['metric']}: old={r['old_value']} new={r['verified_value']}")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / 'audit_verification.md').write_text('\n'.join(md_lines), encoding='utf-8')

    print('Wrote', AUDIT_DIR / 'audit_verification.csv')
    print('Wrote', DOCS_DIR / 'audit_verification.md')


def verify_relationships(verified_results: list[dict], old_rels_df: pd.DataFrame) -> list[dict]:
    """Verify only the relationships present in old_rels_df with confidence HIGH or MEDIUM/HIGH.
    Returns list of verification dicts for relationships."""
    rels_out = []
    # build map of filename->result
    res_map = {r['filename']: r for r in verified_results}

    # consider only high or medium from old
    if old_rels_df is None or old_rels_df.empty:
        return rels_out
    candidates = old_rels_df[old_rels_df['confidence'].isin(['HIGH','MEDIUM','MEDIUM_High','MEDIUM_LOW','MEDIUM'])] if 'confidence' in old_rels_df.columns else old_rels_df
    # fall back to all
    if candidates.empty:
        candidates = old_rels_df

    for _, r in candidates.iterrows():
        parent = r['parent_dataset']
        child = r['child_dataset']
        pcol = r['parent_column']
        ccol = r['child_column']
        # load parent unique set from verified results if available
        p_res = res_map.get(parent)
        c_res = res_map.get(child)
        if not p_res or not c_res:
            continue
        # find unique sets: computed earlier; in id_col_summary keys are original col names
        pset = set()
        cset = set()
        # To compute exact matching keys and coverage, we may need to re-scan files to collect parent set and child counts
        # Build parent set
        parent_path = RAW_DIR / p_res['filename']
        enc = detect_encoding(parent_path)
        for chunk in pd.read_csv(parent_path, encoding=enc, dtype=str, usecols=[pcol], chunksize=CHUNKSIZE, on_bad_lines='warn'):
            vals = chunk[pcol].dropna().astype(str).str.strip()
            pset.update(v for v in vals if v != '')
        # Build child set and counts
        child_path = RAW_DIR / c_res['filename']
        enc2 = detect_encoding(child_path)
        child_rows = 0
        child_no_parent = 0
        child_nulls = 0
        for chunk in pd.read_csv(child_path, encoding=enc2, dtype=str, usecols=[ccol], chunksize=CHUNKSIZE, on_bad_lines='warn'):
            vals = chunk[ccol].astype(str).str.strip()
            child_rows += len(vals)
            for v in vals:
                if v == '' or v.lower() == 'nan':
                    child_nulls += 1
                else:
                    if v not in pset:
                        child_no_parent += 1
                    cset.add(v)
        intersection = len(pset & cset)
        parent_unique = len(pset)
        child_unique = len(cset)
        child_rows_no_parent = child_no_parent
        parent_keys_no_child = max(0, parent_unique - intersection)
        child_coverage_pct = round(100.0 * intersection / child_unique, 4) if child_unique else 0.0
        parent_coverage_pct = round(100.0 * intersection / parent_unique, 4) if parent_unique else 0.0
        rels_out.append({'parent_dataset': parent, 'parent_column': pcol, 'child_dataset': child, 'child_column': ccol, 'parent_unique': parent_unique, 'child_unique': child_unique, 'intersection': intersection, 'child_rows': child_rows, 'child_rows_no_parent': child_rows_no_parent, 'child_coverage_pct': child_coverage_pct, 'parent_coverage_pct': parent_coverage_pct})
    # write to audit dir
    if rels_out:
        pd.DataFrame(rels_out).to_csv(AUDIT_DIR / 'relationship_verification.csv', index=False)
        print('Wrote', AUDIT_DIR / 'relationship_verification.csv')
    return rels_out


def main():
    if not RAW_DIR.exists():
        print('No raw dir', RAW_DIR)
        return 1
    csvs = sorted(RAW_DIR.glob('*.csv'))
    if not csvs:
        print('No CSVs found')
        return 1

    old_ds_df, old_col_df, old_rels_df = load_old_audit()

    verified = []
    errors = []
    for p in csvs:
        try:
            print('Verifying', p.name)
            res = compute_file_metrics(p)
            verified.append(res)
        except Exception as e:
            print('ERROR processing', p.name, e)
            errors.append({'filename': p.name, 'error': str(e)})

    compare_and_emit(old_ds_df, old_col_df, verified)
    rels_verified = verify_relationships(verified, old_rels_df)

    # final verdict heuristics
    # If any critical mismatches in row counts or duplicate counts -> PARTIALLY TRUSTWORTHY or INVALID
    import pandas as _pd
    av = _pd.read_csv(AUDIT_DIR / 'audit_verification.csv')
    mismatches = av[av['status'] == 'MISMATCH']
    critical_mismatch = mismatches[mismatches['metric'].isin(['row_count','duplicate_row_count'])]
    if critical_mismatch.empty and mismatches.empty:
        status = 'TRUSTWORTHY'
    elif not critical_mismatch.empty:
        status = 'INVALID'
    else:
        status = 'PARTIALLY TRUSTWORTHY'

    # summary file
    md = []
    md.append('# Audit Verification Summary')
    md.append('')
    md.append(f'Verification run: {datetime.now().isoformat()}')
    md.append('')
    md.append(f'AUDIT STATUS:')
    md.append(status)
    md.append('')
    md.append('METRICS:')
    md.append(f'- TOTAL CSV FILES FOUND: {len(csvs)}')
    md.append(f'- TOTAL CSV FILES VERIFIED: {len(verified)}')
    md.append(f'- TOTAL ROWS VERIFIED: {sum(r["row_count"] for r in verified):,}')
    md.append(f'- FILES WITH MISMATCHES: {len(mismatches["dataset"].unique())}')
    md.append(f'- FILES WITH ERRORS: {len(errors)}')
    md.append(f'- RELATIONSHIPS VERIFIED: {len(rels_verified)}')
    md.append(f'- RELATIONSHIPS WITH PROBLEMS: {sum(1 for r in rels_verified if r["intersection"]==0)}')
    md.append('')
    if errors:
        md.append('Errors:')
        for e in errors:
            md.append(f"- {e['filename']}: {e['error']}")
    (DOCS_DIR / 'audit_verification.md').write_text('\n'.join(md), encoding='utf-8')
    print('Wrote', DOCS_DIR / 'audit_verification.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
