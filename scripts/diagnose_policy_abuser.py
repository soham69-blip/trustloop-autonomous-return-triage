import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("TRUSTLOOP — POLICY ABUSER ML DIAGNOSTICS")
print("=" * 70)

DATA_PATH = Path("data/processed/trustloop/model_ready.csv")

df = pd.read_csv(DATA_PATH)

TARGET = "abuse_label"

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\n" + "=" * 70)
print("1. CLASS DISTRIBUTION")
print("=" * 70)

counts = df[TARGET].value_counts().sort_index()
percent = df[TARGET].value_counts(normalize=True).sort_index() * 100

for cls in counts.index:
    print(
        f"Class {cls}: "
        f"{counts[cls]:,} rows "
        f"({percent[cls]:.2f}%)"
    )

print("\n" + "=" * 70)
print("2. POLICY ABUSER VS OTHER CLASSES")
print("=" * 70)

policy = df[df[TARGET] == 1]
other = df[df[TARGET] != 1]

print(f"Policy Abuser rows: {len(policy):,}")
print(f"Other rows:         {len(other):,}")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

print("\nNumeric feature comparison:\n")

rows = []

for col in numeric_cols:
    p = pd.to_numeric(policy[col], errors="coerce")
    o = pd.to_numeric(other[col], errors="coerce")

    rows.append({
        "feature": col,
        "policy_mean": p.mean(),
        "other_mean": o.mean(),
        "policy_median": p.median(),
        "other_median": o.median(),
        "difference": abs(p.mean() - o.mean())
    })

comparison = pd.DataFrame(rows)
comparison = comparison.sort_values("difference", ascending=False)

print(comparison.to_string(index=False))

print("\n" + "=" * 70)
print("3. CLASS-SPECIFIC NUMERIC STATISTICS")
print("=" * 70)

for col in numeric_cols:
    print(f"\n--- {col} ---")

    stats = (
        df.groupby(TARGET)[col]
        .agg(["mean", "median", "std", "min", "max"])
        .round(4)
    )

    print(stats)

print("\n" + "=" * 70)
print("4. CATEGORICAL FEATURES")
print("=" * 70)

categorical_cols = [
    c for c in df.columns
    if c != TARGET
    and (
        pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_string_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
    )
]

print("Categorical columns:")
print(categorical_cols)

for col in categorical_cols:
    print("\n" + "-" * 60)
    print(f"FEATURE: {col}")

    table = pd.crosstab(
        df[col],
        df[TARGET],
        normalize="columns"
    ) * 100

    print(table.round(2).to_string())

print("\n" + "=" * 70)
print("5. BINARY FEATURES")
print("=" * 70)

binary_cols = []

for col in numeric_cols:
    values = set(df[col].dropna().unique())

    if values.issubset({0, 1}):
        binary_cols.append(col)

for col in binary_cols:
    print(f"\n--- {col} ---")

    table = pd.crosstab(
        df[col],
        df[TARGET],
        normalize="columns"
    ) * 100

    print(table.round(2).to_string())

print("\n" + "=" * 70)
print("6. POLICY ABUSER CORRELATION")
print("=" * 70)

policy_indicator = (df[TARGET] == 1).astype(int)

corr_rows = []

for col in numeric_cols:
    series = pd.to_numeric(df[col], errors="coerce")

    corr = series.corr(policy_indicator)

    corr_rows.append({
        "feature": col,
        "correlation_with_policy_abuser": corr
    })

corr_df = pd.DataFrame(corr_rows)

corr_df["abs_correlation"] = (
    corr_df["correlation_with_policy_abuser"].abs()
)

corr_df = corr_df.sort_values(
    "abs_correlation",
    ascending=False
)

print(
    corr_df[
        ["feature", "correlation_with_policy_abuser"]
    ].round(4).to_string(index=False)
)

print("\n" + "=" * 70)
print("7. MISSING VALUES")
print("=" * 70)

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("No missing values.")
else:
    print(missing.to_string())

print("\n" + "=" * 70)
print("8. DUPLICATE ROWS")
print("=" * 70)

print("Duplicate rows:", df.duplicated().sum())

print("\n" + "=" * 70)
print("DIAGNOSTICS COMPLETED")
print("=" * 70)