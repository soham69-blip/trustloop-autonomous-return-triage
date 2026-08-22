import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/trustloop/model_ready.csv")
target = "abuse_label"

out = []

out.append("CLASS DISTRIBUTION")
out.append(df[target].value_counts().sort_index().to_string())

out.append("\n\nPOLICY ABUSER NUMERIC COMPARISON")

policy = df[df[target] == 1]
other = df[df[target] != 1]

numeric = df.select_dtypes(include=np.number).columns.tolist()
numeric.remove(target)

rows = []

for c in numeric:
    p = pd.to_numeric(policy[c], errors="coerce")
    o = pd.to_numeric(other[c], errors="coerce")

    rows.append({
        "feature": c,
        "policy_mean": p.mean(),
        "other_mean": o.mean(),
        "difference": abs(p.mean() - o.mean())
    })

comparison = pd.DataFrame(rows).sort_values("difference", ascending=False)

out.append(comparison.to_string(index=False))

out.append("\n\nPOLICY ABUSER CORRELATIONS")

indicator = (df[target] == 1).astype(int)

corr = []

for c in numeric:
    value = pd.to_numeric(df[c], errors="coerce").corr(indicator)
    corr.append({
        "feature": c,
        "correlation": value
    })

corr_df = pd.DataFrame(corr)
corr_df["abs_correlation"] = corr_df["correlation"].abs()
corr_df = corr_df.sort_values("abs_correlation", ascending=False)

out.append(corr_df[["feature", "correlation"]].to_string(index=False))

out.append("\n\nCATEGORICAL FEATURES")

categorical = [
    c for c in df.columns
    if c != target and (
        pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_string_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
    )
]

for c in categorical:
    out.append(f"\n--- {c} ---")
    table = pd.crosstab(
        df[c],
        df[target],
        normalize="columns"
    ) * 100
    out.append(table.round(2).to_string())

with open(
    "reports/policy_abuser_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(out))

print("CREATED: reports/policy_abuser_summary.txt")
