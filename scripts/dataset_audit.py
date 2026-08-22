import sys
from pathlib import Path
import pandas as pd

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/dataset_audit.py data/raw/<file>.csv")
        raise SystemExit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        raise SystemExit(1)

    df = pd.read_csv(path)

    print("\n=== FlipLens Dataset Audit ===")
    print("File:", path)
    print("Shape:", df.shape)
    print("\nColumns:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False).to_string())

    print("\nDuplicate rows:", int(df.duplicated().sum()))
    print("\nPreview:")
    print(df.head().to_string())

if __name__ == "__main__":
    main()
