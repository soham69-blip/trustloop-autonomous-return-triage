"""Placeholder for the FlipLens canonical data pipeline.

Do not modify files under data/raw/. Read raw data and write processed
artifacts under data/processed/.
"""

from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Dataset preparation scaffold ready.")
    print("Next: map the audited Kaggle columns into the FlipLens canonical schema.")

if __name__ == "__main__":
    main()
