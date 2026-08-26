from pathlib import Path
import sys
import pickle
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODEL_PATH = ROOT_DIR / "models" / "lightgbm_model.pkl"
DATA_PATH = ROOT_DIR / "data" / "processed" / "trustloop" / "model_ready.csv"

LABELS = {
    0: "Legitimate",
    1: "Policy Abuser",
    2: "Fraudulent Return",
    3: "Wardrobing",
}


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found: {MODEL_PATH}"
        )

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def get_model_features(model):

    features = getattr(
        model,
        "feature_name_",
        None
    )

    if features:
        return list(features)

    booster = getattr(
        model,
        "booster_",
        None
    )

    if booster is not None:
        return list(
            booster.feature_name()
        )

    raise RuntimeError(
        "Could not determine model feature names."
    )


def inspect():

    print("=" * 70)
    print("TRUSTLOOP MODEL FEATURE INSPECTOR")
    print("=" * 70)

    model = load_model()

    features = get_model_features(model)

    print()
    print(f"Model: {MODEL_PATH}")
    print(f"Feature count: {len(features)}")
    print()

    print("MODEL FEATURES:")

    for i, feature in enumerate(features):
        print(f"{i:02d}. {feature}")

    print()

    if DATA_PATH.exists():

        df = pd.read_csv(
            DATA_PATH,
            nrows=5,
            low_memory=False
        )

        data_features = [
            c for c in df.columns
            if c != "abuse_label"
        ]

        print(
            f"Dataset feature columns: "
            f"{len(data_features)}"
        )

        missing_from_dataset = [
            f
            for f in features
            if f not in data_features
        ]

        extra_dataset_features = [
            f
            for f in data_features
            if f not in features
        ]

        print()

        print("Missing from dataset:")

        if missing_from_dataset:
            for f in missing_from_dataset:
                print(f"  - {f}")
        else:
            print("  NONE")

        print()

        print("Extra dataset columns:")

        if extra_dataset_features:
            for f in extra_dataset_features:
                print(f"  - {f}")
        else:
            print("  NONE")

    print()
    print("=" * 70)
    print("FEATURE INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    inspect()
