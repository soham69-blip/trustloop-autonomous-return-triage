import json
import pickle
from pathlib import Path

from backend.app.ml_feature_builder import build_model_dataframe

ROOT = Path.cwd()

payload_path = ROOT / "tests" / "payload_policy_abuser.json"
prod_path = ROOT / "models" / "lightgbm_model.pkl"
cand_path = ROOT / "models" / "lightgbm_candidate.pkl"

LABELS = {
    0: "Legitimate",
    1: "Policy Abuser",
    2: "Fraudulent Return",
    3: "Wardrobing",
}

with open(payload_path, "r", encoding="utf-8") as f:
    case = json.load(f)

with open(prod_path, "rb") as f:
    prod = pickle.load(f)

with open(cand_path, "rb") as f:
    cand = pickle.load(f)

prod_features = list(prod.feature_name_)
cand_features = list(cand.feature_name_)

print("=" * 80)
print("TRUSTLOOP — POLICY ABUSER ROOT CAUSE DIAGNOSTIC")
print("=" * 80)

print("\nPAYLOAD:")
print(json.dumps(case, indent=2))

print("\n" + "=" * 80)
print("NEW CANDIDATE FEATURES")
print("=" * 80)

new_features = [
    "total_returns_lifetime",
    "total_orders_lifetime",
    "return_rate_pct",
    "customer_support_contacts",
    "previous_dispute_count",
    "refund_amount_requested_usd",
]

for feature in new_features:
    print(
        f"{feature:35s}: "
        f"{case.get(feature, '<MISSING>')}"
    )

print("\n" + "=" * 80)
print("CONSTRUCTED CANDIDATE FEATURES")
print("=" * 80)

X_candidate = build_model_dataframe(
    case,
    feature_names=cand_features
)

for feature in new_features:
    print(
        f"{feature:35s}: "
        f"{X_candidate[feature].iloc[0]}"
    )

print("\n" + "=" * 80)
print("MODEL OUTPUT")
print("=" * 80)

prod_prob = prod.predict_proba(
    build_model_dataframe(case, feature_names=prod_features)
)[0]

cand_prob = cand.predict_proba(X_candidate)[0]

print("\nProduction:")
for cls, prob in zip(prod.classes_, prod_prob):
    print(f"  {LABELS[int(cls)]:20s}: {prob:.6f}")

print("\nCandidate:")
for cls, prob in zip(cand.classes_, cand_prob):
    print(f"  {LABELS[int(cls)]:20s}: {prob:.6f}")

print("\nProduction prediction:",
      LABELS[int(prod.predict(
          build_model_dataframe(case, feature_names=prod_features)
      )[0])])

print("Candidate prediction:",
      LABELS[int(cand.predict(X_candidate)[0])])

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

missing = [
    feature for feature in new_features
    if feature not in case
]

if missing:
    print("\nMISSING NEW FEATURES:")
    for feature in missing:
        print(" -", feature)

    print(
        "\nWARNING: Candidate model is receiving missing "
        "Experiment A features."
    )

    print(
        "Those fields are currently converted into defaults."
    )

    print(
        "This can cause a policy abuser to appear legitimate."
    )
else:
    print("\nAll six Experiment A fields are present.")
    print(
        "The failure is NOT caused by missing candidate fields."
    )

print("\nDiagnostic complete.")
