import json
import pickle
from pathlib import Path

import pandas as pd

from backend.app.ml_feature_builder import build_model_dataframe
from backend.app.decision.decision_engine import make_decision


ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_MODEL_PATH = ROOT / "models" / "lightgbm_model.pkl"
CANDIDATE_MODEL_PATH = ROOT / "models" / "lightgbm_candidate.pkl"
TEST_DIR = ROOT / "tests"
REPORT_DIR = ROOT / "reports"

LABELS = {
    0: "Legitimate",
    1: "Policy Abuser",
    2: "Fraudulent Return",
    3: "Wardrobing",
}


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def model_features(model):
    return list(
        getattr(
            model,
            "feature_name_",
            getattr(model, "feature_names_in_", [])
        )
    )


def run_model(model, case_data):
    features = model_features(model)

    X = build_model_dataframe(
        case_data,
        feature_names=features
    )

    probabilities_raw = model.predict_proba(X)[0]
    prediction_raw = model.predict(X)[0]

    classes = getattr(
        model,
        "classes_",
        range(len(probabilities_raw))
    )

    probabilities = {}

    for class_id, probability in zip(classes, probabilities_raw):
        numeric_class = class_id
        label = LABELS.get(
            numeric_class,
            str(class_id)
        )
        probabilities[label] = float(probability)

    prediction_id = int(prediction_raw)
    ml_label = LABELS[prediction_id]
    ml_confidence = max(probabilities.values())

    decision = make_decision(
        case_data,
        probabilities,
        ml_label
    )

    return {
        "prediction_id": prediction_id,
        "ml_label": ml_label,
        "ml_confidence": ml_confidence,
        "probabilities": probabilities,
        "decision": decision,
    }


def compare_case(name, case_data, production_model, candidate_model):
    production = run_model(
        production_model,
        case_data
    )

    candidate = run_model(
        candidate_model,
        case_data
    )

    prod_decision = production["decision"]["decision"]
    cand_decision = candidate["decision"]["decision"]

    model_prediction_changed = (
        production["prediction_id"]
        != candidate["prediction_id"]
    )

    final_decision_changed = (
        prod_decision != cand_decision
    )

    return {
        "case": name,
        "production_label": production["ml_label"],
        "candidate_label": candidate["ml_label"],
        "production_confidence": round(
            production["ml_confidence"],
            6
        ),
        "candidate_confidence": round(
            candidate["ml_confidence"],
            6
        ),
        "production_decision": prod_decision,
        "candidate_decision": cand_decision,
        "production_risk": production["decision"]["risk_score"],
        "candidate_risk": candidate["decision"]["risk_score"],
        "model_prediction_changed": model_prediction_changed,
        "final_decision_changed": final_decision_changed,
        "production_probabilities": production["probabilities"],
        "candidate_probabilities": candidate["probabilities"],
    }


def main():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    production_model = load_model(
        PRODUCTION_MODEL_PATH
    )

    candidate_model = load_model(
        CANDIDATE_MODEL_PATH
    )

    print("=" * 70)
    print("TRUSTLOOP PHASE 3 — BACKEND SHADOW VALIDATION")
    print("=" * 70)

    print()
    print("Production model:")
    print(PRODUCTION_MODEL_PATH)
    print(
        "Features:",
        len(model_features(production_model))
    )

    print()
    print("Candidate model:")
    print(CANDIDATE_MODEL_PATH)
    print(
        "Features:",
        len(model_features(candidate_model))
    )

    payload_files = sorted(
        TEST_DIR.glob("payload_*.json")
    )

    if not payload_files:
        raise RuntimeError(
            "No tests/payload_*.json files found."
        )

    results = []

    for payload_path in payload_files:
        with open(
            payload_path,
            "r",
            encoding="utf-8"
        ) as f:
            case_data = json.load(f)

        try:
            result = compare_case(
                payload_path.name,
                case_data,
                production_model,
                candidate_model
            )

            results.append(result)

            print()
            print("-" * 70)
            print(payload_path.name)
            print("-" * 70)

            print(
                "Production:",
                result["production_label"],
                "|",
                result["production_decision"],
                "| risk=",
                result["production_risk"]
            )

            print(
                "Candidate :",
                result["candidate_label"],
                "|",
                result["candidate_decision"],
                "| risk=",
                result["candidate_risk"]
            )

            print(
                "Prediction changed:",
                result["model_prediction_changed"]
            )

            print(
                "Final decision changed:",
                result["final_decision_changed"]
            )

        except Exception as exc:
            print()
            print("ERROR:", payload_path.name)
            print(type(exc).__name__, str(exc))

            results.append({
                "case": payload_path.name,
                "error": str(exc),
            })

    successful = [
        r for r in results
        if "error" not in r
    ]

    prediction_changes = sum(
        r["model_prediction_changed"]
        for r in successful
    )

    decision_changes = sum(
        r["final_decision_changed"]
        for r in successful
    )

    errors = len(results) - len(successful)

    summary = {
        "total_cases": len(results),
        "successful_cases": len(successful),
        "errors": errors,
        "model_prediction_changes": prediction_changes,
        "final_decision_changes": decision_changes,
        "production_model": str(
            PRODUCTION_MODEL_PATH
        ),
        "candidate_model": str(
            CANDIDATE_MODEL_PATH
        ),
        "production_model_modified": False,
    }

    output = {
        "summary": summary,
        "cases": results,
    }

    output_path = (
        REPORT_DIR /
        "phase3_shadow_validation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("PHASE 3 SHADOW VALIDATION SUMMARY")
    print("=" * 70)
    print(
        "Total cases:",
        summary["total_cases"]
    )
    print(
        "Successful:",
        summary["successful_cases"]
    )
    print(
        "Errors:",
        summary["errors"]
    )
    print(
        "ML prediction changes:",
        summary["model_prediction_changes"]
    )
    print(
        "Final decision changes:",
        summary["final_decision_changes"]
    )

    print()
    print("Report:")
    print(output_path)

    print()
    print(
        "IMPORTANT: Production model was NOT modified."
    )


if __name__ == "__main__":
    main()
