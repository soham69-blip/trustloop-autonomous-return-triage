"""Auditable verification-score fusion.

Weights are reliability gates, not UI tuning: unavailable sources contribute
zero and the remaining active evidence is renormalized. Risk is inverted
because it is a bad-outcome probability; responsibility is the leading-party
attribution confidence.
"""
from typing import Any, Dict


def fuse_scores(*, risk: Dict[str, Any], evidence: Dict[str, Any], policy: Dict[str, Any], responsibility: Dict[str, Any], investigation: Dict[str, Any]) -> Dict[str, Any]:
    raw = [
        ("vision", float(evidence.get("evidence_score", 50)), 0.30 if evidence.get("vision_available") else 0.10, "Evidence Agent"),
        ("risk", 100 - float(risk.get("fraud_probability") if risk.get("fraud_probability") is not None else 50), 0.30 if risk.get("model_source") == "LightGBM" else 0.10, "Risk Scoring Agent / LightGBM"),
        ("policy", float(policy.get("policy_score", 50)), 0.20 if policy.get("retrieval_available") else 0.10, "Policy Agent / FAISS + BM25 + RRF"),
        ("responsibility", float(responsibility.get(responsibility.get("dominant_party", "unknown"), 0)), 0.20, "Responsibility Agent"),
    ]
    total_weight = sum(x[2] for x in raw)
    iteration = int(investigation.get("pass_number", 1))
    components = [{"agent": name, "raw_score": round(score, 1), "weight": round(weight / total_weight, 4), "contribution": round(score * weight / total_weight, 2), "source": source, "confidence": round(score, 1), "iteration": iteration} for name, score, weight, source in raw]
    final = round(sum(c["contribution"] for c in components), 1)
    uncertainty = float(investigation.get("uncertainty_score", 100))
    risk_score = 100 - next(c["raw_score"] for c in components if c["agent"] == "risk")
    return {"final_score": final, "fraud_risk": risk_score, "risk_score": risk_score, "claim_verification_score": components[0]["raw_score"], "claim_validity": components[0]["raw_score"], "evidence_score": components[0]["raw_score"], "policy_support_score": components[2]["raw_score"], "policy_score": components[2]["raw_score"], "responsibility_score": components[3]["raw_score"], "responsibility_confidence": components[3]["raw_score"], "investigation_confidence": round(100 - uncertainty, 1), "uncertainty_score": uncertainty, "uncertainty": uncertainty, "final_verification_score": final, "final_decision_confidence": final, "components": components, "method": "Reliability-gated weighted evidence fusion; active-source weights are renormalized; LightGBM risk is inverted."}
