"""Live TrustLoop investigation pipeline.

Each function in this module is an executable agent boundary.  The
orchestrator records the input/output of every boundary and keeps the
investigation loop explicit so the API and UI can audit the decision.
"""
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List

from backend.app.services.vision_service import verify_evidence
from backend.app.services.rag_service import analyze_policy
from backend.app.services.responsibility_service import calculate_responsibility
from backend.app.services.shap_service import explain_prediction
from intelligence.decision.expected_loss_engine import calculate_expected_loss
from backend.app.services.score_fusion_service import fuse_scores

logger = logging.getLogger(__name__)
DECISIONS = ("AUTO_ACCEPT", "AUTO_RETURN", "HUMAN_ESCALATION")


def _value(payload: Dict[str, Any]) -> float:
    return float(payload.get("refund_amount_requested_usd") or payload.get("order_value") or payload.get("product_value_usd") or 100)


def run_intake_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    reason = str(payload.get("return_reason") or payload.get("claim_type") or "standard return")
    claim_type = "Damaged item" if any(x in reason.lower() for x in ("damage", "broken", "crush")) else reason.replace("_", " ").title()
    mode = str(payload.get("platform_mode") or ("q_commerce" if payload.get("is_q_commerce") else "e_commerce"))
    return {"agent":"Intake Agent", "status":"completed", "claim_type":claim_type,
            "platform_mode":mode, "order_value":_value(payload),
            "sla":"2 Hours (Q-Commerce)" if mode == "q_commerce" else "24 Hours (Standard)",
            "initial_questions":["Is physical packaging evidence available?", "Does carrier telemetry corroborate the claim?"],
            "customer_id":payload.get("customer_id"), "seller_id":payload.get("seller_id"), "courier_id":payload.get("courier_id") or payload.get("carrier_id")}


def run_evidence_agent(payload: Dict[str, Any], pass_number: int = 1) -> Dict[str, Any]:
    image_path = payload.get("secondary_image_path") if pass_number > 1 and payload.get("secondary_image_path") else payload.get("image_path")
    raw = verify_evidence(image_path, payload.get("return_reason"))
    available = bool(raw.get("available"))
    damage = raw.get("damage_detected") if available else None
    findings = []
    if available:
        findings = [f"Image quality: {raw.get('image_quality') or 'UNKNOWN'}",
                    f"Product condition: {raw.get('product_condition') or 'UNKNOWN'}",
                    f"Packaging condition: {raw.get('packaging_condition') or 'UNKNOWN'}",
                    f"Vision finding: {raw.get('explanation') or 'No explanation returned'}"]
        score = round(float(raw.get("confidence") or 0) * 100 if float(raw.get("confidence") or 0) <= 1 else float(raw.get("confidence") or 0), 1)
        consistency = "CONSISTENT" if raw.get("evidence_consistent") is not False else "CONTRADICTORY"
        contradictions = [] if consistency == "CONSISTENT" else ["Vision result contradicts the submitted claim."]
    else:
        findings = ["VISION UNAVAILABLE", raw.get("explanation") or "No image evidence supplied."]
        # Absence of an image is neutral.  Other supplied evidence may still
        # support the claim, but it is never presented as a vision finding.
        score = 70.0 if payload.get("package_damage_reported") or payload.get("weight_discrepancy_grams") else 50.0
        consistency, contradictions = "UNKNOWN", []
    if payload.get("package_damage_reported") and not available:
        findings.append("Customer-reported damage retained as a neutral metadata signal; no visual conclusion made.")
    if pass_number > 1 and not image_path:
        findings.append("Investigation-requested re-check completed; no new image was supplied.")
    return {"agent":"Evidence Agent", "status":"completed", "pass_number":pass_number,
            "vision_available":available, "vision_status":"ANALYZED" if available else "VISION UNAVAILABLE",
            "damage_detected":damage, "product_condition":raw.get("product_condition") if available else "UNINSPECTED",
            "packaging_condition":raw.get("packaging_condition") if available else "UNKNOWN",
            "item_match":raw.get("item_match", "UNVERIFIED") if available else "UNVERIFIED",
            "evidence_consistency":consistency, "confidence":score, "findings":findings,
            "contradictions":contradictions, "evidence_score":score,
            "source":"Gemini Vision" if available else "No vision source", "recheck_input": "secondary_image_path" if pass_number > 1 and image_path else "investigation request only"}


def run_policy_agent(payload: Dict[str, Any], evidence: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
    policy_input = dict(payload)
    policy_input["evidence_summary"] = evidence.get("findings", [])
    policy_input["fraud_risk"] = risk.get("fraud_probability")
    raw = analyze_policy(policy_input)
    retrieved = raw.get("retrieved_policy", [])
    citations = []
    for item in retrieved:
        if isinstance(item, dict):
            citations.append({"section":item.get("section") or item.get("title") or "Policy clause", "text":item.get("text") or item.get("content") or str(item), "score":item.get("score")})
        else:
            citations.append({"section":"Retrieved policy clause", "text":str(item)})
    compliant = True if raw.get("policy_status") == "POLICY_COMPLIANT" else False if raw.get("policy_status") else None
    score = 94.0 if compliant is True else 25.0 if compliant is False else 50.0
    return {"agent":"Policy Agent", "status":"completed", "policy_match":citations[0].get("section") if citations else "No clause retrieved",
            "citations":citations, "compliant":compliant, "policy_score":score,
            "reasoning": "; ".join(x.get("reason", "") for x in raw.get("flags", [])) or ("Retrieved applicable return policy clauses after reviewing evidence and risk." if citations else "Policy retrieval unavailable."),
            "source":"FAISS + BM25 + RRF", "retrieval_available":raw.get("available", False)}


def run_risk_agent(payload: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from backend.app.main import load_model, build_model_dataframe, MODEL_FEATURES, LABELS
        model = load_model()
        # Demo/API payloads use business vocabulary while the production
        # artifact expects training categories.  These are explicit neutral
        # mappings, not model results, and preserve caller-provided values.
        model_payload = dict(payload)
        model_payload["vision_damage_detected"] = int(evidence.get("damage_detected") is True)
        model_payload["vision_evidence_score"] = float(evidence.get("evidence_score", 50))
        model_payload["country"] = model_payload.get("country") if model_payload.get("country") in {"AU","US","DE","GB","BR","CA","IN","FR"} else "IN"
        model_payload["customer_segment"] = model_payload.get("customer_segment") if model_payload.get("customer_segment") in {"Silver","New","Platinum","Bronze","Gold"} else "Gold"
        model_payload["device_type"] = model_payload.get("device_type") if model_payload.get("device_type") in {"iPad","iPhone","MacBook","Android","Windows PC"} else "Android"
        model_payload["payment_method"] = model_payload.get("payment_method") if model_payload.get("payment_method") in {"Buy Now Pay Later","Debit Card","PayPal","Crypto","Gift Card","Credit Card"} else "Credit Card"
        model_payload["platform"] = model_payload.get("platform") if model_payload.get("platform") in {"Web Browser","Tablet App","Mobile App"} else ("Mobile App" if payload.get("platform_mode") == "q_commerce" else "Web Browser")
        model_payload["product_category"] = model_payload.get("product_category") if model_payload.get("product_category") in {"Toys","Jewelry","Home & Kitchen","Clothing","Furniture","Beauty","Shoes","Grocery","Electronics","Sports","Books","Tools"} else "Electronics"
        reason = str(payload.get("return_reason", "")).lower()
        model_payload["return_reason"] = model_payload.get("return_reason") if model_payload.get("return_reason") in {"Arrived Late","Not As Described","Quality Issue","Changed Mind","Item Not Received","Too Large","Defective/Broken","Wrong Item Sent","Found Better Price","Accidental Order","Too Small","Gift Duplicate"} else ("Defective/Broken" if any(x in reason for x in ("damage", "broken", "defect")) else "Wrong Item Sent" if "wrong" in reason else "Changed Mind")
        model_payload["shipping_carrier"] = model_payload.get("shipping_carrier") if model_payload.get("shipping_carrier") in {"FedEx","UPS","OnTrac","DHL","USPS"} else "FedEx"
        model_payload.setdefault("multiple_accounts_flag", int(bool(payload.get("has_multiple_accounts", False))))
        features = getattr(model, "feature_name_", getattr(model, "feature_names_in_", MODEL_FEATURES))
        df = build_model_dataframe(model_payload, feature_names=features)
        pred = int(model.predict(df)[0]); probs = model.predict_proba(df)[0]
        fraud = round(float(sum(probs[1:])) * 100, 1)
        shap = explain_prediction(model, df, predicted_class_idx=pred, class_labels=LABELS, top_k=5)
        top = shap.get("top_positive_drivers", []) + shap.get("top_negative_drivers", [])
        level = "CRITICAL" if fraud >= 75 else "HIGH" if fraud >= 50 else "MEDIUM" if fraud >= 25 else "LOW"
        disagreement = []
        if evidence.get("damage_detected") is True and fraud >= 60:
            disagreement.append("Vision supports physical damage while LightGBM flags elevated abuse risk.")
        return {"agent":"Risk Scoring Agent", "status":"completed", "fraud_probability":fraud, "risk_level":level,
                "top_features":top[:5], "shap_explanation":shap, "model_source":"LightGBM", "predicted_class":LABELS.get(pred, str(pred)), "disagreements":disagreement}
    except Exception as exc:
        logger.warning("Risk model unavailable: %s", exc)
        return {"agent":"Risk Scoring Agent", "status":"completed", "fraud_probability":None, "risk_level":"UNKNOWN",
                "top_features":[], "shap_explanation":{"available":False,"explanation_summary":str(exc)}, "model_source":"LightGBM unavailable", "error":str(exc)}


def run_responsibility_agent(payload: Dict[str, Any], evidence: Dict[str, Any], risk: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    vision = {"is_transit_damage": evidence.get("damage_detected") is True and evidence.get("packaging_condition") not in ("INTACT", "UNKNOWN")}
    ml = {"Fraudulent Return": (risk.get("fraud_probability") or 0) / 100}
    raw = calculate_responsibility(payload, ml_probabilities=ml, vision_result=vision, rag_result=policy)
    dist = raw["responsibility"]
    return {"agent":"Responsibility Agent", "status":"completed", **dist, "dominant_party":raw["dominant_party"],
            "evidence":sum(raw.get("drivers", {}).values(), []), "source":"Responsibility Agent", "signals":raw.get("drivers", {})}


def run_investigation_agent(payload: Dict[str, Any], evidence: Dict[str, Any], risk: Dict[str, Any], resp: Dict[str, Any], policy: Dict[str, Any], pass_number: int = 1) -> Dict[str, Any]:
    missing = [] if evidence.get("vision_available") else ["Packaging / doorstep photo"]
    if not payload.get("weight_discrepancy_grams") and resp.get("courier", 0) > 35: missing.append("Carrier hub scale telemetry")
    conflicts = list(evidence.get("contradictions", []))
    if risk.get("fraud_probability") is not None and evidence.get("damage_detected") is True and risk["fraud_probability"] >= 60: conflicts.append("Strong physical damage conflicts with elevated abuse risk.")
    uncertainty = min(95.0, len(missing) * 25 + len(conflicts) * 25 + (10 if _value(payload) >= 750 else 0))
    further = pass_number == 1 and uncertainty >= 50
    return {"agent":"Investigation Agent", "status":"completed", "pass_number":pass_number,
            "current_belief":f"{resp.get('dominant_party','unknown')} is the leading responsibility hypothesis.",
            "uncertainty_score":uncertainty, "missing_evidence":missing, "conflicting_evidence":conflicts,
            "next_best_evidence":missing[0] if missing else "Proceed to score fusion", "investigate_further":further,
            "reasoning":("Further evidence requested because uncertainty is above threshold." if further else "Evidence is sufficient for final fusion."), "disagreements":conflicts, "source":"Investigation Agent"}


def _losses(payload, risk, evidence):
    fraud = (risk.get("fraud_probability") if risk.get("fraud_probability") is not None else 50) / 100
    truth = evidence.get("evidence_score", 50) / 100
    result = calculate_expected_loss(fraud, truth, _value(payload), review_cost=float(payload.get("human_review_cost", 50)))
    names = {"AUTO_APPROVE":"AUTO_ACCEPT", "AUTO_REJECT":"AUTO_RETURN", "HUMAN_ESCALATION":"HUMAN_ESCALATION"}
    options = [{"decision":names.get(o.decision, o.decision),"expected_loss":round(o.expected_loss,2),"rationale":o.rationale} for o in result.options]
    return {"formula":"probability_of_bad_outcome × financial_exposure + review_cost where applicable", "inputs":{"fraud_probability":fraud,"claim_truth":truth,"refund_amount":_value(payload),"review_cost":float(payload.get("human_review_cost",50))}, "options":options, "recommended_option":names.get(result.recommended_decision, result.recommended_decision),
            **{names.get(o.decision, o.decision):round(o.expected_loss,2) for o in result.options}, "item_value":_value(payload)}


def _fusion(risk, evidence, policy, resp, investigation):
    return fuse_scores(risk=risk, evidence=evidence, policy=policy, responsibility=resp, investigation=investigation)


def run_decision_agent(payload, fusion, losses, resp, investigation):
    risk = fusion.get("fraud_risk", fusion.get("risk_score", 50))
    claim = fusion.get("claim_verification_score", fusion.get("claim_validity_score", fusion.get("claim_validity", 50)))
    unc = fusion.get("uncertainty", fusion.get("uncertainty_score", 100))
    if unc >= 50 or (fusion["item_value"] if "item_value" in fusion else _value(payload)) >= 1000 and unc >= 25:
        decision = "HUMAN_ESCALATION"; actions = ["send_to_investigator", "request_missing_evidence"]
    elif risk >= 70 and claim < 55:
        decision = "AUTO_RETURN"; actions = ["reject_refund", "flag_customer"]
    elif claim >= 65 and risk < 70 and fusion["policy_score"] >= 60:
        decision = "AUTO_ACCEPT"; actions = ["refund_customer"] + (["open_courier_liability_claim"] if resp.get("dominant_party") == "courier" else [])
    else:
        decision = "HUMAN_ESCALATION"; actions = ["send_to_investigator", "request_missing_evidence"]
    reasons = [f"Claim validity {claim:.1f}%", f"Fraud risk {risk:.1f}%", f"Investigation uncertainty {unc:.1f}%", f"{resp.get('dominant_party','unknown').title()} responsibility leads at {resp.get(resp.get('dominant_party','unknown'),0)}%"]
    rule = {"verification_threshold":65,"fraud_risk_reject_threshold":70,"uncertainty_escalation_threshold":50,"expected_loss_review_cost":payload.get("human_review_cost",50)}
    return {"agent":"Decision Agent","status":"completed","decision":decision,"confidence":fusion["final_decision_confidence"],"reason":"; ".join(reasons),"decision_reason":"; ".join(reasons),"reasons":reasons,"actions":actions,
            "decision_rule":rule,"verification_score":fusion["final_score"],"fraud_risk":risk,"responsibility":resp,"policy_compliance":fusion["policy_score"],"escalation_reasons":investigation.get("missing_evidence", []) + investigation.get("conflicting_evidence", []),"thresholds":rule,"options":losses["options"]}


def build_agent_graph(payload, evidence, policy, risk, resp, decision):
    from backend.app.services.investigation_service import build_evidence_graph
    graph = build_evidence_graph({"case_id":payload.get("case_id","UNKNOWN"), **payload,
                                  "vision_available": evidence.get("vision_available"),
                                  "vision_status": evidence.get("vision_status"),
                                  "evidence_score": evidence.get("evidence_score"),
                                  "policy_score": policy.get("policy_score"),
                                  "policy_match": policy.get("policy_match")}, {"responsibility":resp})
    nodes, edges = graph["nodes"], graph["edges"]
    extras = [("node_vision_agent","Vision Agent","agent"),("node_risk_agent","Risk Scoring Agent","agent"),("node_policy_agent","Policy Agent","agent"),("node_resp_agent","Responsibility Agent","agent"),("node_investigation_agent","Investigation Agent","agent"),("node_vision_finding","Vision Finding","finding"),("node_risk_finding","Risk Finding","finding"),("node_resp_finding","Responsibility Finding","finding"),("node_decision","Decision","decision")]
    for ident,label,typ in extras: nodes.append({"id":ident,"label":f"{label}: {decision.get('decision') if typ=='decision' else 'Recorded'}","type":typ})
    edges += [{"source":"node_evidence_photo","target":"node_vision_agent","label":"provided to","weight":evidence.get("evidence_score")}, {"source":"node_vision_agent","target":"node_risk_agent","label":"vision finding passed","weight":evidence.get("confidence")}, {"source":"node_risk_agent","target":"node_policy_agent","label":"risk context passed","weight":risk.get("confidence")}, {"source":"node_policy_agent","target":"node_resp_agent","label":"policy context passed","weight":policy.get("confidence")}, {"source":"node_resp_agent","target":"node_investigation_agent","label":"attribution passed","weight":resp.get("confidence")}, {"source":"node_vision_agent","target":"node_vision_finding","label":"produced","weight":evidence.get("evidence_score")}, {"source":"node_risk_agent","target":"node_risk_finding","label":"produced","weight":risk.get("confidence")}, {"source":"node_risk_finding","target":"node_resp_finding","label":"challenged by","weight":risk.get("fraud_probability")}, {"source":"node_vision_finding","target":"node_resp_finding","label":"supports","weight":evidence.get("evidence_score")}, {"source":"node_resp_agent","target":"node_resp_finding","label":"produced","weight":resp.get("confidence")}, {"source":"node_investigation_agent","target":"node_decision","label":"synthesis passed","weight":decision.get("confidence")}, {"source":"node_resp_finding","target":"node_decision","label":"informs","weight":resp.get(resp.get("dominant_party","unknown"),0)}]
    return graph


def execute_autonomous_investigation(payload: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time(); trace = []
    context: Dict[str, Any] = {"case": payload, "outputs": {}, "communications": []}
    def communicate(source, target, source_output, target_output, iteration):
        if source == "Evidence Agent":
            message = f"Vision status {source_output.get('vision_status')}; claim verification score {source_output.get('evidence_score')}%."
        elif source == "Risk Scoring Agent":
            message = f"LightGBM fraud risk is {source_output.get('fraud_probability')}%; disagreements: {source_output.get('disagreements', [])}."
        elif source == "Policy Agent":
            message = f"Retrieved {len(source_output.get('citations', []))} policy clauses; support score {source_output.get('policy_score')}%."
        elif source == "Responsibility Agent":
            party = source_output.get("dominant_party", "unknown")
            message = f"Responsibility attribution leads with {party} at {source_output.get(party, 0)}%."
        else:
            message = str(source_output.get("reasoning") or source_output.get("reason") or source_output)
        disagreements = source_output.get("disagreements", []) or target_output.get("conflicting_evidence", [])
        context["communications"].append({"timestamp": datetime.now(timezone.utc).isoformat(), "from_agent": source, "to_agent": target,
            "message_type": "disagreement" if disagreements else ("request" if target.startswith("Evidence") else "finding"),
            "message": message, "source_outputs": [{"agent": source, "output": source_output}], "evidence": source_output.get("findings") or source_output.get("evidence") or [],
            "confidence": source_output.get("confidence"), "iteration": iteration})
    def record(name, fn, *args, source_agents=None):
        t0 = datetime.now(timezone.utc).isoformat(); source_agents = source_agents or []
        out = fn(*args)
        for source in source_agents:
            communicate(source, name, context["outputs"].get(source, {}), out, out.get("pass_number", 1))
        context["outputs"][name] = out
        score = next((out[k] for k in ("evidence_score", "policy_score", "fraud_probability", "uncertainty_score", "confidence") if out.get(k) is not None), None)
        observations = out.get("findings") or out.get("evidence") or out.get("reasons") or out.get("reasoning") or []
        trace.append({"name":name,"agent_name":name,"status":out.get("status","completed"),"started_at":t0,"completed_at":datetime.now(timezone.utc).isoformat(),"timestamp":t0,"iteration":out.get("pass_number",1),
                      "input_received":{"case_id":payload.get("case_id"),"source_outputs":{s:context["outputs"].get(s) for s in source_agents}},"source_agents":source_agents,
                      "observations":observations if isinstance(observations,list) else [observations],"reasoning_summary":out.get("reasoning") or out.get("reason") or out.get("summary",""),
                      "evidence_produced":out.get("evidence") or out.get("findings") or [],"score_produced":score,"confidence":out.get("confidence", score),
                      "output_sent_to_next":out,"disagreements":out.get("disagreements", []),"output":out})
        return out
    intake = record("Intake Agent", run_intake_agent, payload)
    evidence = record("Evidence Agent", run_evidence_agent, payload, 1)
    risk = record("Risk Scoring Agent", run_risk_agent, payload, evidence, source_agents=["Evidence Agent"])
    policy = record("Policy Agent", run_policy_agent, payload, evidence, risk, source_agents=["Evidence Agent", "Risk Scoring Agent"])
    resp = record("Responsibility Agent", run_responsibility_agent, payload, evidence, risk, policy, source_agents=["Evidence Agent", "Risk Scoring Agent", "Policy Agent"])
    investigation = record("Investigation Agent", run_investigation_agent, payload, evidence, risk, resp, policy, 1, source_agents=["Evidence Agent", "Risk Scoring Agent", "Policy Agent", "Responsibility Agent"])
    iterations = 1
    if investigation["investigate_further"]:
        iterations = 2
        second_pass_payload = dict(payload, requested_evidence=investigation["next_best_evidence"])
        evidence = record("Evidence Agent — SECOND PASS", run_evidence_agent, second_pass_payload, 2, source_agents=["Investigation Agent"])
        resp = record("Responsibility Agent — SECOND PASS", run_responsibility_agent, payload, evidence, risk, policy, source_agents=["Evidence Agent — SECOND PASS", "Risk Scoring Agent", "Policy Agent"])
        investigation = record("Investigation Agent — SECOND PASS", run_investigation_agent, payload, evidence, risk, resp, policy, 2, source_agents=["Evidence Agent — SECOND PASS", "Risk Scoring Agent", "Policy Agent", "Responsibility Agent — SECOND PASS"])
    fusion = _fusion(risk, evidence, policy, resp, investigation)
    losses = _losses(payload, risk, evidence); fusion["expected_loss"] = losses
    context["outputs"]["Score Fusion"] = fusion
    decision = record("Decision Agent", run_decision_agent, payload, fusion, losses, resp, investigation, source_agents=["Investigation Agent", "Responsibility Agent", "Score Fusion"])
    public_resp = {k: resp[k] for k in ("customer", "seller", "courier", "unknown")}
    return {"case_id":payload.get("case_id","INVESTIGATION-TEMP"),"investigation":{"status":"completed","iterations":iterations,"duration_ms":int((time.time()-started)*1000)},"agents":trace,"shared_context":{"case":payload,"agent_outputs":context["outputs"],"communications":context["communications"]},"communications":context["communications"],"score_fusion":fusion,"responsibility":public_resp,"responsibility_analysis":resp,"decision":{k:v for k,v in decision.items() if k != "agent"},"expected_loss":losses,"evidence_graph":build_agent_graph(payload,evidence,policy,risk,resp,decision),"shap_explanation":risk.get("shap_explanation"),"vision_analysis":evidence,"policy_analysis":policy,"risk_analysis":risk,"timeline":[],"agent_timeline":[{"label":a["name"],"status":a["status"],"timestamp":a["started_at"],"iteration":a.get("iteration",1)} for a in trace],"trace":trace}
