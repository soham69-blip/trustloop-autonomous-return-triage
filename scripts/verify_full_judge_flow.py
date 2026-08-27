"""
Comprehensive End-to-End Judge Flow & API Verification Script (Pure ASCII).
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8080"

def test_judge_flow():
    print("=" * 60)
    print("TRUSTLOOP JUDGE WALKTHROUGH -- END-TO-END VERIFICATION")
    print("=" * 60)

    # 1. Health & Readiness Probes
    print("\n[STEP 1] Testing Health & Readiness Probes...")
    r_health = requests.get(f"{BASE_URL}/health", timeout=5)
    assert r_health.status_code == 200, f"Health check failed: {r_health.status_code}"
    print("  [PASS] /health: 200 OK ->", r_health.json())

    r_ready = requests.get(f"{BASE_URL}/ready", timeout=5)
    assert r_ready.status_code == 200, f"Ready check failed: {r_ready.status_code}"
    ready_data = r_ready.json()
    assert ready_data["checks"]["production_model"]["sha256_verified"] is True
    print("  [PASS] /ready: 200 OK -> Model Checksums Verified Matching Reference")

    # 2. Demo Case Directory
    print("\n[STEP 2] Fetching Demo Case Directory...")
    r_cases = requests.get(f"{BASE_URL}/api/v1/cases/demo", timeout=5)
    assert r_cases.status_code == 200
    cases = r_cases.json()
    assert len(cases) == 6, f"Expected 6 demo cases, found {len(cases)}"
    print(f"  [PASS] /api/v1/cases/demo: 200 OK -> {len(cases)} Demo Cases Loaded:")
    for c in cases:
        print(f"    - [{c['case_id']}] {c['title']} ({c.get('platform_mode', 'e_commerce')})")

    # 3. Load Case Room Context (CASE-001: Severe Courier Transit Damage)
    print("\n[STEP 3] Loading Case Room Context (CASE-001)...")
    r_case1 = requests.get(f"{BASE_URL}/api/v1/cases/CASE-001", timeout=5)
    assert r_case1.status_code == 200
    case1_data = r_case1.json()
    resp1 = case1_data["responsibility"]
    assert sum(resp1.values()) == 100, f"Responsibility sum != 100: {resp1}"
    assert case1_data["dominant_party"] == "courier"
    timeline1 = case1_data["timeline"]
    assert len(timeline1) == 8, f"Timeline milestones != 8: {len(timeline1)}"
    print(f"  [PASS] /api/v1/cases/CASE-001: 200 OK")
    print(f"    - Dominant Party: {case1_data['dominant_party'].upper()}")
    print(f"    - Normalized Attribution (Exact 100% Invariant): {resp1}")
    print(f"    - Timeline Milestones: {len(timeline1)} stages ({timeline1[0]['stage']} -> {timeline1[-1]['stage']})")

    # 4. Trigger Autonomous Investigation
    print("\n[STEP 4] Executing Autonomous Investigation (/api/v1/investigate)...")
    demo_payload = case1_data["case"]["payload"]
    r_inv = requests.post(f"{BASE_URL}/api/v1/investigate", json=demo_payload, timeout=5)
    assert r_inv.status_code == 200
    inv_data = r_inv.json()
    assert inv_data["dominant_party"] == "courier"
    assert inv_data["recommended_action"] == "AUTO_ACCEPT"
    assert sum(inv_data["responsibility"].values()) == 100
    print(f"  [PASS] /api/v1/investigate: 200 OK")
    print(f"    - Recommended Platform Action: {inv_data['recommended_action']}")
    print(f"    - Action Label: {inv_data['action_label']}")
    print(f"    - Confidence Score: {inv_data['confidence']}%")
    print(f"    - Evidence Drivers Count: {sum(len(v) for v in inv_data['drivers'].values())}")

    # 5. Challenge Decision (The Wow Counterfactual Recalculation)
    print("\n[STEP 5] Testing Counterfactual Challenge Decision Recalculation...")
    challenge_req = {
        "case_payload": demo_payload,
        "disabled_signals": ["courier_incident_history", "packaging_damage"],
    }
    r_chal = requests.post(f"{BASE_URL}/api/v1/challenge", json=challenge_req, timeout=5)
    assert r_chal.status_code == 200
    chal_data = r_chal.json()
    base_cour = chal_data["baseline"]["responsibility"]["courier"]
    cf_cour = chal_data["counterfactual"]["responsibility"]["courier"]
    delta_cour = chal_data["deltas"]["courier"]["delta"]
    print(f"  [PASS] /api/v1/challenge: 200 OK")
    print(f"    - Courier Responsibility Shift: {base_cour}% -> {cf_cour}% (Delta: {delta_cour}%)")
    print(f"    - Recommended Action Transition: {chal_data['baseline']['recommended_action']} -> {chal_data['counterfactual']['recommended_action']}")
    print(f"    - AI Causal Explanation: \"{chal_data['explanation']}\"")

    # 6. Fraud Network Graph
    print("\n[STEP 6] Testing Fraud Network Graph & Cluster Engine...")
    r_net = requests.get(f"{BASE_URL}/api/v1/network/graph", timeout=5)
    assert r_net.status_code == 200
    net_data = r_net.json()
    assert len(net_data["active_clusters"]) > 0
    cluster = net_data["active_clusters"][0]
    print(f"  [PASS] /api/v1/network/graph: 200 OK")
    print(f"    - Active Fraud Cluster: {cluster['cluster_name']} ({cluster['cluster_id']})")
    print(f"    - Risk Severity: {cluster['severity']} | Disputed Value: ${cluster['total_disputed_value_usd']:.2f}")
    print(f"    - Coordinated Link: {cluster['primary_link']}")
    print(f"    - Connected Accounts: {cluster['connected_customers']}")

    # 7. Learning Loop Feedback Submission & Persistence
    print("\n[STEP 7] Submitting Ground-Truth Human Auditor Correction...")
    feedback_req = {
        "case_id": "CASE-001",
        "reviewer_id": "JUDGE-AUDITOR-01",
        "human_verified_label": "Legitimate",
        "human_decision": "Approved",
        "notes": "Verified conveyor compression telemetry matches doorstep photos. No customer fault.",
        "traffic_type": "production",
        "raw_payload": demo_payload,
    }
    r_fb = requests.post(f"{BASE_URL}/api/v1/feedback", json=feedback_req, timeout=5)
    assert r_fb.status_code == 200
    print(f"  [PASS] /api/v1/feedback: 200 OK -> Review persisted to learning store")

    r_fb_hist = requests.get(f"{BASE_URL}/api/v1/feedback/history?limit=5", timeout=5)
    assert r_fb_hist.status_code == 200
    fb_list = r_fb_hist.json()
    assert any(f["reviewer_id"] == "JUDGE-AUDITOR-01" for f in fb_list)
    print(f"  [PASS] /api/v1/feedback/history: 200 OK -> Verified {len(fb_list)} records retrieved")

    # 8. Web Assets Serving
    print("\n[STEP 8] Verifying Static Web Assets Serving...")
    r_html = requests.get(f"{BASE_URL}/", headers={"Accept": "text/html"}, timeout=5)
    assert r_html.status_code == 200 and "TrustLoop" in r_html.text
    r_css = requests.get(f"{BASE_URL}/styles.css", timeout=5)
    assert r_css.status_code == 200 and "--bg-app" in r_css.text
    r_js = requests.get(f"{BASE_URL}/app.js", timeout=5)
    assert r_js.status_code == 200 and "setupNavigation" in r_js.text
    print("  [PASS] GET /: 200 OK (HTML Shell rendered)")
    print("  [PASS] GET /styles.css: 200 OK (CSS Tokens rendered)")
    print("  [PASS] GET /app.js: 200 OK (JS Controller rendered)")

    print("\n" + "=" * 60)
    print("ALL 8 END-TO-END JUDGE FLOW VERIFICATION STEPS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_judge_flow()
