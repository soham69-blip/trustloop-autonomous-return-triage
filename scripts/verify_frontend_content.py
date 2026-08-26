"""Verify frontend content serves correctly from backend."""
import urllib.request

def check(url, checks_dict):
    resp = urllib.request.urlopen(url)
    content = resp.read().decode()
    for name, pattern in checks_dict.items():
        status = "PASS" if pattern in content else "FAIL"
        print("  [%s] %s" % (status, name))

print("FRONTEND HTML VERIFICATION:")
check("http://127.0.0.1:8080/", {
    "TrustLoop branding": "TrustLoop",
    "v1.3.0 version": "v1.3.0",
    "Case Room tab": "Case Room",
    "Evidence Graph tab": "Evidence Graph",
    "Challenge Decision tab": "Challenge Decision",
    "Learning Loop tab": "Learning Loop",
    "Fraud Network tab": "Fraud Network",
    "Investigate button": "btn-investigate-hero",
    "E-Commerce mode": "mode-ecom",
    "Q-Commerce mode": "mode-qcom",
    "Case Dossier": "Case Dossier",
    "Customer Entity": "Customer Entity",
    "Merchant Entity": "Merchant Entity",
    "Logistics Carrier": "Logistics Carrier",
    "Responsibility quad": "quad-val-cust",
    "Three.js container": "three-graph-container",
    "Challenge signals": "challenge-signals-container",
    "Feedback form": "feedback-submission-form",
    "Policy modal": "policy-modal",
    "Demo case select": "demo-case-select",
})

print("\nCSS DESIGN SYSTEM VERIFICATION:")
check("http://127.0.0.1:8080/styles.css", {
    "Design tokens": "--bg-app",
    "Party colors": "--party-courier",
    "Font system": "--font-mono",
    "Responsive rules": "@media",
    "Case room grid": "case-room-grid",
})

print("\nJAVASCRIPT CONTROLLER VERIFICATION:")
check("http://127.0.0.1:8080/app.js", {
    "THREE.js import": "THREE",
    "fetchAPI function": "fetchAPI",
    "loadCase function": "loadCase",
    "renderChallengeResults": "renderChallengeResults",
    "loadFraudNetwork": "loadFraudNetwork",
    "setupFeedbackForm": "setupFeedbackForm",
    "initThree3DGraph": "initThree3DGraph",
    "window.location.origin": "window.location.origin",
})

print("\nALL FRONTEND CONTENT CHECKS COMPLETE")
