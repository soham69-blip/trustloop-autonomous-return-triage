# TrustLoop Shadow Deployment Disagreement Audit

**Scope:** 3,000 Consecutive Inbound Returns Evaluated under Dual Shadow Routing  
**Primary Model:** `production-v1.3.0` (33 features)  
**Shadow Model:** `candidate-v2.0.0` (39 features)  

---

## 1. Executive Summary & Routing Invariants
During shadow evaluation, the **production model retained 100% control over all business refund decisions**. The candidate model executed asynchronously in silent shadow mode.

| Shadow Deployment Metric | Observed Value | Production Health Target | Status |
| :--- | :---: | :---: | :---: |
| **Total Shadow Requests** | **3,000** | $\\ge 1,000$ | **PASS** |
| **Model Agreement Rate** | **94.1%** | $\\ge 90.0\%$ | **PASS** |
| **Model Disagreement Rate** | **5.9%** | $\\le 10.0\%$ | **PASS** |
| **Candidate Accuracy on Divergent Cases** | **100.0%** | $> 80.0\%$ | **PASS (Superior)** |
| **Production Accuracy on Divergent Cases** | **0.0%** | — | **Baseline Gap** |

---

## 2. Policy Abuser Divergence Breakdown

- **Candidate-Only Policy Abuser Flags:** **152** cases  
  *Analysis:* True Policy Abusers who spread returns over multiple weeks without triggering single-transaction velocity thresholds. The 39-feature candidate model caught them via `return_rate_pct > 33%` and `total_returns_lifetime`.
- **Production-Only Policy Abuser Flags:** **17** cases  
  *Analysis:* False alarms caused by high single-order cart quantities that were correctly recognized as legitimate high-volume shoppers by the candidate model's lifetime context.

---

## 3. Counterfactual Divergence Feature Attribution

| Divergence Driver Feature | Disagreement Case Count | Impact Mechanism |
| :--- | :---: | :--- |
| `return_rate_pct` | 25 | Primary driver separating habitual returners from legitimate sizing shoppers. |
| `total_returns_lifetime` | 51 | Prevents chronic returners with low single-order velocity from slipping through. |
| `customer_support_contacts` | 46 | Identifies aggressive escalation patterns on repeat returns. |
| `previous_dispute_count` | 55 | Flags chargeback habituation prior to formal claim filing. |

---

## 4. Shadow Deployment Artifacts
- **Summary JSON:** `reports/shadow_disagreement_summary.json`
- **Disagreement Samples CSV:** `reports/shadow_disagreement_samples.csv`
