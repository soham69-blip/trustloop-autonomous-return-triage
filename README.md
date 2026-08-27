# TrustLoop — Autonomous Return & Refund Triage System

> **TrustLoop** is an AI-powered e-commerce return and refund investigation platform that analyzes customer behavior, transaction history, return policies, and product evidence to determine whether a case should be **Auto-Approved, Auto-Rejected, or Escalated to a Human Investigator**.

---

## 🚨 Problem

E-commerce platforms process thousands of return and refund requests every day.

The problem is that not every return is genuine.

Common forms of return and refund abuse include:

* Item switching
* Fake damage claims
* Serial-number mismatches
* Repeated return abuse
* Missing accessories
* Wardrobing
* Policy abuse
* False refund claims
* Suspicious customer behavior
* Manipulated or inconsistent evidence

Traditional rule-based systems struggle because they usually examine only one dimension of a return request.

For example:

> "Customer says the product is damaged → approve refund."

That approach can be exploited.

TrustLoop instead investigates the **entire case** before making a decision.

---

# 💡 Solution

TrustLoop combines multiple AI and data-analysis systems into one investigation pipeline.

```text
                 RETURN / REFUND REQUEST
                          │
                          ▼
                  CASE INTAKE AGENT
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
    CUSTOMER/FRAUD    POLICY/RAG       VISION EVIDENCE
       ANALYSIS        ANALYSIS          VERIFICATION
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                  EVIDENCE FUSION
                          │
                          ▼
                  DECISION ENGINE
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
        AUTO-APPROVE  AUTO-REJECT  HUMAN ESCALATION
```

The goal is not simply to predict "fraud."

The goal is to determine:

> **"What should the platform do with this return case, and why?"**

---

# 🧠 Core Technology

## 1. Multi-Agent AI — LangGraph

TrustLoop uses **LangGraph** to orchestrate specialized investigation agents.

### Fraud/Risk Agent

Analyzes:

* Customer return frequency
* Historical purchasing behavior
* Previous suspicious activity
* High-value purchases
* Recent return patterns
* Behavioral anomalies

### Policy Agent

Determines:

* Whether the return is within the allowed window
* Whether the product category has special rules
* Whether the customer's reason is policy-compliant
* Which policy clauses apply

### Vision Agent

Analyzes uploaded evidence such as:

* Product photographs
* Packaging
* Damage
* Product identity
* Serial numbers
* Accessories
* Product/return image consistency

### Decision Agent

Combines the outputs from the other agents and produces the final recommendation.

---

# 📊 2. Explainable Machine Learning

TrustLoop uses:

**LightGBM + SHAP**

LightGBM produces a fraud/risk probability based on behavioral and case-level features.

Example:

```text
Fraud Risk: 87%

Top Risk Factors:

+22%  Serial-number mismatch
+19%  Extremely high return frequency
+15%  Previous suspicious returns
+11%  High-value product
```

SHAP makes the prediction explainable instead of presenting an unexplained AI score.

---

# 📚 3. Policy-Aware RAG

TrustLoop does not rely entirely on an LLM to remember return policies.

A curated policy corpus is indexed using:

* FAISS
* BM25
* Reciprocal Rank Fusion (RRF)

Pipeline:

```text
Policy Documents
       │
       ├──────────► FAISS
       │
       └──────────► BM25
                       │
                       ▼
                 RRF Ranking
                       │
                       ▼
             Relevant Policy Rules
                       │
                       ▼
                Policy Agent
```

The system can therefore explain:

```text
Policy: POL-ELEC-007

Rule:
Returned product serial number must match
the original shipment record.

Result:
Serial mismatch detected.

Recommended action:
HUMAN ESCALATION
```

---

# 👁️ 4. Vision-Based Evidence Verification

TrustLoop uses **Gemini Vision** to analyze return evidence.

For example:

```json
{
  "product_match_score": 0.34,
  "damage_score": 0.12,
  "packaging_match_score": 0.81,
  "serial_visible": true,
  "serial_match": false,
  "confidence": 0.94
}
```

The vision system does not make the final decision independently.

Its output becomes evidence for the TrustLoop decision engine.

---

# ⚖️ 5. Decision Engine

TrustLoop produces three primary outcomes.

## AUTO-APPROVE

Used when:

* Return is policy eligible
* Evidence strongly matches the order
* Fraud risk is low
* No critical violation exists

```text
AUTO_APPROVE
```

---

## AUTO-REJECT

Used when:

* A critical violation is confirmed
* Strong evidence indicates item switching
* Serial number mismatch is confirmed
* Return clearly violates the applicable policy

```text
AUTO_REJECT
```

---

## HUMAN ESCALATION

Used when:

* Evidence is ambiguous
* AI systems disagree
* Vision confidence is low
* Policy interpretation is unclear
* High-value cases require additional verification
* Fraud probability is suspicious but not conclusive

```text
HUMAN_ESCALATION
```

This prevents TrustLoop from blindly trusting AI when the evidence is uncertain.

---

# 🔍 Investigation Timeline

Every TrustLoop case should provide an explainable investigation trail.

Example:

```text
10:41:02  Case Received

10:41:03  Customer Agent
          → 7 returns in the last 30 days

10:41:04  Fraud Agent
          → Risk Score: 87%

10:41:05  Vision Agent
          → Product mismatch detected

10:41:06  Policy Agent
          → POL-ELEC-007 retrieved

10:41:07  Decision Engine
          → HUMAN ESCALATION
```

This makes the AI reasoning process observable instead of hiding it behind a single "AI decision" button.

---

# 🗂️ Dataset Strategy

TrustLoop uses multiple datasets where appropriate.

The primary data foundation is e-commerce order/return information.

Additional datasets may provide:

* Customer behavior
* Product information
* Sales history
* Category information
* Ice-cream/quick-commerce product information
* Fraud scenarios

### Important

Raw datasets are preserved unchanged under:

```text
data/raw/
```

They should never be directly modified.

Instead:

```text
RAW DATA
   ↓
AUDIT
   ↓
CLEANING
   ↓
CANONICAL TRUSTLOOP SCHEMA
   ↓
FEATURE ENGINEERING
   ↓
MODEL / RAG / VISION
```

Synthetic fraud scenarios must be explicitly identified as synthetic.

Synthetic TrustLoop policy rules must not be represented as the official policy of Amazon, Flipkart, Zepto, or another retailer.

---

# 🧊 Category-Specific Intelligence

TrustLoop can support category-specific policies.

For example, temperature-sensitive products such as ice cream may require a different workflow from electronics.

```text
Product Category
       │
       ▼
Category Policy
       │
       ├── Electronics
       ├── Fashion
       ├── Furniture
       ├── Grocery
       └── Ice Cream
```

For temperature-sensitive products, the system can investigate cases such as:

* Melted product
* Damaged packaging
* Wrong item
* Missing item
* Delivery delay
* Repeated false damage claims

This allows TrustLoop to become a **policy-aware return/refund investigation platform**, rather than a generic fraud classifier.

---

# 🏗️ System Architecture

```text
                         TRUSTLOOP
                             │
                             ▼
                      FastAPI Backend
                             │
                             ▼
                       LangGraph
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
 Fraud/Risk Agent       Policy Agent         Vision Agent
       │                     │                     │
       ▼                     ▼                     ▼
 LightGBM + SHAP       FAISS + BM25          Gemini Vision
                             │
                             ▼
                          RRF
                             │
                             ▼
                    Evidence Fusion
                             │
                             ▼
                     Decision Engine
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
          APPROVE          REJECT        ESCALATE
```

---

# 🛠️ Technology Stack

| Component           | Technology                          |
| ------------------- | ----------------------------------- |
| Backend             | Python + FastAPI                    |
| Agent orchestration | LangGraph                           |
| Fraud model         | LightGBM                            |
| Explainability      | SHAP                                |
| Vector search       | FAISS                               |
| Keyword search      | BM25                                |
| Retrieval fusion    | RRF                                 |
| Vision              | Gemini Vision                       |
| Database            | Supabase / PostgreSQL               |
| Frontend            | Lovable-generated React application |
| Observability       | LangSmith                           |
| Version control     | Git + GitHub                        |
| Deployment          | Vercel + Render/Railway             |

---

# 📁 Project Structure

```text
TrustLoop_VSCode_Starter/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── agents/
│   │   ├── decision/
│   │   ├── ml/
│   │   ├── rag/
│   │   ├── vision/
│   │   ├── schemas/
│   │   └── services/
│   │
│   └── tests/
│
├── data/
│   ├── raw/
│   ├── audit/
│   ├── processed/
│   ├── demo/
│   └── evidence/
│
├── policies/
│   ├── general/
│   ├── electronics/
│   ├── fashion/
│   ├── high_value/
│   └── fraud/
│
├── models/
│
├── scripts/
│
├── docs/
│
├── frontend/
│
├── .env.example
├── requirements.txt
└── README.md
```

---

# 🚀 Development Roadmap

## Phase 1 — Dataset Audit

* [ ] Audit all raw datasets
* [ ] Identify schemas
* [ ] Identify primary/foreign keys
* [ ] Identify relationships
* [ ] Detect duplicate and invalid records
* [ ] Determine primary dataset
* [ ] Create canonical TrustLoop schema

## Phase 2 — Data Pipeline

* [ ] Clean data
* [ ] Normalize categories
* [ ] Engineer customer behavior features
* [ ] Generate return-case records
* [ ] Validate processed data

## Phase 3 — Policy Engine

* [ ] Create TrustLoop Demo Return Policy
* [ ] Create category-specific policies
* [ ] Implement deterministic policy checks
* [ ] Add policy tests

## Phase 4 — RAG

* [ ] Build policy corpus
* [ ] Chunk documents
* [ ] Build FAISS index
* [ ] Build BM25 index
* [ ] Implement RRF
* [ ] Return structured policy evidence

## Phase 5 — Fraud Detection

* [ ] Define fraud scenarios
* [ ] Engineer behavioral features
* [ ] Train LightGBM
* [ ] Evaluate precision/recall/F1/ROC-AUC
* [ ] Implement SHAP explanations

## Phase 6 — Vision

* [ ] Create evidence cases
* [ ] Integrate Gemini Vision
* [ ] Detect product mismatch
* [ ] Detect damage
* [ ] Verify packaging
* [ ] Verify serial evidence
* [ ] Implement confidence thresholds

## Phase 7 — Multi-Agent System

* [ ] Create LangGraph state
* [ ] Implement Fraud Agent
* [ ] Implement Policy Agent
* [ ] Implement Vision Agent
* [ ] Implement Evidence Fusion
* [ ] Implement Decision Agent
* [ ] Implement Human Escalation

## Phase 8 — Product

* [ ] FastAPI endpoints
* [ ] Supabase database
* [ ] Evidence storage
* [ ] TrustLoop dashboard
* [ ] Investigation timeline
* [ ] Human review interface
* [ ] Analytics

## Phase 9 — Testing & Deployment

* [ ] Unit tests
* [ ] ML evaluation
* [ ] RAG evaluation
* [ ] Vision evaluation
* [ ] End-to-end testing
* [ ] Create demonstration cases
* [ ] Deploy backend
* [ ] Deploy frontend
* [ ] Configure monitoring

---

# 🎯 Target User Experience

A TrustLoop investigator should be able to open a case and immediately see:

```text
┌─────────────────────────────────────────────┐
│ RETURN CASE #TL-10482                       │
├─────────────────────────────────────────────┤
│ Product: Premium Electronics                │
│ Order Value: ₹42,999                        │
│                                             │
│ Fraud Risk              87%                 │
│ Decision                AUTO-REJECT         │
│                                             │
│ ───────────── EVIDENCE ─────────────        │
│                                             │
│ Product Match           34%                 │
│ Serial Match             8%                 │
│ Damage Score            12%                 │
│                                             │
│ ───────────── POLICY ──────────────        │
│                                             │
│ POL-ELEC-007                               │
│ Serial number mismatch                     │
│                                             │
│ ───────────── WHY? ────────────────        │
│                                             │
│ • Serial number mismatch                   │
│ • High return frequency                    │
│ • Product image mismatch                   │
│ • Previous suspicious activity             │
│                                             │
└─────────────────────────────────────────────┘
```

The objective is to turn a return request into a **traceable investigation**, not simply produce an AI-generated yes/no answer.

---

# 🔐 Design Principles

TrustLoop follows five principles:

### 1. Evidence over assumption

AI decisions should be backed by observable evidence.

### 2. Policy over hallucination

Critical policy enforcement should use deterministic rules and retrieved policy evidence.

### 3. Explainability

Every risk score and decision should have an understandable explanation.

### 4. Human-in-the-loop

Ambiguous or high-risk cases should be escalated rather than blindly automated.

### 5. Data provenance

Raw, derived, and synthetic data must remain distinguishable.

---

# 📌 Current Status

**Current development stage: Dataset Integration & Audit**

The next engineering milestone is:

```text
Raw CSV datasets
       ↓
Dataset Audit
       ↓
Relationship Mapping
       ↓
Canonical TrustLoop Schema
```

No ML model, RAG system, or agent should be considered production-ready until the underlying data relationships and labels have been validated.

---

## TrustLoop

**Investigate smarter. Decide with evidence.**
~Contribution:-
Money Goyal 
Keshav Gupta
