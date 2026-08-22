# FlipLens Implementation Plan

## Milestone 1 — Data
- [ ] Download Kaggle E-Commerce Dataset Order and Return
- [ ] Put untouched source file in `data/raw/`
- [ ] Run dataset audit
- [ ] Document actual columns; do not invent fields
- [ ] Build canonical processed return-case dataset

## Milestone 2 — Policy
- [ ] Create synthetic FlipLens Demo Return Policy
- [ ] Create 40–60 rules
- [ ] Implement deterministic policy engine
- [ ] Add policy unit tests

## Milestone 3 — RAG
- [ ] Chunk policy documents
- [ ] Build BM25 index
- [ ] Build FAISS index
- [ ] Implement Reciprocal Rank Fusion
- [ ] Return structured policy evidence

## Milestone 4 — ML
- [ ] Define synthetic benchmark fraud scenarios separately from raw data
- [ ] Engineer behavioral/evidence features
- [ ] Train LightGBM
- [ ] Evaluate precision/recall/F1/ROC-AUC
- [ ] Add SHAP explanations

## Milestone 5 — Vision
- [ ] Create evidence-case folder structure
- [ ] Integrate Gemini Vision
- [ ] Force structured JSON output
- [ ] Add confidence thresholds

## Milestone 6 — Orchestration
- [ ] Build LangGraph state
- [ ] Parallel fraud / vision / policy agents
- [ ] Evidence fusion
- [ ] Deterministic decision engine
- [ ] Human escalation path

## Milestone 7 — Product
- [ ] FastAPI endpoints
- [ ] Supabase
- [ ] Lovable dashboard
- [ ] Agent investigation timeline
- [ ] Analytics
- [ ] End-to-end testing
- [ ] Deployment
