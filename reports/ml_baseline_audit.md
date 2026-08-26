# TrustLoop ML Baseline & MLOps Infrastructure Audit

**Date:** 2026-08-26  
**Auditor:** Senior ML Validation & MLOps Engineer  
**Scope:** Machine Learning Pipeline, Models, Datasets, Feature Contracts, Calibration, SHAP, Drift, and Validation Architecture  

---

## 1. Executive Summary

TrustLoop operates a hybrid AI decisioning and fraud detection architecture with a 4-class multi-classification model at its core:
- **Class 0:** Legitimate (70.1% prevalence)
- **Class 1:** Policy Abuser (12.0% prevalence)
- **Class 2:** Fraudulent Return (10.2% prevalence)
- **Class 3:** Wardrobing (7.7% prevalence)

The current production LightGBM model (`models/lightgbm_model.pkl`, 33 features) demonstrates strong overall accuracy (91.70%) and macro F1 (87.21%), with exceptional performance on Fraudulent Return (96.89% F1) and Wardrobing (97.24% F1). However, the primary operational vulnerability is **Policy Abuser detection (47.27% recall, 60.38% F1)**.

This audit establishes the ground truth across all existing artifacts, datasets, split mechanisms, feature contracts, and evaluation pipelines prior to formal baseline generation.

---

## 2. Inventory of Existing ML Artifacts & Checksums

| Artifact | Path | Size | Features | Verified SHA256 Checksum | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Production LightGBM Model** | `models/lightgbm_model.pkl` | 1.97 MB | 33 | `db3a6c03149fa096c7df0d2df214043236f9cb8a8b0fe8ea198d1c24ea94d485` | ACTIVE (Frozen) |
| **Candidate LightGBM Model** | `models/lightgbm_candidate.pkl` | 1.38 MB | 39 | `6dec9ceeb10b2f9c7148b531bb1306e38344803250b162d0b5f0ff3ae0296f04` | SHADOW (Isolated) |
| **Categorical Mappings** | `models/categorical_mappings.pkl` | 783 B | 8 cat cols | `432e9539c295367d700c3307dbb439a5e674f17432e84d2b3d52089c852997ad` | ACTIVE |
| **Production Backup Model** | `models/lightgbm_model_backup.pkl` | 1.97 MB | 33 | `db3a6c03149fa096c7df0d2df214043236f9cb8a8b0fe8ea198d1c24ea94d485` | BACKUP |
| **Primary Dataset** | `data/processed/trustloop/model_ready.csv` | 8.94 MB | 33 + Target | - | 60,000 Rows |

---

## 3. Existing Evaluation Scripts & Tools

1. **`scripts/train_lightgbm.py`**:
   - Implements temporal feature extraction (`order_date`, `return_date`, `calculated_days_to_return`).
   - Implements chronological split based on `(order_date, return_date)` sort order (70% train / 15% validation / 15% test).
   - Generates candidate model artifacts and evaluations.
2. **`scripts/eval_hard_cases.py`**:
   - Evaluates 5 curated boundary cases against both production and candidate models.
   - Highlights policy abuser and wardrobing boundary failures.
3. **`scripts/forensic_eval.py`**:
   - Diagnostic script analyzing test dataset predictions, confusion matrix, and feature distributions.
4. **`scripts/policy_threshold_validation.py` & `scripts/policy_abuser_experiment.py`**:
   - Explores class weight and threshold tuning specifically for Class 1 (Policy Abuser).

---

## 4. Existing Train / Validation / Test Split Strategy

- **Total Rows:** 60,000 rows
- **Sorting Mechanism:** Chronological sort via `np.lexsort((order_date, return_date))`
- **Split Breakdown:**
  - **Train (70%):** Rows 0 to 41,999 (42,000 samples)
  - **Validation (15%):** Rows 42,000 to 50,999 (9,000 samples)
  - **Test (15%):** Rows 51,000 to 59,999 (9,000 samples)
- **Leakage Prevention:** Chronological ordering ensures future transactions never leak into training or validation splits.

---

## 5. Feature Contract Architecture

### Production Model (33 Features):
1. `age`
2. `account_age_days`
3. `customer_segment` (Categorical)
4. `country` (Categorical)
5. `platform` (Categorical)
6. `device_type` (Categorical)
7. `payment_method` (Categorical)
8. `product_category` (Categorical)
9. `avg_order_value_usd`
10. `is_high_value_item`
11. `discount_used`
12. `days_to_return`
13. `return_reason` (Categorical)
14. `shipping_carrier` (Categorical)
15. `multiple_accounts_flag`
16. `wishlist_to_cart_time_hrs`
17. `customer_return_count_prior`
18. `returns_last_30d_prior`
19. `returns_last_90d_prior`
20. `total_returns_lifetime_prior`
21. `order_date_year`
22. `order_date_month`
23. `order_date_day`
24. `order_date_dayofweek`
25. `order_date_dayofyear`
26. `order_date_is_weekend`
27. `return_date_year`
28. `return_date_month`
29. `return_date_day`
30. `return_date_dayofweek`
31. `return_date_dayofyear`
32. `return_date_is_weekend`
33. `calculated_days_to_return`

### Candidate Model (39 Features):
Includes all 33 production features plus 6 cumulative behavioral features:
- `total_returns_lifetime`
- `total_orders_lifetime`
- `return_rate_pct`
- `customer_support_contacts`
- `previous_dispute_count`
- `refund_amount_requested_usd`

---

## 6. Existing Service Implementations

- **TreeSHAP Explainability (`backend/app/services/shap_service.py`)**:
  - Implements cached `shap.TreeExplainer(model)` instances for fast local feature attribution.
  - Correctly handles multi-class SHAP dimensions `(1, n_features, n_classes)`.
- **Drift Monitoring (`backend/app/services/drift_service.py`)**:
  - Implements categorical & binned numerical Population Stability Index (PSI).
  - Maintains baseline training class priors: Legitimate (70.10%), Policy Abuser (11.99%), Fraudulent Return (10.19%), Wardrobing (7.73%).
- **Shadow Mode Candidate Execution (`backend/app/services/shadow_service.py`)**:
  - Executes candidate model in background without mutating production decisions.
  - Logs disagreements to `data/feedback/shadow_log.jsonl`.
- **Model Promotion Gate (`backend/app/services/retraining_service.py`)**:
  - Strict promotion criteria: Macro F1 improvement $\ge +0.5\%$, Class Recall regression $\le 2\%$, Legitimate precision regression $\le 2\%$.

---

## 7. Gap Analysis & Missing Baseline Components

1. **Missing Unified Baseline Script**: No single automated script exists that comprehensively evaluates all metrics, calibration, ROC/PR curves, thresholds, and bootstrap confidence intervals.
2. **Missing Formal Calibration Metrics**: Brier score (multi-class and per-class) and Expected Calibration Error (ECE) are not yet systematically tracked as report artifacts.
3. **Hard-Case Suite Expansion**: Existing hard-case evaluation only covers 5 cases; needs expansion to 15+ comprehensive adversarial cases covering missing fields, unknown categories, and boundary anomalies.
4. **Automated Robustness & Sensitivity Suite**: Missing systematic perturbation and counterfactual boundary sweeps across key operational features.
5. **Statistical Confidence Intervals**: No bootstrap confidence intervals currently computed for accuracy and class-specific F1/recall.

---

## 8. Potential Data Leakage & Evaluation Risks Identified

- **Risk 1 (Test Contamination in Calibration):** Fitting probability calibrators (Platt scaling / Isotonic regression) on test data would invalidate generalization claims. *Mitigation:* Use validation split only for any calibration modeling or evaluate raw model probabilities on the test set.
- **Risk 2 (Telemetry Contamination):** Unit and integration test runs creating synthetic records in `data/feedback/verified_feedback.jsonl` or `shadow_log.jsonl`. *Mitigation:* Explicitly isolate test telemetry from production drift monitoring baselines.
- **Risk 3 (Target Leakage):** Features derived after return decision. *Verification:* All 33 production features represent pre-decision signals captured at or before return triage intake.

---

## 9. Implementation Plan

- **Phase 1:** Dataset audit script & baseline reports (`reports/dataset_baseline.json`, `.csv`, `.md`).
- **Phase 2–6:** Core ML evaluation engine (`scripts/ml_baseline.py`) computing all overall & per-class metrics, confusion matrices, ROC/PR curves, calibration curves, and Policy Abuser threshold analysis.
- **Phase 7–9:** Class imbalance analysis, gain/split feature importance, and TreeSHAP global/class-wise attributions.
- **Phase 10–12:** 15+ Adversarial hard cases (`reports/hard_case_baseline.*`), schema robustness testing, and counterfactual sensitivity sweeps.
- **Phase 13–15:** Production vs Candidate rigorous benchmark, bootstrap 95% confidence intervals, and drift monitoring baseline.
- **Phase 16–19:** Environment manifest, model hash validation, automated test suite (`tests/test_ml_baseline.py`), and final summary report (`reports/ML_BASELINE_FINAL_REPORT.md`).
