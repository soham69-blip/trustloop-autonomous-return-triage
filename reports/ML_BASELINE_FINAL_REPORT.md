# TrustLoop — Complete ML Baseline & Validation Final Report

```
================================================================================
ML BASELINE STATUS: COMPLETE
- Evaluation Components Executed: 19 / 19 (100%)
- Automated Tests Passing: 106 / 106
- Production Model SHA256 Integrity: MATCH (db3a6c03149fa096...)
- Candidate Model SHA256 Integrity:  MATCH (6dec9ceeb10b2f9c...)
- Dataset Quality Integrity:         VERIFIED (60,000 samples, 0 leakage)
================================================================================
```

---

## 1. Dataset Summary & Split Architecture
- **Dataset Source:** `data/processed/trustloop/model_ready.csv` (60,000 samples)
- **Chronological Split:**
  - **Train (70%):** 42,000 samples
  - **Validation (15%):** 9,000 samples
  - **Test (15%):** 9,000 samples
- **Data Quality:** 0 duplicate rows, 0 missing values in feature set, 0 infinite values, 0 train/test overlap leakage.

---

## 2. Model Architecture & Feature Count
- **Production Model:** LightGBM Gradient Boosted Decision Tree (33 active features, `models/lightgbm_model.pkl`)
- **Candidate Model:** LightGBM Classifier with Extended Behavioral Features (39 features, `models/lightgbm_candidate.pkl`)
- **Multi-Class Strategy:** Softmax objective with 4-class output distribution.

---

## 3. Production Model Performance Benchmark (9,000 Test Samples)

### Overall Classification Metrics
| Metric | Score | Benchmark Target | Status |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **0.9170** (91.70%) | $\ge 90.0\%$ | PASS |
| **Balanced Accuracy** | **0.8521** (85.21%) | $\ge 80.0\%$ | PASS |
| **Macro F1 Score** | **0.8721** (87.21%) | $\ge 85.0\%$ | PASS |
| **Weighted F1 Score** | **0.9082** (90.82%) | $\ge 90.0\%$ | PASS |
| **Cohen's Kappa** | **0.8192** | $\ge 0.75$ | PASS |
| **Matthews Corr Coef (MCC)** | **0.8254** | $\ge 0.75$ | PASS |
| **Log Loss** | **0.2113** | $\le 0.35$ | PASS |
| **Multiclass Brier Score** | **0.1265** | $\le 0.15$ | PASS |
| **Macro ROC-AUC** | **0.9685** | $\ge 0.95$ | PASS |
| **Weighted ROC-AUC** | **0.9609** | $\ge 0.95$ | PASS |
| **Expected Calibration Error** | **0.0068** | $\le 0.08$ | PASS |

### Per-Class Performance Breakdown
| Class ID | Class Name | Precision | Recall | F1 Score | Specificity | FPR | FNR | ROC-AUC | PR-AUC | Brier | Support |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **Legitimate** | 0.9128 | 0.9757 | 0.9432 | 0.7886 | 0.2114 | 0.0243 | 0.9579 | 0.9786 | 0.0620 | 6,247 |
| **1** | **Policy Abuser** | 0.8355 | 0.4727 | 0.6038 | 0.9875 | 0.0125 | 0.5273 | 0.9167 | 0.7181 | 0.0570 | 1,064 |
| **2** | **Fraudulent Return** | 0.9508 | 0.9878 | 0.9689 | 0.9949 | 0.0051 | 0.0122 | 0.9997 | 0.9964 | 0.0039 | 821 |
| **3** | **Wardrobing** | 0.9724 | 0.9724 | 0.9724 | 0.9970 | 0.0030 | 0.0276 | 0.9998 | 0.9981 | 0.0037 | 868 |

---

## 4. Root Cause Analysis of Primary ML Weakness (Policy Abuser Detection)

1. **Class Imbalance & Subtle Boundary Overlap:**
   - Policy Abusers represent only 12.0% of the dataset.
   - Standard 33-feature space lacks direct cumulative behavioral metrics (e.g. `return_rate_pct`, `lifetime_dispute_count`, `customer_support_contacts`).
   - Consequently, the model exhibits conservative recall on Class 1 (47.27%) while maintaining high precision (83.56%).
2. **Threshold Optimization Finding:**
   - Operating Policy Abuser probability threshold at **$	au = 0.39$** (instead of standard argmax 0.50) increases Policy Abuser F1 to **0.6335** with significantly higher recall.

---

## 5. Summary of Generated Baseline Artifacts

| Category | Artifact Path | Description |
| :--- | :--- | :--- |
| **Dataset Baseline** | `reports/dataset_baseline.json`, `.csv`, `.md` | Full data quality, leakage, and distribution audit. |
| **Classification Metrics** | `reports/ml_baseline_metrics.json`, `.csv` | Comprehensive overall and per-class metrics. |
| **Classification Report** | `reports/ml_classification_report.csv` | Precision, recall, specificity, Brier, ROC/PR per class. |
| **Confusion Matrices** | `reports/confusion_matrix.csv`, `_normalized.csv`, `.png` | Raw and row-normalized 4x4 confusion matrix plots. |
| **Curves & Calibration** | `reports/roc_curves.png`, `precision_recall_curves.png`, `calibration_curve.png` | One-vs-Rest ROC, PR, and reliability diagrams. |
| **Calibration Metrics** | `reports/calibration_metrics.json` | Multiclass Brier, ECE, and MCE metrics. |
| **Threshold Analysis** | `reports/policy_abuser_threshold_analysis.csv`, `.png` | 46-step probability threshold sensitivity sweep. |
| **Feature Importance** | `reports/feature_importance.csv`, `.png` | Native LightGBM gain and split importances. |
| **TreeSHAP Explainability** | `reports/shap_global.csv`, `shap_classwise.csv`, `shap_summary.png` | Global and classwise feature attribution values. |
| **Adversarial Hard Cases** | `reports/hard_case_baseline.json`, `.csv`, `.md` | 15-case adversarial and boundary scenario suite. |
| **Robustness & Perturbation** | `reports/robustness_report.json`, `.md` | Schema-valid input perturbation resilience analysis. |
| **Counterfactual Analysis** | `reports/counterfactual_baseline.csv` | Feature sweep probability transition grids. |
| **Model Comparison** | `reports/production_vs_candidate.csv`, `.md` | Production (33) vs Candidate (39) side-by-side benchmark. |
| **Bootstrap CIs** | `reports/bootstrap_confidence_intervals.csv` | 1,000-iteration 95% bootstrap confidence intervals. |
| **Drift Monitoring** | `reports/drift_baseline.json`, `.md` | Population Stability Index baseline and priors. |
| **Environment Manifest** | `reports/ml_environment.json` | Exact dependency versions and model SHA256 checksums. |
