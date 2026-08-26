# TrustLoop 15-Case Adversarial & Hard-Case Baseline

| Case ID | Case Description | Expected | Production Prediction | Conf | Correct? | Candidate | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| HC-01 | 1. Legitimate high-frequency shopper ( | Legitimate | Legitimate | 85.1% | PASS | Legitimate | Nominal Evaluation |
| HC-02 | 2. Borderline Policy Abuser with 22% r | Policy Abuser | Legitimate | 64.9% | FAIL | Legitimate | DATASET / EVALUATION MISMATCH: 3 |
| HC-03 | 3. Fraudulent return with multi-accoun | Fraudulent Return | Fraudulent Return | 98.1% | PASS | Fraudulent Return | Nominal Evaluation |
| HC-04 | 4. Wardrobing with 12-day return windo | Wardrobing | Legitimate | 80.2% | FAIL | Legitimate | DATASET / EVALUATION MISMATCH: 1 |
| HC-05 | 5. Legitimate luxury item ($600 high v | Legitimate | Legitimate | 92.0% | PASS | Legitimate | Nominal Evaluation |
| HC-06 | 6. Strong Policy Abuser (70% return ra | Policy Abuser | Wardrobing | 67.2% | FAIL | Wardrobing | Nominal Evaluation |
| HC-07 | 7. Rapid turnaround return (Delivered  | Legitimate | Legitimate | 97.2% | PASS | Legitimate | Nominal Evaluation |
| HC-08 | 8. Out-of-policy late return window (4 | Policy Abuser | Wardrobing | 100.0% | FAIL | Wardrobing | Nominal Evaluation |
| HC-09 | 9. Chronic Chargeback / Dispute Ring ( | Fraudulent Return | Fraudulent Return | 100.0% | PASS | Fraudulent Return | Nominal Evaluation |
| HC-10 | 10. Aggressive escalation abuser (14 s | Policy Abuser | Legitimate | 95.9% | FAIL | Policy Abuser | Nominal Evaluation |
| HC-11 | 11. High return-rate habitual returner | Policy Abuser | Legitimate | 83.4% | FAIL | Policy Abuser | Nominal Evaluation |
| HC-12 | 12. Conflicting Signals (High value $8 | Legitimate | Legitimate | 93.0% | PASS | Legitimate | Nominal Evaluation |
| HC-13 | 13. Missing optional telemetry fields  | Legitimate | Legitimate | 93.3% | PASS | Legitimate | Nominal Evaluation |
| HC-14 | 14. Extreme Outlier Values (Order $12, | Legitimate | Legitimate | 91.1% | PASS | Legitimate | Nominal Evaluation |
| HC-15 | 15. Novel / Unseen Categorical Encodin | SCHEMA_CONTRACT_REJECTION | SCHEMA_REJECTED | 100.0% | PASS | SCHEMA_REJECTED | Unseen category values correctly |
