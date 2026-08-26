# TrustLoop Dataset Baseline Report

- **Total Samples:** 60,000
- **Features in Raw Dataset:** 28
- **Chronological Split:**
  - **Train (70%):** 42,000 samples
  - **Validation (15%):** 9,000 samples
  - **Test (15%):** 9,000 samples

## Class Distribution
| Class ID | Class Name | Total Count | Total % | Train Count | Val Count | Test Count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Legitimate | 42,060 | 70.10% | 29,429 | 6,384 | 6,247 |
| 1 | Policy Abuser | 7,192 | 11.99% | 5,082 | 1,046 | 1,064 |
| 2 | Fraudulent Return | 6,112 | 10.19% | 4,416 | 875 | 821 |
| 3 | Wardrobing | 4,636 | 7.73% | 3,073 | 695 | 868 |

## Data Quality Integrity
- **Duplicate Rows:** 0
- **Total Missing Values:** 0
- **Total Infinite Values:** 0
- **Constant Features:** 0
- **Train/Test Overlap:** 0
- **Data Leakage Risk:** NONE (pre-decision feature formulation verified)
