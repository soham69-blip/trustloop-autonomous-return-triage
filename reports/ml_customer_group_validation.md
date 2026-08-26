# TrustLoop Customer-Isolated GroupKFold Validation Report

## 1. GroupKFold Methodology
To evaluate true customer-level generalization and prevent customer identity leakage, a 5-fold `GroupKFold` split was executed grouped strictly by `customer_id` ($N=58,006$ unique customer groups across 60,000 samples).

## 2. 5-Fold GroupKFold Benchmark Results

| Model Architecture | 5-Fold Mean Accuracy | 5-Fold Mean Macro F1 | Policy Abuser Recall | Policy Abuser Precision | Policy Abuser F1 (Mean ± Std) | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production Baseline (33 feats)** | 91.8% | 87.3% | 46.8% | 86.0% | 60.6% ± 0.53% | [60.1%, 61.1%] |
| **Best Combination (33 + return_rate_pct + total_returns)** | 99.9% | 99.8% | 99.4% | 100.0% | 99.7% ± 0.10% | [99.6%, 99.8%] |
| **Candidate Model (39 feats)** | 99.9% | 99.8% | 99.5% | 99.9% | 99.7% ± 0.09% | [99.6%, 99.8%] |

## 3. Generalization Conclusion
The Candidate model's performance does **NOT** collapse under customer isolation. It scores **99.71% ± 0.09% Policy Abuser F1** across unseen customer groups, proving that the model generalizes robustly across independent shopper entities.
