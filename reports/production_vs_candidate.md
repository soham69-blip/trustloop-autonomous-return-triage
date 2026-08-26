# Production vs Candidate Model Benchmark

- **Test Evaluation Samples:** 9,000 identical samples
- **Production Architecture:** LightGBM Classifier (33 baseline features)
- **Candidate Architecture:** LightGBM Classifier (39 extended features)

## Overall Performance Comparison
| Metric | Production (33 Feats) | Candidate (39 Feats) | Delta |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.9170 | 0.9994 | +0.0824 |
| **Balanced Accuracy** | 0.8521 | 0.9988 | +0.1467 |
| **Macro F1** | 0.8721 | 0.9987 | +0.1266 |
| **Weighted F1** | 0.9082 | 0.9994 | +0.0912 |
| **Log Loss** | 0.2113 | 0.0032 | -0.2081 |
| **Multiclass Brier** | 0.1265 | 0.0011 | -0.1254 |

## Policy Abuser Focus Comparison
| Metric | Production | Candidate | Delta |
| :--- | :--- | :--- | :--- |
| **Policy Abuser Recall** | 0.4727 | 0.9953 | +0.5226 |
| **Policy Abuser Precision** | 0.8355 | 1.0000 | +0.1645 |
| **Policy Abuser F1** | 0.6038 | 0.9976 | +0.3938 |
