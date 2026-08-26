# TrustLoop Model Robustness & Perturbation Report

| Test Scenario | Baseline Prediction | Perturbed Prediction | Flipped? | Max Prob Delta | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Unknown Country ('ZZ') | Legitimate | CONTRACT_REJECTED | NO | 0.0000 | SCHEMA_CONTRACT_REJECTED (SAFE DEFENSE) |
| Unknown Payment ('CryptoToken') | Legitimate | CONTRACT_REJECTED | NO | 0.0000 | SCHEMA_CONTRACT_REJECTED (SAFE DEFENSE) |
| Zero Account Age (0 days) | Legitimate | Legitimate | NO | 0.0558 | RESILIENT |
| High Order Value ($5,000) | Legitimate | Legitimate | NO | 0.1235 | RESILIENT |
| Small Noise (+5% Age, +5% Wishlist) | Legitimate | Legitimate | NO | 0.0000 | RESILIENT |
| Extreme Wishlist Duration (720 hrs) | Legitimate | Legitimate | NO | 0.0456 | RESILIENT |
| Boundary Return Window (30.0 days) | Legitimate | Legitimate | NO | 0.0345 | RESILIENT |
