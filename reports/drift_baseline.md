# TrustLoop Production Drift Baseline

- **Training Class Priors:** {'Legitimate': 0.701, 'Policy Abuser': 0.1199, 'Fraudulent Return': 0.1019, 'Wardrobing': 0.0773}

## Feature PSI (Train vs Test Baseline Split)
| Feature | PSI Score | Drift Status |
| :--- | :--- | :--- |
| `age` | 0.0011 | STABLE |
| `account_age_days` | 0.0005 | STABLE |
| `avg_order_value_usd` | 0.0014 | STABLE |
| `calculated_days_to_return` | 0.0144 | STABLE |
| `wishlist_to_cart_time_hrs` | 0.0020 | STABLE |
