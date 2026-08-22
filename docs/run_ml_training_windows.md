# TrustLoop Stage 2: ML Training on Windows

This document provides exact instructions to run the LightGBM baseline training locally on Windows in VS Code.

## Prerequisites

- Windows 10/11
- Python 3.8+ installed and accessible from PowerShell
- Project cloned/downloaded to: `C:\Users\khura\Downloads\TrustLoop_VSCode_Starter`
- Stage 1 feature engineering completed (verified: `data/processed/trustloop/model_ready.csv` exists)

## Step-by-Step Setup

### 1. Open PowerShell in VS Code

- Press `Ctrl + Shift + ~` to open the integrated terminal in VS Code
- Or manually open PowerShell as Administrator if required
- Terminal should show: `PS C:\Users\khura\Downloads\TrustLoop_VSCode_Starter>`

### 2. Verify Python Installation

```powershell
python --version
```

Expected output: `Python 3.8.x` or newer (3.10+ recommended)

If the command fails, Python is not in PATH. Install from https://www.python.org and ensure "Add Python to PATH" is checked.

### 3. Create Virtual Environment

Virtual environments isolate project dependencies and prevent conflicts with system Python.

```powershell
python -m venv .venv
```

This creates a `.venv` folder in the project root.

### 4. Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

Expected: terminal prompt changes to `(.venv) PS ...`

#### If Activation Fails (Execution Policy Error)

You may see: `...cannot be loaded because running scripts is disabled...`

Run this command ONCE (for the current PowerShell session only):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

**Note:** This only affects the current PowerShell window. Opening a new PowerShell requires rerunning the policy command.

### 5. Upgrade pip

Ensures the latest package installer:

```powershell
python -m pip install --upgrade pip
```

### 6. Install ML Dependencies

```powershell
python -m pip install -r requirements-ml.txt
```

This installs:
- pandas, numpy (data handling)
- scikit-learn (ML utilities)
- lightgbm (gradient boosting)
- shap (model interpretability)
- pyarrow (parquet I/O)
- joblib (model serialization)
- matplotlib (optional, for future visualization)

Installation may take 5–10 minutes depending on internet speed.

### 7. Verify Installation

```powershell
python -c "import pandas, numpy, sklearn, lightgbm, shap, pyarrow; print('✓ ML environment OK')"
```

Expected output: `✓ ML environment OK`

If this fails, review the error message and re-run step 6.

## Running the Training Script

### Execute Training

```powershell
python scripts\train_lightgbm.py
```

Expected behavior:

1. **Validation checks** (1–2 seconds):
   - Verifies `data/processed/trustloop/model_ready.csv` exists
   - Checks for required columns (features, target)
   - Verifies 4 target classes
   - Ensures no leakage/ID columns

2. **Data preprocessing** (5–10 seconds):
   - Loads 60,000 rows
   - Handles missing values (median for numeric, "unknown" for categorical)
   - Encodes categorical features
   - Performs chronological 70/15/15 split by return_date, order_date

3. **Model training** (1–3 minutes):
   - Trains LightGBM multiclass on 42,000 train rows
   - Validates on 9,000 validation rows with early stopping
   - Prints progress

4. **Evaluation & SHAP** (2–5 minutes):
   - Computes metrics (Accuracy, Precision, Recall, F1, Weighted F1)
   - Generates confusion matrix and per-class classification report
   - Computes feature importance
   - Computes SHAP values for model explainability

5. **Output** (final summary):
   ```
   ===============================================================================
   TRAINING COMPLETE
   ===============================================================================
   
   Metrics (test set):
     accuracy: 0.8123
     precision_macro: 0.7845
     recall_macro: 0.7654
     f1_macro: 0.7745
     f1_weighted: 0.8089
   
   All artifacts written to:
     - models/
     - reports/
     - docs/lightgbm_baseline.md
   ```

### Total Runtime

Expected **10–15 minutes** on a modern laptop (Intel i5+, 8GB+ RAM).

## Output Files

After successful training, verify these files were created:

### Models (in `models/`)
```
models/
├── lightgbm_model.pkl              # Trained LightGBM model (binary)
└── label_encoder.pkl               # Target class encoder (binary)
```

### Reports (in `reports/`)
```
reports/
├── metrics.json                    # Train/Val/Test metrics
├── confusion_matrix.csv            # Test confusion matrix
├── classification_report.csv       # Per-class precision/recall/F1
├── feature_importance.csv          # Feature importance ranks
├── shap_values.parquet             # Per-sample SHAP values
└── shap_summary.csv                # Mean abs SHAP per feature per class
```

### Documentation (in `docs/`)
```
docs/lightgbm_baseline.md           # Human-readable training summary
```

## Next Steps After Training

### 1. Review Metrics

Open `reports/metrics.json` to inspect:
- Accuracy, precision, recall, F1 across train/val/test
- Identify class imbalance effects (class 3 has fewer samples)

### 2. Analyze Feature Importance

Open `reports/feature_importance.csv` to see which features drive predictions:
```csv
feature,importance
return_reason,0.450
days_to_return,0.320
customer_return_count_prior,0.180
...
```

### 3. Inspect SHAP Explanations

Open `reports/shap_summary.csv` to see mean absolute SHAP contribution per feature:
```csv
feature,class,mean_abs_shap
return_reason,0,0.125
return_reason,1,0.110
...
```

Use this to understand which features matter for each abuse class.

### 4. Archive & Report Results

After confirming successful training:
- Make a note of test-set accuracy/F1
- Save the outputs directory for later reference (this is your baseline)
- Proceed to Stage 3+ if needed (API deployment, frontend, etc.)

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'pandas'`
- Ensure virtual environment is activated (`(.venv)` prefix visible)
- Re-run: `python -m pip install -r requirements-ml.txt`

### Error: `FileNotFoundError: ... model_ready.csv`
- Ensure Stage 1 feature engineering completed successfully
- Check: `data/processed/trustloop/model_ready.csv` exists
- If missing, re-run: `python scripts\build_trustloop_features.py`

### Error: `ValueError: Expected 4 target classes, found X`
- The dataset has wrong target classes
- Verify `model_ready.csv` contains `abuse_label` column with values 0, 1, 2, 3
- Do NOT modify `data/raw/` — re-check Stage 1 output

### Error: `LightGBMError: ... boost from empty train set`
- Feature encoding failed or training data is empty
- Ensure features were properly encoded (categorical → codes)
- Check: `len(train_df) > 0` and feature matrix has no all-NaN columns

### Script hangs or is very slow
- SHAP computation on large datasets can take 5+ minutes
- For testing, reduce X_train sample size temporarily (edit script line ~130)
- On powerful machines (CPU cores), increase `n_jobs=-1` to use all cores

### PowerShell execution policy issue
Run this ONCE per session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Deactivating Virtual Environment

When finished training, you can deactivate the virtual environment:

```powershell
deactivate
```

The prompt returns to `PS ...` (without `.venv` prefix).

To reuse the environment for another training run:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\train_lightgbm.py
```

No need to re-install dependencies; they persist in `.venv/`.

## Files to Send Back

After training completes successfully, send these artifacts:

1. **models/lightgbm_model.pkl** — trained model (for deployment)
2. **models/label_encoder.pkl** — class encoder (for deployment)
3. **reports/metrics.json** — numerical results
4. **reports/feature_importance.csv** — feature rankings
5. **reports/shap_summary.csv** — SHAP analysis
6. **docs/lightgbm_baseline.md** — training summary

Optional:
- Full `reports/` directory (all CSV/parquet files for detailed analysis)
- Screenshots of metrics/feature importance for documentation

## Summary

| Step | Time | Command |
|------|------|---------|
| Setup venv | 1 min | `python -m venv .venv` |
| Activate | 1 sec | `.\.venv\Scripts\Activate.ps1` |
| Install | 5 min | `python -m pip install -r requirements-ml.txt` |
| Verify | 1 sec | `python -c "import pandas, sklearn, lightgbm, shap; print('OK')"` |
| Train | 10 min | `python scripts\train_lightgbm.py` |
| **Total** | **~17 min** | (one-time setup, training only ~10 min on repeat) |

---

**For questions or issues, check the troubleshooting section above or review console output for specific error messages.**
