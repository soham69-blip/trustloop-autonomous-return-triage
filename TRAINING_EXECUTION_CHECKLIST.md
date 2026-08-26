# TrustLoop Stage 2 — LightGBM Training: Final Checklist

## Pre-Execution Checklist

Before running `python scripts\train_lightgbm.py`:

- [ ] Virtual environment created: `.\.venv\Scripts\Activate.ps1` works
- [ ] ML packages installed: `python -m pip install -r requirements-ml.txt` completed
- [ ] Stage 1 artifacts exist: `data/processed/trustloop/model_ready.csv` (60,000 rows)
- [ ] Script syntax verified: `python -m py_compile .\scripts\train_lightgbm.py` passed
- [ ] Current directory: `C:\Users\khura\Downloads\TrustLoop_VSCode_Starter`
- [ ] Sufficient disk space: ~100 MB for models/ and reports/
- [ ] Sufficient RAM: 4+ GB available

## Execution Command

```powershell
cd "C:\Users\khura\Downloads\TrustLoop_VSCode_Starter"
.\.venv\Scripts\Activate.ps1
python scripts\train_lightgbm.py
```

## Expected Console Output (In Order)

### Phase 1: Loading & Validation (30 seconds)

```
Loading dataset: C:\...\data\processed\trustloop\model_ready.csv
Found NaNs in feature matrix; filling with median for numeric and "unknown" for categorical
Extracting temporal features from datetime columns...
  Found datetime columns: ['order_date', 'return_date']
  Derived temporal features; feature_cols updated: 22 -> 30
Split sizes: train=42000, val=9000, test=9000
```

**CHECK: Feature count changed from 22 to 30 ✓**

### Phase 2: Pre-Training Validation (5 seconds)

```
======================================================================
PRE-TRAINING VALIDATION
======================================================================
FEATURE COUNT: 30
DATETIME COLUMNS REMOVED FROM X: ['order_date', 'return_date']
CATEGORICAL COLUMNS: ['country', 'customer_segment', 'device_type', 'payment_method', 'platform', 'product_category', 'return_reason', 'shipping_carrier']
NUMERIC COLUMNS: 22
✓ No datetime64 columns in feature matrix
✓ No object dtype columns in feature matrix
✓ No infinite values in feature matrix
✓ Identical feature columns across splits
✓ Target has exactly 4 classes
✓ No NaN values in feature matrices
======================================================================
```

**CHECK: All validation checks passed ✓**

### Phase 3: Model Training (2-3 minutes)

```
Training LightGBM multiclass, num_class=4
[LightGBM training progress messages]
```

### Phase 4: SHAP Computation (5-8 minutes)

```
Computing SHAP values (may take time)...
Saving SHAP values...
```

### Phase 5: Output Saving (1-2 minutes)

```
Saving model and encoder...
  Model saved: C:\...\models\lightgbm_model.pkl
  Label encoder saved: C:\...\models\label_encoder.pkl
Writing documentation...
  Documentation saved: C:\...\docs\lightgbm_baseline.md
```

### Phase 6: Final Summary

```
===============================================================================
TRAINING COMPLETE
===============================================================================

Metrics (test set):
  accuracy: 0.7500-0.8500
  precision_macro: 0.7000-0.8000
  recall_macro: 0.7000-0.8000
  f1_macro: 0.7000-0.8000
  f1_weighted: 0.7500-0.8500

All artifacts written to:
  - models/
  - reports/
  - docs/lightgbm_baseline.md
```

**CHECK: Script exits normally (exit code 0) ✓**

## File Existence Verification

After training completes, verify all 8 files were created:

```powershell
# Models
Test-Path .\models\lightgbm_model.pkl        # Should be True (2-5 MB)
Test-Path .\models\label_encoder.pkl         # Should be True (< 1 KB)

# Reports
Test-Path .\reports\metrics.json             # Should be True
Test-Path .\reports\classification_report.csv
Test-Path .\reports\confusion_matrix.csv
Test-Path .\reports\feature_importance.csv
Test-Path .\reports\shap_values.parquet      # Should be True (10-50 MB)
Test-Path .\reports\shap_summary.csv         # Should be True

# Documentation
Test-Path .\docs\lightgbm_baseline.md        # Should be True
```

All should return `True`.

## Post-Training Verification

### 1. Check Metrics

```powershell
cat .\reports\metrics.json | ConvertFrom-Json | Format-Table
```

Expected: accuracy between 0.75-0.85 for test set

### 2. Check Feature Importance

```powershell
cat .\reports\feature_importance.csv | Select-Object -First 10
```

Top features should include temporal features derived from order_date/return_date

### 3. Check Training Summary

```powershell
cat .\docs\lightgbm_baseline.md
```

Should describe:
- 30 features (derived from original 22)
- Temporal features extracted from dates
- 4-class multiclass classification
- Chronological split

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'sklearn'`

**Fix:** Virtual environment not activated or pip install incomplete
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ml.txt
python scripts\train_lightgbm.py
```

### Error: `FileNotFoundError: ...model_ready.csv`

**Fix:** Stage 1 not completed
```powershell
python scripts\build_trustloop_features.py
python scripts\train_lightgbm.py
```

### Error: `ERROR: Datetime columns found in X_train`

**Fix:** Temporal feature extraction failed
- Check `data/processed/trustloop/model_ready.csv` has order_date/return_date columns
- Verify pandas version supports `.dt` accessor
- Try: `python -c "import pandas as pd; print(pd.__version__)"`

### Error: `ERROR: Expected 4 classes in y_train`

**Fix:** Target encoding issue
- Do NOT modify `data/raw/`
- Do NOT rebuild Stage 1 without understanding the issue
- Check: abuse_label values in model_ready.csv are {0, 1, 2, 3}

### Script hangs (> 30 minutes)

**Likely cause:** SHAP computation on large dataset
- This is normal; SHAP can take 5-15 minutes
- Wait for completion
- For testing, you can temporarily reduce X_train sample size in script

### Out of Memory error

**Fix:** Insufficient RAM
- Close other applications
- Minimum 4 GB required; 8 GB recommended
- If still fails, reduce sample size temporarily for testing

## Success Criteria

✅ Training is successful if ALL of these are true:

1. Script exits with code 0 (no errors)
2. PRE-TRAINING VALIDATION section shows all 6 checks passed
3. FEATURE COUNT printed as 30
4. DATETIME COLUMNS REMOVED shows ['order_date', 'return_date']
5. "TRAINING COMPLETE" message appears
6. Test set accuracy is printed (expect 0.75-0.85)
7. All 8 output files exist and are non-empty
8. No unhandled exceptions in console output

## After Successful Training

1. Review metrics: `cat .\reports\metrics.json`
2. Review feature importance: `cat .\reports\feature_importance.csv`
3. Optional: Review SHAP: `cat .\reports\shap_summary.csv`
4. Read summary: `cat .\docs\lightgbm_baseline.md`
5. Archive output files for later reference
6. Send models/ and reports/ to next stage

## Important Notes

- ✓ Do NOT modify data/raw/
- ✓ Do NOT rebuild Stage 1
- ✓ Do NOT change feature_cols manually
- ✓ Do NOT delete model_ready.csv
- ✓ This is a one-time training run; outputs will be overwritten if run again
- ✓ For reproducibility, keep random_state=42 and use same input data

## Timeline

| Phase | Expected Time | Notes |
|-------|----------------|-------|
| Setup (venv, pip) | 8-10 min | One-time (if not done) |
| Data load & validation | 30 sec | Fast |
| Temporal feature extraction | 10 sec | New in this version |
| Model training | 2-3 min | LightGBM is efficient |
| SHAP computation | 5-8 min | Most time-consuming |
| I/O & metrics | 1-2 min | File writing |
| **TOTAL** | **10-15 min** | (if setup already done) |

---

**You are ready to train. Run the command above and monitor the console output for success indicators.**
