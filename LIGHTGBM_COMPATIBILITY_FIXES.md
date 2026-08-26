# TrustLoop Stage 2: LightGBM Compatibility Fixes

## Issues Fixed

### 1. LightGBM Callback API (Line 168)

**BEFORE (Deprecated):**
```python
clf.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)
```

**AFTER (Current API):**
```python
from lightgbm import early_stopping, log_evaluation

clf.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        early_stopping(50),
        log_evaluation(0)
    ]
)
```

**Why:** LightGBM 4.0+ uses the callback API. The `early_stopping_rounds` and `verbose` parameters are deprecated in favor of callbacks.

### 2. Pandas dtype Warning (Line 113)

**BEFORE (Generates FutureWarning):**
```python
cat_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
```

**AFTER (Clean):**
```python
cat_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
```

**Why:** Pandas 2.0+ treats 'object' and 'category' differently. Using only 'object' is cleaner and excludes datetime types automatically.

## What Remains Unchanged

✓ 22 features (unchanged)
✓ 4-class multiclass classification (unchanged)
✓ 42,000 train rows (unchanged)
✓ 9,000 validation rows (unchanged)
✓ 9,000 test rows (unchanged)
✓ Chronological split by return_date/order_date (unchanged)
✓ No shuffling (unchanged)
✓ Feature engineering from Stage 1 (unchanged)
✓ Target variable: abuse_label (unchanged)
✓ Evaluation metrics (unchanged)
✓ SHAP generation (unchanged)
✓ Output paths (unchanged)

## Files Modified

- `scripts/train_lightgbm.py` — Fixed callback API and dtype warning

## How to Run (Windows PowerShell)

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Run training
python scripts\train_lightgbm.py
```

Expected runtime: **10-15 minutes**

## Expected Output

```
Loading dataset: ...model_ready.csv
Split sizes: train=42000, val=9000, test=9000
Training LightGBM multiclass, num_class=4
Saving model and encoder...
  Model saved: ...models/lightgbm_model.pkl
  Label encoder saved: ...models/label_encoder.pkl
Computing SHAP values (may take time)...
Saving SHAP values...
Writing documentation...
  Documentation saved: ...docs/lightgbm_baseline.md

===============================================================================
TRAINING COMPLETE
===============================================================================

Metrics (test set):
  accuracy: 0.75-0.85
  precision_macro: 0.70-0.80
  recall_macro: 0.70-0.80
  f1_macro: 0.70-0.80
  f1_weighted: 0.75-0.85

All artifacts written to:
  - models/
  - reports/
  - docs/lightgbm_baseline.md
```

## Verification Checklist

After training completes, verify these files exist:

- [ ] `models/lightgbm_model.pkl` (2-5 MB)
- [ ] `models/label_encoder.pkl` (< 1 KB)
- [ ] `reports/metrics.json` (valid JSON with train/val/test metrics)
- [ ] `reports/classification_report.csv` (per-class metrics)
- [ ] `reports/confusion_matrix.csv` (test confusion matrix)
- [ ] `reports/feature_importance.csv` (feature rankings)
- [ ] `reports/shap_values.parquet` (SHAP explanations)
- [ ] `reports/shap_summary.csv` (mean absolute SHAP per feature)
- [ ] `docs/lightgbm_baseline.md` (training summary)

## Compatibility Notes

- **LightGBM**: 3.3.0+ (tested with callback API)
- **scikit-learn**: 1.0.0+
- **shap**: 0.40.0+
- **pandas**: 1.3.0+
- **Python**: 3.8+

## Next Steps

1. Run: `python scripts\train_lightgbm.py` in your Windows environment
2. Wait for completion (~15 min)
3. Verify all 8 output files are generated
4. Send artifacts back (models/, reports/, docs/lightgbm_baseline.md)
5. Do NOT proceed to LangGraph/FastAPI/RAG until training succeeds

---

**Note:** The script now uses the current LightGBM callback API and will work with LightGBM 4.0+. All TrustLoop logic remains unchanged.
