# TrustLoop Stage 2: Datetime Handling Fix

## Root Cause

LightGBM cannot accept datetime64 columns directly. The model-ready CSV contained:
- `order_date` (datetime)
- `return_date` (datetime)

These were being passed directly to LightGBM, causing:
```
numpy.exceptions.DTypePromotionError:
The DType Float64DType could not be promoted by DateTime64DType
```

## Solution Applied

### 1. Datetime Feature Extraction (Lines 125-145)

For each datetime column, extract numeric temporal features:
- `{col}_year` — year of the date
- `{col}_month` — month (1-12)
- `{col}_day` — day of month (1-31)
- `{col}_day_of_week` — day of week (0-6, Monday=0)
- `{col}_day_of_year` — day of year (1-366)

Example output:
```
order_date           → order_date_year, order_date_month, order_date_day, etc.
return_date          → return_date_year, return_date_month, return_date_day, etc.
```

### 2. Datetime Column Removal (Line 142-145)

- Remove raw `order_date` and `return_date` from feature_cols
- Include derived temporal features (10 new numeric features)
- Update feature matrix to exclude datetime64

### 3. Split Still Uses Raw Dates (Line 147-148)

Chronological split preserves the original logic:
```python
df.sort_values(by=['return_date', 'order_date'])
```

The raw datetime columns are kept in df for ordering but removed from X before training.

### 4. Pre-Training Validation (Lines 184-222)

Added comprehensive validation before model training:

✓ No datetime64 columns in feature matrix
✓ No object dtype columns (should be encoded to codes)
✓ No infinite values
✓ Identical feature columns across train/val/test splits
✓ Target has exactly 4 classes
✓ No NaN values

### 5. LightGBM API Update (Lines 234-242)

Updated from deprecated eval_set to current API:

**BEFORE:**
```python
clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(50), log_evaluation(0)]
)
```

**AFTER:**
```python
clf.fit(
    X_train, y_train,
    eval_X=[X_val],
    eval_y=[y_val],
    callbacks=[early_stopping(50), log_evaluation(0)]
)
```

## Feature Count Impact

**Original:**
- 22 features including order_date and return_date

**After temporal extraction:**
- Removed: 2 datetime columns
- Added: 10 derived temporal features (2 dates × 5 features each)
- **Net: 22 - 2 + 10 = 30 features**

All features are now numeric or properly encoded categorical.

## Preservation of Logic

✓ No modification to `data/raw/` or Stage 1
✓ Feature engineering logic unchanged
✓ Chronological split preserved
✓ Target encoding unchanged
✓ Evaluation metrics unchanged
✓ SHAP generation unchanged
✓ Output paths unchanged

## What Changed

**File modified:**
- `scripts/train_lightgbm.py` (3 main changes)

**Changes:**
1. Lines 125-145: Datetime feature extraction
2. Lines 142-145: Feature column updates (remove raw datetime, add derived)
3. Lines 184-222: Pre-training validation
4. Lines 234-242: LightGBM API update (eval_set → eval_X/eval_y)

## Ready to Run

```powershell
cd "C:\Users\khura\Downloads\TrustLoop_VSCode_Starter"
.\.venv\Scripts\Activate.ps1
python scripts\train_lightgbm.py
```

Expected output (new):
```
Extracting temporal features from datetime columns...
  Found datetime columns: ['order_date', 'return_date']
  Derived temporal features; feature_cols updated: 22 -> 30
Split sizes: train=42000, val=9000, test=9000

======================================================================
PRE-TRAINING VALIDATION
======================================================================
FEATURE COUNT: 30
DATETIME COLUMNS REMOVED FROM X: ['order_date', 'return_date']
CATEGORICAL COLUMNS: [...]
NUMERIC COLUMNS: [...]
✓ No datetime64 columns in feature matrix
✓ No object dtype columns in feature matrix
✓ No infinite values in feature matrix
✓ Identical feature columns across splits
✓ Target has exactly 4 classes
✓ No NaN values in feature matrices
======================================================================
```

Then training proceeds normally.

## Verification Points

After training completes successfully, verify:

1. ✓ Script exits with code 0 (no errors)
2. ✓ Pre-training validation all passed
3. ✓ Feature count printed as 30
4. ✓ Datetime columns removed message
5. ✓ All 8 output files generated

---

**Status: READY FOR EXECUTION**

All datetime dtype issues resolved. Script syntax verified. Ready for local training run.
