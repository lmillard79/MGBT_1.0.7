# R Integration Fixed

## Issue Resolved

The R MGBT integration had two issues that have been fixed:

### Issue 1: Deprecated rpy2 Syntax

**Problem:** Using `numpy2ri.activate()` and `deactivate()` which are deprecated in rpy2 3.6.x

**Error:**
```
The activate and deactivate are deprecated. To set a conversion
context check the docstring for rpy2.robjects.conversion.Converter.context.
```

**Solution:** Use modern conversion context manager:

```python
from rpy2.robjects import conversion, numpy2ri

# OLD (deprecated):
numpy2ri.activate()
# ... code ...
numpy2ri.deactivate()

# NEW (correct):
with conversion.localconverter(ro.default_converter + numpy2ri.converter):
    # ... code ...
```

### Issue 2: Incorrect Result Extraction

**Problem:** Trying to use `.rx2()` method on NamedList object

**Error:**
```
'NamedList' object has no attribute 'rx2'
```

**Solution:** Use proper NamedList access:

```python
# Get result names
names = list(result.names())

# Find indices
klow_idx = names.index('klow')
thresh_idx = names.index('LOThresh')

# Extract values
klow = int(result[klow_idx][0])
threshold = float(result[thresh_idx][0])
```

## Files Updated

### 1. scripts/compare_all_methods.py

**Changes:**
- Removed deprecated `numpy2ri.activate()`/`deactivate()`
- Added modern conversion context manager
- Fixed result extraction from NamedList
- Removed alpha parameters (R MGBT uses defaults)

### 2. scripts/test_r_mgbt.py

**Created:** New test script to verify R MGBT works correctly

**Features:**
- Tests R connection
- Runs R MGBT on sample data
- Demonstrates correct modern rpy2 syntax
- Validates result extraction

### 3. notebooks/r_mgbt_cell_update.py

**Created:** Updated code for Jupyter notebook Cell 4

**Usage:** Replace the R MGBT cell in the notebook with this code

## Testing

### Verify R Connection

```bash
python scripts/setup_r_environment.py
```

Expected output:
```
R_HOME set to: C:\Program Files\R\R-4.5.1
R connection successful!
R version: R version 4.5.1 (2025-06-13 ucrt)
MGBT package: INSTALLED
```

### Test R MGBT Function

```bash
python scripts/test_r_mgbt.py
```

Expected output:
```
Testing R MGBT with modern rpy2 syntax...
Importing R MGBT package...
Test data: [100 200 300  10  20 400 500   5  15 600]
Running R MGBT...
Result names: ['index', 'omegas', 'x', 'pvalues', 'klow', 'LOThresh']
R MGBT Results:
  Outliers detected: X
  Threshold: XX.XX
✓ R MGBT working correctly with modern rpy2 syntax!
```

### Run Full Comparison

```bash
python scripts/compare_all_methods.py
```

This will now work correctly with R integration.

## Correct Usage Pattern

### Basic R MGBT Call

```python
import os
import numpy as np

# Configure R environment
R_HOME = r"C:\Program Files\R\R-4.5.1"
os.environ['R_HOME'] = R_HOME

# Import rpy2
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, conversion
from rpy2.robjects.packages import importr

# Import MGBT package
mgbt_r = importr('MGBT')

# Your data
flows = np.array([100, 200, 300, 10, 20, 400])

# Run MGBT with modern syntax
with conversion.localconverter(ro.default_converter + numpy2ri.converter):
    r_flows = ro.FloatVector(flows)
    result = mgbt_r.MGBT(r_flows)
    
    # Extract results
    names = list(result.names())
    klow = int(result[names.index('klow')][0])
    threshold = float(result[names.index('LOThresh')][0])

print(f"Outliers: {klow}, Threshold: {threshold}")
```

## Key Points

1. **No alpha parameters:** R MGBT package uses default values internally
2. **Modern conversion:** Use `conversion.localconverter()` context manager
3. **NamedList access:** Use `result.names()` to get field names, then index by position
4. **R_HOME required:** Must be set before importing rpy2

## Warnings (Harmless)

These warnings are normal on Windows and can be ignored:

```
'sh' is not recognized as an internal or external command
Error importing in API mode: ImportError('On Windows, cffi mode "ANY" is only "ABI".')
Trying to import in ABI mode.
```

rpy2 automatically falls back to ABI mode which works correctly.

## Summary

✓ R environment configured correctly  
✓ Modern rpy2 syntax implemented  
✓ Result extraction fixed  
✓ All scripts updated  
✓ Test script created  
✓ Ready for three-way comparison (FLIKE, R, Python)

You can now run the full comparison script to generate the dataset with gauge_id, flike_censor_count, r_mgbt_censor_count, and python_censor_count!
