# R Environment Setup Guide for rpy2

## Overview

This guide explains how to configure rpy2 to use your R installation for MGBT validation.

## Your R Installation

**Location:** `C:\Program Files\R\R-4.5.1`

## Setup Steps

### 1. Set R_HOME Environment Variable

rpy2 needs to know where R is installed. You have two options:

#### Option A: Set in Python Script (Recommended)

Add this to the top of any script using rpy2:

```python
import os

# Set R_HOME
R_HOME = r"C:\Program Files\R\R-4.5.1"
os.environ['R_HOME'] = R_HOME

# Add R to PATH
R_BIN = os.path.join(R_HOME, 'bin', 'x64')
if R_BIN not in os.environ['PATH']:
    os.environ['PATH'] = R_BIN + os.pathsep + os.environ['PATH']
```

This is already configured in:
- `scripts/compare_all_methods.py`
- `scripts/setup_r_environment.py`
- `scripts/install_r_mgbt.py`

#### Option B: Set System Environment Variable (Permanent)

1. Open System Properties (Win + Pause/Break)
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "System variables", click "New"
5. Variable name: `R_HOME`
6. Variable value: `C:\Program Files\R\R-4.5.1`
7. Click OK
8. Restart your IDE/terminal

### 2. Verify R Connection

Run the test script:

```bash
python scripts/setup_r_environment.py
```

Expected output:
```
R_HOME set to: C:\Program Files\R\R-4.5.1
R binary path: C:\Program Files\R\R-4.5.1\bin\x64
R connection successful!
R version: R version 4.5.1 (2025-06-13 ucrt)
MGBT package: INSTALLED
```

### 3. Install MGBT Package in R

The MGBT package is now installed. If you need to reinstall:

```bash
python scripts/install_r_mgbt.py
```

Or manually in R console:
```r
install.packages("MGBT")
```

## Using R MGBT in Python

### Basic Example

```python
import os
import numpy as np

# Configure R
R_HOME = r"C:\Program Files\R\R-4.5.1"
os.environ['R_HOME'] = R_HOME

# Import rpy2
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

# Activate numpy conversion
numpy2ri.activate()

# Import MGBT package
mgbt_r = importr('MGBT')

# Your flow data
flows = np.array([100, 200, 300, 10, 20, 400, 500])

# Convert to R vector
r_flows = ro.FloatVector(flows)

# Run MGBT
result = mgbt_r.MGBT(r_flows, alpha1=0.01, alpha10=0.10)

# Extract results
klow = int(result.rx2('klow')[0])
threshold = float(result.rx2('LOThresh')[0])

print(f"Outliers detected: {klow}")
print(f"Threshold: {threshold}")

numpy2ri.deactivate()
```

### In Jupyter Notebook

The notebook `notebooks/MGBT_Demonstration.ipynb` is already configured. Just run:

```bash
cd notebooks
jupyter notebook
# Open MGBT_Demonstration.ipynb
# Run the R comparison cells
```

### In Comparison Script

Now that R is configured, you can run the full comparison:

```bash
# With R comparison (no --no-r flag)
python scripts/compare_all_methods.py
```

This will compare:
- FLIKE censored counts
- R MGBT censored counts
- Python MGBT censored counts

## Troubleshooting

### Issue: "R_HOME not found"

**Solution:** Verify R installation path:
```bash
dir "C:\Program Files\R\R-4.5.1"
```

If R is in a different location, update `R_HOME` in the scripts.

### Issue: "MGBT package not found"

**Solution:** Install MGBT:
```bash
python scripts/install_r_mgbt.py
```

Or in R console:
```r
install.packages("MGBT")
```

### Issue: "sh not recognized" warning

This is a harmless warning on Windows. rpy2 falls back to ABI mode automatically.

### Issue: rpy2 import fails

**Solution:** Reinstall rpy2:
```bash
pip uninstall rpy2
pip install rpy2
```

### Issue: Different R version

If you have a different R version, update the path:
```python
R_HOME = r"C:\Program Files\R\R-X.Y.Z"  # Your version
```

## Files Modified

The following files now include R configuration:

1. **scripts/compare_all_methods.py** - Main comparison script
2. **scripts/setup_r_environment.py** - Test R connection
3. **scripts/install_r_mgbt.py** - Install MGBT package

## Testing R Integration

### Quick Test

```bash
python scripts/setup_r_environment.py
```

Should show:
- R connection successful
- R version
- MGBT package status

### Full Comparison Test

```bash
python scripts/compare_all_methods.py
```

This will:
1. Load all FLIKE files
2. Run Python MGBT
3. Run R MGBT
4. Compare all three methods
5. Generate comparison CSV

## Expected Output

After running the comparison, you'll get:

**File:** `data/test_results/mgbt_three_way_comparison.csv`

**Columns:**
- `station_id`
- `flike_censored` - Reference from FLIKE
- `r_censored` - R MGBT results
- `python_censored` - Python MGBT results
- `r_matches_flike` - Boolean
- `py_matches_flike` - Boolean
- `py_matches_r` - Boolean

## Summary

Your R environment is now configured:

- ✓ R installed at: `C:\Program Files\R\R-4.5.1`
- ✓ rpy2 installed and working
- ✓ MGBT package installed in R
- ✓ Scripts configured with R_HOME
- ✓ Ready for three-way comparison

You can now run the full validation comparing FLIKE, R MGBT, and Python MGBT!
