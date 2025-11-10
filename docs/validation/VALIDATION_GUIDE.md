# MGBT Validation Guide

## Overview

This guide explains how to validate the Python MGBT implementation against FLIKE and R MGBT.

## Three Validation Methods

### 1. Interactive Jupyter Notebook (Recommended for Learning)

**File:** `notebooks/MGBT_Demonstration.ipynb`

**Best for:**
- Understanding how MGBT works
- Exploring individual stations
- Visual comparison of results
- Learning the algorithm

**Usage:**
```bash
cd notebooks
jupyter notebook
# Open MGBT_Demonstration.ipynb
```

**Features:**
- Load any FLIKE file
- See censored flows from FLIKE
- Run Python MGBT
- Compare with R MGBT (optional)
- Visualize results with plots
- Interactive station selection

### 2. Comprehensive Comparison Script

**File:** `scripts/compare_all_methods.py`

**Best for:**
- Validating all stations at once
- Generating comparison datasets
- Production validation
- Automated testing

**Usage:**
```bash
# Python vs FLIKE only (no R required)
python scripts/compare_all_methods.py --no-r

# Include R comparison (requires rpy2 and R MGBT)
python scripts/compare_all_methods.py
```

**Output:**
- `data/test_results/mgbt_three_way_comparison.csv` - Detailed results
- `data/test_results/mgbt_three_way_summary.txt` - Summary report

**CSV Columns:**
- `station_id` - Gauge identifier
- `n_flows` - Total number of flows
- `flike_censored` - FLIKE censored count (reference)
- `python_censored` - Python MGBT censored count
- `r_censored` - R MGBT censored count (if available)
- `py_matches_flike` - Boolean: Python matches FLIKE
- `r_matches_flike` - Boolean: R matches FLIKE
- `py_matches_r` - Boolean: Python matches R

### 3. Quick Test Script

**File:** `scripts/quick_test.py`

**Best for:**
- Rapid validation
- Debugging
- Performance testing
- Quick checks

**Usage:**
```bash
python scripts/quick_test.py
```

## Validation Workflow

### Step 1: Extract Validation Data

```bash
python scripts/extract_validation_data.py
```

This creates the validation database from FLIKE files.

### Step 2: Quick Validation

```bash
python scripts/quick_test.py
```

Tests a single station to verify implementations work.

### Step 3: Interactive Exploration (Optional)

```bash
cd notebooks
jupyter notebook
# Open MGBT_Demonstration.ipynb
# Select different stations to explore
```

### Step 4: Full Validation

```bash
python scripts/compare_all_methods.py --no-r
```

Runs all stations and generates comprehensive comparison.

### Step 5: Review Results

```bash
# View summary
cat data/test_results/mgbt_three_way_summary.txt

# View detailed results in Excel/pandas
# Open: data/test_results/mgbt_three_way_comparison.csv
```

## Understanding Results

### Expected Outcome

For a correctly implemented MGBT:
- **Python censored** should match **FLIKE censored**
- **R censored** should match **FLIKE censored**
- **Python censored** should match **R censored**

### Discrepancies

If counts don't match, possible causes:

1. **P-value calculation accuracy** - Integration tolerance issues
2. **Algorithm differences** - Subtle implementation variations
3. **Numerical precision** - Floating point differences
4. **Data interpretation** - How flows are sorted/processed

### Investigating Discrepancies

Use the Jupyter notebook to:
1. Load the problematic station
2. Examine the flow data
3. Check p-values at each position
4. Compare thresholds
5. Visualize outlier selection

## Example Validation Session

### Using Jupyter Notebook

```python
# Load station with known results
flike_data = load_flike_file('416040')
print(f"FLIKE censored: {flike_data['n_censored']}")  # Expected: 16

# Run Python MGBT
py_result = MGBT_optimized(flike_data['flows'])
print(f"Python censored: {py_result.klow}")  # Should be: 16

# Check match
if py_result.klow == flike_data['n_censored']:
    print("VALIDATION PASSED")
else:
    print(f"DISCREPANCY: {py_result.klow - flike_data['n_censored']:+d}")
```

### Using Comparison Script

```bash
# Run full comparison
python scripts/compare_all_methods.py --no-r

# Check summary
cat data/test_results/mgbt_three_way_summary.txt
```

Expected output:
```
Total stations: 28
Python successful: 28/28
Python matches FLIKE: 28/28 (100.0%)
```

## Validation Metrics

### Success Criteria

- **Implementation Success:** All stations process without errors
- **Accuracy:** >95% match rate with FLIKE
- **Performance:** <5 minutes for all 28 stations (parallel)
- **Consistency:** Python and R produce identical results

### Current Status

Based on initial testing:
- **Stations tested:** 28
- **Python success rate:** 100%
- **Match rate:** Under investigation
- **Performance:** ~20 minutes (parallel, 4 workers)

## Troubleshooting

### Issue: Different outlier counts

**Investigation steps:**
1. Load station in Jupyter notebook
2. Check flow data distribution
3. Examine p-values near threshold
4. Compare test statistics
5. Check for edge cases (ties, zeros, etc.)

### Issue: Performance too slow

**Solutions:**
- Use optimized implementation (default)
- Enable parallel processing
- Increase worker count
- Use caching (enabled by default)

### Issue: R comparison fails

**Note:** R comparison is optional. Python vs FLIKE is sufficient.

**To enable R:**
```bash
pip install rpy2
# Install R and MGBT package
```

## Validation Checklist

- [ ] Extract validation data
- [ ] Run quick test
- [ ] Explore 3-5 stations in notebook
- [ ] Run full comparison script
- [ ] Review summary report
- [ ] Investigate any discrepancies
- [ ] Document findings
- [ ] Verify performance metrics

## Reference Data

### FLIKE Output Files

**Location:** `UnitTests/flike_Bayes_*.txt`

**Contains:**
- Gauged annual maximum discharge
- Censored flow list
- Flood frequency model
- Zero flow threshold

### Validation Database

**Location:** `data/validation/`

**Contains:**
- Extracted flow data
- Expected censored counts
- Station metadata
- Summary statistics

## Next Steps

After validation:

1. **If all match:** Python implementation is validated
2. **If discrepancies:** Investigate using notebook
3. **If systematic bias:** Review algorithm implementation
4. **If random errors:** Check numerical precision

## Support Files

- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `scripts/README.md` - Testing framework guide
- `notebooks/README.md` - Notebook usage guide
- Test logs in `data/test_results/`

## References

- USGS FLIKE Software
- Cohn et al. (2013) - MGBT Paper
- USGS Bulletin 17C
- R MGBT Package Documentation
