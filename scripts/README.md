# MGBT Testing and Validation Scripts

This directory contains utilities for testing and validating the Python MGBT implementation against the R package.

## Scripts

### 1. extract_validation_data.py

Extracts annual maxima series from FLIKE output files for validation testing.

**Usage:**
```bash
python scripts/extract_validation_data.py
```

**Output:**
- `data/validation/` - Validation database with 28 stations
- `data/validation/validation_summary.csv` - Summary of all stations
- Individual station directories with flow data and metadata

### 2. test_mgbt_comparison.py

Comprehensive testing framework with detailed logging and performance tracking.

**Usage:**
```bash
# Parallel execution (default, fastest)
python scripts/test_mgbt_comparison.py

# Sequential execution with DEBUG logging
python scripts/test_mgbt_comparison.py --no-parallel --log-level DEBUG

# Parallel with 8 workers
python scripts/test_mgbt_comparison.py --workers 8

# Show help
python scripts/test_mgbt_comparison.py --help
```

**Options:**
- `--parallel` / `--no-parallel` - Enable/disable parallel processing (default: enabled)
- `--workers N` - Number of parallel workers (default: CPU count)
- `--log-level LEVEL` - Console logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)

**Output:**
- `data/test_results/mgbt_comparison_results.csv` - Detailed results for all stations
- `data/test_results/mgbt_comparison_summary.txt` - Summary report
- `data/test_results/mgbt_test_YYYYMMDD_HHMMSS.log` - Full DEBUG log

### 3. quick_test.py

Quick validation test on a single station for rapid debugging.

**Usage:**
```bash
python scripts/quick_test.py
```

**Output:**
- Console output comparing corrected vs optimized implementations
- Performance timing
- Validation against expected results

## Workflow

### Initial Setup

1. Extract validation data:
```bash
python scripts/extract_validation_data.py
```

2. Run quick test to verify implementations:
```bash
python scripts/quick_test.py
```

### Full Validation

3. Run comprehensive tests (parallel):
```bash
python scripts/test_mgbt_comparison.py --parallel --workers 4
```

4. Review results:
```bash
# View summary
cat data/test_results/mgbt_comparison_summary.txt

# View detailed results
# Open data/test_results/mgbt_comparison_results.csv in Excel/pandas

# View full log
# Open data/test_results/mgbt_test_*.log
```

## Logging

### Log Levels

**DEBUG:**
- Detailed execution flow
- P-value calculations
- Test statistics
- Cache performance

**INFO:**
- Station processing
- Results summary
- Performance metrics
- Progress updates

**WARNING:**
- Result mismatches
- Numerical issues
- Outlier count warnings

**ERROR:**
- Calculation failures
- File errors
- Critical failures

### Log Files

**Console:** Configurable level (default: INFO)
**File:** Always DEBUG with full details
**Location:** `data/test_results/mgbt_test_YYYYMMDD_HHMMSS.log`

## Performance

### Optimization Features

1. **Vectorized Operations** - Batch computation where possible
2. **LRU Caching** - Cached p-value calculations
3. **Early Stopping** - Stops when no significance found
4. **Parallel Processing** - Multi-core execution
5. **Memory Optimization** - Pre-allocated arrays

### Benchmarks

**Single Station (416040, 44 flows):**
- Corrected: 332.5s
- Optimized: 172.4s
- Speedup: 1.93x

**Full Suite (28 stations):**
- Sequential (optimized): ~1.3 hours
- Parallel 4 workers: ~20 minutes

## Troubleshooting

### Issue: "Validation directory not found"

**Solution:** Run `extract_validation_data.py` first

### Issue: Slow performance

**Solutions:**
- Use `--parallel` flag
- Increase `--workers` count
- Use optimized implementation (default)

### Issue: "rpy2 not available"

**Note:** R comparison is optional. Python-only testing works without rpy2.

**To enable R comparison:**
```bash
pip install rpy2
# Install R and MGBT package
```

### Issue: Memory errors with parallel processing

**Solution:** Reduce worker count:
```bash
python scripts/test_mgbt_comparison.py --workers 2
```

## Output Files

### validation_summary.csv

Columns:
- `station_id` - Station identifier
- `n_gauged` - Number of gauged flows
- `n_censored` - Number of censored flows (expected outliers)
- `n_total` - Total flows
- `zero_threshold` - Zero flow threshold
- `flood_model` - Flood frequency model used

### mgbt_comparison_results.csv

Columns:
- `station_id` - Station identifier
- `n_flows` - Total number of flows
- `expected_censored` - Expected censored count from R
- `py_klow` - Python detected outliers
- `py_threshold` - Python outlier threshold
- `match_expected` - Boolean: Python matches expected
- `py_time` - Python execution time (seconds)
- `test_time` - Total test time (seconds)

### mgbt_comparison_summary.txt

Contains:
- Test statistics
- Success rate
- Performance metrics
- List of discrepancies
- R comparison (if available)

## Examples

### Example 1: Quick validation
```bash
# Extract data
python scripts/extract_validation_data.py

# Quick test
python scripts/quick_test.py
```

### Example 2: Full validation with detailed logging
```bash
python scripts/test_mgbt_comparison.py --no-parallel --log-level DEBUG
```

### Example 3: Fast parallel execution
```bash
python scripts/test_mgbt_comparison.py --parallel --workers 8 --log-level INFO
```

### Example 4: Review specific station
```python
import pandas as pd
import numpy as np
from pymgbt.core.mgbt_optimized import MGBT

# Load station data
flows = np.loadtxt('data/validation/416040/flows.txt')

# Run MGBT
result = MGBT(flows)

# View results
print(f"Outliers: {result.klow}")
print(f"Threshold: {result.low_outlier_threshold}")
print(f"P-values: {result.p_values[:10]}")  # First 10
```

## Implementation Details

### MGBT Implementations

1. **Original** (`pymgbt/core/mgbt.py`) - Initial implementation (incorrect)
2. **Corrected** (`pymgbt/core/mgbt_corrected.py`) - Exact R algorithm
3. **Optimized** (`pymgbt/core/mgbt_optimized.py`) - Performance-enhanced (default)

### Key Corrections from R

- Log10 transformation: `log10(pmax(1e-8, Q))`
- Two-level alpha: `alpha1=0.01`, `alpha10=0.10`
- Consecutive outlier logic
- Proper test statistic calculation
- Recursive retry for excessive outliers

## References

- R MGBT Package: CRAN
- Cohn et al. (2013): Water Resources Research, v. 49, no. 8
- USGS Bulletin 17C: Flood Flow Frequency Guidelines
