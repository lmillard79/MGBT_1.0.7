# MGBT Performance Analysis

## Overview

This document analyzes the performance difference between:
1. **Pure Python MGBT** (pymgbt)
2. **R MGBT via rpy2** (R package called from Python)

## Benchmark Script

**Location:** `scripts/benchmark_performance.py`

**What it does:**
- Loads all FLIKE unit test files
- Runs each dataset through both Python and R implementations
- Times each execution with high precision (`time.perf_counter()`)
- Calculates speedup factors
- Provides detailed statistics and extrapolations

## Usage

```bash
# Test both Python and R
python scripts/benchmark_performance.py

# Test Python only
python scripts/benchmark_performance.py --python-only

# Test R only
python scripts/benchmark_performance.py --r-only
```

## Expected Results

### Performance Factors

**Python Advantages:**
- No subprocess overhead
- No data conversion (Python → R → Python)
- Optimized NumPy/SciPy operations
- No rpy2 marshalling costs
- Direct memory access

**R via rpy2 Overhead:**
- Subprocess communication
- Data type conversion (numpy → R vectors)
- Result extraction and conversion back
- rpy2 wrapper overhead
- R environment initialization (one-time)

### Typical Speedup

Based on similar implementations:
- **Small datasets (n<30):** 2-5x faster (overhead dominates)
- **Medium datasets (n=30-100):** 3-10x faster
- **Large datasets (n>100):** 5-15x faster (computation dominates)

## Benchmark Output

### Summary Statistics

```
Python MGBT:
  Total time: X.XXXs
  Mean time:  XX.XXms per station
  Median time: XX.XXms per station

R MGBT (via rpy2):
  Total time: X.XXXs
  Mean time:  XX.XXms per station
  Median time: XX.XXms per station

Performance Comparison:
  Mean speedup:   X.XXx
  Median speedup: X.XXx
  Time saved:     X.XXXs (XX.X%)
```

### Extrapolation for Large Projects

The benchmark provides extrapolations for larger projects:

```
Extrapolation for larger projects:
  100 stations: Python Xs vs R Xs (save Xs = Xmin)
  500 stations: Python Xs vs R Xs (save Xs = Xmin)
  1000 stations: Python Xs vs R Xs (save Xs = Xmin)
  5000 stations: Python Xs vs R Xs (save Xs = Xmin)
```

### Performance by Dataset Size

```
Size       Count    Python(ms)   R(ms)        Speedup
----------------------------------------------------------
<20        X        XX.XX        XX.XX        X.XXx
20-40      X        XX.XX        XX.XX        X.XXx
40-60      X        XX.XX        XX.XX        X.XXx
60-100     X        XX.XX        XX.XX        X.XXx
100+       X        XX.XX        XX.XX        X.XXx
```

## Real-World Impact

### For Your Current Project

If you're processing many stations with R via rpy2:

**Example: 1000 stations**
- R via rpy2: ~X minutes
- Pure Python: ~X minutes
- **Time saved: ~X minutes per run**

### Development Workflow

**Iterative testing:**
- Testing 10 stations repeatedly during development
- R: X seconds per iteration
- Python: X seconds per iteration
- **Faster feedback loop = faster development**

### Production Processing

**Large-scale analysis:**
- Processing entire catchment databases
- Batch processing for reports
- Automated workflows

**Benefits:**
- Reduced processing time
- Lower computational costs
- Faster turnaround for clients
- More responsive applications

## Memory Usage

**Python MGBT:**
- Lower memory footprint
- No R environment overhead
- Direct NumPy arrays

**R via rpy2:**
- R environment memory
- Duplicate data in both Python and R
- rpy2 conversion buffers

## Recommendations

### When to Use Python MGBT

✓ **Production workflows** - Maximum performance
✓ **Large datasets** - Significant speedup
✓ **Batch processing** - Cumulative time savings
✓ **Embedded applications** - No R dependency
✓ **Cloud deployment** - Simpler dependencies

### When R Might Be Acceptable

- **One-off analyses** - Performance less critical
- **R-heavy workflows** - Already using R extensively
- **Legacy code** - Existing R scripts
- **Validation** - Cross-checking against R reference

## Migration Strategy

### From R to Python

1. **Validate first** - Ensure Python matches R results
2. **Benchmark** - Measure actual speedup for your data
3. **Gradual migration** - Start with new code
4. **Keep R for validation** - Use for cross-checking

### Hybrid Approach

- Use Python for production
- Keep R for validation/verification
- Best of both worlds

## Optimization Notes

### Python MGBT Optimizations

The `mgbt_optimized.py` implementation includes:
- Vectorized operations
- Cached p-value computations
- Early stopping conditions
- Efficient memory usage
- NumPy/SciPy optimizations

### Further Optimization Potential

- Numba JIT compilation
- Cython for critical loops
- Parallel processing for multiple stations
- GPU acceleration (for very large datasets)

## Conclusion

**Expected Outcome:**
Pure Python MGBT should be **3-10x faster** than R via rpy2 for typical flood frequency datasets.

**Key Benefits:**
- ✓ Faster processing
- ✓ Lower memory usage
- ✓ Simpler deployment
- ✓ No R dependency
- ✓ Better integration with Python workflows

**Trade-offs:**
- Need to validate against R
- Different numerical precision (minor)
- Maintain Python implementation

## Results

After running the benchmark, results are saved to:
- `data/test_results/performance_benchmark.csv`

This file contains detailed timing for each station, allowing you to:
- Analyze performance patterns
- Identify outliers
- Understand speedup distribution
- Make informed decisions about migration

## Next Steps

1. **Run the benchmark** on your actual data
2. **Review the results** - Check speedup factors
3. **Extrapolate** to your project size
4. **Calculate ROI** - Time saved vs migration effort
5. **Decide** - Pure Python or hybrid approach
