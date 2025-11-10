# MGBT Demonstration Notebooks

This directory contains Jupyter notebooks for interactive demonstration and validation of the Python MGBT implementation.

## Notebooks

### MGBT_Demonstration.ipynb

Interactive demonstration notebook that allows you to:

1. **Load FLIKE output files** - Select any station from the UnitTests directory
2. **View censored flows** - See which flows FLIKE identified as outliers
3. **Run Python MGBT** - Apply the Python implementation to the same data
4. **Compare with R MGBT** - Optional comparison with R package (requires rpy2)
5. **Visualize results** - Plot flows with outliers highlighted
6. **Validate across stations** - Test multiple stations and compare results

## Getting Started

### Prerequisites

```bash
# Install Jupyter
pip install jupyter notebook

# Install required packages
pip install numpy pandas matplotlib seaborn

# Optional: For R comparison
pip install rpy2
# Also need R and MGBT package installed
```

### Launch Notebook

```bash
# Navigate to notebooks directory
cd d:\GitRepos\MGBT_1.0.7\notebooks

# Start Jupyter
jupyter notebook

# Open MGBT_Demonstration.ipynb in your browser
```

## Notebook Features

### 1. Interactive Station Selection

Choose any station from the 28 available stations to analyze:

```python
STATION_ID = '416040'  # Change to any available station
```

### 2. Three-Way Comparison

Compare results from:
- **FLIKE** - Reference values from USGS software
- **R MGBT** - R package implementation
- **Python MGBT** - Python implementation

### 3. Visualization

Automatic plots showing:
- Sorted flow data with outliers highlighted
- P-value progression through MGBT algorithm
- Threshold lines for significance levels

### 4. Batch Testing

Test multiple stations at once:

```python
test_stations = ['416040', '416050', '416060']
# Run comparison across all selected stations
```

## Example Usage

### Basic Analysis

```python
# Load FLIKE data
flike_data = load_flike_file('416040')

# Run Python MGBT
result = MGBT_optimized(flike_data['flows'])

# Compare
print(f"FLIKE censored: {flike_data['n_censored']}")
print(f"Python censored: {result.klow}")
print(f"Match: {result.klow == flike_data['n_censored']}")
```

### With Visualization

```python
# Analyze station and plot
data, result = analyze_station('416040')

# Creates automatic plots showing:
# - Flow data with outliers
# - P-value progression
# - Comparison statistics
```

## Understanding the Results

### FLIKE Censored Count

Number of flows identified as low outliers by USGS FLIKE software. This is the **reference/expected** value.

### Python Censored Count

Number of flows identified as low outliers by Python MGBT implementation using:
- `alpha1 = 0.01` (primary significance level)
- `alpha10 = 0.10` (secondary significance level for consecutive outliers)

### R Censored Count

Number of flows identified by R MGBT package (if available). Should match Python if implementation is correct.

### Match Status

- **YES** - Python matches FLIKE (validation successful)
- **NO** - Discrepancy detected (requires investigation)

## Validation Workflow

1. **Select a station** with known censored flows
2. **Load FLIKE data** to see expected results
3. **Run Python MGBT** to get Python results
4. **Compare results** - check if counts match
5. **Visualize** - inspect which flows are flagged
6. **Investigate discrepancies** - if counts don't match

## Common Stations for Testing

### Station 416040
- Total flows: 44
- FLIKE censored: 16 (all zero flows)
- Good test case with clear outliers

### Station 416050
- Total flows: 29
- FLIKE censored: 0
- Good test case with no outliers

### Station 416060
- Total flows: 72
- FLIKE censored: 7
- Good test case with moderate outliers

## Troubleshooting

### Issue: "FLIKE file not found"

**Solution:** Ensure you're running from the notebooks directory and UnitTests directory exists:

```python
repo_root = Path.cwd().parent
print(repo_root)  # Should be d:\GitRepos\MGBT_1.0.7
```

### Issue: "rpy2 not available"

**Note:** R comparison is optional. The notebook works fine without it for Python vs FLIKE comparison.

**To enable R:**
```bash
pip install rpy2
# Install R from https://www.r-project.org/
# In R console: install.packages("MGBT")
```

### Issue: Plots not showing

**Solution:** Ensure matplotlib backend is set:
```python
%matplotlib inline
```

## Output Files

The notebook can save results to:
- CSV files with comparison data
- PNG files with plots
- Summary reports

## Advanced Usage

### Custom Alpha Values

```python
# Test with different significance levels
result = MGBT_optimized(
    flows,
    alpha1=0.05,  # Less stringent
    alpha10=0.15
)
```

### Performance Testing

```python
import time

start = time.time()
result = MGBT_optimized(flows)
elapsed = time.time() - start

print(f"Processing time: {elapsed:.3f}s")
```

### Detailed P-value Analysis

```python
# Examine p-values for each position
for i, pval in enumerate(result.p_values[:10]):
    print(f"Position {i+1}: p-value = {pval:.6f}")
```

## References

- USGS FLIKE Software Documentation
- Cohn et al. (2013) - Multiple Grubbs-Beck Test
- USGS Bulletin 17C - Flood Frequency Guidelines
- R MGBT Package Documentation

## Support

For issues or questions:
1. Check the main `IMPLEMENTATION_SUMMARY.md`
2. Review test logs in `data/test_results/`
3. Run `scripts/quick_test.py` for rapid validation
