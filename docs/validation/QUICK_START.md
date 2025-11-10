# MGBT Quick Start Guide

## For Interactive Demonstration

### Option 1: Jupyter Notebook (Recommended)

```bash
# Install Jupyter
pip install jupyter notebook matplotlib seaborn

# Navigate to notebooks
cd d:\GitRepos\MGBT_1.0.7\notebooks

# Launch Jupyter
jupyter notebook

# Open: MGBT_Demonstration.ipynb
# Run cells to see FLIKE vs Python vs R comparison
```

**What you'll see:**
- Load any station's FLIKE data
- View censored flows from FLIKE
- Run Python MGBT
- Compare results
- Visualize with plots

### Option 2: Command Line Comparison

```bash
# Run comprehensive comparison
cd d:\GitRepos\MGBT_1.0.7
python scripts/compare_all_methods.py --no-r

# View results
cat data/test_results/mgbt_three_way_summary.txt

# Open CSV in Excel
# File: data/test_results/mgbt_three_way_comparison.csv
```

**What you'll get:**
- CSV with: gauge_id, flike_censor_count, r_mgbt_censor_count, python_censor_count
- Summary report with match statistics
- Detailed comparison for all 28 stations

## Quick Test Single Station

```bash
python scripts/quick_test.py
```

Shows corrected vs optimized Python implementation on one station.

## Files Overview

| File | Purpose |
|------|---------|
| `notebooks/MGBT_Demonstration.ipynb` | Interactive demo |
| `scripts/compare_all_methods.py` | Automated validation |
| `scripts/quick_test.py` | Quick single-station test |
| `data/test_results/mgbt_three_way_comparison.csv` | Results dataset |

## Expected Dataset Format

```csv
station_id,n_flows,flike_censored,python_censored,r_censored,py_matches_flike,r_matches_flike,py_matches_r
416040,44,16,23,16,False,True,False
416050,29,0,0,0,True,True,True
...
```

## Next Steps

1. **Try the notebook** - Best for understanding
2. **Run comparison script** - Get full dataset
3. **Review results** - Check match rates
4. **Investigate discrepancies** - Use notebook for details

## Documentation

- `DEMONSTRATION_SUMMARY.md` - Complete overview
- `VALIDATION_GUIDE.md` - Validation workflow
- `notebooks/README.md` - Notebook guide
- `IMPLEMENTATION_SUMMARY.md` - Technical details
