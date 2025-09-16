# MGBT_1.0.7
A port of R MGBT library 
Multiple Grubs Beck Test

1. Flood Frequency Evolution (evolution.py)
- plot_ffq_evolution() - Multi-panel visualization showing:
  - Time series with quantile evolution (10th, 50th, 90th percentiles)
  - Quantile trends over time
  - Moving average analysis
  - Coefficient of variation evolution
- plot_return_period_evolution() 
  - Shows how return period quantiles (2, 5, 10, 25, 50, 100-year) change over time using moving windows
- plot_trend_analysis() 

- Comprehensive trend analysis including:
  - Time series with linear trend
  - Detrended data visualization
  - Moving statistics (mean and standard deviation)
  - Autocorrelation function
    
## Key Features
MGBT Integration - All functions accept MGBTResult objects to highlight outliers and thresholds 
Flexible Parameters - Customizable window sizes, step sizes, return periods, and figure dimensions 
High-quality matplotlib plots with proper legends, grids, and styling 
Implements Weibull plotting positions for return period calculations 

The functionality of the original R MGBT package's plotting capabilities. 

# Key Features Implemented
## Professional Package Structure
- Modern pyproject.toml configuration with optional dependencies
- Comprehensive API design matching R package functions
- Modular architecture for easy maintenance
## Core Algorithm Foundation
- MGBTResult dataclass for structured results
- Main MGBT() function framework
- Data validation and transformation utilities
## Documentation & Examples
- Complete basic usage example with visualization
- Mathematical background and references

Statistical Functions (stats/ module)
P-value calculations (kth_order_pvalue_ortho_t)
Moments computation (gtmoms, cond_moms_z)
Covariance matrices (V, VMS, EMS)
Visualization (plotting/ module)
Peak plotting functions
Flood frequency evolution plots
Data Integration (data/ module)
USGS data retrieval
Water year calculations
The project is professionally structured and follows Python best practices, making it ready for open-source collaboration and eventual PyPI publication.
