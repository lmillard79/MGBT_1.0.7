# Documentation Organization Script
# Organizes all documentation files into a clean folder structure

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Organizing Documentation Files" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$root = "D:\GitRepos\MGBT_1.0.7"

# Define folder structure
$folders = @{
    "docs/validation" = @(
        "VALIDATION_GUIDE.md",
        "VALIDATION_FINDINGS.md",
        "DEMONSTRATION_SUMMARY.md",
        "QUICK_START.md"
    )
    "docs/setup_guides" = @(
        "R_SETUP_GUIDE.md",
        "R_INTEGRATION_COMPLETE.md",
        "R_INTEGRATION_FIXED.md",
        "CSHARP_SETUP_GUIDE.md",
        "CSHARP_SETUP_WALKTHROUGH.md"
    )
    "docs/analysis" = @(
        "MGBT_IMPLEMENTATION_COMPARISON.md",
        "FOUR_WAY_COMPARISON_SUMMARY.md",
        "LOGGING_SUPPRESSION.md"
    )
    "docs" = @(
        "IMPLEMENTATION_SUMMARY.md",
        "README.md"
    )
}

# Move files
$moved = 0
$notFound = 0

foreach ($folder in $folders.Keys) {
    $targetPath = Join-Path $root $folder
    
    Write-Host "Processing folder: $folder" -ForegroundColor Yellow
    
    foreach ($file in $folders[$folder]) {
        $sourcePath = Join-Path $root $file
        $destPath = Join-Path $targetPath $file
        
        if (Test-Path $sourcePath) {
            Move-Item -Path $sourcePath -Destination $destPath -Force
            Write-Host "  ✓ Moved: $file" -ForegroundColor Green
            $moved++
        } else {
            Write-Host "  ⚠ Not found: $file" -ForegroundColor Gray
            $notFound++
        }
    }
    Write-Host ""
}

# Create index files
Write-Host "Creating index files..." -ForegroundColor Yellow

# Main README
$mainReadme = @"
# MGBT Python Implementation

Multiple Grubbs-Beck Test (MGBT) implementation in Python for low outlier detection in flood frequency analysis.

## Quick Links

### Documentation
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)
- [Validation Guide](docs/validation/VALIDATION_GUIDE.md)
- [Quick Start](docs/validation/QUICK_START.md)

### Setup Guides
- [R Setup Guide](docs/setup_guides/R_SETUP_GUIDE.md)
- [C# Setup Guide](docs/setup_guides/CSHARP_SETUP_GUIDE.md)
- [C# Setup Walkthrough](docs/setup_guides/CSHARP_SETUP_WALKTHROUGH.md)

### Analysis & Validation
- [Four-Way Comparison](docs/analysis/FOUR_WAY_COMPARISON_SUMMARY.md)
- [Implementation Comparison](docs/analysis/MGBT_IMPLEMENTATION_COMPARISON.md)
- [Validation Findings](docs/validation/VALIDATION_FINDINGS.md)

## Directory Structure

``````
MGBT_1.0.7/
├── pymgbt/              # Python MGBT package
│   └── pymgbt/
│       ├── core/        # Core MGBT implementations
│       └── stats/       # Statistical functions
├── scripts/             # Validation and comparison scripts
│   ├── compare_all_methods.py      # Four-way comparison
│   ├── test_mgbt_comparison.py     # Validation tests
│   ├── quick_test.py               # Quick validation
│   └── setup_csharp_mgbt.ps1       # C# setup automation
├── notebooks/           # Jupyter notebooks for demos
├── data/               # Test data and results
│   ├── validation/     # Extracted validation data
│   └── test_results/   # Comparison results
├── docs/               # Documentation
│   ├── validation/     # Validation guides
│   ├── setup_guides/   # Setup instructions
│   └── analysis/       # Analysis documents
└── UnitTests/          # FLIKE reference data
``````

## Usage

### Basic MGBT
``````python
from pymgbt.core.mgbt_optimized import MGBT

result = MGBT(flows, alpha1=0.01, alpha10=0.10)
print(f"Low outliers: {result.klow}")
print(f"Threshold: {result.low_outlier_threshold}")
``````

### Four-Way Comparison
``````bash
# Compare FLIKE, R, Python, and C# implementations
python scripts/compare_all_methods.py

# Skip C# if not set up
python scripts/compare_all_methods.py --no-csharp
``````

### Validation
``````bash
# Quick test
python scripts/quick_test.py

# Full comparison
python scripts/test_mgbt_comparison.py --parallel --workers 4
``````

## Installation

See [docs/IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md) for detailed installation instructions.

## Validation

This implementation has been validated against:
- FLIKE (USACE reference)
- R MGBT package
- C# MGBT (USACE-RMC Numerics - Fortran port)

See [docs/validation/](docs/validation/) for validation results and guides.
"@

$mainReadme | Out-File -FilePath "$root\README.md" -Encoding UTF8 -Force
Write-Host "  ✓ Created: README.md" -ForegroundColor Green

# Validation index
$validationIndex = @"
# Validation Documentation

## Overview
This directory contains validation guides, findings, and demonstration materials for the MGBT Python implementation.

## Documents

### [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)
Comprehensive guide to validating the Python MGBT implementation against FLIKE and R MGBT.

### [VALIDATION_FINDINGS.md](VALIDATION_FINDINGS.md)
Test results and analysis from comparing Python MGBT against FLIKE, R, and C# implementations.

### [DEMONSTRATION_SUMMARY.md](DEMONSTRATION_SUMMARY.md)
Summary of the demonstration and validation system, including Jupyter notebooks and comparison scripts.

### [QUICK_START.md](QUICK_START.md)
Quick start guide for running validation tests and comparisons.

## Quick Validation

``````bash
# Quick test
python scripts/quick_test.py

# Full comparison (FLIKE, R, Python, C#)
python scripts/compare_all_methods.py

# Interactive notebook
jupyter notebook notebooks/MGBT_Demonstration.ipynb
``````

## Validation Results

Results are saved to:
- ``data/test_results/mgbt_four_way_comparison.csv``
- ``data/test_results/mgbt_four_way_summary.txt``
"@

$validationIndex | Out-File -FilePath "$root\docs\validation\README.md" -Encoding UTF8 -Force
Write-Host "  ✓ Created: docs/validation/README.md" -ForegroundColor Green

# Setup guides index
$setupIndex = @"
# Setup Guides

## Overview
This directory contains setup instructions for R and C# MGBT integrations.

## R Setup

### [R_SETUP_GUIDE.md](R_SETUP_GUIDE.md)
Complete guide for setting up R environment and MGBT package for validation.

### [R_INTEGRATION_COMPLETE.md](R_INTEGRATION_COMPLETE.md)
Summary of successful R integration with Python validation system.

### [R_INTEGRATION_FIXED.md](R_INTEGRATION_FIXED.md)
Documentation of rpy2 syntax fixes and result extraction improvements.

## C# Setup

### [CSHARP_SETUP_WALKTHROUGH.md](CSHARP_SETUP_WALKTHROUGH.md) ⭐
**Start here!** Step-by-step walkthrough for setting up C# MGBT comparison.

### [CSHARP_SETUP_GUIDE.md](CSHARP_SETUP_GUIDE.md)
Detailed reference guide for C# integration.

## Quick Setup

### R Setup
``````bash
# Install R and MGBT package
python scripts/setup_r_environment.py
python scripts/install_r_mgbt.py
``````

### C# Setup
``````powershell
# Automated setup (after installing .NET SDK)
.\scripts\setup_csharp_mgbt.ps1
``````
"@

$setupIndex | Out-File -FilePath "$root\docs\setup_guides\README.md" -Encoding UTF8 -Force
Write-Host "  ✓ Created: docs/setup_guides/README.md" -ForegroundColor Green

# Analysis index
$analysisIndex = @"
# Analysis Documentation

## Overview
This directory contains analysis documents comparing different MGBT implementations.

## Documents

### [FOUR_WAY_COMPARISON_SUMMARY.md](FOUR_WAY_COMPARISON_SUMMARY.md)
Overview of the four-way comparison system (FLIKE, R, Python, C#).

### [MGBT_IMPLEMENTATION_COMPARISON.md](MGBT_IMPLEMENTATION_COMPARISON.md)
Detailed comparison of MGBT implementations, including algorithm differences and parameter analysis.

### [LOGGING_SUPPRESSION.md](LOGGING_SUPPRESSION.md)
Documentation of logging suppression for cleaner output from rpy2 and R.

## Key Findings

### Alpha Parameters
- **FLIKE/Fortran:** alpha1 = 0.005
- **Python (current):** alpha1 = 0.01
- **R MGBT:** Unknown defaults

### Implementation Differences
- C# is a documented Fortran port (USACE-RMC Numerics)
- Different integration methods (Simpson's vs adaptive quadrature)
- Varying zero-flow handling approaches

## Running Comparisons

``````bash
# Four-way comparison
python scripts/compare_all_methods.py

# Results saved to:
# - data/test_results/mgbt_four_way_comparison.csv
# - data/test_results/mgbt_four_way_summary.txt
``````
"@

$analysisIndex | Out-File -FilePath "$root\docs\analysis\README.md" -Encoding UTF8 -Force
Write-Host "  ✓ Created: docs/analysis/README.md" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Organization Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Files moved: $moved" -ForegroundColor White
Write-Host "  Not found: $notFound" -ForegroundColor Gray
Write-Host ""
Write-Host "New structure:" -ForegroundColor Cyan
Write-Host "  docs/" -ForegroundColor White
Write-Host "    ├── validation/      (validation guides)" -ForegroundColor White
Write-Host "    ├── setup_guides/    (R and C# setup)" -ForegroundColor White
Write-Host "    └── analysis/        (implementation analysis)" -ForegroundColor White
Write-Host ""
Write-Host "Start here:" -ForegroundColor Cyan
Write-Host "  README.md" -ForegroundColor White
Write-Host "  docs/validation/QUICK_START.md" -ForegroundColor White
Write-Host "  docs/setup_guides/CSHARP_SETUP_WALKTHROUGH.md" -ForegroundColor White
Write-Host ""
