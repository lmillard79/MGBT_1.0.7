# MGBT Implementation Comparison Analysis

## Overview

This document compares four MGBT implementations to validate the Python implementation against established references:

1. **FLIKE (USACE)** - Reference implementation used in flood frequency analysis
2. **R MGBT Package** - R implementation
3. **USACE-RMC Numerics (C#)** - Port of original Fortran code for RMC-BestFit
4. **Python MGBT (pymgbt)** - Our implementation

## Key Finding: Critical Parameter Differences

### Alpha Values

The implementations use **different default alpha values**:

| Implementation | Alphaout | Alphain | Alphazeroin |
|---------------|----------|---------|-------------|
| **USACE C# (Fortran port)** | 0.005 | 0.0 | 0.10 |
| **R MGBT Package** | 0.01 | ? | 0.10 |
| **Python (pymgbt)** | 0.01 | 0.10 | 0.10 |
| **FLIKE** | Unknown (likely 0.005) | Unknown | Unknown |

### Critical Insight

The C# implementation from USACE-RMC states:
```csharp
double Alphaout = 0.005d;  // Step 1 outward sweep
double Alphain = 0.0d;     // Step 2 inward sweep  
double Alphazeroin = 0.1d; // Step 3 zero-flow sweep
```

**This is the original Fortran specification!**

## Source Code Analysis

### 1. USACE-RMC Numerics (C#) - Fortran Port

**Source:** https://github.com/USACE-RMC/Numerics/blob/main/Numerics/Data/Statistics/MultipleGrubbsBeckTest.cs

**Key Features:**
- Direct port from Fortran PeakfqSA
- Uses **Alphaout = 0.005** (not 0.01)
- Uses **Alphain = 0.0** (effectively disabled)
- Three-sweep approach:
  1. Outward sweep from median (alpha = 0.005)
  2. Inward sweep from largest outlier (alpha = 0.0)
  3. Inward sweep from smallest observation (alpha = 0.10)
- Returns maximum of three sweeps

**Documentation states:**
> "This code converted from the FORTRAN source code for PeakfqSA, which can be downloaded at:
> https://sites.google.com/a/alumni.colostate.edu/jengland/resources"

### 2. Original Fortran (PeakfqSA)

**Source:** Available from John England's site (USACE/USBR)
- https://sites.google.com/a/alumni.colostate.edu/jengland/bulletin-17c
- Implements Bulletin 17C guidelines
- Written by Tim Cohn (USGS) and John England
- Reference implementation for flood frequency analysis

**Expected Parameters:**
- Alphaout = 0.005 (based on C# port)
- Alphain = 0.0
- Alphazeroin = 0.10

### 3. R MGBT Package

**Current Usage:**
```r
result <- MGBT(flows)  # Uses default alpha values
```

**Issue:** The R package wrapper may use different defaults than the underlying Fortran code.

### 4. Python pymgbt

**Current Implementation:**
```python
MGBT(flows, alpha1=0.01, alpha10=0.10)
```

**Issue:** Using alpha1=0.01 instead of 0.005

## Validation Strategy

### Step 1: Verify FLIKE Parameters

FLIKE likely uses the original Fortran parameters (0.005, 0.0, 0.10). We need to:

1. Check FLIKE documentation
2. Compare FLIKE results with C# implementation using same parameters
3. Verify if FLIKE matches the Fortran specification

### Step 2: Test Python with Correct Parameters

Modify Python implementation to match Fortran:

```python
# Test with Fortran parameters
result = MGBT(flows, alpha1=0.005, alpha10=0.10)
```

### Step 3: Compare All Implementations

Run comparison with corrected parameters:
- FLIKE (reference)
- C# USACE-RMC (Fortran port)
- Python with alpha1=0.005
- R MGBT (verify its defaults)

## Expected Outcome

**Hypothesis:** Python MGBT will match FLIKE when using alpha1=0.005 instead of 0.01.

**Reasoning:**
1. FLIKE is USACE software, likely uses USACE Fortran code
2. C# implementation is documented as Fortran port
3. C# uses alpha=0.005
4. Therefore, FLIKE likely uses alpha=0.005

## Implementation Differences

### P-Value Calculation

**C# (Fortran port):**
```csharp
// Uses Adaptive Simpson's Rule for integration
var sr = new AdaptiveSimpsonsRule(FGGB, 1E-16, 1 - 1E-16);
sr.MaxDepth = 25;
```

**Python:**
```python
# Uses scipy.integrate.quad for adaptive quadrature
result, error = quad(peta, 0, 1, args=(n, r, eta))
```

Both methods should produce equivalent results, but numerical precision may differ slightly.

### Three-Sweep Algorithm

**All implementations follow the same logic:**

1. **Outward Sweep** (from median toward smallest)
   - Find first p-value < Alphaout
   - Count = J1

2. **Inward Sweep** (from J1 toward median)
   - Find first p-value >= Alphain
   - Count = J2

3. **Zero-Flow Sweep** (from smallest toward median)
   - Find first p-value >= Alphazeroin
   - Count = J3

4. **Final Count** = max(J1, J2, J3)

## Recommendations

### Immediate Actions

1. **Update Python default to match Fortran:**
   ```python
   def MGBT(flows, alpha1=0.005, alpha10=0.10):  # Changed from 0.01
   ```

2. **Test with station 416040:**
   - FLIKE: 16 censored
   - Python (alpha=0.01): 23 censored
   - Python (alpha=0.005): ? censored (expected: 16)

3. **Verify R MGBT defaults:**
   - Check R package source code
   - Determine if it uses 0.005 or 0.01

### Validation Tests

Create test script that compares:

```python
# Test 1: Python with Fortran parameters
py_fortran = MGBT(flows, alpha1=0.005, alpha10=0.10)

# Test 2: Python with current parameters  
py_current = MGBT(flows, alpha1=0.01, alpha10=0.10)

# Test 3: C# implementation (via external call or manual verification)
# Test 4: FLIKE (from existing files)
# Test 5: R MGBT (from rpy2)

# Compare all results
```

## References

### Primary Sources

1. **Cohn et al. (2013)** - "A generalized Grubbs-Beck test statistic for detecting multiple potentially influential low outliers in flood series"
   - Water Resources Research, 49(8), 5047-5058
   - Defines the MGBT algorithm

2. **Bulletin 17C (2018)** - USGS Guidelines for flood flow frequency
   - Recommends MGBT for low outlier detection
   - Specifies alpha = 0.005 for outward sweep

3. **PeakfqSA (Fortran)** - Original implementation
   - By Tim Cohn (USGS) and John England (USBR/USACE)
   - Available: https://sites.google.com/a/alumni.colostate.edu/jengland/resources

4. **USACE-RMC Numerics** - C# port of Fortran
   - GitHub: https://github.com/USACE-RMC/Numerics
   - File: Numerics/Data/Statistics/MultipleGrubbsBeckTest.cs
   - Documented as Fortran port

### Software Implementations

- **PeakFQ 7.3** (USGS) - Implements Bulletin 17C
- **HEC-SSP 2.2** (USACE) - Implements Bulletin 17C  
- **RMC-BestFit** (USACE-RMC) - Uses Numerics library
- **FLIKE** (USACE) - Flood frequency analysis

## Conclusion

The discrepancy between Python MGBT and FLIKE is likely due to **different alpha values**:

- **FLIKE/Fortran:** alpha1 = 0.005 (stricter threshold)
- **Python (current):** alpha1 = 0.01 (more lenient threshold)

**Next Steps:**
1. Update Python to use alpha1=0.005 as default
2. Re-run validation tests
3. Verify Python matches FLIKE with corrected parameters
4. Document the alpha parameter choice in pymgbt

**Expected Result:** Python MGBT with alpha1=0.005 should match FLIKE results.
