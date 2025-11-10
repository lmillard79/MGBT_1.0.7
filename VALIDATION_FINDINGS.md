# MGBT Validation Findings

## Executive Summary

After examining the USACE-RMC C# implementation (Fortran port) and testing different alpha parameters, we've discovered important insights about MGBT implementation differences.

## Test Results - Station 416040

| Implementation | Censored Count | Matches FLIKE? |
|---------------|----------------|----------------|
| **FLIKE (Reference)** | 16 | - |
| **Python (α=0.01)** | 24 | ✗ |
| **Python (α=0.005)** | 24 | ✗ |
| **R MGBT** | 20 | ✗ |

**Total flows:** 45 (28 gauged + 17 censored zeros)

## Key Findings

### 1. Alpha Parameter Difference

The USACE-RMC C# implementation (Fortran port) uses:
- **Alphaout = 0.005** (not 0.01)
- **Alphain = 0.0** (effectively disabled)
- **Alphazeroin = 0.10**

Source: `MultipleGrubbsBeckTest.cs` line 82-84

### 2. Alpha Alone Doesn't Explain Discrepancy

Testing Python with both α=0.01 and α=0.005 produced **identical results** (24 censored), suggesting:
- The discrepancy is NOT solely due to alpha parameter differences
- There may be other algorithmic differences

### 3. All Implementations Disagree

- FLIKE: 16 censored
- R MGBT: 20 censored  
- Python: 24 censored

This suggests **different algorithm implementations or interpretations**.

## Possible Explanations

### Hypothesis 1: Zero-Flow Handling

**FLIKE shows:** "The following gauged flows were censored" - 16 zero flows

**Question:** Does FLIKE pre-censor zeros before running MGBT, or does MGBT identify them?

**Impact:** If FLIKE pre-censors zeros, MGBT would only run on the 28 non-zero flows, potentially identifying additional low outliers among them.

### Hypothesis 2: Three-Sweep Implementation Differences

The C# code shows three sweeps:
1. Outward sweep (α=0.005)
2. Inward sweep (α=0.0) 
3. Zero-flow sweep (α=0.10)

**Final count = max(J1, J2, J3)**

Different implementations may:
- Use different sweep strategies
- Have different stopping criteria
- Handle edge cases differently

### Hypothesis 3: P-Value Calculation Differences

**C# Implementation:**
```csharp
// Uses Adaptive Simpson's Rule
var sr = new AdaptiveSimpsonsRule(FGGB, 1E-16, 1 - 1E-16);
sr.MaxDepth = 25;
```

**Python Implementation:**
```python
# Uses scipy.integrate.quad
result, error = quad(peta, 0, 1, args=(n, r, eta))
```

**R Implementation:**
- Unknown integration method

Numerical integration differences could lead to slightly different p-values, affecting outlier detection.

### Hypothesis 4: Data Preprocessing

**FLIKE may:**
- Apply log transformation differently
- Handle zero/negative values differently
- Use different rounding or precision

## Recommendations

### Immediate Actions

1. **Examine FLIKE source code or documentation** to understand:
   - How zeros are handled
   - What alpha values are used
   - Pre-processing steps

2. **Test with non-zero data only:**
   ```python
   # Remove zeros and test
   non_zero_flows = flows[flows > 0]
   result = MGBT(non_zero_flows, alpha1=0.005, alpha10=0.10)
   ```

3. **Compare p-values directly:**
   - Extract p-values from each implementation
   - Compare for the same dataset
   - Identify where they diverge

4. **Contact USACE/FLIKE developers:**
   - Request FLIKE algorithm specifications
   - Clarify zero-flow handling
   - Verify alpha parameters used

### Validation Strategy

Since we have access to:
1. **USACE-RMC C# code** (documented Fortran port)
2. **Original Fortran code** (PeakfqSA - available for download)
3. **R MGBT package**
4. **Python implementation**

**Recommended approach:**

1. **Use C# as reference** (it's a documented Fortran port)
2. **Verify C# matches Fortran** (download PeakfqSA)
3. **Align Python with C#/Fortran**
4. **Then compare with FLIKE**

This establishes a clear lineage: Fortran → C# → Python

### Testing Protocol

```python
# Test suite for validation
test_cases = [
    {
        'name': 'Station 416040 - All flows',
        'flows': all_flows_including_zeros,
        'expected_flike': 16
    },
    {
        'name': 'Station 416040 - Non-zero only',
        'flows': non_zero_flows,
        'expected_flike': unknown
    },
    {
        'name': 'Synthetic - No zeros',
        'flows': [100, 200, 300, 10, 20, 400, 500],
        'expected_fortran': verify_with_peakfqsa
    }
]
```

## C# Implementation Details

### Key Code Sections

**Main Function:**
```csharp
public static int Function(double[] X)
{
    double Alphaout = 0.005d;
    double Alphain = 0.0d;
    double Alphazeroin = 0.1d;
    double maxFracLO = 0.5d;
    
    // Log transform
    for (int i = 0; i < N; i++)
        zt[i] = Math.Log10(Math.Max(1.0E-88d, X[i]));
    
    // Sort
    Array.Sort(zt);
    
    // Three sweeps...
    int MGBTP = Math.Max(J1, Math.Max(J2, J3));
    return MGBTP;
}
```

**P-Value Calculation:**
```csharp
private static double GGBCRITP(int N, int R, double ETA)
{
    if (N < 10 | R > N / 2d)
        return 0.5d;
    
    // Adaptive Simpson's Rule integration
    var sr = new AdaptiveSimpsonsRule(FGGB, 1E-16, 1 - 1E-16);
    sr.MaxDepth = 25;
    sr.ReportFailure = false;
    sr.Integrate();
    return sr.Status != IntegrationStatus.Failure ? sr.Result : double.NaN;
}
```

### Differences from Python

1. **Minimum value handling:**
   - C#: `Math.Max(1.0E-88d, X[i])`
   - Python: `np.maximum(1e-8, flows)`
   
2. **Integration method:**
   - C#: Adaptive Simpson's Rule
   - Python: Adaptive quadrature (quad)

3. **Alpha defaults:**
   - C#: 0.005, 0.0, 0.10
   - Python: 0.01, 0.10, 0.10

## Next Steps

1. **Download PeakfqSA** from John England's site
2. **Run PeakfqSA on station 416040** to get Fortran reference
3. **Compare C# vs Fortran** to verify C# port accuracy
4. **Align Python with verified reference**
5. **Document differences between FLIKE and Fortran/C#**

## References

- **USACE-RMC Numerics:** https://github.com/USACE-RMC/Numerics
- **PeakfqSA (Fortran):** https://sites.google.com/a/alumni.colostate.edu/jengland/bulletin-17c
- **Bulletin 17C:** USGS Guidelines for flood flow frequency
- **Cohn et al. (2013):** Water Resources Research, 49(8), 5047-5058

## Conclusion

The validation reveals that:

1. ✓ **C# implementation found** - documented Fortran port
2. ✓ **Alpha parameter difference identified** - 0.005 vs 0.01
3. ✗ **Alpha alone doesn't explain discrepancy** - same result with both values
4. ✗ **No implementation matches FLIKE exactly**

**Recommendation:** Use USACE-RMC C# (Fortran port) as the reference implementation, verify against original Fortran, then align Python accordingly. FLIKE may have additional preprocessing or different handling that needs separate investigation.
