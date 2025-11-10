# C# MGBT Integration Setup Guide

## Overview

The comparison script now supports four-way comparison:
1. **FLIKE** - Reference implementation
2. **R MGBT** - R package
3. **Python MGBT** - Our implementation
4. **C# MGBT** - USACE-RMC Numerics (Fortran port)

## Why Include C#?

The USACE-RMC Numerics C# implementation is a **documented port of the original Fortran code** from PeakfqSA. This makes it an excellent reference for validation:

- Direct conversion from Fortran (Tim Cohn/John England)
- Used in RMC-BestFit software
- Well-documented and maintained
- Uses alpha1=0.005 (Fortran default)

## Setup Instructions

### Step 1: Clone USACE-RMC Numerics

```bash
cd d:\GitRepos
git clone https://github.com/USACE-RMC/Numerics.git
```

### Step 2: Build the Numerics Library

**Requirements:**
- .NET SDK 6.0 or later
- Visual Studio 2022 (or VS Code with C# extension)

**Build with Visual Studio:**
1. Open `Numerics.sln` in Visual Studio
2. Build Solution (Ctrl+Shift+B)
3. Find `Numerics.dll` in `bin\Release\net6.0\`

**Build with command line:**
```bash
cd d:\GitRepos\Numerics
dotnet build -c Release
```

### Step 3: Compile the C# Wrapper

The wrapper is located at: `d:\GitRepos\MGBT_1.0.7\scripts\csharp_mgbt_wrapper.cs`

**Option A: Using Visual Studio Developer Command Prompt:**

```bash
cd d:\GitRepos\MGBT_1.0.7\scripts
csc /reference:..\..\Numerics\bin\Release\net6.0\Numerics.dll /out:csharp_mgbt_wrapper.exe csharp_mgbt_wrapper.cs
```

**Option B: Using dotnet:**

Create a project file `csharp_mgbt_wrapper.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <Reference Include="Numerics">
      <HintPath>..\..\Numerics\bin\Release\net6.0\Numerics.dll</HintPath>
    </Reference>
  </ItemGroup>
</Project>
```

Then build:
```bash
dotnet build -c Release
```

### Step 4: Copy Dependencies

Copy the Numerics DLL to the scripts directory:

```bash
copy d:\GitRepos\Numerics\bin\Release\net6.0\Numerics.dll d:\GitRepos\MGBT_1.0.7\scripts\
```

### Step 5: Test the Wrapper

```bash
cd d:\GitRepos\MGBT_1.0.7\scripts
csharp_mgbt_wrapper.exe 100 200 300 10 20 400 500
```

Expected output:
```json
{"klow": 2, "threshold": 100.0}
```

## Usage

### Run Four-Way Comparison

```bash
# With all methods (FLIKE, R, Python, C#)
python scripts/compare_all_methods.py

# Without R
python scripts/compare_all_methods.py --no-r

# Without C#
python scripts/compare_all_methods.py --no-csharp

# Python vs FLIKE only
python scripts/compare_all_methods.py --no-r --no-csharp
```

### Output Files

**CSV:** `data/test_results/mgbt_four_way_comparison.csv`

Columns:
- `station_id`
- `n_flows`
- `flike_censored`
- `python_censored`
- `r_censored`
- `csharp_censored`
- `python_threshold`
- `r_threshold`
- `csharp_threshold`
- `py_matches_flike`
- `r_matches_flike`
- `cs_matches_flike`
- `py_matches_r`
- `py_matches_cs`
- `r_matches_cs`

**Summary:** `data/test_results/mgbt_four_way_summary.txt`

## Troubleshooting

### C# Wrapper Not Found

If you see: `C# not available: C# wrapper not found`

**Solution:**
1. Verify `csharp_mgbt_wrapper.exe` exists in `scripts/` directory
2. Check that `Numerics.dll` is in the same directory
3. Rebuild the wrapper if necessary

### Missing Numerics.dll

If you see: `Could not load file or assembly 'Numerics'`

**Solution:**
```bash
copy d:\GitRepos\Numerics\bin\Release\net6.0\Numerics.dll d:\GitRepos\MGBT_1.0.7\scripts\
```

### .NET Runtime Not Found

If you see: `You must install .NET to run this application`

**Solution:**
1. Download .NET 6.0 Runtime: https://dotnet.microsoft.com/download/dotnet/6.0
2. Install the runtime
3. Restart your terminal

### C# Process Timeout

If comparison hangs on C# step:

**Solution:**
- Check if wrapper runs standalone
- Reduce dataset size for testing
- Increase timeout in `run_csharp_mgbt()` function

## Alternative: Manual C# Testing

If you can't get the wrapper working, you can manually test C# MGBT:

### Create a C# Console App

```csharp
using System;
using System.Linq;
using Numerics.Data.Statistics;

class Program
{
    static void Main()
    {
        // Test data from station 416040
        double[] flows = new double[] {
            1894.38, 1221.90, 1108.43, 672.61, 655.84, 610.87, 578.46,
            534.04, 501.34, 455.95, 408.63, 358.47, 336.82, 253.72,
            198.25, 182.64, 179.82, 163.71, 161.11, 146.59, 126.97,
            90.61, 80.65, 76.98, 61.95, 38.01, 20.18, 16.53,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };
        
        int klow = MultipleGrubbsBeckTest.Function(flows);
        Console.WriteLine($"C# MGBT Result: {klow} outliers");
    }
}
```

### Compare Results Manually

Run the C# program and compare with:
- FLIKE: 16 censored
- Python: 24 censored
- R: 20 censored
- C#: ? censored

## Expected Results

Based on the C# implementation being a Fortran port with alpha1=0.005:

**Hypothesis:** C# should match FLIKE more closely than Python (which uses alpha1=0.01)

**Test this by:**
1. Running the four-way comparison
2. Checking C# vs FLIKE match rate
3. Comparing C# vs Python discrepancies

## Benefits of C# Integration

1. **Reference Implementation** - Direct Fortran port
2. **Validation** - Verify Python against documented reference
3. **Alpha Parameter Verification** - C# uses 0.005 (Fortran default)
4. **Algorithm Clarity** - Well-documented C# code to compare against
5. **Independent Verification** - Not dependent on R or FLIKE

## Summary

The C# integration provides:
- ✓ Four-way comparison (FLIKE, R, Python, C#)
- ✓ Reference implementation (Fortran port)
- ✓ Independent validation
- ✓ Clear algorithm documentation
- ✓ Known alpha parameters (0.005, 0.0, 0.10)

Once set up, the comparison script will automatically include C# results alongside FLIKE, R, and Python, giving you a comprehensive validation of all MGBT implementations.
