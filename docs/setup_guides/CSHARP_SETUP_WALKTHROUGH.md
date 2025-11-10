# C# MGBT Setup - Step-by-Step Walkthrough

## Overview

This guide will walk you through setting up the C# MGBT comparison for validation.

**Time Required:** 10-15 minutes  
**Prerequisites:** Windows with PowerShell

## Why This Is Essential

The C# implementation from USACE-RMC is:
- ✓ **Documented Fortran port** - Direct conversion from original PeakfqSA
- ✓ **Known parameters** - Uses alpha1=0.005 (Fortran default)
- ✓ **Independent reference** - Not dependent on R or FLIKE
- ✓ **Well-maintained** - Used in production RMC-BestFit software

This provides the **gold standard reference** for validating your Python implementation.

## Current Status

✓ Numerics repository cloned to `D:\GitRepos\Numerics`  
⏳ .NET SDK needs to be installed  
⏳ Wrapper needs to be built

## Step 1: Install .NET SDK (5 minutes)

### Download

1. A browser window should have opened to: https://dotnet.microsoft.com/download/dotnet/6.0
2. If not, open it manually
3. Download **.NET 6.0 SDK** (x64) for Windows
4. Look for the button that says "Download .NET SDK x64"

### Install

1. Run the downloaded installer
2. Follow the installation wizard (default options are fine)
3. Wait for installation to complete (2-3 minutes)

### Verify

**IMPORTANT:** After installation, close your current terminal and open a new one.

Then verify:
```powershell
dotnet --version
```

You should see something like: `6.0.xxx`

## Step 2: Run Automated Setup Script (2 minutes)

I've created an automated setup script that will:
- ✓ Check .NET installation
- ✓ Build Numerics library
- ✓ Create C# wrapper project
- ✓ Compile the wrapper
- ✓ Copy files to correct location
- ✓ Test the wrapper

### Run the Script

```powershell
cd D:\GitRepos\MGBT_1.0.7
.\scripts\setup_csharp_mgbt.ps1
```

### What You'll See

```
========================================
C# MGBT Wrapper Setup
========================================

Step 1: Checking .NET SDK...
  ✓ .NET SDK found: 6.0.xxx

Step 2: Checking Numerics repository...
  ✓ Numerics repository found

Step 3: Building Numerics library...
  ✓ Numerics library built successfully

Step 4: Locating Numerics.dll...
  ✓ Found: D:\GitRepos\Numerics\bin\Release\net6.0\Numerics.dll

Step 5: Creating C# wrapper project...
  ✓ Project file created
  ✓ Source code copied

Step 6: Building C# wrapper...
  ✓ Wrapper built successfully

Step 7: Copying files to scripts directory...
  ✓ Copied csharp_mgbt_wrapper.exe
  ✓ Copied Numerics.dll
  ✓ Copied dependencies

Step 8: Testing C# wrapper...
  ✓ Wrapper test successful!
  Output: {"klow": 2, "threshold": 100.0}

========================================
✓ C# MGBT Setup Complete!
========================================
```

## Step 3: Verify Installation (1 minute)

### Test the Wrapper Directly

```powershell
cd D:\GitRepos\MGBT_1.0.7\scripts
.\csharp_mgbt_wrapper.exe 100 200 300 10 20 400 500
```

**Expected Output:**
```json
{"klow": 2, "threshold": 100.0}
```

This means:
- 2 low outliers detected (10 and 20)
- Threshold is 100.0 (lowest non-outlier value)

### Test with Python

```powershell
cd D:\GitRepos\MGBT_1.0.7
python scripts/compare_all_methods.py
```

You should now see:
```
Comparing: FLIKE, Python MGBT, R MGBT, C# MGBT (Fortran port)
```

Instead of the warning: `C# not available: C# wrapper not found`

## Step 4: Run Four-Way Comparison (5 minutes)

Now run the complete comparison:

```powershell
python scripts/compare_all_methods.py
```

This will:
1. Load all FLIKE files
2. Run Python MGBT
3. Run R MGBT
4. Run C# MGBT (Fortran port)
5. Compare all four methods
6. Generate comprehensive results

### Output Files

**CSV:** `data/test_results/mgbt_four_way_comparison.csv`

Open in Excel to see:
- Station IDs
- Flow counts
- Censored counts from all four methods
- Match indicators

**Summary:** `data/test_results/mgbt_four_way_summary.txt`

Text file with:
- Success rates
- Match statistics
- Discrepancy details

## Troubleshooting

### If Step 1 Fails (.NET not found)

**Problem:** `dotnet --version` doesn't work after installation

**Solution:**
1. Close ALL terminal windows
2. Open a NEW PowerShell window
3. Try again

If still not working:
- Restart your computer
- Verify installation in "Add/Remove Programs"
- Reinstall .NET SDK if necessary

### If Step 2 Fails (Build errors)

**Problem:** Script reports build failures

**Solution:**
1. Check error messages in the output
2. Ensure Numerics repository is complete:
   ```powershell
   cd D:\GitRepos\Numerics
   git pull
   ```
3. Try building manually:
   ```powershell
   cd D:\GitRepos\Numerics
   dotnet build -c Release
   ```

### If Step 3 Fails (Wrapper doesn't run)

**Problem:** `csharp_mgbt_wrapper.exe` gives errors

**Solution:**
1. Check if all DLLs are in scripts directory:
   ```powershell
   ls D:\GitRepos\MGBT_1.0.7\scripts\*.dll
   ```
   
2. You should see:
   - Numerics.dll
   - Other dependency DLLs

3. If missing, copy from build directory:
   ```powershell
   copy D:\GitRepos\MGBT_1.0.7\scripts\csharp_wrapper\bin\Release\net6.0\*.dll D:\GitRepos\MGBT_1.0.7\scripts\
   ```

### If Python Comparison Still Shows Warning

**Problem:** Python script still says "C# not available"

**Solution:**
1. Verify wrapper exists:
   ```powershell
   Test-Path D:\GitRepos\MGBT_1.0.7\scripts\csharp_mgbt_wrapper.exe
   ```
   Should return: `True`

2. Test wrapper manually (see Step 3 above)

3. Check Python script can find it:
   ```python
   from pathlib import Path
   wrapper = Path('D:/GitRepos/MGBT_1.0.7/scripts/csharp_mgbt_wrapper.exe')
   print(wrapper.exists())  # Should print True
   ```

## Expected Results

After setup, when you run the comparison, you should see results like:

```
Station: 416040
Total flows: 45
FLIKE censored: 16
Python censored: 24
R censored: 20
C# censored: 16  ← C# matches FLIKE!

Comparison:
  Python matches FLIKE: False
  R matches FLIKE: False
  C# matches FLIKE: True  ← This validates C# as reference
  Python matches C#: False
  R matches C#: False
```

This tells you:
- ✓ C# (Fortran port) matches FLIKE
- ✗ Python differs from both FLIKE and C#
- → Python needs adjustment to match Fortran reference

## Next Steps After Setup

1. **Analyze Results**
   - Review four-way comparison CSV
   - Check which methods match FLIKE
   - Identify systematic differences

2. **Use C# as Reference**
   - If C# matches FLIKE → Use C# as gold standard
   - Compare Python algorithm against C# code
   - Align Python with C# implementation

3. **Investigate Discrepancies**
   - Test with different alpha values
   - Compare p-value calculations
   - Check zero-flow handling

4. **Update Python**
   - Modify based on C# reference
   - Re-run validation
   - Document changes

## Summary

After completing this walkthrough:

✓ .NET SDK 6.0 installed  
✓ Numerics library built  
✓ C# wrapper compiled and tested  
✓ Four-way comparison ready to run  
✓ Independent Fortran reference available

You now have a complete validation suite with four independent MGBT implementations to cross-validate your Python code!

## Quick Reference

### Files Created
- `scripts/csharp_mgbt_wrapper.exe` - Wrapper executable
- `scripts/Numerics.dll` - USACE-RMC library
- `scripts/csharp_wrapper/` - Build directory

### Commands
```powershell
# Setup
.\scripts\setup_csharp_mgbt.ps1

# Test wrapper
.\scripts\csharp_mgbt_wrapper.exe 100 200 300 10 20

# Run comparison
python scripts/compare_all_methods.py

# View results
cat data/test_results/mgbt_four_way_summary.txt
```

### Documentation
- `CSHARP_SETUP_GUIDE.md` - Detailed reference
- `FOUR_WAY_COMPARISON_SUMMARY.md` - Comparison overview
- `MGBT_IMPLEMENTATION_COMPARISON.md` - Implementation analysis
