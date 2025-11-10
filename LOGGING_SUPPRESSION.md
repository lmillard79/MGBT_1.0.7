# Logging Suppression for rpy2

## Summary

The verbose rpy2 logging messages have been suppressed to provide cleaner output during the MGBT comparison runs.

## What Was Suppressed

### 1. rpy2 INFO Messages (SUPPRESSED ✓)

**Before:**
```
INFO: cffi mode is CFFI_MODE.ANY
INFO: R home found: C:\Program Files\R\R-4.5.1
INFO: R exec path: C:\Program Files\R\R-4.5.1\bin\x64\R
INFO: Default options to initialize R: rpy2, --quiet, --no-save
INFO: Environment variable "PATH" redefined by R and overriding existing variable...
INFO: Environment variable "R_HOME" redefined by R and overriding existing variable...
INFO: R is already initialized. No need to initialize.
```

**After:** (Clean output - no INFO messages)

**Solution:**
```python
# Suppress verbose rpy2 logging
logging.getLogger('rpy2').setLevel(logging.ERROR)
logging.getLogger('rpy2.rinterface_lib.embedded').setLevel(logging.ERROR)

# Suppress R console output and warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='rpy2')
```

### 2. "sh not recognized" Warning (CANNOT SUPPRESS)

**Message:**
```
'sh' is not recognized as an internal or external command,
operable program or batch file.
```

**Status:** This warning appears during rpy2's subprocess initialization before Python code executes. It cannot be easily suppressed but is **completely harmless** and can be ignored.

**Why it appears:** rpy2 tries to use Unix shell commands on Windows, which don't exist. It automatically falls back to Windows-compatible methods.

## Files Updated

All scripts that use rpy2 have been updated with logging suppression:

1. **scripts/compare_all_methods.py** - Main comparison script
2. **scripts/setup_r_environment.py** - R connection test
3. **scripts/test_r_mgbt.py** - R MGBT test

## Before vs After

### Before (Verbose)
```
INFO: cffi mode is CFFI_MODE.ANY
INFO: R home found: C:\Program Files\R\R-4.5.1
INFO: R exec path: C:\Program Files\R\R-4.5.1\bin\x64\R
'sh' is not recognized as an internal or external command
WARNING: Error importing in API mode: ImportError('On Windows, cffi mode "ANY" is only "ABI".')
WARNING: Trying to import in ABI mode.
INFO: Default options to initialize R: rpy2, --quiet, --no-save
INFO: Environment variable "PATH" redefined by R and overriding existing variable...
INFO: Environment variable "R_HOME" redefined by R and overriding existing variable...
INFO: R is already initialized. No need to initialize.
INFO: Station: 416040 Model
INFO: Total flows: 29
INFO: FLIKE censored: 16
INFO: R censored: 0
```

### After (Clean)
```
'sh' is not recognized as an internal or external command
INFO: ==========================================================
INFO: MGBT Comprehensive Comparison
INFO: ==========================================================
INFO: Station: 416040 Model
INFO: Total flows: 29
INFO: FLIKE censored: 16
INFO: R censored: 0
```

Much cleaner! Only the essential comparison information is shown.

## Implementation Details

### Logging Configuration

Added to all rpy2-using scripts:

```python
import logging

# Suppress verbose rpy2 logging
logging.getLogger('rpy2').setLevel(logging.ERROR)
logging.getLogger('rpy2.rinterface_lib.embedded').setLevel(logging.ERROR)

# Suppress R console output and warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='rpy2')
```

### What Gets Suppressed

- ✓ rpy2 INFO messages about cffi mode
- ✓ R home and exec path messages
- ✓ R initialization messages
- ✓ Environment variable redefinition messages
- ✓ R console callback messages
- ✗ "sh not recognized" (subprocess level, harmless)

### What Remains Visible

- Your script's INFO/WARNING/ERROR messages
- MGBT comparison results
- Station processing status
- Match/mismatch information
- Actual errors (if any occur)

## Usage

No changes needed to your workflow. The scripts automatically suppress verbose logging:

```bash
# Clean output
python scripts/compare_all_methods.py

# Clean output
python scripts/test_r_mgbt.py

# Clean output
python scripts/setup_r_environment.py
```

## Note on "sh not recognized"

This single-line warning cannot be suppressed because:
1. It comes from Windows cmd.exe, not Python
2. It occurs during subprocess initialization
3. It happens before Python can intercept stderr

**This is completely safe to ignore.** rpy2 automatically handles this and falls back to ABI mode, which works perfectly on Windows.

## Benefits

- **Cleaner console output** - Focus on actual results
- **Easier debugging** - Important messages stand out
- **Professional appearance** - Less clutter
- **Same functionality** - All features work identically

## Summary

✓ Verbose rpy2 INFO messages suppressed  
✓ Environment variable warnings suppressed  
✓ R console callbacks suppressed  
✓ Clean, focused output  
✓ All functionality preserved  
⚠ One harmless "sh not recognized" warning remains (safe to ignore)

Your MGBT comparison scripts now produce clean, professional output!
