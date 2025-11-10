"""
Setup R environment for rpy2.

This script configures rpy2 to use the R installation at:
C:\Program Files\R\R-4.5.1
"""

import os
import sys
from pathlib import Path

# Set R_HOME environment variable
R_HOME = r"C:\Program Files\R\R-4.5.1"
os.environ['R_HOME'] = R_HOME

# Add R to PATH
R_BIN = os.path.join(R_HOME, 'bin', 'x64')
if R_BIN not in os.environ['PATH']:
    os.environ['PATH'] = R_BIN + os.pathsep + os.environ['PATH']

print(f"R_HOME set to: {os.environ['R_HOME']}")
print(f"R binary path: {R_BIN}")

# Test rpy2 connection
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    
    # Get R version (no conversion needed for simple string)
    r_version = ro.r('R.version.string')[0]
    
    print(f"\nR connection successful!")
    print(f"R version: {r_version}")
    
    # Check if MGBT package is installed
    utils = importr('utils')
    installed_packages = ro.r('installed.packages()')
    package_names = list(installed_packages.rx(True, 1))
    
    if 'MGBT' in package_names:
        print("MGBT package: INSTALLED")
    else:
        print("MGBT package: NOT INSTALLED")
        print("\nTo install MGBT in R, run:")
        print('  install.packages("MGBT")')
        
except ImportError as e:
    print(f"\nError: rpy2 not installed")
    print("Install with: pip install rpy2")
    
except Exception as e:
    print(f"\nError connecting to R: {e}")
    print("\nTroubleshooting:")
    print("1. Verify R is installed at:", R_HOME)
    print("2. Check that R_HOME path is correct")
    print("3. Try restarting Python/IDE after setting R_HOME")
