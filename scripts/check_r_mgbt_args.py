"""
Check R MGBT function signature.
"""

import os

# Configure R
R_HOME = r"C:\Program Files\R\R-4.5.1"
os.environ['R_HOME'] = R_HOME

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# Import MGBT
mgbt_r = importr('MGBT')

# Get function signature
print("Checking R MGBT function signature...")
ro.r('library(MGBT)')

# Get the arguments
args_result = ro.r('args(MGBT)')
print("\nMGBT function signature:")
print(args_result)

# Try to get help
try:
    help_text = ro.r('capture.output(help(MGBT))')
    print("\nHelp text (first 20 lines):")
    for i, line in enumerate(help_text[:20]):
        print(line)
except:
    pass
