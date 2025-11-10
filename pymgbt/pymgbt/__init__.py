"""
PyMGBT: Python implementation of Multiple Grubbs-Beck Low-Outlier Test

A complete Python port of the R MGBT package for detecting low outliers
in hydrological data, particularly annual peak streamflow data.
"""

from .core import MGBT, MGBTResult, mgbt_simple, mgbt
from .stats import gtmoms, cond_moms_z, cond_moms_chi2, V, VMS, EMS
from .core.critical_values import crit_k

__version__ = "1.0.7"
__author__ = "PyMGBT Development Team"

__all__ = [
    # Main MGBT functions
    "MGBT",
    "MGBTResult",
    "mgbt_simple", 
    "mgbt",
    
    # Statistical functions
    "gtmoms",
    "cond_moms_z",
    "cond_moms_chi2", 
    "V",
    "VMS",
    "EMS",
    
    # Critical values
    "crit_k",
]
