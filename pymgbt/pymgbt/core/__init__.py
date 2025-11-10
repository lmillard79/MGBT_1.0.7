"""
Core MGBT algorithms and functions.

This module contains the main MGBT implementation and supporting functions
for critical value calculations and outlier detection.
"""

from .mgbt import MGBT, MGBTResult, mgbt_simple, mgbt
from .critical_values import crit_k, crit_k10

__all__ = [
    "MGBT",
    "MGBTResult", 
    "mgbt_simple",
    "mgbt",
    "crit_k",
    "crit_k10"
]
