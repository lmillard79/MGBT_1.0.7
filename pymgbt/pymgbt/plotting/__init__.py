"""
Plotting module for MGBT visualization functions.

This module provides comprehensive visualization capabilities for MGBT analysis
including peak streamflow plots, diagnostic plots, and evolution analysis.
"""

from .peaks import plot_peaks, plot_peaks_batch, plot_diagnostic
from .evolution import (
    plot_ffq_evolution,
    plot_return_period_evolution, 
    plot_trend_analysis
)

__all__ = [
    'plot_peaks',
    'plot_peaks_batch', 
    'plot_diagnostic',
    'plot_ffq_evolution',
    'plot_return_period_evolution',
    'plot_trend_analysis'
]
