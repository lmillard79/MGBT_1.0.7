"""
Peak streamflow plotting functions for MGBT analysis.

This module provides visualization capabilities for annual peak streamflow data
with MGBT outlier identification results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, Union, Tuple, Dict, Any
import warnings

from ..core.mgbt import MGBTResult


def plot_peaks(
    data: np.ndarray,
    mgbt_result: Optional[MGBTResult] = None,
    years: Optional[np.ndarray] = None,
    station_id: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    show_threshold: bool = True,
    show_legend: bool = True,
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot annual peak streamflow data with MGBT outlier identification.
    
    Creates a comprehensive visualization showing the time series of annual peaks,
    identified outliers, and statistical thresholds from MGBT analysis.
    
    Parameters
    ----------
    data : np.ndarray
        Annual peak streamflow data
    mgbt_result : MGBTResult, optional
        Results from MGBT analysis
    years : np.ndarray, optional
        Years corresponding to data. If None, uses sequential numbering
    station_id : str, optional
        USGS station identifier for plot title
    title : str, optional
        Custom plot title
    figsize : tuple, default=(12, 8)
        Figure size (width, height) in inches
    show_threshold : bool, default=True
        Whether to show the low outlier threshold line
    show_legend : bool, default=True
        Whether to show the legend
    save_path : str, optional
        Path to save the figure
    **kwargs
        Additional arguments passed to matplotlib plotting functions
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
        
    Notes
    -----
    This implements functionality similar to the plotPeaks function in the
    original R MGBT package, with enhanced Python/matplotlib styling.
    """
    # Validate inputs
    if len(data) == 0:
        raise ValueError("Data array cannot be empty")
    
    # Set up years if not provided
    if years is None:
        years = np.arange(1, len(data) + 1)
    elif len(years) != len(data):
        raise ValueError("Years array must have same length as data")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot main data
    line_color = kwargs.get('color', 'steelblue')
    marker_size = kwargs.get('markersize', 6)
    
    ax.plot(years, data, 'o-', color=line_color, markersize=marker_size, 
            alpha=0.8, linewidth=1.5, label='Annual peaks')
    
    # Add MGBT results if provided
    if mgbt_result is not None:
        # Highlight outliers
        if mgbt_result.klow > 0:
            outlier_years = years[mgbt_result.outlier_indices]
            outlier_values = mgbt_result.outlier_values
            
            ax.plot(outlier_years, outlier_values, 'ro', markersize=marker_size + 2,
                   markerfacecolor='red', markeredgecolor='darkred', 
                   markeredgewidth=1.5, label=f'Low outliers (n={mgbt_result.klow})')
            
            # Show threshold line
            if show_threshold and mgbt_result.threshold > 0:
                ax.axhline(y=mgbt_result.threshold, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label='MGBT threshold')
                
                # Add threshold annotation
                ax.annotate(f'Threshold: {mgbt_result.threshold:.1f}',
                           xy=(years[len(years)//4], mgbt_result.threshold),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                           fontsize=10)
    
    # Customize plot
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Peak Streamflow', fontsize=12)
    
    # Set title
    if title is not None:
        plot_title = title
    elif station_id is not None:
        plot_title = f'Annual Peak Streamflow - USGS Station {station_id}'
        if mgbt_result is not None:
            plot_title += f'\nMGBT Analysis: {mgbt_result.klow} low outliers detected'
    else:
        plot_title = 'Annual Peak Streamflow'
        if mgbt_result is not None:
            plot_title += f' - MGBT: {mgbt_result.klow} outliers'
    
    ax.set_title(plot_title, fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Show legend
    if show_legend:
        ax.legend(loc='upper right', framealpha=0.9)
    
    # Format axes
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_peaks_batch(
    data_dict: Dict[str, np.ndarray],
    mgbt_results: Optional[Dict[str, MGBTResult]] = None,
    years_dict: Optional[Dict[str, np.ndarray]] = None,
    ncols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create batch plots for multiple stations or datasets.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary with station IDs as keys and data arrays as values
    mgbt_results : dict, optional
        Dictionary with station IDs as keys and MGBTResult objects as values
    years_dict : dict, optional
        Dictionary with station IDs as keys and years arrays as values
    ncols : int, default=2
        Number of columns in subplot grid
    figsize : tuple, optional
        Figure size. If None, calculated based on number of plots
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
        
    Notes
    -----
    This implements functionality similar to the plotPeaks_batch function
    in the original R MGBT package.
    """
    n_stations = len(data_dict)
    if n_stations == 0:
        raise ValueError("Data dictionary cannot be empty")
    
    # Calculate subplot layout
    nrows = int(np.ceil(n_stations / ncols))
    
    # Set figure size if not provided
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_stations == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each station
    for i, (station_id, data) in enumerate(data_dict.items()):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col] if nrows > 1 else axes[col]
        
        # Get corresponding data
        years = years_dict.get(station_id) if years_dict else None
        mgbt_result = mgbt_results.get(station_id) if mgbt_results else None
        
        # Plot on current axis
        plt.sca(ax)
        plot_peaks(
            data, 
            mgbt_result=mgbt_result,
            years=years,
            station_id=station_id,
            figsize=figsize,  # This won't be used since we're plotting on existing axis
            show_legend=True
        )
    
    # Hide empty subplots
    for i in range(n_stations, nrows * ncols):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col] if nrows > 1 else axes[col]
        ax.set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_diagnostic(
    mgbt_result: MGBTResult,
    figsize: Tuple[float, float] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create diagnostic plots for MGBT analysis.
    
    Parameters
    ----------
    mgbt_result : MGBTResult
        Results from MGBT analysis
    figsize : tuple, default=(15, 10)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Plot 1: Original data time series
    ax1 = axes[0, 0]
    years = np.arange(1, len(mgbt_result.data_sorted) + 1)
    ax1.plot(years, mgbt_result.data_sorted, 'bo-', alpha=0.7)
    if mgbt_result.klow > 0:
        ax1.plot(mgbt_result.outlier_indices + 1, mgbt_result.outlier_values, 
                'ro', markersize=8)
        ax1.axhline(y=mgbt_result.threshold, color='r', linestyle='--')
    ax1.set_title('Time Series (Sorted)')
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-transformed data
    ax2 = axes[0, 1]
    ax2.plot(years, mgbt_result.log_data_sorted, 'go-', alpha=0.7)
    if mgbt_result.klow > 0:
        ax2.plot(mgbt_result.outlier_indices + 1, 
                np.log10(mgbt_result.outlier_values), 'ro', markersize=8)
        ax2.axhline(y=mgbt_result.log_threshold, color='r', linestyle='--')
    ax2.set_title('Log-Transformed Data')
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Log10(Value)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: P-values
    ax3 = axes[0, 2]
    valid_pvals = mgbt_result.pvalues[mgbt_result.pvalues > -99]
    positions = np.arange(1, len(valid_pvals) + 1)
    ax3.semilogy(positions, valid_pvals, 'bo-', alpha=0.7)
    ax3.axhline(y=mgbt_result.alpha1, color='r', linestyle='-', 
               label=f'α₁ = {mgbt_result.alpha1}')
    ax3.axhline(y=mgbt_result.alpha10, color='orange', linestyle='--', 
               label=f'α₁₀ = {mgbt_result.alpha10}')
    ax3.set_title('MGBT P-values')
    ax3.set_xlabel('Outlier Position')
    ax3.set_ylabel('P-value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Histogram of original data
    ax4 = axes[1, 0]
    ax4.hist(mgbt_result.data_sorted, bins=20, alpha=0.7, color='skyblue', 
            edgecolor='black')
    if mgbt_result.klow > 0:
        ax4.hist(mgbt_result.outlier_values, bins=20, alpha=0.8, color='red', 
                edgecolor='darkred')
        ax4.axvline(x=mgbt_result.threshold, color='r', linestyle='--', linewidth=2)
    ax4.set_title('Distribution')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Q-Q plot of log data
    ax5 = axes[1, 1]
    from scipy import stats
    stats.probplot(mgbt_result.log_data_sorted, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot (Log Data)')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""
MGBT Analysis Summary

Sample Size: {mgbt_result.n_total}
Low Outliers Detected: {mgbt_result.klow}
Threshold: {mgbt_result.threshold:.3f}

Significance Levels:
  α₁ = {mgbt_result.alpha1}
  α₁₀ = {mgbt_result.alpha10}

Data Statistics:
  Min: {np.min(mgbt_result.data_sorted):.3f}
  Max: {np.max(mgbt_result.data_sorted):.3f}
  Mean: {np.mean(mgbt_result.data_sorted):.3f}
  Std: {np.std(mgbt_result.data_sorted):.3f}
"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
