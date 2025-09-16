"""
Flood frequency evolution plotting functions for MGBT analysis.

This module provides visualization capabilities for flood frequency evolution
and related statistical analyses.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any
import warnings

from ..core.mgbt import MGBTResult


def plot_ffq_evolution(
    data: np.ndarray,
    mgbt_result: Optional[MGBTResult] = None,
    years: Optional[np.ndarray] = None,
    window_size: int = 20,
    step_size: int = 5,
    figsize: Tuple[float, float] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot flood frequency quantile evolution over time.
    
    Creates a visualization showing how flood frequency quantiles change
    over time using a moving window approach, with MGBT outlier identification.
    
    Parameters
    ----------
    data : np.ndarray
        Annual peak streamflow data
    mgbt_result : MGBTResult, optional
        Results from MGBT analysis
    years : np.ndarray, optional
        Years corresponding to data
    window_size : int, default=20
        Size of moving window for quantile calculation
    step_size : int, default=5
        Step size for moving window
    figsize : tuple, default=(14, 10)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
        
    Notes
    -----
    This implements functionality similar to the plotFFQevol function
    in the original R MGBT package.
    """
    if len(data) < window_size:
        raise ValueError(f"Data length ({len(data)}) must be >= window_size ({window_size})")
    
    # Set up years if not provided
    if years is None:
        years = np.arange(1, len(data) + 1)
    elif len(years) != len(data):
        raise ValueError("Years array must have same length as data")
    
    # Calculate moving window statistics
    n_windows = (len(data) - window_size) // step_size + 1
    window_centers = []
    quantiles_10 = []
    quantiles_50 = []
    quantiles_90 = []
    means = []
    
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        window_data = data[start_idx:end_idx]
        window_years = years[start_idx:end_idx]
        
        # Calculate center year of window
        center_year = np.mean(window_years)
        window_centers.append(center_year)
        
        # Calculate quantiles and statistics
        quantiles_10.append(np.percentile(window_data, 10))
        quantiles_50.append(np.percentile(window_data, 50))
        quantiles_90.append(np.percentile(window_data, 90))
        means.append(np.mean(window_data))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Time series with quantile evolution
    ax1 = axes[0, 0]
    ax1.plot(years, data, 'o-', color='lightblue', alpha=0.6, markersize=4, 
             label='Annual peaks')
    
    # Add MGBT outliers if available
    if mgbt_result is not None and mgbt_result.klow > 0:
        outlier_years = years[mgbt_result.outlier_indices]
        outlier_values = mgbt_result.outlier_values
        ax1.plot(outlier_years, outlier_values, 'ro', markersize=6, 
                label=f'MGBT outliers (n={mgbt_result.klow})')
        
        if mgbt_result.threshold > 0:
            ax1.axhline(y=mgbt_result.threshold, color='red', linestyle='--', 
                       alpha=0.7, label='MGBT threshold')
    
    # Plot quantile evolution
    ax1.plot(window_centers, quantiles_90, 'g-', linewidth=2, label='90th percentile')
    ax1.plot(window_centers, quantiles_50, 'b-', linewidth=2, label='50th percentile')
    ax1.plot(window_centers, quantiles_10, 'orange', linewidth=2, label='10th percentile')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Streamflow')
    ax1.set_title('Flood Frequency Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Quantile trends
    ax2 = axes[0, 1]
    ax2.plot(window_centers, quantiles_90, 'g-', linewidth=2, marker='o', 
             markersize=4, label='90th percentile')
    ax2.plot(window_centers, quantiles_50, 'b-', linewidth=2, marker='s', 
             markersize=4, label='50th percentile')
    ax2.plot(window_centers, quantiles_10, 'orange', linewidth=2, marker='^', 
             markersize=4, label='10th percentile')
    
    ax2.set_xlabel('Window Center Year')
    ax2.set_ylabel('Quantile Value')
    ax2.set_title('Quantile Trends Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Moving average
    ax3 = axes[1, 0]
    ax3.plot(years, data, 'o', color='lightgray', alpha=0.5, markersize=3, 
             label='Annual peaks')
    ax3.plot(window_centers, means, 'r-', linewidth=3, marker='o', 
             markersize=5, label=f'{window_size}-year moving average')
    
    if mgbt_result is not None and mgbt_result.klow > 0:
        ax3.plot(outlier_years, outlier_values, 'ro', markersize=6, 
                label='MGBT outliers')
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Streamflow')
    ax3.set_title('Moving Average Trend')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Coefficient of variation evolution
    ax4 = axes[1, 1]
    cv_values = []
    
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window_data = data[start_idx:end_idx]
        
        cv = np.std(window_data) / np.mean(window_data)
        cv_values.append(cv)
    
    ax4.plot(window_centers, cv_values, 'purple', linewidth=2, marker='d', 
             markersize=4)
    ax4.set_xlabel('Window Center Year')
    ax4.set_ylabel('Coefficient of Variation')
    ax4.set_title('Variability Evolution')
    ax4.grid(True, alpha=0.3)
    
    # Add overall mean CV line
    overall_cv = np.std(data) / np.mean(data)
    ax4.axhline(y=overall_cv, color='red', linestyle='--', alpha=0.7, 
               label=f'Overall CV = {overall_cv:.3f}')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_return_period_evolution(
    data: np.ndarray,
    mgbt_result: Optional[MGBTResult] = None,
    years: Optional[np.ndarray] = None,
    return_periods: List[float] = [2, 5, 10, 25, 50, 100],
    window_size: int = 30,
    step_size: int = 5,
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot return period quantile evolution over time.
    
    Parameters
    ----------
    data : np.ndarray
        Annual peak streamflow data
    mgbt_result : MGBTResult, optional
        Results from MGBT analysis
    years : np.ndarray, optional
        Years corresponding to data
    return_periods : list, default=[2, 5, 10, 25, 50, 100]
        Return periods to calculate and plot
    window_size : int, default=30
        Size of moving window
    step_size : int, default=5
        Step size for moving window
    figsize : tuple, default=(12, 8)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    if len(data) < window_size:
        raise ValueError(f"Data length ({len(data)}) must be >= window_size ({window_size})")
    
    # Set up years if not provided
    if years is None:
        years = np.arange(1, len(data) + 1)
    
    # Calculate moving window return period quantiles
    n_windows = (len(data) - window_size) // step_size + 1
    window_centers = []
    return_period_quantiles = {rp: [] for rp in return_periods}
    
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        window_data = data[start_idx:end_idx]
        window_years = years[start_idx:end_idx]
        
        center_year = np.mean(window_years)
        window_centers.append(center_year)
        
        # Calculate return period quantiles using Weibull plotting position
        sorted_data = np.sort(window_data)[::-1]  # Descending order
        n = len(sorted_data)
        
        for rp in return_periods:
            # Weibull plotting position: P = m/(n+1) where m is rank
            # For return period T: P = 1 - 1/T
            prob = 1 - 1/rp
            
            # Find quantile using linear interpolation
            rank = prob * (n + 1)
            
            if rank <= 1:
                quantile = sorted_data[0]
            elif rank >= n:
                quantile = sorted_data[-1]
            else:
                lower_idx = int(np.floor(rank)) - 1
                upper_idx = int(np.ceil(rank)) - 1
                weight = rank - np.floor(rank)
                quantile = (1 - weight) * sorted_data[lower_idx] + weight * sorted_data[upper_idx]
            
            return_period_quantiles[rp].append(quantile)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color map for different return periods
    colors = plt.cm.viridis(np.linspace(0, 1, len(return_periods)))
    
    # Plot return period evolution
    for i, rp in enumerate(return_periods):
        ax.plot(window_centers, return_period_quantiles[rp], 
               color=colors[i], linewidth=2, marker='o', markersize=3,
               label=f'{rp}-year return period')
    
    # Add MGBT outliers if available
    if mgbt_result is not None and mgbt_result.klow > 0:
        outlier_years = years[mgbt_result.outlier_indices]
        outlier_values = mgbt_result.outlier_values
        
        # Plot outliers at their actual years
        ax.scatter(outlier_years, outlier_values, color='red', s=50, 
                  marker='x', linewidth=2, label='MGBT outliers', zorder=10)
    
    ax.set_xlabel('Window Center Year')
    ax.set_ylabel('Discharge')
    ax.set_title('Return Period Quantile Evolution')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_trend_analysis(
    data: np.ndarray,
    years: Optional[np.ndarray] = None,
    mgbt_result: Optional[MGBTResult] = None,
    figsize: Tuple[float, float] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive trend analysis plots.
    
    Parameters
    ----------
    data : np.ndarray
        Annual peak streamflow data
    years : np.ndarray, optional
        Years corresponding to data
    mgbt_result : MGBTResult, optional
        Results from MGBT analysis
    figsize : tuple, default=(14, 10)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    if years is None:
        years = np.arange(1, len(data) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Time series with trend line
    ax1 = axes[0, 0]
    ax1.plot(years, data, 'bo-', alpha=0.7, markersize=4, label='Annual peaks')
    
    # Add linear trend
    z = np.polyfit(years, data, 1)
    p = np.poly1d(z)
    ax1.plot(years, p(years), 'r--', linewidth=2, 
            label=f'Trend: {z[0]:.2f}/year')
    
    # Add MGBT outliers
    if mgbt_result is not None and mgbt_result.klow > 0:
        outlier_years = years[mgbt_result.outlier_indices]
        outlier_values = mgbt_result.outlier_values
        ax1.plot(outlier_years, outlier_values, 'ro', markersize=6, 
                label='MGBT outliers')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Discharge')
    ax1.set_title('Time Series with Linear Trend')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Detrended data
    ax2 = axes[0, 1]
    detrended = data - p(years)
    ax2.plot(years, detrended, 'go-', alpha=0.7, markersize=4)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Detrended Discharge')
    ax2.set_title('Detrended Time Series')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Moving statistics
    ax3 = axes[1, 0]
    window = min(10, len(data) // 4)
    
    # Calculate moving mean and std
    moving_mean = []
    moving_std = []
    moving_years = []
    
    for i in range(window, len(data) - window + 1):
        window_data = data[i-window:i+window]
        moving_mean.append(np.mean(window_data))
        moving_std.append(np.std(window_data))
        moving_years.append(years[i])
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(moving_years, moving_mean, 'b-', linewidth=2, 
                    label='Moving mean')
    line2 = ax3_twin.plot(moving_years, moving_std, 'r-', linewidth=2, 
                         label='Moving std')
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Moving Mean', color='b')
    ax3_twin.set_ylabel('Moving Std', color='r')
    ax3.set_title(f'{2*window}-Year Moving Statistics')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Autocorrelation
    ax4 = axes[1, 1]
    
    # Calculate autocorrelation
    max_lag = min(20, len(data) // 4)
    lags = np.arange(0, max_lag + 1)
    autocorr = []
    
    for lag in lags:
        if lag == 0:
            autocorr.append(1.0)
        else:
            corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            autocorr.append(corr)
    
    ax4.bar(lags, autocorr, alpha=0.7, color='skyblue', edgecolor='navy')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='±0.2')
    ax4.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Lag (years)')
    ax4.set_ylabel('Autocorrelation')
    ax4.set_title('Autocorrelation Function')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
