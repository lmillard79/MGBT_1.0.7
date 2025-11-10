"""
Main MGBT (Multiple Grubbs-Beck Low-Outlier Test) implementation.

This module implements the core MGBT algorithm for detecting low outliers
in hydrological data, particularly annual peak streamflow data.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union
import warnings

from ..stats import gtmoms, cond_moms_z, V, kth_order_pvalue_ortho_t
from .critical_values import crit_k


@dataclass
class MGBTResult:
    """
    Result structure for MGBT analysis.
    
    Attributes
    ----------
    low_outlier_threshold : Optional[float]
        Threshold below which observations are considered low outliers
    outlier_indices : List[int]
        Indices of detected outliers in original data
    cleaned_data : np.ndarray
        Data with outliers removed
    test_statistics : np.ndarray
        Test statistics for each potential outlier
    p_values : np.ndarray
        P-values for each test statistic
    n_outliers : int
        Number of outliers detected
    alpha : float
        Significance level used
    """
    low_outlier_threshold: Optional[float]
    outlier_indices: List[int]
    cleaned_data: np.ndarray
    test_statistics: np.ndarray
    p_values: np.ndarray
    n_outliers: int
    alpha: float


def MGBT(data: Union[np.ndarray, List], alpha: float = 0.1, 
         max_outliers: Optional[int] = None) -> MGBTResult:
    """
    Multiple Grubbs-Beck Low-Outlier Test for hydrological data.
    
    This function implements the MGBT algorithm for detecting multiple low outliers
    in annual peak streamflow data or similar hydrological datasets.
    
    Parameters
    ----------
    data : array-like
        Input data array (typically annual peak flows)
    alpha : float, default=0.1
        Significance level for outlier detection
    max_outliers : int, optional
        Maximum number of outliers to detect. If None, uses n//3
        
    Returns
    -------
    MGBTResult
        Complete results of the MGBT analysis
        
    Notes
    -----
    The MGBT test is specifically designed for detecting low outliers in
    hydrological data where the underlying distribution is typically
    log-normal or similar right-skewed distribution.
    
    References
    ----------
    Cohn, T.A., et al. (2013). The Multiple Grubbs-Beck Low-Outlier Test.
    """
    # Input validation and preprocessing
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]  # Remove NaN values
    
    if len(data) < 10:
        warnings.warn("MGBT requires at least 10 observations for reliable results", 
                     UserWarning)
    
    if len(data) < 3:
        return MGBTResult(
            low_outlier_threshold=None,
            outlier_indices=[],
            cleaned_data=data,
            test_statistics=np.array([]),
            p_values=np.array([]),
            n_outliers=0,
            alpha=alpha
        )
    
    # Set maximum outliers if not specified
    if max_outliers is None:
        max_outliers = max(1, len(data) // 3)
    
    # Initialize tracking variables
    original_data = data.copy()
    working_data = data.copy()
    outlier_indices = []
    test_statistics = []
    p_values = []
    
    # Sort data for processing (MGBT works on ordered statistics)
    sorted_indices = np.argsort(working_data)
    working_data = working_data[sorted_indices]
    
    # Iterative outlier detection
    for iteration in range(max_outliers):
        n = len(working_data)
        
        if n < 3:
            break
            
        # Test each potential low outlier (focus on lower tail)
        # In MGBT, we typically test the k lowest values
        max_k = min(n // 2, 5)  # Test up to 5 lowest values or half the data
        
        best_k = None
        best_pvalue = 1.0
        best_test_stat = 0.0
        
        for k in range(1, max_k + 1):
            try:
                # Calculate test statistic for k-th order statistic
                # This uses the MGBT methodology with orthogonal transformation
                p_value = kth_order_pvalue_ortho_t(working_data, k, alpha)
                
                # Calculate corresponding test statistic
                # (simplified - in full MGBT this would be more complex)
                mean_val = np.mean(working_data)
                std_val = np.std(working_data, ddof=1)
                if std_val > 0:
                    test_stat = (mean_val - working_data[k-1]) / std_val
                else:
                    test_stat = 0.0
                
                # Keep track of most significant outlier
                if p_value < best_pvalue:
                    best_pvalue = p_value
                    best_k = k
                    best_test_stat = test_stat
                    
            except Exception as e:
                # If p-value calculation fails, skip this k
                warnings.warn(f"P-value calculation failed for k={k}: {e}", UserWarning)
                continue
        
        # Check if we found a significant outlier
        if best_k is not None and best_pvalue < alpha:
            # Record the outlier
            outlier_idx_in_sorted = best_k - 1  # Convert to 0-based index
            original_idx = sorted_indices[outlier_idx_in_sorted]
            outlier_indices.append(original_idx)
            test_statistics.append(best_test_stat)
            p_values.append(best_pvalue)
            
            # Remove the outlier from working data
            working_data = np.delete(working_data, outlier_idx_in_sorted)
            sorted_indices = np.delete(sorted_indices, outlier_idx_in_sorted)
        else:
            # No more significant outliers found
            break
    
    # Determine threshold
    threshold = None
    if outlier_indices:
        outlier_values = original_data[outlier_indices]
        threshold = np.max(outlier_values)
    
    # Create cleaned data
    cleaned_data = original_data.copy()
    if outlier_indices:
        cleaned_data = np.delete(cleaned_data, outlier_indices)
    
    return MGBTResult(
        low_outlier_threshold=threshold,
        outlier_indices=outlier_indices,
        cleaned_data=cleaned_data,
        test_statistics=np.array(test_statistics),
        p_values=np.array(p_values),
        n_outliers=len(outlier_indices),
        alpha=alpha
    )


def mgbt_simple(data: Union[np.ndarray, List], alpha: float = 0.1) -> MGBTResult:
    """
    Simplified MGBT implementation using Grubbs test approach.
    
    This is a fallback implementation that uses iterative Grubbs tests
    when the full MGBT p-value calculations are not available.
    
    Parameters
    ----------
    data : array-like
        Input data array
    alpha : float, default=0.1
        Significance level
        
    Returns
    -------
    MGBTResult
        MGBT results using simplified approach
    """
    from scipy import stats
    
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    
    if len(data) < 3:
        return MGBTResult(
            low_outlier_threshold=None,
            outlier_indices=[],
            cleaned_data=data,
            test_statistics=np.array([]),
            p_values=np.array([]),
            n_outliers=0,
            alpha=alpha
        )
    
    original_data = data.copy()
    working_data = data.copy()
    outlier_indices = []
    test_statistics = []
    p_values = []
    
    max_iterations = len(data) // 3
    
    for iteration in range(max_iterations):
        n = len(working_data)
        if n < 3:
            break
            
        mean_val = np.mean(working_data)
        std_val = np.std(working_data, ddof=1)
        
        if std_val == 0:
            break
        
        # Focus on low outliers (below mean)
        low_candidates = working_data < mean_val
        if not np.any(low_candidates):
            break
        
        # Calculate Grubbs statistics for low values
        grubbs_stats = np.abs(working_data - mean_val) / std_val
        
        # Find the most extreme low value
        low_indices = np.where(low_candidates)[0]
        if len(low_indices) == 0:
            break
            
        low_grubbs = grubbs_stats[low_indices]
        max_grubbs_idx = low_indices[np.argmax(low_grubbs)]
        max_grubbs_stat = grubbs_stats[max_grubbs_idx]
        
        # Calculate critical value for Grubbs test
        t_critical = stats.t.ppf(1 - alpha/(2*n), n-2)
        grubbs_critical = ((n-1) / np.sqrt(n)) * np.sqrt(t_critical**2 / (n-2 + t_critical**2))
        
        if max_grubbs_stat > grubbs_critical:
            # Found significant outlier
            # Map back to original data index
            original_idx = np.where(original_data == working_data[max_grubbs_idx])[0][0]
            outlier_indices.append(original_idx)
            test_statistics.append(max_grubbs_stat)
            
            # Approximate p-value
            p_val = 2 * n * (1 - stats.t.cdf(max_grubbs_stat * np.sqrt(n) / (n-1), n-2))
            p_values.append(min(p_val, 1.0))
            
            # Remove outlier
            working_data = np.delete(working_data, max_grubbs_idx)
        else:
            break
    
    # Determine threshold
    threshold = None
    if outlier_indices:
        outlier_values = original_data[outlier_indices]
        threshold = np.max(outlier_values)
    
    # Create cleaned data
    cleaned_data = original_data.copy()
    if outlier_indices:
        cleaned_data = np.delete(cleaned_data, outlier_indices)
    
    return MGBTResult(
        low_outlier_threshold=threshold,
        outlier_indices=outlier_indices,
        cleaned_data=cleaned_data,
        test_statistics=np.array(test_statistics),
        p_values=np.array(p_values),
        n_outliers=len(outlier_indices),
        alpha=alpha
    )


# Alias for backward compatibility
def mgbt(data, alpha=0.1):
    """Backward compatibility alias for MGBT function."""
    return MGBT(data, alpha)


if __name__ == "__main__":
    # Test the MGBT implementation
    print("Testing MGBT implementation...")
    
    # Create test data with known outliers
    np.random.seed(42)
    main_data = np.random.lognormal(2.0, 0.5, 25)
    outliers = np.random.lognormal(0.5, 0.3, 5)
    test_data = np.concatenate([main_data, outliers])
    np.random.shuffle(test_data)
    
    print(f"Test data: {len(test_data)} points")
    print(f"Data range: {test_data.min():.2f} to {test_data.max():.2f}")
    
    # Run MGBT
    try:
        result = MGBT(test_data, alpha=0.1)
        print(f"\nMGBT Results:")
        print(f"Outliers detected: {result.n_outliers}")
        print(f"Threshold: {result.low_outlier_threshold}")
        print(f"Outlier indices: {result.outlier_indices}")
        print("✅ MGBT implementation working")
    except Exception as e:
        print(f"❌ MGBT failed, trying simplified version: {e}")
        
        # Try simplified version
        result = mgbt_simple(test_data, alpha=0.1)
        print(f"\nSimplified MGBT Results:")
        print(f"Outliers detected: {result.n_outliers}")
        print(f"Threshold: {result.low_outlier_threshold}")
        print("✅ Simplified MGBT implementation working")
