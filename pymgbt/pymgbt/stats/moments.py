"""
Moments calculations for truncated normal distributions.

This module implements the gtmoms function and conditional moments calculations
that are essential for the MGBT algorithm.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Union


def gtmoms(xsi: float, k: int) -> float:
    """
    Compute k-th moment of observations above threshold xsi for standard normal.
    
    This function calculates the k-th moment of a standard normal distribution
    truncated below at xsi (i.e., moments of X given X > xsi where X ~ N(0,1)).
    
    Parameters
    ----------
    xsi : float
        Truncation threshold (standardized)
    k : int
        Moment order (0, 1, 2, ...)
        
    Returns
    -------
    float
        k-th moment of truncated standard normal distribution
        
    Notes
    -----
    This implements the recursive formula from the original R code:
    - gtmoms(xsi, 0) = 1
    - gtmoms(xsi, 1) = H(xsi) where H(xsi) = φ(xsi)/(1-Φ(xsi))
    - gtmoms(xsi, k) = (k-1)*gtmoms(xsi, k-2) + H(xsi)*xsi^(k-1) for k > 1
    
    where φ is the standard normal PDF and Φ is the standard normal CDF.
    """
    def H(x: float) -> float:
        """Hazard function: φ(x)/(1-Φ(x))"""
        return stats.norm.pdf(x) / (1 - stats.norm.cdf(x))
    
    if k == 0:
        return 1.0
    elif k == 1:
        return H(xsi)
    elif k > 1:
        return (k - 1) * gtmoms(xsi, k - 2) + H(xsi) * (xsi ** (k - 1))
    else:
        raise ValueError("Moment order k must be non-negative integer")


def cond_moms_z(n: int, r: int, xsi: float) -> np.ndarray:
    """
    Conditional moments of mean and variance for observations above threshold.
    
    Computes the conditional mean and variance of the sample mean and variance
    for observations above the threshold xsi.
    
    Parameters
    ----------
    n : int
        Total sample size
    r : int
        Number of observations below threshold
    xsi : float
        Threshold value (standardized)
        
    Returns
    -------
    np.ndarray
        Array with [conditional_mean, conditional_variance/(n-r)]
        
    Notes
    -----
    This corresponds to the CondMomsZ function in the original R code.
    """
    mu1 = gtmoms(xsi, 1)
    mu2 = gtmoms(xsi, 2)
    
    conditional_mean = mu1
    conditional_variance_scaled = (mu2 - mu1**2) / (n - r)
    
    return np.array([conditional_mean, conditional_variance_scaled])


def cond_moms_chi2(n: int, r: int, xsi: float) -> np.ndarray:
    """
    Conditional moments for chi-squared related calculations.
    
    Computes moments needed for chi-squared approximations in the MGBT algorithm.
    
    Parameters
    ----------
    n : int
        Total sample size
    r : int
        Number of observations below threshold
    xsi : float
        Threshold value (standardized)
        
    Returns
    -------
    np.ndarray
        Array with [variance_moment, variance_of_variance]
        
    Notes
    -----
    This corresponds to the CondMomsChi2 function in the original R code.
    The second element comes from the V function's [2,2] component.
    """
    from .covariance import V  # Import here to avoid circular imports
    
    mu1 = gtmoms(xsi, 1)
    mu2 = gtmoms(xsi, 2)
    
    # Convert xsi back to probability for V function
    p_threshold = stats.norm.cdf(xsi)
    
    variance_moment = mu2 - mu1**2
    covariance_matrix = V(n, r, p_threshold)
    variance_of_variance = covariance_matrix[1, 1]
    
    return np.array([variance_moment, variance_of_variance])


def validate_moment_parameters(n: int, r: int, xsi: float, k: int = None) -> None:
    """
    Validate parameters for moment calculations.
    
    Parameters
    ----------
    n : int
        Sample size
    r : int
        Number of truncated observations
    xsi : float
        Threshold value
    k : int, optional
        Moment order
        
    Raises
    ------
    ValueError
        If parameters are invalid
    """
    if n <= 0:
        raise ValueError("Sample size n must be positive")
    
    if r < 0 or r >= n:
        raise ValueError(f"Number of truncated observations r must be in [0, n-1], got r={r}, n={n}")
    
    if not np.isfinite(xsi):
        raise ValueError("Threshold xsi must be finite")
    
    if k is not None and k < 0:
        raise ValueError("Moment order k must be non-negative")
    
    # Warn about extreme thresholds that might cause numerical issues
    if xsi > 8:
        import warnings
        warnings.warn(
            f"Very large threshold xsi={xsi:.2f} may cause numerical instability",
            UserWarning
        )
    elif xsi < -8:
        import warnings
        warnings.warn(
            f"Very small threshold xsi={xsi:.2f} may cause numerical instability",
            UserWarning
        )
