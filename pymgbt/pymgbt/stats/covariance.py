"""
Covariance matrix calculations for MGBT analysis.

This module implements the V, VMS, and EMS functions that compute covariance
matrices and expected values for the statistical components of MGBT.
"""

import numpy as np
from scipy import stats, special
from typing import Tuple
from .moments import gtmoms


def V(n: int, r: int, qmin: float) -> np.ndarray:
    """
    Covariance matrix of sample mean (M) and sample variance (S²).
    
    Computes the 2x2 covariance matrix for the sample mean and sample variance
    of observations above the threshold.
    
    Parameters
    ----------
    n : int
        Total sample size
    r : int
        Number of observations below threshold
    qmin : float
        Probability threshold (P(X ≤ threshold))
        
    Returns
    -------
    np.ndarray
        2x2 covariance matrix [[Var(M), Cov(M,S²)], [Cov(M,S²), Var(S²)]]
        
    Notes
    -----
    This implements the V function from the original R code, computing the
    covariance matrix of M and S² where:
    - M is the sample mean of observations above threshold
    - S² is the sample variance of observations above threshold
    """
    n2 = n - r  # Number of observations above threshold
    
    if n2 <= 1:
        raise ValueError("Need at least 2 observations above threshold")
    
    # Convert probability threshold to standardized value
    zr = stats.norm.ppf(qmin)
    
    # Compute moments E[X^k] for k=1,2,3,4 given X > zr
    E = np.array([gtmoms(zr, k) for k in range(1, 5)])
    
    # Central moments
    cm = np.array([
        E[0],  # μ
        E[1] - E[0]**2,  # σ²
        E[2] - 3*E[1]*E[0] + 2*E[0]**3,  # μ₃
        E[3] - 4*E[2]*E[0] + 6*E[1]*E[0]**2 - 3*E[0]**4  # μ₄
    ])
    
    # Covariance matrix elements
    var_m = (E[1] - E[0]**2) / n2
    
    cov_m_s2 = (E[2] - 3*E[0]*E[1] + 2*E[0]**3) / np.sqrt(n2 * (n2 - 1))
    
    var_s2 = (cm[3] - cm[1]**2) / n2 + 2 / ((n2 - 1) * n2) * cm[1]**2
    
    return np.array([
        [var_m, cov_m_s2],
        [cov_m_s2, var_s2]
    ])


def EMS(n: int, r: int, qmin: float) -> np.ndarray:
    """
    Expected values of sample mean (M) and sample standard deviation (S).
    
    Computes the expected values of the sample mean and sample standard
    deviation for observations above the threshold.
    
    Parameters
    ----------
    n : int
        Total sample size
    r : int
        Number of observations below threshold
    qmin : float
        Probability threshold (P(X ≤ threshold))
        
    Returns
    -------
    np.ndarray
        Array [E[M], E[S]] where M is sample mean, S is sample std dev
        
    Notes
    -----
    This implements the EMS function from the original R code.
    Uses gamma function approximation for E[S] from E[S²].
    """
    # Convert probability threshold to standardized value
    zr = stats.norm.ppf(qmin)
    
    # Expected value of sample mean
    Em = gtmoms(zr, 1)
    
    # For expected value of sample standard deviation, we use the
    # chi-squared approximation from the original R code
    from .moments import cond_moms_chi2
    mom_s2 = cond_moms_chi2(n, r, zr)
    
    # Parameters for gamma approximation
    alpha = mom_s2[0]**2 / mom_s2[1]
    beta = mom_s2[1] / mom_s2[0]
    
    # E[S] = sqrt(beta) * Γ(α + 0.5) / Γ(α)
    Es = np.sqrt(beta) * np.exp(special.loggamma(alpha + 0.5) - special.loggamma(alpha))
    
    return np.array([Em, Es])


def VMS(n: int, r: int, qmin: float) -> np.ndarray:
    """
    Covariance matrix of sample mean (M) and sample standard deviation (S).
    
    Computes the 2x2 covariance matrix for the sample mean and sample
    standard deviation of observations above the threshold.
    
    Parameters
    ----------
    n : int
        Total sample size
    r : int
        Number of observations below threshold
    qmin : float
        Probability threshold (P(X ≤ threshold))
        
    Returns
    -------
    np.ndarray
        2x2 covariance matrix [[Var(M), Cov(M,S)], [Cov(M,S), Var(S)]]
        
    Notes
    -----
    This implements the VMS function from the original R code, transforming
    the covariance matrix from (M, S²) to (M, S).
    """
    # Convert probability threshold to standardized value
    zr = stats.norm.ppf(qmin)
    
    # Get moments
    E = np.array([gtmoms(zr, k) for k in range(1, 3)])
    
    # Expected values
    ems_values = EMS(n, r, qmin)
    Es = ems_values[1]
    
    # Variance of S² and covariance matrix for (M, S²)
    Es2 = E[1] - E[0]**2  # E[S²]
    V2 = V(n, r, qmin)
    
    # Transform to covariance matrix for (M, S)
    var_m = V2[0, 0]
    cov_m_s = V2[0, 1] / (2 * Es)  # ∂S/∂S² = 1/(2√S²) = 1/(2S)
    var_s = Es2 - Es**2  # Var(S) ≈ E[S²] - E[S]²
    
    return np.array([
        [var_m, cov_m_s],
        [cov_m_s, var_s]
    ])


def validate_covariance_parameters(n: int, r: int, qmin: float) -> None:
    """
    Validate parameters for covariance calculations.
    
    Parameters
    ----------
    n : int
        Sample size
    r : int
        Number of truncated observations
    qmin : float
        Probability threshold
        
    Raises
    ------
    ValueError
        If parameters are invalid
    """
    if n <= 2:
        raise ValueError("Sample size n must be > 2 for covariance calculations")
    
    if r < 0 or r >= n - 1:
        raise ValueError(f"Number of truncated observations r must be in [0, n-2], got r={r}, n={n}")
    
    if not (0 < qmin < 1):
        raise ValueError(f"Probability threshold qmin must be in (0, 1), got {qmin}")
    
    # Check for extreme probabilities that might cause numerical issues
    if qmin < 1e-10 or qmin > 1 - 1e-10:
        import warnings
        warnings.warn(
            f"Extreme probability threshold qmin={qmin:.2e} may cause numerical instability",
            UserWarning
        )
