"""
P-value calculations for MGBT orthogonal evaluation.

This module implements the most complex part of the MGBT algorithm: the orthogonal
p-value evaluation using integration and non-central t-distribution calculations.
"""

import numpy as np
from scipy import stats, integrate
from typing import Union, Callable
import warnings

from .moments import gtmoms
from .covariance import VMS, EMS
from .moments import cond_moms_chi2


def peta(pzr: float, n: int, r: int, eta: float) -> float:
    """
    Integrand function for orthogonal p-value evaluation.
    
    This is the core integrand function used in the KthOrderPValueOrthoT
    calculation. It implements the complex statistical computation involving
    non-central t-distributions.
    
    Parameters
    ----------
    pzr : float
        Beta distribution quantile parameter
    n : int
        Total sample size
    r : int
        Order statistic position (1-indexed)
    eta : float
        Studentized test statistic
        
    Returns
    -------
    float
        Integrand value for p-value calculation
        
    Notes
    -----
    This implements the peta function from the original R code, which involves:
    1. Beta distribution quantile transformation
    2. Covariance matrix calculations
    3. Non-central t-distribution evaluation
    """
    try:
        # Convert beta quantile to normal quantile
        zr = stats.norm.ppf(stats.beta.ppf(pzr, a=r, b=n + 1 - r))
        
        # Get covariance matrix and expected values
        qmin = stats.norm.cdf(zr)
        CV = VMS(n, r, qmin)
        EMp = EMS(n, r, qmin)
        
        # Compute lambda (regression coefficient)
        lambda_coef = CV[0, 1] / CV[1, 1]
        
        # Adjusted eta
        etap = eta + lambda_coef
        
        # Conditional mean
        muMp = EMp[0] - lambda_coef * EMp[1]
        
        # Conditional standard deviation
        SigmaMp = np.sqrt(CV[0, 0] - CV[0, 1]**2 / CV[1, 1])
        
        # Chi-squared moments for degrees of freedom calculation
        mom_s2 = cond_moms_chi2(n, r, zr)
        shape = mom_s2[0]**2 / mom_s2[1]
        scale = mom_s2[1] / mom_s2[0]
        
        sqrt_s2 = np.sqrt(mom_s2[0])
        df = 2 * shape
        
        # Non-centrality parameter
        ncp = (muMp - zr) / SigmaMp
        
        # Test statistic for t-distribution
        q = -(sqrt_s2 / SigmaMp) * etap
        
        # Return complementary CDF of non-central t-distribution
        return 1 - stats.nct.cdf(q, df=df, nc=ncp)
        
    except (ValueError, RuntimeError, OverflowError) as e:
        # Handle numerical issues gracefully
        warnings.warn(f"Numerical issue in peta calculation: {e}", RuntimeWarning)
        return 0.0


def kth_order_pvalue_ortho_t(n: int, r: int, eta: float, 
                            method: str = "adaptive") -> float:
    """
    Compute p-value for k-th order statistic using orthogonal evaluation.
    
    This is the main p-value calculation function that implements the orthogonal
    evaluation method described in Cohn et al. (2013). It integrates the peta
    function over the appropriate domain.
    
    Parameters
    ----------
    n : int
        Total sample size
    r : int
        Order statistic position (1-indexed)
    eta : float
        Studentized test statistic value
    method : str, default="adaptive"
        Integration method ("adaptive" or "gaussian")
        
    Returns
    -------
    float
        P-value for the test statistic
        
    Notes
    -----
    This implements the KthOrderPValueOrthoT function from the original R code.
    The integration is performed over [1e-7, 1-1e-7] to avoid numerical issues
    at the boundaries.
    
    References
    ----------
    Cohn, T.A., England, J.F., Berenbrock, C.E., Mason, R.R., Stedinger, J.R., 
    and Lamontagne, J.R., 2013, A generalized Grubbs-Beck test statistic for 
    detecting multiple potentially influential low outliers in flood series: 
    Water Resources Research, v. 49, no. 8, p. 5047-5058.
    """
    # Validate inputs
    if n <= 0:
        raise ValueError("Sample size n must be positive")
    if r <= 0 or r > n:
        raise ValueError(f"Order statistic r must be in [1, n], got r={r}, n={n}")
    if not np.isfinite(eta):
        raise ValueError("Test statistic eta must be finite")
    
    # Integration bounds (avoid exact 0 and 1 for numerical stability)
    lower_bound = 1e-7
    upper_bound = 1 - 1e-7
    
    # Define integrand function
    def integrand(pzr_val: float) -> float:
        return peta(pzr_val, n, r, eta)
    
    try:
        if method == "adaptive":
            # Use adaptive quadrature (similar to R's integrate function)
            result, error = integrate.quad(
                integrand, 
                lower_bound, 
                upper_bound,
                epsabs=1e-10,
                epsrel=1e-8,
                limit=100
            )
            
            # Check integration error
            if error > 1e-6:
                warnings.warn(
                    f"Large integration error: {error:.2e}. "
                    "P-value may be inaccurate.",
                    RuntimeWarning
                )
            
            return max(0.0, min(1.0, result))  # Ensure valid probability
            
        elif method == "gaussian":
            # Use Gaussian quadrature (faster but potentially less accurate)
            return _gaussian_quadrature_pvalue(n, r, eta)
            
        else:
            raise ValueError(f"Unknown integration method: {method}")
            
    except Exception as e:
        warnings.warn(
            f"P-value calculation failed: {e}. Returning conservative value.",
            RuntimeWarning
        )
        return 1.0  # Conservative p-value


def _gaussian_quadrature_pvalue(n: int, r: int, eta: float, 
                               n_points: int = 50) -> float:
    """
    Compute p-value using Gaussian quadrature integration.
    
    This is a faster but potentially less accurate alternative to adaptive
    quadrature, corresponding to the KthOrderPValueOrthoTb function in R.
    
    Parameters
    ----------
    n : int
        Sample size
    r : int
        Order statistic position
    eta : float
        Test statistic
    n_points : int, default=50
        Number of quadrature points
        
    Returns
    -------
    float
        P-value estimate
    """
    try:
        # Get Gaussian quadrature points and weights for [0,1]
        # Transform from [-1,1] to [0,1]
        points, weights = np.polynomial.legendre.leggauss(n_points)
        points = 0.5 * (points + 1)  # Transform to [0,1]
        weights = 0.5 * weights      # Adjust weights
        
        # Avoid exact boundaries
        points = np.clip(points, 1e-7, 1 - 1e-7)
        
        # Evaluate integrand at quadrature points
        integrand_values = np.array([peta(p, n, r, eta) for p in points])
        
        # Compute weighted sum
        result = np.sum(integrand_values * weights)
        
        return max(0.0, min(1.0, result))
        
    except Exception as e:
        warnings.warn(
            f"Gaussian quadrature p-value calculation failed: {e}",
            RuntimeWarning
        )
        return 1.0


def validate_pvalue_parameters(n: int, r: int, eta: float) -> None:
    """
    Validate parameters for p-value calculations.
    
    Parameters
    ----------
    n : int
        Sample size
    r : int
        Order statistic position
    eta : float
        Test statistic
        
    Raises
    ------
    ValueError
        If parameters are invalid
    """
    if n < 3:
        raise ValueError("Sample size n must be at least 3 for p-value calculations")
    
    if r <= 0 or r > n // 2:
        raise ValueError(f"Order statistic r must be in [1, n//2], got r={r}, n={n}")
    
    if not np.isfinite(eta):
        raise ValueError("Test statistic eta must be finite")
    
    # Warn about extreme values that might cause numerical issues
    if abs(eta) > 10:
        warnings.warn(
            f"Extreme test statistic |eta|={abs(eta):.2f} may cause numerical issues",
            UserWarning
        )
    
    if r > n // 3:
        warnings.warn(
            f"Large number of outliers r={r} relative to sample size n={n} "
            "may reduce test reliability",
            UserWarning
        )
