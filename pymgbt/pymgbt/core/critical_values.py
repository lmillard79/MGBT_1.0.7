"""
Critical value computations for MGBT analysis.

This module implements functions to compute critical values for the MGBT test,
including the CritK function and related utilities.
"""

import numpy as np
from scipy import optimize
from typing import Optional, Union
import warnings

from ..stats.pvalue import kth_order_pvalue_ortho_t, peta


def crit_k(n: int, r: int, p: float, method: str = "adaptive") -> float:
    """
    Compute critical value for k-th order statistic at significance level p.
    
    This function finds the critical value eta such that P(W_r ≤ eta) = p,
    where W_r is the r-th order statistic test statistic.
    
    Parameters
    ----------
    n : int
        Total sample size
    r : int
        Order statistic position (1-indexed)
    p : float
        Significance level (probability)
    method : str, default="adaptive"
        Method for p-value calculation ("adaptive" or "gaussian")
        
    Returns
    -------
    float
        Critical value eta
        
    Notes
    -----
    This implements the CritK function from the original R code using
    a two-stage root-finding approach:
    1. Initial guess based on 50th percentile approximation
    2. Refined search using the full p-value calculation
    """
    # Validate inputs
    if n <= 2:
        raise ValueError("Sample size n must be > 2")
    if r <= 0 or r > n:
        raise ValueError(f"Order statistic r must be in [1, n], got r={r}")
    if not (0 < p < 1):
        raise ValueError(f"Significance level p must be in (0, 1), got {p}")
    
    # Stage 1: Get initial guess using 50th percentile approximation
    def peta_50th_percentile(eta: float) -> float:
        """Approximate p-value using 50th percentile of integration domain."""
        return peta(0.5, n, r, eta) - p
    
    try:
        # Find initial guess
        initial_guess = optimize.brentq(
            peta_50th_percentile,
            a=-10.0,
            b=10.0,
            xtol=1e-6
        )
    except ValueError:
        # If brentq fails, use a broader search
        warnings.warn("Initial guess search failed, using fallback method", RuntimeWarning)
        initial_guess = 0.0
    
    # Stage 2: Refine using full p-value calculation
    def pvalue_difference(eta: float) -> float:
        """Difference between computed p-value and target p."""
        computed_p = kth_order_pvalue_ortho_t(n, r, eta, method=method)
        return computed_p - p
    
    try:
        # Search around initial guess
        search_range = 1.0
        lower_bound = initial_guess - search_range
        upper_bound = initial_guess + search_range
        
        # Expand search range if needed
        max_iterations = 10
        for i in range(max_iterations):
            try:
                critical_value = optimize.brentq(
                    pvalue_difference,
                    a=lower_bound,
                    b=upper_bound,
                    xtol=1e-8
                )
                return critical_value
            except ValueError:
                # Expand search range
                search_range *= 2
                lower_bound = initial_guess - search_range
                upper_bound = initial_guess + search_range
                
                if search_range > 50:
                    break
        
        # If all else fails, use a very broad search
        critical_value = optimize.brentq(
            pvalue_difference,
            a=-20.0,
            b=20.0,
            xtol=1e-6
        )
        return critical_value
        
    except Exception as e:
        warnings.warn(
            f"Critical value computation failed: {e}. "
            f"Returning approximate value based on normal distribution.",
            RuntimeWarning
        )
        # Fallback to normal approximation
        from scipy import stats
        return stats.norm.ppf(p)


def crit_k10(n: int, alpha: float = 0.10) -> np.ndarray:
    """
    Compute critical values for multiple order statistics at 10% significance level.
    
    This function computes critical values for the first several order statistics,
    which is useful for the MGBT algorithm's sequential testing procedure.
    
    Parameters
    ----------
    n : int
        Sample size
    alpha : float, default=0.10
        Significance level
        
    Returns
    -------
    np.ndarray
        Array of critical values for r = 1, 2, ..., min(10, n//2)
        
    Notes
    -----
    This corresponds to the critK10 function in the original R code.
    """
    if n <= 2:
        raise ValueError("Sample size n must be > 2")
    if not (0 < alpha < 1):
        raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
    
    # Determine how many critical values to compute
    max_r = min(10, n // 2)
    
    critical_values = np.zeros(max_r)
    
    for r in range(1, max_r + 1):
        try:
            critical_values[r - 1] = crit_k(n, r, alpha)
        except Exception as e:
            warnings.warn(
                f"Failed to compute critical value for r={r}: {e}. "
                f"Using normal approximation.",
                RuntimeWarning
            )
            from scipy import stats
            critical_values[r - 1] = stats.norm.ppf(alpha)
    
    return critical_values


def monte_carlo_critical_values(
    n: int,
    r_values: Union[int, list] = None,
    alpha_levels: Union[float, list] = None,
    n_simulations: int = 10000,
    seed: Optional[int] = None
) -> dict:
    """
    Compute critical values using Monte Carlo simulation.
    
    This function provides an alternative method for computing critical values
    using Monte Carlo simulation, which can be useful for validation or when
    the analytical method fails.
    
    Parameters
    ----------
    n : int
        Sample size
    r_values : int or list, optional
        Order statistic positions to compute. If None, uses [1, 2, 3]
    alpha_levels : float or list, optional
        Significance levels. If None, uses [0.01, 0.05, 0.10]
    n_simulations : int, default=10000
        Number of Monte Carlo simulations
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary with critical values organized by r and alpha
        
    Notes
    -----
    This implements a Monte Carlo version similar to the CritValuesMC
    function in the original R code.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Set default values
    if r_values is None:
        r_values = [1, 2, 3]
    elif isinstance(r_values, int):
        r_values = [r_values]
    
    if alpha_levels is None:
        alpha_levels = [0.01, 0.05, 0.10]
    elif isinstance(alpha_levels, (int, float)):
        alpha_levels = [alpha_levels]
    
    results = {}
    
    for r in r_values:
        if r > n // 2:
            continue
            
        # Generate test statistics under null hypothesis
        test_statistics = []
        
        for _ in range(n_simulations):
            # Generate standard normal sample
            sample = np.random.standard_normal(n)
            sample_sorted = np.sort(sample)
            
            # Compute test statistic for r-th order statistic
            remaining_data = sample_sorted[r:]
            if len(remaining_data) >= 2:
                mean_remaining = np.mean(remaining_data)
                std_remaining = np.std(remaining_data, ddof=1)
                
                if std_remaining > 0:
                    w_stat = (sample_sorted[r-1] - mean_remaining) / std_remaining
                    test_statistics.append(w_stat)
        
        # Compute empirical quantiles
        test_statistics = np.array(test_statistics)
        results[r] = {}
        
        for alpha in alpha_levels:
            quantile = np.percentile(test_statistics, 100 * alpha)
            results[r][alpha] = quantile
    
    return results


def validate_critical_value_parameters(n: int, r: int, p: float) -> None:
    """
    Validate parameters for critical value calculations.
    
    Parameters
    ----------
    n : int
        Sample size
    r : int
        Order statistic position
    p : float
        Significance level
        
    Raises
    ------
    ValueError
        If parameters are invalid
    """
    if n <= 2:
        raise ValueError("Sample size n must be > 2 for critical value calculations")
    
    if r <= 0 or r > n // 2:
        raise ValueError(f"Order statistic r must be in [1, n//2], got r={r}, n={n}")
    
    if not (0 < p < 1):
        raise ValueError(f"Significance level p must be in (0, 1), got {p}")
    
    # Warn about extreme cases
    if p < 0.001 or p > 0.5:
        warnings.warn(
            f"Extreme significance level p={p} may cause numerical issues",
            UserWarning
        )
    
    if r > n // 4:
        warnings.warn(
            f"Large number of outliers r={r} relative to sample size n={n} "
            "may reduce critical value accuracy",
            UserWarning
        )
