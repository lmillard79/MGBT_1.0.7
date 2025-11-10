"""
Fast MGBT implementation with improved retry logic.

This version fixes the performance issue where recursive retries
caused complete recalculation. Instead, it extends the search
incrementally.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from functools import lru_cache
import logging

from ..stats import kth_order_pvalue_ortho_t

logger = logging.getLogger(__name__)


@dataclass
class MGBTResult:
    """Result structure for MGBT analysis."""
    klow: int
    low_outlier_threshold: Optional[float]
    outlier_indices: List[int]
    p_values: np.ndarray
    test_statistics: np.ndarray
    n_outliers: int
    alpha1: float
    alpha10: float


@lru_cache(maxsize=2048)
def cached_pvalue(n: int, r: int, eta: float) -> float:
    """Cached p-value calculation."""
    try:
        return kth_order_pvalue_ortho_t(n, r, eta)
    except:
        return 1.0


def MGBT(Q, alpha1=0.01, alpha10=0.10, n2=None, use_cache=True):
    """
    Fast Multiple Grubbs-Beck Test with optimized retry logic.
    
    Key optimization: Instead of recursive retry, extends search
    incrementally without recalculating previous values.
    """
    # Convert and validate
    Q = np.asarray(Q, dtype=float)
    Q = Q[~np.isnan(Q)]
    
    if len(Q) < 3:
        logger.warning("MGBT requires at least 3 observations")
        return MGBTResult(
            klow=0,
            low_outlier_threshold=None,
            outlier_indices=[],
            p_values=np.array([]),
            test_statistics=np.array([]),
            n_outliers=0,
            alpha1=alpha1,
            alpha10=alpha10
        )
    
    # Log transform and sort
    zt = np.sort(np.log10(np.maximum(1e-8, Q)))
    n = len(zt)
    
    # Set initial n2 - be more conservative to avoid excessive computation
    if n2 is None:
        # Start with smaller search space
        n2_initial = min(n // 2, 25)  # Cap at 25 initially
    else:
        n2_initial = n2
    
    n2_max = min(n - 1, n // 2 + 10)  # Don't go beyond n/2 + 10
    n2_current = min(n2_initial, n2_max)
    
    # Pre-allocate arrays for maximum possible size
    pvalueW = np.full(n2_max, -99.0, dtype=np.float64)
    w = np.full(n2_max, -99.0, dtype=np.float64)
    
    # Compute test statistics and p-values incrementally
    j1 = 0
    j2 = 0
    retry_count = 0
    max_retries = 3
    
    while retry_count <= max_retries:
        # Compute test statistics for positions we haven't computed yet
        for i in range(1, n2_current + 1):
            if w[i-1] == -99.0:  # Not yet computed
                remaining = zt[i:n]
                if len(remaining) > 1:
                    mean_remaining = np.mean(remaining)
                    var_remaining = np.var(remaining, ddof=1)
                    
                    if var_remaining > 0:
                        w[i-1] = (zt[i-1] - mean_remaining) / np.sqrt(var_remaining)
                    else:
                        w[i-1] = 0.0
        
        # Compute p-values for positions we haven't computed yet
        consecutive_high_pvalues = 0
        for i in range(1, n2_current + 1):
            if pvalueW[i-1] == -99.0:  # Not yet computed
                eta_rounded = round(w[i-1], 6) if use_cache else w[i-1]
                
                try:
                    if use_cache:
                        pvalueW[i-1] = cached_pvalue(n, i, eta_rounded)
                    else:
                        pvalueW[i-1] = kth_order_pvalue_ortho_t(n, i, w[i-1])
                except Exception as e:
                    logger.debug(f"P-value calculation failed for position {i}: {e}")
                    pvalueW[i-1] = 1.0
                
                # Early stopping if we get many consecutive high p-values
                if pvalueW[i-1] > alpha10:
                    consecutive_high_pvalues += 1
                    if consecutive_high_pvalues >= 5:
                        logger.debug(f"Early stopping at position {i} (5 consecutive high p-values)")
                        n2_current = i  # Reduce search space
                        break
                else:
                    consecutive_high_pvalues = 0
        
        # Determine outliers based on current range
        j1 = 0
        j2 = 0
        last_significant = -1
        
        for i in range(1, n2_current + 1):
            if pvalueW[i-1] < alpha1:
                j1 = i
                j2 = i
                last_significant = i
            
            if (pvalueW[i-1] < alpha10) and (j2 == i - 1):
                j2 = i
                last_significant = i
        
        # Check if we need to extend the search
        if j2 == n2_current and n2_current < n2_max:
            # Extend search range
            n2_new = min(n2_current + 10, n2_max)
            logger.debug(f"Extending search from {n2_current} to {n2_new}")
            n2_current = n2_new
            retry_count += 1
        else:
            # Found stable result or reached maximum
            break
    
    if retry_count > max_retries:
        logger.warning(f"MGBT reached maximum retries at n2={n2_current}")
    
    # Determine threshold
    sorted_Q = np.sort(Q)
    if j2 > 0:
        threshold = sorted_Q[j2]
    else:
        threshold = None
    
    # Get outlier indices
    outlier_indices = list(range(j2)) if j2 > 0 else []
    
    # Trim arrays to actual size used
    pvalueW_trimmed = pvalueW[:n2_current]
    w_trimmed = w[:n2_current]
    
    logger.debug(f"MGBT completed: {j2} outliers detected, threshold={threshold}")
    
    return MGBTResult(
        klow=j2,
        low_outlier_threshold=threshold,
        outlier_indices=outlier_indices,
        p_values=pvalueW_trimmed,
        test_statistics=w_trimmed,
        n_outliers=j2,
        alpha1=alpha1,
        alpha10=alpha10
    )


def clear_cache():
    """Clear the p-value cache."""
    cached_pvalue.cache_clear()
    logger.debug("P-value cache cleared")
