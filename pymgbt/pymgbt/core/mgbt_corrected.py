"""
Corrected MGBT implementation matching R package behavior.

This module implements the MGBT algorithm exactly as specified in the R package,
including log10 transformation and the alpha1/alpha10 consecutive outlier logic.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union
import warnings

from ..stats import kth_order_pvalue_ortho_t


@dataclass
class MGBTResult:
    """
    Result structure for MGBT analysis.
    
    Attributes
    ----------
    klow : int
        Number of low outliers detected
    low_outlier_threshold : Optional[float]
        Threshold below which observations are considered low outliers (in original units)
    outlier_indices : List[int]
        Indices of detected outliers in sorted data (0-based)
    p_values : np.ndarray
        P-values for each tested position
    test_statistics : np.ndarray
        Test statistics (w values) for each position
    n_outliers : int
        Number of outliers detected (same as klow)
    alpha1 : float
        Primary significance level (default 0.01)
    alpha10 : float
        Secondary significance level for consecutive outliers (default 0.10)
    """
    klow: int
    low_outlier_threshold: Optional[float]
    outlier_indices: List[int]
    p_values: np.ndarray
    test_statistics: np.ndarray
    n_outliers: int
    alpha1: float
    alpha10: float


def MGBT(Q: Union[np.ndarray, List], 
         alpha1: float = 0.01, 
         alpha10: float = 0.10,
         n2: Optional[int] = None) -> MGBTResult:
    """
    Multiple Grubbs-Beck Test (MGBT) - Exact R implementation.
    
    This function implements the MGBT algorithm exactly as specified in the R package
    by Cohn et al. (2013), including:
    - Log10 transformation of flow data
    - Two-level alpha testing (alpha1 and alpha10)
    - Consecutive outlier detection logic
    - Recursive handling when too many outliers detected
    
    Parameters
    ----------
    Q : array-like
        Input flow data (annual peak flows)
    alpha1 : float, default=0.01
        Primary significance level for outlier detection
    alpha10 : float, default=0.10
        Secondary significance level for consecutive outliers
    n2 : int, optional
        Maximum number of outliers to test. If None, uses floor(length(Q)/2)
        
    Returns
    -------
    MGBTResult
        Complete results of the MGBT analysis
        
    Notes
    -----
    The MGBT test follows the algorithm from:
    Cohn, T.A., et al. (2013). The Multiple Grubbs-Beck Low-Outlier Test.
    
    R Implementation Reference:
    ```R
    MGBT <- function(Q,alpha1=0.01,alpha10=0.10,n2=floor(length(Q)/2)){
          zt      <- sort(log10(pmax(1e-8,Q)))
          n       <- length(zt)
          pvalueW <-rep(-99,n2);w<-rep(-99,n2)
          j1=0;j2=0
        for(i in 1:n2) {
           w[i]<-(zt[i]-mean(zt[(i+1):n]))/sqrt(var(zt[(i+1):n]))
           pvalueW[i]<-KthOrderPValueOrthoT(n,i,w[i])$value
           if(pvalueW[i]<alpha1){j1<-i;j2<-i}
           if( (pvalueW[i]<alpha10) & (j2==i-1)){j2<-i}
           }
        ...
        return(list(klow=j2,pvalues=pvalueW,LOThresh=ifelse(j2>0,sort(Q)[j2+1],0)))
    }
    ```
    """
    # Convert to numpy array
    Q = np.asarray(Q, dtype=float)
    Q = Q[~np.isnan(Q)]  # Remove NaN values
    
    # Apply log10 transformation with floor at 1e-8 (matching R: pmax(1e-8, Q))
    zt = np.sort(np.log10(np.maximum(1e-8, Q)))
    n = len(zt)
    
    # Set maximum outliers to test
    if n2 is None:
        n2 = n // 2  # floor division
    
    # Ensure we have enough data
    if n < 3:
        warnings.warn("MGBT requires at least 3 observations", UserWarning)
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
    
    # Ensure n2 is valid
    n2 = min(n2, n - 1)
    
    # Initialize arrays
    pvalueW = np.full(n2, -99.0)
    w = np.full(n2, -99.0)
    j1 = 0
    j2 = 0
    
    # Main MGBT loop - test each position from 1 to n2
    for i in range(1, n2 + 1):  # R uses 1-based indexing
        # Calculate test statistic w[i]
        # R: w[i]<-(zt[i]-mean(zt[(i+1):n]))/sqrt(var(zt[(i+1):n]))
        # Note: R indexing is 1-based, Python is 0-based
        # R's zt[i] is Python's zt[i-1]
        # R's zt[(i+1):n] is Python's zt[i:n]
        
        if i < n:  # Need at least one value for mean/var
            remaining = zt[i:n]  # Values above the i-th smallest
            mean_remaining = np.mean(remaining)
            
            # R's var() uses ddof=1 (sample variance)
            if len(remaining) > 1:
                var_remaining = np.var(remaining, ddof=1)
                
                if var_remaining > 0:
                    w[i-1] = (zt[i-1] - mean_remaining) / np.sqrt(var_remaining)
                else:
                    w[i-1] = 0.0
            else:
                w[i-1] = 0.0
        else:
            w[i-1] = 0.0
        
        # Calculate p-value using orthogonal transformation
        try:
            pvalue_result = kth_order_pvalue_ortho_t(n, i, w[i-1])
            pvalueW[i-1] = pvalue_result
        except Exception as e:
            warnings.warn(f"P-value calculation failed for i={i}: {e}", UserWarning)
            pvalueW[i-1] = 1.0  # Conservative: no outlier
        
        # Update j1 and j2 based on significance
        # R logic:
        # if(pvalueW[i]<alpha1){j1<-i;j2<-i}
        # if( (pvalueW[i]<alpha10) & (j2==i-1)){j2<-i}
        
        if pvalueW[i-1] < alpha1:
            j1 = i
            j2 = i
        
        if (pvalueW[i-1] < alpha10) and (j2 == i - 1):
            j2 = i
    
    # Check if we hit the limit
    if j2 == n2:
        if n2 < len(Q) - 5:  # Set a limit of at least 5 retained observations
            warnings.warn(
                f"Number of low outliers equals or exceeds limit of {n2}. "
                f"Retrying with n2={len(Q)-5}",
                UserWarning
            )
            return MGBT(Q, alpha1=alpha1, alpha10=alpha10, n2=len(Q)-5)
        else:
            warnings.warn(
                "MGBT identifies too many low outliers; use caution and judgment",
                UserWarning
            )
    
    # Determine threshold
    # R: LOThresh=ifelse(j2>0,sort(Q)[j2+1],0)
    # R uses 1-based indexing, so sort(Q)[j2+1] is the (j2+1)-th smallest value
    # In Python 0-based: that's sorted_Q[j2]
    sorted_Q = np.sort(Q)
    if j2 > 0:
        threshold = sorted_Q[j2]  # The value just above the j2-th outlier
    else:
        threshold = None
    
    # Get outlier indices (0-based)
    outlier_indices = list(range(j2)) if j2 > 0 else []
    
    return MGBTResult(
        klow=j2,
        low_outlier_threshold=threshold,
        outlier_indices=outlier_indices,
        p_values=pvalueW,
        test_statistics=w,
        n_outliers=j2,
        alpha1=alpha1,
        alpha10=alpha10
    )


def MGBTnb(Q: Union[np.ndarray, List], 
           alpha1: float = 0.01, 
           alpha10: float = 0.10,
           n2: Optional[int] = None) -> MGBTResult:
    """
    Multiple Grubbs-Beck Test (MGBTnb) - No backup version.
    
    This eliminates the backup/consecutive procedure and only uses alpha1.
    Added for JL/JRS study 24 Aug 2012 (TAC).
    
    Parameters
    ----------
    Q : array-like
        Input flow data
    alpha1 : float, default=0.01
        Primary significance level
    alpha10 : float, default=0.10
        Secondary significance level (used for first observation only)
    n2 : int, optional
        Maximum number of outliers to test
        
    Returns
    -------
    MGBTResult
        MGBT results without backup procedure
    """
    Q = np.asarray(Q, dtype=float)
    Q = Q[~np.isnan(Q)]
    
    zt = np.sort(np.log10(np.maximum(1e-8, Q)))
    n = len(zt)
    
    if n2 is None:
        n2 = n // 2
    
    if n < 3:
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
    
    n2 = min(n2, n - 1)
    
    pvalueW = np.full(n2, -99.0)
    w = np.full(n2, -99.0)
    j1 = 0
    j2 = 0
    
    for i in range(1, n2 + 1):
        if i < n:
            remaining = zt[i:n]
            mean_remaining = np.mean(remaining)
            
            if len(remaining) > 1:
                var_remaining = np.var(remaining, ddof=1)
                if var_remaining > 0:
                    w[i-1] = (zt[i-1] - mean_remaining) / np.sqrt(var_remaining)
                else:
                    w[i-1] = 0.0
            else:
                w[i-1] = 0.0
        else:
            w[i-1] = 0.0
        
        try:
            pvalue_result = kth_order_pvalue_ortho_t(n, i, w[i-1])
            pvalueW[i-1] = pvalue_result
        except Exception as e:
            warnings.warn(f"P-value calculation failed for i={i}: {e}", UserWarning)
            pvalueW[i-1] = 1.0
        
        # MGBTnb logic: only update if p-value < alpha1
        if pvalueW[i-1] < alpha1:
            j1 = i
            j2 = i
    
    # Special case: if first observation is < alpha10 and j2==0, set j2=1
    if (pvalueW[0] < alpha10) and (j2 == 0):
        j2 = 1
    
    if j2 == n2:
        if n2 < len(Q) - 5:
            warnings.warn(
                f"Number of low outliers equals or exceeds limit of {n2}",
                UserWarning
            )
            return MGBT(Q, alpha1=alpha1, alpha10=alpha10, n2=len(Q)-5)
        else:
            warnings.warn(
                "MGBT identifies too many low outliers; use caution and judgment",
                UserWarning
            )
    
    sorted_Q = np.sort(Q)
    threshold = sorted_Q[j2] if j2 > 0 else None
    outlier_indices = list(range(j2)) if j2 > 0 else []
    
    return MGBTResult(
        klow=j2,
        low_outlier_threshold=threshold,
        outlier_indices=outlier_indices,
        p_values=pvalueW,
        test_statistics=w,
        n_outliers=j2,
        alpha1=alpha1,
        alpha10=alpha10
    )
