"""
Updated R MGBT cell for Jupyter notebook.
Replace the R MGBT cell (Cell 4) with this code.
"""

# Modern rpy2 usage (no deprecated activate/deactivate)
try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects import conversion
    
    # Import R MGBT package
    mgbt_r = importr('MGBT')
    
    # Use modern conversion context
    with conversion.localconverter(ro.default_converter + numpy2ri.converter):
        # Convert flows to R vector
        r_flows = ro.FloatVector(flike_data['flows'])
        
        # Run MGBT
        r_result = mgbt_r.MGBT(r_flows, alpha1=0.01, alpha10=0.10)
        
        # Extract results
        r_klow = int(r_result.rx2('klow')[0])
        r_threshold = float(r_result.rx2('LOThresh')[0]) if r_klow > 0 else None
    
    print("R MGBT Results:")
    print(f"  Outliers detected: {r_klow}")
    print(f"  Threshold: {r_threshold}")
    
    # Get R outliers
    sorted_flows = np.sort(flike_data['flows'])
    r_outliers = sorted_flows[:r_klow] if r_klow > 0 else []
    print(f"\nR identified outliers:")
    print(r_outliers)
    
    print(f"\nThree-way Comparison:")
    print(f"  FLIKE: {flike_data['n_censored']}")
    print(f"  R: {r_klow}")
    print(f"  Python: {py_result.klow}")
    print(f"  R matches FLIKE: {r_klow == flike_data['n_censored']}")
    print(f"  Python matches FLIKE: {py_result.klow == flike_data['n_censored']}")
    print(f"  Python matches R: {py_result.klow == r_klow}")
    
except ImportError:
    print("rpy2 not installed - skipping R comparison")
    print("To enable R comparison: pip install rpy2")
except Exception as e:
    print(f"R MGBT failed: {e}")
