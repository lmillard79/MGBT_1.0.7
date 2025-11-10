"""
Test R MGBT function with modern rpy2 syntax.
"""

import os
import numpy as np

# Configure R
R_HOME = r"C:\Program Files\R\R-4.5.1"
os.environ['R_HOME'] = R_HOME
R_BIN = os.path.join(R_HOME, 'bin', 'x64')
if R_BIN not in os.environ['PATH']:
    os.environ['PATH'] = R_BIN + os.pathsep + os.environ['PATH']

print("Testing R MGBT with modern rpy2 syntax...")

# Suppress verbose rpy2 logging
import logging
logging.getLogger('rpy2').setLevel(logging.ERROR)
logging.getLogger('rpy2.rinterface_lib.embedded').setLevel(logging.ERROR)

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects import conversion
    
    print("Importing R MGBT package...")
    mgbt_r = importr('MGBT')
    
    # Test data
    flows = np.array([100, 200, 300, 10, 20, 400, 500, 5, 15, 600])
    print(f"\nTest data: {flows}")
    print(f"Number of flows: {len(flows)}")
    
    # Use modern conversion context
    print("\nRunning R MGBT...")
    with conversion.localconverter(ro.default_converter + numpy2ri.converter):
        # Convert to R vector
        r_flows = ro.FloatVector(flows)
        
        # Run MGBT (use default alpha values)
        result = mgbt_r.MGBT(r_flows)
        
        # Extract results from NamedList - use names() method to find indices
        names = list(result.names())
        print(f"Result names: {names}")
        
        klow_idx = names.index('klow')
        thresh_idx = names.index('LOThresh')
        
        klow = int(result[klow_idx][0])
        threshold = float(result[thresh_idx][0]) if klow > 0 else None
    
    print(f"\nR MGBT Results:")
    print(f"  Outliers detected: {klow}")
    print(f"  Threshold: {threshold}")
    
    # Show which values are outliers
    sorted_flows = np.sort(flows)
    if klow > 0:
        outliers = sorted_flows[:klow]
        print(f"  Outlier values: {outliers}")
    
    print("\n✓ R MGBT working correctly with modern rpy2 syntax!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
