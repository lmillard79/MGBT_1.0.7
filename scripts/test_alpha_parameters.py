"""
Test MGBT with different alpha parameters to validate against FLIKE.

This script tests the hypothesis that FLIKE uses alpha1=0.005 (Fortran default)
rather than alpha1=0.01 (current Python default).
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Configure R environment
R_HOME = r"C:\Program Files\R\R-4.5.1"
os.environ['R_HOME'] = R_HOME
R_BIN = os.path.join(R_HOME, 'bin', 'x64')
if R_BIN not in os.environ['PATH']:
    os.environ['PATH'] = R_BIN + os.pathsep + os.environ['PATH']

# Add pymgbt to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'pymgbt'))

from pymgbt.core.mgbt_optimized import MGBT as MGBT_optimized

# Suppress logging
import logging
logging.getLogger('rpy2').setLevel(logging.ERROR)
logging.getLogger('rpy2.rinterface_lib.embedded').setLevel(logging.ERROR)


def load_flike_data(flike_file: Path):
    """Load annual maxima from FLIKE file."""
    with open(flike_file, 'r') as f:
        lines = f.readlines()
    
    # Find gauged flows section
    flows = []
    censored_flows = []
    in_gauged = False
    in_censored = False
    
    for line in lines:
        # Start of gauged section
        if 'Gauged Annual Maximum Discharge' in line:
            in_gauged = True
            continue
        # Start of censored section
        elif 'following gauged flows were censored' in line:
            in_gauged = False
            in_censored = True
            continue
        # End of sections
        elif 'Flood model:' in line or 'Zero flow threshold' in line:
            in_gauged = False
            in_censored = False
            break
        
        # Skip header lines
        if line.strip().startswith('---') or line.strip().startswith('Obs'):
            continue
        
        # Parse gauged flows
        if in_gauged and line.strip():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # Skip the observation number, get discharge
                    flow = float(parts[1])
                    flows.append(flow)
                except:
                    pass
        
        # Parse censored flows
        if in_censored and line.strip():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # Skip the observation number, get discharge
                    flow = float(parts[1])
                    censored_flows.append(flow)
                except:
                    pass
    
    # Combine all flows
    all_flows = np.array(flows + censored_flows)
    n_censored = len(censored_flows)
    
    return all_flows, n_censored


def test_station(station_id: str):
    """Test a single station with different alpha values."""
    
    # Load FLIKE data
    flike_file = repo_root / 'UnitTests' / f'flike_Bayes_{station_id}.txt'
    if not flike_file.exists():
        print(f"FLIKE file not found: {flike_file}")
        return
    
    flows, flike_censored = load_flike_data(flike_file)
    
    print(f"\n{'='*70}")
    print(f"Station: {station_id}")
    print(f"Total flows: {len(flows)}")
    print(f"FLIKE censored: {flike_censored}")
    print(f"{'='*70}")
    
    # Test 1: Python with alpha1=0.01 (current default)
    result_01 = MGBT_optimized(flows, alpha1=0.01, alpha10=0.10)
    print(f"\nPython MGBT (alpha1=0.01, alpha10=0.10):")
    print(f"  Censored: {result_01.klow}")
    print(f"  Threshold: {result_01.low_outlier_threshold}")
    print(f"  Matches FLIKE: {result_01.klow == flike_censored}")
    
    # Test 2: Python with alpha1=0.005 (Fortran default)
    result_005 = MGBT_optimized(flows, alpha1=0.005, alpha10=0.10)
    print(f"\nPython MGBT (alpha1=0.005, alpha10=0.10) [Fortran default]:")
    print(f"  Censored: {result_005.klow}")
    print(f"  Threshold: {result_005.low_outlier_threshold}")
    print(f"  Matches FLIKE: {result_005.klow == flike_censored}")
    
    # Test 3: R MGBT (default parameters)
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, conversion
        from rpy2.robjects.packages import importr
        
        mgbt_r = importr('MGBT')
        
        with conversion.localconverter(ro.default_converter + numpy2ri.converter):
            r_flows = ro.FloatVector(flows)
            r_result = mgbt_r.MGBT(r_flows)
            
            names = list(r_result.names())
            r_klow = int(r_result[names.index('klow')][0])
        
        print(f"\nR MGBT (default parameters):")
        print(f"  Censored: {r_klow}")
        print(f"  Matches FLIKE: {r_klow == flike_censored}")
        
    except Exception as e:
        print(f"\nR MGBT: Failed ({e})")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  FLIKE:              {flike_censored}")
    print(f"  Python (α=0.01):    {result_01.klow} {'✓' if result_01.klow == flike_censored else '✗'}")
    print(f"  Python (α=0.005):   {result_005.klow} {'✓' if result_005.klow == flike_censored else '✗'}")
    try:
        print(f"  R MGBT:             {r_klow} {'✓' if r_klow == flike_censored else '✗'}")
    except:
        pass
    print(f"{'='*70}")


def test_all_stations():
    """Test all available stations."""
    
    # Find all FLIKE files
    unittest_dir = repo_root / 'UnitTests'
    flike_files = list(unittest_dir.glob('flike_Bayes_*.txt'))
    
    print(f"\nTesting {len(flike_files)} stations...")
    print(f"Hypothesis: FLIKE uses alpha1=0.005 (Fortran default)")
    print(f"Current Python default: alpha1=0.01")
    
    results = []
    
    for flike_file in flike_files:
        # Extract station ID
        station_id = flike_file.stem.replace('flike_Bayes_', '')
        
        # Skip model files for now
        if 'Model' in station_id or 'Out' in station_id:
            continue
        
        try:
            flows, flike_censored = load_flike_data(flike_file)
            
            # Test with both alpha values
            result_01 = MGBT_optimized(flows, alpha1=0.01, alpha10=0.10)
            result_005 = MGBT_optimized(flows, alpha1=0.005, alpha10=0.10)
            
            results.append({
                'station_id': station_id,
                'n_flows': len(flows),
                'flike': flike_censored,
                'py_01': result_01.klow,
                'py_005': result_005.klow,
                'match_01': result_01.klow == flike_censored,
                'match_005': result_005.klow == flike_censored
            })
            
        except Exception as e:
            print(f"Error processing {station_id}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("ALPHA PARAMETER VALIDATION RESULTS")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    
    # Statistics
    print(f"\n{'='*80}")
    print("MATCH STATISTICS:")
    print(f"  Python (α=0.01):  {df['match_01'].sum()}/{len(df)} matches ({df['match_01'].mean()*100:.1f}%)")
    print(f"  Python (α=0.005): {df['match_005'].sum()}/{len(df)} matches ({df['match_005'].mean()*100:.1f}%)")
    print(f"{'='*80}")
    
    # Save results
    output_file = repo_root / 'data' / 'test_results' / 'alpha_parameter_test.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Conclusion
    if df['match_005'].sum() > df['match_01'].sum():
        print(f"\n✓ HYPOTHESIS CONFIRMED: alpha1=0.005 provides better match to FLIKE")
    elif df['match_01'].sum() > df['match_005'].sum():
        print(f"\n✗ HYPOTHESIS REJECTED: alpha1=0.01 provides better match to FLIKE")
    else:
        print(f"\n? INCONCLUSIVE: Both alpha values provide similar matches")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MGBT alpha parameters')
    parser.add_argument('--station', type=str, help='Test single station')
    parser.add_argument('--all', action='store_true', help='Test all stations')
    
    args = parser.parse_args()
    
    if args.station:
        test_station(args.station)
    elif args.all:
        test_all_stations()
    else:
        # Default: test a few key stations
        print("Testing key stations (use --all for complete test)...")
        for station in ['416040', '416050', '416002']:
            test_station(station)
