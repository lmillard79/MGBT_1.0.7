"""
Comprehensive comparison of FLIKE, R MGBT, Python MGBT, and C# MGBT censoring.

This script compares censored flow counts from four sources:
1. FLIKE output files (expected/reference)
2. R MGBT package
3. Python MGBT package (optimized)
4. C# MGBT (USACE-RMC Numerics - Fortran port) [if available]

Generates a comprehensive comparison table for all stations.

Note: C# comparison requires the USACE-RMC Numerics library to be compiled
and accessible. If not available, the script will run without C# comparison.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional

# Configure R environment for rpy2
R_HOME = r"C:\Program Files\R\R-4.5.1"
os.environ['R_HOME'] = R_HOME
R_BIN = os.path.join(R_HOME, 'bin', 'x64')
if R_BIN not in os.environ['PATH']:
    os.environ['PATH'] = R_BIN + os.pathsep + os.environ['PATH']

# Add pymgbt to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'pymgbt'))

from pymgbt.core.mgbt_optimized import MGBT as MGBT_optimized

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose rpy2 logging
logging.getLogger('rpy2').setLevel(logging.ERROR)
logging.getLogger('rpy2.rinterface_lib.embedded').setLevel(logging.ERROR)

# Suppress R console output and warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='rpy2')

# Note: The "sh not recognized" warning from rpy2 initialization cannot be easily
# suppressed as it comes from subprocess before Python can intercept it.
# This is harmless and can be ignored.


def load_flike_data(flike_file: Path) -> Dict:
    """
    Load data from FLIKE output file.
    
    Returns dict with:
    - station_id
    - flows (all flows including zeros)
    - n_censored (from FLIKE)
    - censored_flows
    """
    with open(flike_file, 'r') as f:
        lines = f.readlines()
    
    # Extract station ID from filename
    station_id = flike_file.stem.replace('flike_Bayes_', '')
    
    # Parse gauged flows
    gauged_flows = []
    in_gauged = False
    
    for i, line in enumerate(lines):
        if 'Gauged Annual Maximum Discharge' in line:
            in_gauged = True
            continue
        if in_gauged:
            if line.strip() == '' or 'following gauged flows' in line:
                break
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    flow = float(parts[1])
                    gauged_flows.append(flow)
                except ValueError:
                    continue
    
    # Parse censored count
    n_censored = 0
    censored_flows = []
    in_censored = False
    
    for line in lines:
        if 'following gauged flows were censored:' in line:
            in_censored = True
            continue
        if in_censored:
            if 'Flood model:' in line or line.strip() == '':
                if 'Flood model:' in line:
                    break
                continue
            if '---' in line or 'Obs' in line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    flow = float(parts[1])
                    censored_flows.append(flow)
                    n_censored += 1
                except ValueError:
                    continue
    
    # Combine all flows
    all_flows = np.array(gauged_flows)
    
    return {
        'station_id': station_id,
        'flows': all_flows,
        'n_censored_flike': n_censored,
        'censored_flows': censored_flows,
        'n_total': len(all_flows)
    }


def run_python_mgbt(flows: np.ndarray) -> Dict:
    """Run Python MGBT and return results."""
    try:
        result = MGBT_optimized(flows, alpha1=0.01, alpha10=0.10)
        return {
            'success': True,
            'n_censored': result.klow,
            'threshold': result.low_outlier_threshold,
            'outlier_indices': result.outlier_indices
        }
    except Exception as e:
        logger.error(f"Python MGBT failed: {e}")
        return {
            'success': False,
            'n_censored': None,
            'error': str(e)
        }


def run_r_mgbt(flows: np.ndarray) -> Dict:
    """Run R MGBT and return results."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        from rpy2.robjects import conversion
        
        # Import R MGBT package
        mgbt_r = importr('MGBT')
        
        # Use modern conversion context instead of deprecated activate/deactivate
        with conversion.localconverter(ro.default_converter + numpy2ri.converter):
            # Convert flows to R vector
            r_flows = ro.FloatVector(flows)
            
            # Run MGBT (uses default alpha values)
            result = mgbt_r.MGBT(r_flows)
            
            # Extract results from NamedList
            names = list(result.names())
            klow_idx = names.index('klow')
            thresh_idx = names.index('LOThresh')
            
            klow = int(result[klow_idx][0])
            threshold = float(result[thresh_idx][0]) if klow > 0 else None
        
        return {
            'success': True,
            'n_censored': klow,
            'threshold': threshold
        }
    except ImportError:
        logger.warning("rpy2 not available - skipping R comparison")
        return {
            'success': False,
            'n_censored': None,
            'error': 'rpy2 not installed'
        }
    except Exception as e:
        logger.error(f"R MGBT failed: {e}")
        logger.error(f"Traceback: ", exc_info=True)
        return {
            'success': False,
            'n_censored': None,
            'error': str(e)
        }


def run_csharp_mgbt(flows: np.ndarray) -> Dict:
    """Run C# MGBT (USACE-RMC Numerics) and return results."""
    try:
        import subprocess
        import json
        
        # Check if C# wrapper exists
        wrapper_path = repo_root / 'scripts' / 'csharp_mgbt_wrapper.exe'
        if not wrapper_path.exists():
            return {
                'success': False,
                'n_censored': None,
                'error': 'C# wrapper not found'
            }
        
        # Convert flows to command line arguments
        flow_args = [str(f) for f in flows]
        
        # Run C# wrapper
        result = subprocess.run(
            [str(wrapper_path)] + flow_args,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return {
                'success': False,
                'n_censored': None,
                'error': f'C# process failed: {result.stderr}'
            }
        
        # Parse JSON output
        output = json.loads(result.stdout.strip())
        
        if 'error' in output:
            return {
                'success': False,
                'n_censored': None,
                'error': output['error']
            }
        
        return {
            'success': True,
            'n_censored': output['klow'],
            'threshold': output.get('threshold')
        }
        
    except FileNotFoundError:
        return {
            'success': False,
            'n_censored': None,
            'error': 'C# wrapper not found'
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'n_censored': None,
            'error': 'C# process timeout'
        }
    except Exception as e:
        logger.error(f"C# MGBT failed: {e}")
        return {
            'success': False,
            'n_censored': None,
            'error': str(e)
        }


def compare_station(flike_file: Path, use_r: bool = True, use_csharp: bool = True) -> Dict:
    """
    Compare FLIKE, R MGBT, Python MGBT, and C# MGBT for a single station.
    
    Returns dict with comparison results.
    """
    # Load FLIKE data
    flike_data = load_flike_data(flike_file)
    station_id = flike_data['station_id']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Station: {station_id}")
    logger.info(f"Total flows: {flike_data['n_total']}")
    logger.info(f"FLIKE censored: {flike_data['n_censored_flike']}")
    
    # Run Python MGBT
    py_result = run_python_mgbt(flike_data['flows'])
    if py_result['success']:
        logger.info(f"Python censored: {py_result['n_censored']}")
        logger.info(f"Python threshold: {py_result['threshold']}")
    else:
        logger.error(f"Python FAILED: {py_result.get('error', 'Unknown error')}")
    
    # Run R MGBT if requested
    r_result = None
    if use_r:
        r_result = run_r_mgbt(flike_data['flows'])
        if r_result['success']:
            logger.info(f"R censored: {r_result['n_censored']}")
            logger.info(f"R threshold: {r_result['threshold']}")
        else:
            logger.warning(f"R not available: {r_result.get('error', 'Unknown error')}")
    
    # Run C# MGBT if requested
    cs_result = None
    if use_csharp:
        cs_result = run_csharp_mgbt(flike_data['flows'])
        if cs_result['success']:
            logger.info(f"C# censored: {cs_result['n_censored']}")
            logger.info(f"C# threshold: {cs_result['threshold']}")
        else:
            logger.warning(f"C# not available: {cs_result.get('error', 'Unknown error')}")
    
    # Compare results
    py_matches_flike = (py_result['success'] and 
                        py_result['n_censored'] == flike_data['n_censored_flike'])
    r_matches_flike = (r_result and r_result['success'] and 
                       r_result['n_censored'] == flike_data['n_censored_flike'])
    cs_matches_flike = (cs_result and cs_result['success'] and 
                        cs_result['n_censored'] == flike_data['n_censored_flike'])
    py_matches_r = (py_result['success'] and r_result and r_result['success'] and
                    py_result['n_censored'] == r_result['n_censored'])
    py_matches_cs = (py_result['success'] and cs_result and cs_result['success'] and
                     py_result['n_censored'] == cs_result['n_censored'])
    r_matches_cs = (r_result and r_result['success'] and cs_result and cs_result['success'] and
                    r_result['n_censored'] == cs_result['n_censored'])
    
    logger.info(f"\nComparison:")
    logger.info(f"  Python matches FLIKE: {py_matches_flike}")
    if use_r and r_result:
        logger.info(f"  R matches FLIKE: {r_matches_flike}")
        logger.info(f"  Python matches R: {py_matches_r}")
    if use_csharp and cs_result:
        logger.info(f"  C# matches FLIKE: {cs_matches_flike}")
        logger.info(f"  Python matches C#: {py_matches_cs}")
        if use_r and r_result:
            logger.info(f"  R matches C#: {r_matches_cs}")
    
    return {
        'station_id': station_id,
        'n_flows': flike_data['n_total'],
        'flike_censored': flike_data['n_censored_flike'],
        'python_censored': py_result['n_censored'] if py_result['success'] else None,
        'r_censored': r_result['n_censored'] if (r_result and r_result['success']) else None,
        'csharp_censored': cs_result['n_censored'] if (cs_result and cs_result['success']) else None,
        'python_threshold': py_result.get('threshold'),
        'r_threshold': r_result.get('threshold') if r_result else None,
        'csharp_threshold': cs_result.get('threshold') if cs_result else None,
        'py_matches_flike': py_matches_flike,
        'r_matches_flike': r_matches_flike,
        'cs_matches_flike': cs_matches_flike,
        'py_matches_r': py_matches_r,
        'py_matches_cs': py_matches_cs,
        'r_matches_cs': r_matches_cs,
        'python_success': py_result['success'],
        'r_success': r_result['success'] if r_result else False,
        'csharp_success': cs_result['success'] if cs_result else False
    }


def main(use_r: bool = True, use_csharp: bool = True):
    """Run comparison for all stations."""
    logger.info("="*60)
    logger.info("MGBT Comprehensive Comparison")
    methods = ["FLIKE", "Python MGBT"]
    if use_r:
        methods.append("R MGBT")
    if use_csharp:
        methods.append("C# MGBT (Fortran port)")
    logger.info(f"Comparing: {', '.join(methods)}")
    logger.info("="*60)
    
    # Find all FLIKE files
    unit_tests_dir = repo_root / 'UnitTests'
    flike_files = sorted(unit_tests_dir.glob('flike_Bayes_*.txt'))
    
    logger.info(f"\nFound {len(flike_files)} FLIKE files")
    
    # Process all stations
    results = []
    for flike_file in flike_files:
        try:
            result = compare_station(flike_file, use_r=use_r, use_csharp=use_csharp)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {flike_file.name}: {e}")
            continue
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_dir = repo_root / 'data' / 'test_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / 'mgbt_four_way_comparison.csv'
    df.to_csv(csv_file, index=False)
    logger.info(f"\nResults saved to: {csv_file}")
    
    # Generate summary report
    summary_file = output_dir / 'mgbt_four_way_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MGBT Four-Way Comparison Summary\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total stations: {len(df)}\n\n")
        
        # Python statistics
        py_success = df['python_success'].sum()
        f.write(f"Python MGBT:\n")
        f.write(f"  Successful: {py_success}/{len(df)}\n")
        if py_success > 0:
            py_match = df['py_matches_flike'].sum()
            f.write(f"  Matches FLIKE: {py_match}/{py_success} ({py_match/py_success*100:.1f}%)\n")
        f.write("\n")
        
        # R statistics
        if use_r:
            r_success = df['r_success'].sum()
            f.write(f"R MGBT:\n")
            f.write(f"  Successful: {r_success}/{len(df)}\n")
            if r_success > 0:
                r_match = df['r_matches_flike'].sum()
                f.write(f"  Matches FLIKE: {r_match}/{r_success} ({r_match/r_success*100:.1f}%)\n")
            f.write("\n")
        
        # C# statistics
        if use_csharp:
            cs_success = df['csharp_success'].sum()
            f.write(f"C# MGBT (Fortran port):\n")
            f.write(f"  Successful: {cs_success}/{len(df)}\n")
            if cs_success > 0:
                cs_match = df['cs_matches_flike'].sum()
                f.write(f"  Matches FLIKE: {cs_match}/{cs_success} ({cs_match/cs_success*100:.1f}%)\n")
            f.write("\n")
        
        # Discrepancies
        f.write("Stations with discrepancies:\n")
        f.write("-"*60 + "\n")
        
        discrepancies = df[~df['py_matches_flike']]
        if len(discrepancies) > 0:
            for _, row in discrepancies.iterrows():
                f.write(f"\nStation {row['station_id']}:\n")
                f.write(f"  Flows: {row['n_flows']}\n")
                f.write(f"  FLIKE: {row['flike_censored']}\n")
                f.write(f"  Python: {row['python_censored']}\n")
                if use_r and row['r_success']:
                    f.write(f"  R: {row['r_censored']}\n")
                if use_csharp and row['csharp_success']:
                    f.write(f"  C#: {row['csharp_censored']}\n")
        else:
            f.write("  None - all Python results match FLIKE!\n")
        
        f.write("\n" + "="*60 + "\n")
        
        # Detailed results table
        f.write("\nDetailed Results:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Station':<10} {'Flows':<6} {'FLIKE':<6} {'Python':<6}")
        if use_r:
            f.write(f" {'R':<6}")
        if use_csharp:
            f.write(f" {'C#':<6}")
        f.write(f" {'Py=FL':<6}")
        if use_r:
            f.write(f" {'R=FL':<6} {'Py=R':<6}")
        if use_csharp:
            f.write(f" {'CS=FL':<6} {'Py=CS':<6}")
        f.write("\n")
        f.write("-"*80 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['station_id']:<10} {row['n_flows']:<6} "
                   f"{row['flike_censored']:<6} {row['python_censored'] or 'N/A':<6}")
            if use_r:
                f.write(f" {row['r_censored'] or 'N/A':<6}")
            if use_csharp:
                f.write(f" {row['csharp_censored'] or 'N/A':<6}")
            f.write(f" {'Y' if row['py_matches_flike'] else 'N':<6}")
            if use_r:
                f.write(f" {'Y' if row['r_matches_flike'] else 'N':<6} "
                       f"{'Y' if row['py_matches_r'] else 'N':<6}")
            if use_csharp:
                f.write(f" {'Y' if row['cs_matches_flike'] else 'N':<6} "
                       f"{'Y' if row['py_matches_cs'] else 'N':<6}")
            f.write("\n")
    
    logger.info(f"Summary saved to: {summary_file}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total stations: {len(df)}")
    print(f"Python successful: {df['python_success'].sum()}/{len(df)}")
    print(f"Python matches FLIKE: {df['py_matches_flike'].sum()}/{len(df)}")
    if use_r:
        print(f"R successful: {df['r_success'].sum()}/{len(df)}")
        print(f"R matches FLIKE: {df['r_matches_flike'].sum()}/{len(df)}")
        print(f"Python matches R: {df['py_matches_r'].sum()}/{len(df)}")
    if use_csharp:
        print(f"C# successful: {df['csharp_success'].sum()}/{len(df)}")
        print(f"C# matches FLIKE: {df['cs_matches_flike'].sum()}/{len(df)}")
        print(f"Python matches C#: {df['py_matches_cs'].sum()}/{len(df)}")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare FLIKE, R MGBT, Python MGBT, and C# MGBT censoring'
    )
    parser.add_argument(
        '--no-r',
        dest='use_r',
        action='store_false',
        help='Skip R comparison'
    )
    parser.add_argument(
        '--no-csharp',
        dest='use_csharp',
        action='store_false',
        help='Skip C# comparison'
    )
    
    args = parser.parse_args()
    main(use_r=args.use_r, use_csharp=args.use_csharp)
