"""
Test MGBT Python implementation against R package results.

This script:
1. Loads validation data from flike_bayes files
2. Runs both Python and R MGBT implementations
3. Compares results and identifies discrepancies
4. Generates detailed comparison reports
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging

# Add pymgbt to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'pymgbt'))

from pymgbt.core.mgbt_corrected import MGBT as MGBT_corrected
from pymgbt.core.mgbt import MGBT as MGBT_original

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_r_mgbt(flows: np.ndarray, alpha1: float = 0.01, alpha10: float = 0.10):
    """
    Run R MGBT implementation using rpy2.
    
    Parameters
    ----------
    flows : np.ndarray
        Flow data
    alpha1 : float
        Primary alpha
    alpha10 : float
        Secondary alpha
        
    Returns
    -------
    dict
        R MGBT results
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        
        # Activate automatic numpy to R conversion
        numpy2ri.activate()
        
        # Load R MGBT package
        mgbt_pkg = importr('MGBT')
        
        # Convert flows to R vector
        r_flows = ro.FloatVector(flows)
        
        # Run MGBT
        result = mgbt_pkg.MGBT(r_flows, alpha1=alpha1, alpha10=alpha10)
        
        # Extract results
        klow = int(result.rx2('klow')[0])
        lo_thresh = float(result.rx2('LOThresh')[0]) if klow > 0 else None
        pvalues = np.array(result.rx2('pvalues'))
        
        numpy2ri.deactivate()
        
        return {
            'klow': klow,
            'threshold': lo_thresh,
            'pvalues': pvalues,
            'success': True,
            'error': None
        }
        
    except ImportError as e:
        logger.warning(f"rpy2 not available: {e}")
        return {
            'success': False,
            'error': 'rpy2 not installed'
        }
    except Exception as e:
        logger.error(f"R MGBT failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def run_python_mgbt(flows: np.ndarray, 
                    alpha1: float = 0.01, 
                    alpha10: float = 0.10,
                    use_corrected: bool = True):
    """
    Run Python MGBT implementation.
    
    Parameters
    ----------
    flows : np.ndarray
        Flow data
    alpha1 : float
        Primary alpha
    alpha10 : float
        Secondary alpha
    use_corrected : bool
        Use corrected implementation
        
    Returns
    -------
    dict
        Python MGBT results
    """
    try:
        if use_corrected:
            result = MGBT_corrected(flows, alpha1=alpha1, alpha10=alpha10)
        else:
            result = MGBT_original(flows, alpha=alpha1)
        
        return {
            'klow': result.klow if hasattr(result, 'klow') else result.n_outliers,
            'threshold': result.low_outlier_threshold,
            'pvalues': result.p_values,
            'test_stats': result.test_statistics,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Python MGBT failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def compare_results(station_id: str, 
                   flows: np.ndarray,
                   expected_censored: int,
                   r_result: dict = None,
                   py_result: dict = None) -> dict:
    """
    Compare MGBT results from different implementations.
    
    Parameters
    ----------
    station_id : str
        Station identifier
    flows : np.ndarray
        Flow data
    expected_censored : int
        Expected number of censored observations from validation data
    r_result : dict, optional
        R MGBT results
    py_result : dict, optional
        Python MGBT results
        
    Returns
    -------
    dict
        Comparison results
    """
    comparison = {
        'station_id': station_id,
        'n_flows': len(flows),
        'expected_censored': expected_censored,
        'r_klow': r_result['klow'] if r_result and r_result['success'] else None,
        'py_klow': py_result['klow'] if py_result and py_result['success'] else None,
        'r_threshold': r_result['threshold'] if r_result and r_result['success'] else None,
        'py_threshold': py_result['threshold'] if py_result and py_result['success'] else None,
        'match': False,
        'match_expected': False
    }
    
    # Check if results match
    if r_result and py_result and r_result['success'] and py_result['success']:
        comparison['match'] = (r_result['klow'] == py_result['klow'])
        
        # Check threshold match (with tolerance for floating point)
        if r_result['threshold'] is not None and py_result['threshold'] is not None:
            threshold_match = np.isclose(r_result['threshold'], py_result['threshold'], rtol=1e-5)
            comparison['threshold_match'] = threshold_match
        else:
            comparison['threshold_match'] = (r_result['threshold'] == py_result['threshold'])
    
    # Check if Python matches expected
    if py_result and py_result['success']:
        comparison['match_expected'] = (py_result['klow'] == expected_censored)
    
    # Check if R matches expected
    if r_result and r_result['success']:
        comparison['r_match_expected'] = (r_result['klow'] == expected_censored)
    
    return comparison


def test_single_station(station_dir: Path, use_r: bool = False) -> dict:
    """
    Test MGBT on a single station.
    
    Parameters
    ----------
    station_dir : Path
        Directory containing station data
    use_r : bool
        Whether to run R comparison
        
    Returns
    -------
    dict
        Test results
    """
    station_id = station_dir.name
    
    # Load data
    flows_file = station_dir / 'flows.txt'
    metadata_file = station_dir / 'metadata.txt'
    
    if not flows_file.exists():
        logger.error(f"Flows file not found for {station_id}")
        return None
    
    flows = np.loadtxt(flows_file)
    
    # Read expected censored count from metadata
    expected_censored = 0
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            for line in f:
                if 'Censored Flows:' in line:
                    expected_censored = int(line.split(':')[1].strip())
                    break
    
    logger.info(f"\nTesting {station_id}: {len(flows)} flows, {expected_censored} expected censored")
    
    # Run Python MGBT (corrected version)
    py_result = run_python_mgbt(flows, use_corrected=True)
    
    if py_result['success']:
        logger.info(f"  Python: {py_result['klow']} outliers, threshold={py_result['threshold']}")
    else:
        logger.error(f"  Python failed: {py_result['error']}")
    
    # Run R MGBT if requested
    r_result = None
    if use_r:
        r_result = run_r_mgbt(flows)
        if r_result['success']:
            logger.info(f"  R:      {r_result['klow']} outliers, threshold={r_result['threshold']}")
        else:
            logger.warning(f"  R not available: {r_result['error']}")
    
    # Compare results
    comparison = compare_results(station_id, flows, expected_censored, r_result, py_result)
    
    # Report
    if comparison['match_expected']:
        logger.info(f"  ✓ Python matches expected censored count")
    else:
        logger.warning(f"  ✗ Python: {py_result['klow']} vs Expected: {expected_censored}")
    
    if r_result and r_result['success']:
        if comparison['match']:
            logger.info(f"  ✓ Python matches R")
        else:
            logger.warning(f"  ✗ Python vs R mismatch")
    
    return comparison


def test_all_stations(validation_dir: Path, use_r: bool = False) -> pd.DataFrame:
    """
    Test MGBT on all validation stations.
    
    Parameters
    ----------
    validation_dir : Path
        Directory containing validation data
    use_r : bool
        Whether to run R comparison
        
    Returns
    -------
    pd.DataFrame
        Comparison results for all stations
    """
    results = []
    
    # Get all station directories
    station_dirs = [d for d in validation_dir.iterdir() if d.is_dir()]
    station_dirs = sorted(station_dirs, key=lambda x: x.name)
    
    logger.info(f"Testing {len(station_dirs)} stations")
    logger.info("=" * 80)
    
    for station_dir in station_dirs:
        comparison = test_single_station(station_dir, use_r=use_r)
        if comparison:
            results.append(comparison)
    
    return pd.DataFrame(results)


def generate_report(results_df: pd.DataFrame, output_dir: Path):
    """
    Generate detailed comparison report.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Comparison results
    output_dir : Path
        Output directory for reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    results_df.to_csv(output_dir / 'mgbt_comparison_results.csv', index=False)
    logger.info(f"\nSaved results to {output_dir / 'mgbt_comparison_results.csv'}")
    
    # Generate summary statistics
    total = len(results_df)
    py_matches_expected = results_df['match_expected'].sum()
    
    summary = f"""
MGBT Validation Summary
{'=' * 80}

Total Stations Tested: {total}

Python Implementation:
  Matches Expected Censored Count: {py_matches_expected}/{total} ({100*py_matches_expected/total:.1f}%)
  
Stations with Discrepancies:
"""
    
    # List discrepancies
    discrepancies = results_df[~results_df['match_expected']]
    if len(discrepancies) > 0:
        for _, row in discrepancies.iterrows():
            summary += f"\n  {row['station_id']}: Python={row['py_klow']}, Expected={row['expected_censored']}"
    else:
        summary += "\n  None - All stations match!"
    
    # Check R comparison if available
    if 'r_klow' in results_df.columns and results_df['r_klow'].notna().any():
        r_available = results_df['r_klow'].notna().sum()
        py_r_matches = results_df['match'].sum()
        summary += f"""

R vs Python Comparison:
  R Results Available: {r_available}/{total}
  Python Matches R: {py_r_matches}/{r_available} ({100*py_r_matches/r_available:.1f}%)
"""
    
    print(summary)
    
    # Save summary
    with open(output_dir / 'mgbt_comparison_summary.txt', 'w') as f:
        f.write(summary)
    
    logger.info(f"Saved summary to {output_dir / 'mgbt_comparison_summary.txt'}")


def main():
    """Main execution function."""
    # Set up paths
    validation_dir = repo_root / 'data' / 'validation'
    output_dir = repo_root / 'data' / 'test_results'
    
    if not validation_dir.exists():
        logger.error(f"Validation directory not found: {validation_dir}")
        logger.info("Please run extract_validation_data.py first")
        return
    
    # Test all stations
    logger.info("Starting MGBT validation tests")
    logger.info(f"Validation data: {validation_dir}")
    
    # Try to use R if available, but don't require it
    use_r = False
    try:
        import rpy2
        use_r = True
        logger.info("R comparison enabled (rpy2 available)")
    except ImportError:
        logger.info("R comparison disabled (rpy2 not available)")
    
    results_df = test_all_stations(validation_dir, use_r=use_r)
    
    # Generate report
    logger.info("\n" + "=" * 80)
    generate_report(results_df, output_dir)
    
    logger.info("\nValidation testing complete")


if __name__ == '__main__':
    main()
