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
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import traceback

# Add pymgbt to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'pymgbt'))

from pymgbt.core.mgbt_corrected import MGBT as MGBT_corrected
from pymgbt.core.mgbt import MGBT as MGBT_original
from pymgbt.core.mgbt_optimized import MGBT as MGBT_optimized

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Capture warnings in logging
logging.captureWarnings(True)
warnings_logger = logging.getLogger('py.warnings')


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
                    implementation: str = 'optimized',
                    station_id: str = 'unknown'):
    """
    Run Python MGBT implementation with detailed timing and logging.
    
    Parameters
    ----------
    flows : np.ndarray
        Flow data
    alpha1 : float
        Primary alpha
    alpha10 : float
        Secondary alpha
    implementation : str
        Which implementation to use: 'optimized', 'corrected', or 'original'
    station_id : str
        Station identifier for logging
        
    Returns
    -------
    dict
        Python MGBT results with timing information
    """
    start_time = time.perf_counter()
    logger.debug(f"[{station_id}] Starting Python MGBT ({implementation}): {len(flows)} flows, alpha1={alpha1}, alpha10={alpha10}")
    
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            if implementation == 'optimized':
                result = MGBT_optimized(flows, alpha1=alpha1, alpha10=alpha10, use_cache=True, early_stop=True)
            elif implementation == 'corrected':
                result = MGBT_corrected(flows, alpha1=alpha1, alpha10=alpha10)
            else:
                result = MGBT_original(flows, alpha=alpha1)
            
            # Log any warnings
            if w:
                for warning in w:
                    logger.warning(f"[{station_id}] {warning.category.__name__}: {warning.message}")
        
        elapsed = time.perf_counter() - start_time
        logger.debug(f"[{station_id}] Python MGBT completed in {elapsed:.4f}s: {result.klow if hasattr(result, 'klow') else result.n_outliers} outliers")
        
        return {
            'klow': result.klow if hasattr(result, 'klow') else result.n_outliers,
            'threshold': result.low_outlier_threshold,
            'pvalues': result.p_values,
            'test_stats': result.test_statistics,
            'success': True,
            'error': None,
            'elapsed_time': elapsed
        }
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(f"[{station_id}] Python MGBT failed after {elapsed:.4f}s: {e}")
        logger.debug(f"[{station_id}] Traceback:\n{traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e),
            'elapsed_time': elapsed
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
    Test MGBT on a single station with detailed logging.
    
    Parameters
    ----------
    station_dir : Path
        Directory containing station data
    use_r : bool
        Whether to run R comparison
        
    Returns
    -------
    dict
        Test results with timing information
    """
    station_id = station_dir.name
    test_start = time.perf_counter()
    
    logger.info(f"{'='*60}")
    logger.info(f"Testing Station: {station_id}")
    logger.info(f"{'='*60}")
    
    # Load data
    flows_file = station_dir / 'flows.txt'
    metadata_file = station_dir / 'metadata.txt'
    
    if not flows_file.exists():
        logger.error(f"[{station_id}] Flows file not found: {flows_file}")
        return None
    
    logger.debug(f"[{station_id}] Loading flow data from {flows_file}")
    flows = np.loadtxt(flows_file)
    logger.debug(f"[{station_id}] Loaded {len(flows)} flow values")
    logger.debug(f"[{station_id}] Flow range: {flows.min():.2f} to {flows.max():.2f}")
    
    # Read expected censored count from metadata
    expected_censored = 0
    if metadata_file.exists():
        logger.debug(f"[{station_id}] Reading metadata from {metadata_file}")
        with open(metadata_file, 'r') as f:
            for line in f:
                if 'Censored Flows:' in line:
                    expected_censored = int(line.split(':')[1].strip())
                    logger.debug(f"[{station_id}] Expected censored flows: {expected_censored}")
                    break
    
    logger.info(f"[{station_id}] Dataset: {len(flows)} flows, {expected_censored} expected censored")
    
    # Run Python MGBT (optimized version by default)
    logger.info(f"[{station_id}] Running Python MGBT (optimized implementation)...")
    py_result = run_python_mgbt(flows, implementation='optimized', station_id=station_id)
    
    if py_result['success']:
        logger.info(f"[{station_id}] Python Result: {py_result['klow']} outliers detected")
        logger.info(f"[{station_id}] Python Threshold: {py_result['threshold']}")
        logger.info(f"[{station_id}] Python Time: {py_result['elapsed_time']:.4f}s")
    else:
        logger.error(f"[{station_id}] Python MGBT FAILED: {py_result['error']}")
    
    # Run R MGBT if requested
    r_result = None
    if use_r:
        logger.info(f"[{station_id}] Running R MGBT for comparison...")
        r_result = run_r_mgbt(flows)
        if r_result['success']:
            logger.info(f"[{station_id}] R Result: {r_result['klow']} outliers, threshold={r_result['threshold']}")
        else:
            logger.warning(f"[{station_id}] R MGBT not available: {r_result['error']}")
    
    # Compare results
    comparison = compare_results(station_id, flows, expected_censored, r_result, py_result)
    
    # Add timing info
    comparison['test_time'] = time.perf_counter() - test_start
    if py_result['success']:
        comparison['py_time'] = py_result['elapsed_time']
    
    # Report comparison
    logger.info(f"[{station_id}] Comparison Results:")
    if comparison['match_expected']:
        logger.info(f"[{station_id}]   ✓ Python matches expected censored count ({expected_censored})")
    else:
        logger.warning(f"[{station_id}]   ✗ MISMATCH: Python={py_result['klow']}, Expected={expected_censored}")
    
    if r_result and r_result['success']:
        if comparison['match']:
            logger.info(f"[{station_id}]   ✓ Python matches R implementation")
        else:
            logger.warning(f"[{station_id}]   ✗ Python vs R mismatch: Python={py_result['klow']}, R={r_result['klow']}")
    
    logger.info(f"[{station_id}] Total test time: {comparison['test_time']:.4f}s")
    
    return comparison


def test_all_stations(validation_dir: Path, use_r: bool = False, parallel: bool = False, max_workers: int = None) -> pd.DataFrame:
    """
    Test MGBT on all validation stations with optional parallelization.
    
    Parameters
    ----------
    validation_dir : Path
        Directory containing validation data
    use_r : bool
        Whether to run R comparison
    parallel : bool
        Use parallel processing (Note: R comparison not supported in parallel mode)
    max_workers : int, optional
        Maximum number of parallel workers (defaults to CPU count)
        
    Returns
    -------
    pd.DataFrame
        Comparison results for all stations
    """
    overall_start = time.perf_counter()
    results = []
    
    # Get all station directories
    station_dirs = [d for d in validation_dir.iterdir() if d.is_dir()]
    station_dirs = sorted(station_dirs, key=lambda x: x.name)
    
    logger.info(f"{'#'*80}")
    logger.info(f"MGBT VALIDATION TEST SUITE")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*80}")
    logger.info(f"Total stations to test: {len(station_dirs)}")
    logger.info(f"Parallel processing: {'ENABLED' if parallel else 'DISABLED'}")
    if parallel:
        logger.info(f"Max workers: {max_workers if max_workers else 'CPU count'}")
    logger.info(f"R comparison: {'ENABLED' if use_r else 'DISABLED'}")
    logger.info(f"")
    
    if parallel and not use_r:
        # Parallel processing (R not supported in parallel)
        logger.info("Running tests in parallel...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_station = {}
            for station_dir in station_dirs:
                future = executor.submit(test_single_station, station_dir, False)
                future_to_station[future] = station_dir.name
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_station):
                station_id = future_to_station[future]
                try:
                    comparison = future.result()
                    if comparison:
                        results.append(comparison)
                    completed += 1
                    logger.info(f"Progress: {completed}/{len(station_dirs)} stations completed")
                except Exception as e:
                    logger.error(f"Station {station_id} failed: {e}")
                    logger.debug(traceback.format_exc())
    else:
        # Sequential processing
        logger.info("Running tests sequentially...")
        for idx, station_dir in enumerate(station_dirs, 1):
            logger.info(f"\nProgress: {idx}/{len(station_dirs)}")
            comparison = test_single_station(station_dir, use_r=use_r)
            if comparison:
                results.append(comparison)
    
    overall_elapsed = time.perf_counter() - overall_start
    
    logger.info(f"\n{'#'*80}")
    logger.info(f"ALL TESTS COMPLETED")
    logger.info(f"Total time: {overall_elapsed:.2f}s ({overall_elapsed/60:.2f} minutes)")
    logger.info(f"Average time per station: {overall_elapsed/len(station_dirs):.2f}s")
    logger.info(f"{'#'*80}\n")
    
    return pd.DataFrame(results)


def generate_report(results_df: pd.DataFrame, output_dir: Path):
    """
    Generate detailed comparison report with performance metrics.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Comparison results
    output_dir : Path
        Output directory for reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating comprehensive test report...")
    
    # Save full results
    results_df.to_csv(output_dir / 'mgbt_comparison_results.csv', index=False)
    logger.info(f"Saved detailed results to {output_dir / 'mgbt_comparison_results.csv'}")
    
    # Generate summary statistics
    total = len(results_df)
    py_matches_expected = results_df['match_expected'].sum()
    
    # Performance statistics
    if 'py_time' in results_df.columns:
        avg_time = results_df['py_time'].mean()
        min_time = results_df['py_time'].min()
        max_time = results_df['py_time'].max()
        total_py_time = results_df['py_time'].sum()
    else:
        avg_time = min_time = max_time = total_py_time = 0
    
    summary = f"""
MGBT Validation Summary
{'=' * 80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Test Statistics:
  Total Stations Tested: {total}
  
Python Implementation Results:
  Matches Expected Censored Count: {py_matches_expected}/{total} ({100*py_matches_expected/total:.1f}%)
  Success Rate: {100*py_matches_expected/total:.1f}%

Performance Metrics:
  Total Processing Time: {total_py_time:.2f}s
  Average Time per Station: {avg_time:.4f}s
  Fastest Station: {min_time:.4f}s
  Slowest Station: {max_time:.4f}s
  
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


def main(parallel: bool = True, max_workers: int = None):
    """
    Main execution function with performance optimization.
    
    Parameters
    ----------
    parallel : bool
        Enable parallel processing for faster execution
    max_workers : int, optional
        Maximum number of parallel workers
    """
    # Set up paths
    validation_dir = repo_root / 'data' / 'validation'
    output_dir = repo_root / 'data' / 'test_results'
    
    # Set up log file
    log_file = output_dir / f'mgbt_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler for detailed logging
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Detailed log file: {log_file}")
    
    if not validation_dir.exists():
        logger.error(f"Validation directory not found: {validation_dir}")
        logger.info("Please run extract_validation_data.py first")
        return
    
    logger.info(f"Validation data directory: {validation_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Try to use R if available, but don't require it
    use_r = False
    try:
        import rpy2
        use_r = True
        logger.info("R comparison: AVAILABLE (rpy2 installed)")
        if parallel:
            logger.warning("R comparison not supported in parallel mode - will run sequentially")
            parallel = False
    except ImportError:
        logger.info("R comparison: NOT AVAILABLE (rpy2 not installed)")
    
    # Run tests
    results_df = test_all_stations(validation_dir, use_r=use_r, parallel=parallel, max_workers=max_workers)
    
    # Generate report
    generate_report(results_df, output_dir)
    
    logger.info(f"\nLog file saved to: {log_file}")
    logger.info("Validation testing complete")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test MGBT Python implementation against validation data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--parallel', 
        action='store_true',
        default=True,
        help='Enable parallel processing for faster execution'
    )
    parser.add_argument(
        '--no-parallel',
        dest='parallel',
        action='store_false',
        help='Disable parallel processing (run sequentially)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Maximum number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Console logging level'
    )
    
    args = parser.parse_args()
    
    # Set console log level (file will still be DEBUG)
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    main(parallel=args.parallel, max_workers=args.workers)
