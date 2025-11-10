"""
Quick test script to verify MGBT implementations work correctly.
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add pymgbt to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'pymgbt'))

from pymgbt.core.mgbt_corrected import MGBT as MGBT_corrected
from pymgbt.core.mgbt_optimized import MGBT as MGBT_optimized

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_implementation(name, mgbt_func, flows):
    """Test a single MGBT implementation."""
    import time
    
    logger.info(f"\nTesting {name}:")
    logger.info(f"  Input: {len(flows)} flows")
    
    start = time.perf_counter()
    result = mgbt_func(flows)
    elapsed = time.perf_counter() - start
    
    logger.info(f"  Outliers detected: {result.klow if hasattr(result, 'klow') else result.n_outliers}")
    logger.info(f"  Threshold: {result.low_outlier_threshold}")
    logger.info(f"  Time: {elapsed:.4f}s")
    
    return result


def main():
    """Run quick tests."""
    logger.info("="*60)
    logger.info("MGBT Quick Test")
    logger.info("="*60)
    
    # Load a test dataset
    validation_dir = repo_root / 'data' / 'validation'
    
    if not validation_dir.exists():
        logger.error("Validation data not found. Run extract_validation_data.py first.")
        return
    
    # Find first station with censored data
    station_dirs = [d for d in validation_dir.iterdir() if d.is_dir()]
    test_station = None
    
    for station_dir in sorted(station_dirs):
        metadata_file = station_dir / 'metadata.txt'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                content = f.read()
                if 'Censored Flows:' in content:
                    for line in content.split('\n'):
                        if 'Censored Flows:' in line:
                            n_censored = int(line.split(':')[1].strip())
                            if n_censored > 0:
                                test_station = station_dir
                                break
        if test_station:
            break
    
    if not test_station:
        logger.warning("No stations with censored data found, using first station")
        test_station = station_dirs[0]
    
    logger.info(f"\nTest Station: {test_station.name}")
    
    # Load flows
    flows_file = test_station / 'flows.txt'
    flows = np.loadtxt(flows_file)
    
    # Read expected result
    metadata_file = test_station / 'metadata.txt'
    expected_censored = 0
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            for line in f:
                if 'Censored Flows:' in line:
                    expected_censored = int(line.split(':')[1].strip())
                    break
    
    logger.info(f"Expected censored: {expected_censored}")
    
    # Test corrected implementation
    result_corrected = test_implementation("Corrected MGBT", MGBT_corrected, flows)
    
    # Test optimized implementation
    result_optimized = test_implementation("Optimized MGBT", MGBT_optimized, flows)
    
    # Compare results
    logger.info("\n" + "="*60)
    logger.info("Comparison:")
    logger.info(f"  Expected:  {expected_censored} outliers")
    logger.info(f"  Corrected: {result_corrected.klow} outliers")
    logger.info(f"  Optimized: {result_optimized.klow} outliers")
    
    if result_corrected.klow == result_optimized.klow:
        logger.info("  ✓ Implementations agree")
    else:
        logger.warning("  ✗ Implementations disagree!")
    
    if result_optimized.klow == expected_censored:
        logger.info("  ✓ Optimized matches expected")
    else:
        logger.warning(f"  ✗ Optimized does not match expected")
    
    logger.info("="*60)


if __name__ == '__main__':
    main()
