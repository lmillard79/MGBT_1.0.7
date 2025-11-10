"""
Performance Benchmark: Python MGBT vs R MGBT

Compares processing time for all unit tests between:
1. Pure Python implementation (pymgbt)
2. R implementation via rpy2

Provides detailed timing statistics and speedup analysis.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List

# Configure R environment
R_HOME = r"C:\Program Files\R\R-4.5.1"
os.environ['R_HOME'] = R_HOME
R_BIN = os.path.join(R_HOME, 'bin', 'x64')
if R_BIN not in os.environ['PATH']:
    os.environ['PATH'] = R_BIN + os.pathsep + os.environ['PATH']

# Add pymgbt to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'pymgbt'))

from pymgbt.core.mgbt_fast import MGBT as MGBT_fast

# Suppress logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger('rpy2').setLevel(logging.ERROR)
logging.getLogger('rpy2.rinterface_lib.embedded').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='rpy2')


def load_flike_data(flike_file: Path) -> np.ndarray:
    """Load flow data from FLIKE file."""
    with open(flike_file, 'r') as f:
        lines = f.readlines()
    
    flows = []
    censored_flows = []
    in_gauged = False
    in_censored = False
    
    for line in lines:
        if 'Gauged Annual Maximum Discharge' in line:
            in_gauged = True
            continue
        elif 'following gauged flows were censored' in line:
            in_gauged = False
            in_censored = True
            continue
        elif 'Flood model:' in line or 'Zero flow threshold' in line:
            break
        
        if line.strip().startswith('---') or line.strip().startswith('Obs'):
            continue
        
        if in_gauged and line.strip():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    flow = float(parts[1])
                    flows.append(flow)
                except:
                    pass
        
        if in_censored and line.strip():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    flow = float(parts[1])
                    censored_flows.append(flow)
                except:
                    pass
    
    return np.array(flows + censored_flows)


def benchmark_python(flows: np.ndarray) -> Dict:
    """Benchmark Python MGBT."""
    start = time.perf_counter()
    try:
        result = MGBT_fast(flows, alpha1=0.01, alpha10=0.10)
        elapsed = time.perf_counter() - start
        return {
            'success': True,
            'time': elapsed,
            'klow': result.klow
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            'success': False,
            'time': elapsed,
            'error': str(e)
        }


def benchmark_r(flows: np.ndarray) -> Dict:
    """Benchmark R MGBT."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, conversion
        from rpy2.robjects.packages import importr
        
        mgbt_r = importr('MGBT')
        
        start = time.perf_counter()
        
        with conversion.localconverter(ro.default_converter + numpy2ri.converter):
            r_flows = ro.FloatVector(flows)
            r_result = mgbt_r.MGBT(r_flows)
            
            names = list(r_result.names())
            klow = int(r_result[names.index('klow')][0])
        
        elapsed = time.perf_counter() - start
        
        return {
            'success': True,
            'time': elapsed,
            'klow': klow
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            'success': False,
            'time': elapsed,
            'error': str(e)
        }


def run_benchmark(test_r: bool = True, test_python: bool = True):
    """Run performance benchmark on all unit tests."""
    
    print("="*70)
    print("MGBT Performance Benchmark")
    print("="*70)
    print()
    
    # Find all FLIKE files
    unit_tests_dir = repo_root / 'UnitTests'
    flike_files = sorted(unit_tests_dir.glob('flike_Bayes_*.txt'))
    
    print(f"Found {len(flike_files)} FLIKE files")
    print()
    
    results = []
    
    # Benchmark each station
    for i, flike_file in enumerate(flike_files, 1):
        station_id = flike_file.stem.replace('flike_Bayes_', '')
        
        try:
            flows = load_flike_data(flike_file)
            
            if len(flows) < 3:
                continue
            
            print(f"[{i}/{len(flike_files)}] {station_id:<15} (n={len(flows):<3})", end=" ")
            
            result = {
                'station_id': station_id,
                'n_flows': len(flows)
            }
            
            # Benchmark Python
            if test_python:
                py_result = benchmark_python(flows)
                result['python_time'] = py_result['time']
                result['python_success'] = py_result['success']
                result['python_klow'] = py_result.get('klow')
                print(f"Py: {py_result['time']*1000:6.2f}ms", end=" ")
            
            # Benchmark R
            if test_r:
                r_result = benchmark_r(flows)
                result['r_time'] = r_result['time']
                result['r_success'] = r_result['success']
                result['r_klow'] = r_result.get('klow')
                print(f"R: {r_result['time']*1000:6.2f}ms", end=" ")
            
            # Calculate speedup
            if test_python and test_r and py_result['success'] and r_result['success']:
                speedup = r_result['time'] / py_result['time']
                result['speedup'] = speedup
                print(f"Speedup: {speedup:.2f}x", end="")
            
            print()
            results.append(result)
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    print()
    print("="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print()
    
    if test_python:
        py_success = df['python_success'].sum()
        py_total = df['python_time'].sum()
        py_mean = df['python_time'].mean()
        py_median = df['python_time'].median()
        
        print(f"Python MGBT:")
        print(f"  Successful: {py_success}/{len(df)}")
        print(f"  Total time: {py_total:.3f}s ({py_total*1000:.1f}ms)")
        print(f"  Mean time:  {py_mean*1000:.2f}ms per station")
        print(f"  Median time: {py_median*1000:.2f}ms per station")
        print()
    
    if test_r:
        r_success = df['r_success'].sum()
        r_total = df['r_time'].sum()
        r_mean = df['r_time'].mean()
        r_median = df['r_time'].median()
        
        print(f"R MGBT (via rpy2):")
        print(f"  Successful: {r_success}/{len(df)}")
        print(f"  Total time: {r_total:.3f}s ({r_total*1000:.1f}ms)")
        print(f"  Mean time:  {r_mean*1000:.2f}ms per station")
        print(f"  Median time: {r_median*1000:.2f}ms per station")
        print()
    
    if test_python and test_r:
        speedup_mean = df['speedup'].mean()
        speedup_median = df['speedup'].median()
        time_saved = r_total - py_total
        time_saved_pct = (time_saved / r_total) * 100
        
        print(f"Performance Comparison:")
        print(f"  Mean speedup:   {speedup_mean:.2f}x")
        print(f"  Median speedup: {speedup_median:.2f}x")
        print(f"  Time saved:     {time_saved:.3f}s ({time_saved*1000:.1f}ms)")
        print(f"  Time saved:     {time_saved_pct:.1f}%")
        print()
        
        # Extrapolation
        print(f"Extrapolation for larger projects:")
        for n_stations in [100, 500, 1000, 5000]:
            py_time = py_mean * n_stations
            r_time = r_mean * n_stations
            saved = r_time - py_time
            
            print(f"  {n_stations:5d} stations: Python {py_time:.1f}s vs R {r_time:.1f}s "
                  f"(save {saved:.1f}s = {saved/60:.1f}min)")
        print()
    
    # Save results
    output_dir = repo_root / 'data' / 'test_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / 'performance_benchmark.csv'
    df.to_csv(csv_file, index=False)
    print(f"Results saved to: {csv_file}")
    
    # Detailed breakdown by dataset size
    print()
    print("="*70)
    print("PERFORMANCE BY DATASET SIZE")
    print("="*70)
    print()
    
    if test_python and test_r:
        # Bin by flow count
        df['size_bin'] = pd.cut(df['n_flows'], bins=[0, 20, 40, 60, 100, 200], 
                                 labels=['<20', '20-40', '40-60', '60-100', '100+'])
        
        size_stats = df.groupby('size_bin').agg({
            'n_flows': 'count',
            'python_time': 'mean',
            'r_time': 'mean',
            'speedup': 'mean'
        }).round(4)
        
        print(f"{'Size':<10} {'Count':<8} {'Python(ms)':<12} {'R(ms)':<12} {'Speedup':<10}")
        print("-"*70)
        for idx, row in size_stats.iterrows():
            print(f"{idx:<10} {int(row['n_flows']):<8} "
                  f"{row['python_time']*1000:<12.2f} "
                  f"{row['r_time']*1000:<12.2f} "
                  f"{row['speedup']:<10.2f}x")
    
    print()
    print("="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Benchmark Python MGBT vs R MGBT performance'
    )
    parser.add_argument(
        '--python-only',
        action='store_true',
        help='Test Python only (skip R)'
    )
    parser.add_argument(
        '--r-only',
        action='store_true',
        help='Test R only (skip Python)'
    )
    
    args = parser.parse_args()
    
    test_python = not args.r_only
    test_r = not args.python_only
    
    df = run_benchmark(test_r=test_r, test_python=test_python)
