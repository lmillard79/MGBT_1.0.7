"""Quick test of fast MGBT implementation."""

import sys
from pathlib import Path
import numpy as np
import time

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'pymgbt'))

from pymgbt.core.mgbt_fast import MGBT as MGBT_fast

# Test data
flows = np.array([
    1894.38, 1221.90, 1108.43, 672.61, 655.84, 610.87, 578.46,
    534.04, 501.34, 455.95, 408.63, 358.47, 336.82, 253.72,
    198.25, 182.64, 179.82, 163.71, 161.11, 146.59, 126.97,
    90.61, 80.65, 76.98, 61.95, 38.01, 20.18, 16.53,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
])

print(f"Testing fast MGBT with {len(flows)} flows...")
print()

start = time.perf_counter()
result = MGBT_fast(flows, alpha1=0.01, alpha10=0.10)
elapsed = time.perf_counter() - start

print(f"Result:")
print(f"  klow: {result.klow}")
print(f"  threshold: {result.low_outlier_threshold}")
print(f"  time: {elapsed*1000:.2f}ms")
print()

if elapsed > 1.0:
    print(f"⚠ WARNING: Took {elapsed:.2f}s - still too slow!")
else:
    print(f"✓ Fast enough: {elapsed*1000:.2f}ms")
