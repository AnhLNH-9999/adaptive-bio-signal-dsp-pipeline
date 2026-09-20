"""Week 2 -- wall-clock benchmark of LMS / RLS, pure Python vs Cython (N=5000 samples, 8 taps).

Build the Cython extensions first:  python setup.py build_ext --inplace
"""

import time

import _bootstrap  # noqa: F401
import numpy as np

from src.filters.lms import lms_filter
from src.filters.rls import rls_filter

try:
    from src.filters.lms_cython import lms_filter_cy
    from src.filters.rls_cython import rls_filter_cy
except ImportError:  # extensions not built
    lms_filter_cy = rls_filter_cy = None


def timed(func, *args):
    t0 = time.perf_counter()
    func(*args)
    return (time.perf_counter() - t0) * 1000.0


np.random.seed(0)
d = np.random.randn(5000)
x_ref = np.random.randn(5000)
mu, lam, N = 0.01, 0.99, 8

rows = [
    ("LMS (pure Python)", timed(lms_filter, d, x_ref, mu, N)),
    ("RLS (pure Python)", timed(rls_filter, d, x_ref, lam, N)),
]
if lms_filter_cy is not None:
    rows += [
        ("LMS (Cython)", timed(lms_filter_cy, d, x_ref, mu, N)),
        ("RLS (Cython)", timed(rls_filter_cy, d, x_ref, lam, N)),
    ]
else:
    print("Cython extensions not built -- run `python setup.py build_ext --inplace` to include them.\n")

print(f"{'Filter':25s} | {'Time (ms)':>12s}")
print("-" * 40)
for name, ms in rows:
    print(f"{name:25s} | {ms:12.2f}")
