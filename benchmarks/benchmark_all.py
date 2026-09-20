import sys, time
sys.path.insert(0, 'src/filters')
import numpy as np

from lms import lms_filter
from lms_cython import lms_filter_cy
from rls import rls_filter
from rls_cython import rls_filter_cy

np.random.seed(0)
d = np.random.randn(5000)
x_ref = np.random.randn(5000)
mu, N = 0.01, 8
lam = 0.99

t0 = time.perf_counter()
e1, w1 = lms_filter(d, x_ref, mu, N)
t1 = time.perf_counter()
t_lms_py = (t1 - t0) * 1000

t0 = time.perf_counter()
e2, w2 = lms_filter_cy(d, x_ref, mu, N)
t1 = time.perf_counter()
t_lms_cy = (t1 - t0) * 1000

t0 = time.perf_counter()
e3, w3 = rls_filter(d, x_ref, lam, N)
t1 = time.perf_counter()
t_rls_py = (t1 - t0) * 1000

t0 = time.perf_counter()
e4, w4 = rls_filter_cy(d, x_ref, lam, N)
t1 = time.perf_counter()
t_rls_cy = (t1 - t0) * 1000

print(f"{'Bo loc':25s} | {'Thoi gian (ms)':>15s}")
print("-" * 45)
print(f"{'LMS (Python thuan)':25s} | {t_lms_py:15.2f}")
print(f"{'LMS (Cython)':25s} | {t_lms_cy:15.2f}")
print(f"{'RLS (Python thuan)':25s} | {t_rls_py:15.2f}")
print(f"{'RLS (Cython)':25s} | {t_rls_cy:15.2f}")