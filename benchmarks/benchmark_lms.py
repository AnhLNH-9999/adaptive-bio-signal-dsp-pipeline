import sys, time
sys.path.insert(0, 'src/filters')
import numpy as np

from lms import lms_filter
from lms_cython import lms_filter_cy

np.random.seed(0)
d = np.random.randn(5000)
x_ref = np.random.randn(5000)
mu, N = 0.01, 8

t0 = time.perf_counter()
e1, w1 = lms_filter(d, x_ref, mu, N)
t1 = time.perf_counter()
print(f"Python thuan: {(t1 - t0)*1000:.2f} ms")

t0 = time.perf_counter()
e2, w2 = lms_filter_cy(d, x_ref, mu, N)
t1 = time.perf_counter()
print(f"Cython:       {(t1 - t0)*1000:.2f} ms")

print(f"Ket qua giong nhau: {np.allclose(e1, e2)}")