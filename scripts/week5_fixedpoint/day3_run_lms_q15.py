"""Week 5 -- run float and Q15 LMS on the same normalised 10 Hz test signal and save both outputs."""

import _bootstrap  # noqa: F401
import numpy as np

from src.filters.lms import lms_filter
from src.fixedpoint.lms_q15 import lms_filter_q15
from src.fixedpoint.q15 import q15_to_float

fs = 250
t = np.arange(0, 2, 1 / fs)
rng = np.random.default_rng(0)
d = 2.0 * np.sin(2 * np.pi * 10 * t) + 0.1 * rng.standard_normal(len(t))
x_ref = np.sin(2 * np.pi * 10 * t)
mu, N = 0.05, 4

# normalise d into [-1, 1) with 5 % headroom -- without this the Q15 path saturates
d_max = np.max(np.abs(d))
d = d / (d_max * 1.05)
print(f"Normalised d: max |d| before={d_max:.4f}, after={np.max(np.abs(d)):.4f}")

e_q15, w_q15 = lms_filter_q15(d, x_ref, mu_float=mu, N=N)
e_float, _ = lms_filter(d, x_ref, mu, N)

out = _bootstrap.RESULTS_DIR / "q15_lms_output.npz"
np.savez(out, e_float=e_float, e_q15_as_float=q15_to_float(e_q15), e_q15_raw=e_q15)
print("Final Q15 weights:", w_q15)
print("Saved", out)
