"""Week 2 -- effect of the RLS forgetting factor on a synthetic 10 Hz sine (N=4)."""

import _bootstrap  # noqa: F401
import numpy as np

from src.filters.lms import lms_filter
from src.filters.rls import rls_filter

fs = 250
t = np.arange(0, 2, 1 / fs)
rng = np.random.default_rng(0)
d = 2.0 * np.sin(2 * np.pi * 10 * t) + 0.1 * rng.standard_normal(len(t))
x_ref = np.sin(2 * np.pi * 10 * t)


def rms_head_tail(e, frac=0.1):
    k = int(len(e) * frac)
    return np.sqrt(np.mean(e[:k] ** 2)), np.sqrt(np.mean(e[-k:] ** 2))


print(f"{'lambda':>8s} | {'RMS first 10%':>14s} | {'RMS last 10%':>13s}")
print("-" * 42)
for lam in [0.90, 0.95, 0.99, 0.995, 1.0]:
    e, _ = rls_filter(d, x_ref, lam=lam, N=4)
    head, tail = rms_head_tail(e)
    flag = "  <- unstable (covariance wind-up)" if tail > head else ""
    print(f"{lam:8.3f} | {head:14.4f} | {tail:13.4f}{flag}")

e_lms, _ = lms_filter(d, x_ref, mu=0.05, N=4)
head, tail = rms_head_tail(e_lms)
print(f"\nLMS mu=0.05 | RMS first 10% = {head:.4f} | RMS last 10% = {tail:.4f}")
