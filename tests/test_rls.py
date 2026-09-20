import numpy as np

from src.filters.rls import rls_filter


def test_rls_no_nan():
    rng = np.random.default_rng(0)
    e, w = rls_filter(rng.standard_normal(200), rng.standard_normal(200), lam=0.99, N=8)
    assert not np.any(np.isnan(e)) and not np.any(np.isnan(w))


def test_rls_converges_on_known_signal():
    t = np.arange(0, 2, 1 / 250)
    rng = np.random.default_rng(0)
    d = 2.0 * np.sin(2 * np.pi * 10 * t) + 0.1 * rng.standard_normal(len(t))
    x_ref = np.sin(2 * np.pi * 10 * t)
    e, _ = rls_filter(d, x_ref, lam=0.99, N=4)
    n = len(e)
    rms_first, rms_last = np.sqrt(np.mean(e[: n // 10] ** 2)), np.sqrt(np.mean(e[-n // 10:] ** 2))
    assert rms_last < rms_first
