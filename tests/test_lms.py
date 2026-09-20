import numpy as np

from src.filters.lms import lms_filter


def test_lms_no_nan():
    rng = np.random.default_rng(0)
    e, w = lms_filter(rng.standard_normal(200), rng.standard_normal(200), mu=0.01, N=8)
    assert not np.any(np.isnan(e)) and not np.any(np.isnan(w))


def test_lms_converges_on_known_signal():
    t = np.arange(0, 2, 1 / 250)
    rng = np.random.default_rng(0)
    d = 2.0 * np.sin(2 * np.pi * 10 * t) + 0.1 * rng.standard_normal(len(t))
    x_ref = np.sin(2 * np.pi * 10 * t)
    e, _ = lms_filter(d, x_ref, mu=0.05, N=4)
    n = len(e)
    assert np.sqrt(np.mean(e[-n // 10:] ** 2)) < np.sqrt(np.mean(e[: n // 10] ** 2))
