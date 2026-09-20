import numpy as np

from src.fixedpoint.lms_q15 import lms_filter_q15
from src.fixedpoint.q15 import float_to_q15, q15_to_float
from src.utils.eval_utils import compute_sqnr
from src.filters.lms import lms_filter


def test_q15_boundaries():
    assert float_to_q15(-1.0) == -32768
    assert float_to_q15(0.0) == 0
    assert float_to_q15(0.99999) == 32767      # clamps instead of wrapping negative
    assert float_to_q15(1.5) == 32767
    assert float_to_q15(-2.0) == -32768


def test_q15_roundtrip_small_error():
    x = np.array([-0.7, 0.3, 0.123456])
    assert np.max(np.abs(x - q15_to_float(float_to_q15(x)))) < 2 ** -14


def test_lms_q15_sqnr_after_normalisation():
    t = np.arange(0, 2, 1 / 250)
    rng = np.random.default_rng(0)
    d = 2.0 * np.sin(2 * np.pi * 10 * t) + 0.1 * rng.standard_normal(len(t))
    x_ref = np.sin(2 * np.pi * 10 * t)
    d = d / (np.max(np.abs(d)) * 1.05)
    e_q15, _ = lms_filter_q15(d, x_ref, mu_float=0.05, N=4)
    e_float, _ = lms_filter(d, x_ref, 0.05, 4)
    assert compute_sqnr(e_float, q15_to_float(e_q15)) >= 30
