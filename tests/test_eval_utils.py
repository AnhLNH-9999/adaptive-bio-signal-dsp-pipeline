import numpy as np

from src.utils.eval_utils import compute_sqnr, count_macs_lms, count_macs_rls


def test_sqnr_identical_is_inf():
    x = np.ones(10)
    assert compute_sqnr(x, x) == float("inf")


def test_macs_formulas():
    assert count_macs_lms(8, 250) == 4000
    assert count_macs_rls(8, 250) == 64000
