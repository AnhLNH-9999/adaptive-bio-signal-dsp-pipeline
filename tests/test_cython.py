"""Cython ports must match the pure-Python reference bit-for-bit (up to float rounding)."""

import numpy as np
import pytest

from src.filters.lms import lms_filter
from src.filters.rls import rls_filter

lms_cy = pytest.importorskip("src.filters.lms_cython")
rls_cy = pytest.importorskip("src.filters.rls_cython")


def _signals():
    rng = np.random.default_rng(1)
    return rng.standard_normal(1000), rng.standard_normal(1000)


def test_lms_cython_matches_python():
    d, x = _signals()
    e_py, w_py = lms_filter(d, x, 0.01, 8)
    e_cy, w_cy = lms_cy.lms_filter_cy(d, x, 0.01, 8)
    np.testing.assert_allclose(e_cy, e_py, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(w_cy, w_py, rtol=1e-10, atol=1e-12)


def test_rls_cython_matches_python():
    d, x = _signals()
    e_py, w_py = rls_filter(d, x, 0.99, 8)
    e_cy, w_cy = rls_cy.rls_filter_cy(d, x, 0.99, 8)
    np.testing.assert_allclose(e_cy, e_py, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(w_cy, w_py, rtol=1e-8, atol=1e-10)
