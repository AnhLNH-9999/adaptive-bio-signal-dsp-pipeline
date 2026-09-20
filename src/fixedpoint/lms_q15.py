"""LMS filter simulated entirely in Q15 integer arithmetic.

Every multiply is Q15 x Q15 = Q30 and is shifted right by 15 bits to return
to Q15, mirroring what a 16-bit DSP or FPGA datapath would do. int64 is used
only as the accumulator to avoid intermediate overflow; every stored value is
saturated back to the int16 range.

Inputs **must** be normalised into ``[-1, 1)`` before calling, otherwise the
signal saturates and SQNR collapses (see ``docs/report/week5_fixedpoint.md``).
"""

from __future__ import annotations

import numpy as np

from .q15 import float_to_q15

_INT16_MIN, _INT16_MAX = -32768, 32767


def lms_filter_q15(d_float, x_ref_float, mu_float: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Q15 LMS filter. Same semantics as :func:`src.filters.lms.lms_filter`.

    Parameters
    ----------
    d_float, x_ref_float : array_like, shape (n_samples,)
        Float inputs in ``[-1, 1)``; converted to Q15 internally.
    mu_float : float
        Step size in ``[0, 1)``; converted to Q15 internally.
    N : int
        Number of taps.

    Returns
    -------
    e_q15 : ndarray of int16, shape (n_samples,)
    w_q15 : ndarray of int16, shape (N,)
    """
    d_q15 = float_to_q15(d_float).astype(np.int64)
    x_q15 = float_to_q15(x_ref_float).astype(np.int64)
    mu_q15 = int(float_to_q15(mu_float))

    n_samples = len(d_q15)
    w_q15 = np.zeros(N, dtype=np.int64)
    e_q15 = np.zeros(n_samples, dtype=np.int64)

    for n in range(N, n_samples):
        x_window = x_q15[n - N:n]

        acc = np.sum(w_q15 * x_window)              # Q30 accumulator
        y_n = acc >> 15                             # back to Q15

        e_n = np.clip(d_q15[n] - y_n, _INT16_MIN, _INT16_MAX)
        e_q15[n] = e_n

        me = (mu_q15 * e_n) >> 15                   # mu * e   (Q15)
        delta_w = (me * x_window) >> 15             # mu*e*x   (Q15)
        w_q15 = np.clip(w_q15 + delta_w, _INT16_MIN, _INT16_MAX)

    return e_q15.astype(np.int16), w_q15.astype(np.int16)
