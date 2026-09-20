# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython port of :func:`src.filters.lms.lms_filter` (identical numerics, no Python loop overhead)."""

import numpy as np
cimport numpy as cnp

cnp.import_array()


def lms_filter_cy(double[::1] d, double[::1] x_ref, double mu, int N):
    """See :func:`src.filters.lms.lms_filter`. Returns ``(e, w)``."""
    cdef Py_ssize_t n_samples = d.shape[0]
    if x_ref.shape[0] != n_samples:
        raise ValueError("d and x_ref must have the same length")

    cdef cnp.ndarray[cnp.float64_t, ndim=1] e_arr = np.zeros(n_samples, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] w_arr = np.zeros(N, dtype=np.float64)
    cdef double[::1] e = e_arr
    cdef double[::1] w = w_arr

    cdef Py_ssize_t n, i
    cdef double y_n, err, scale

    for n in range(N, n_samples):
        y_n = 0.0
        for i in range(N):
            y_n += w[i] * x_ref[n - N + i]
        err = d[n] - y_n
        e[n] = err
        scale = mu * err
        for i in range(N):
            w[i] += scale * x_ref[n - N + i]

    return e_arr, w_arr
