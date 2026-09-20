# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython port of :func:`src.filters.rls.rls_filter` (identical numerics, no Python loop overhead)."""

import numpy as np
cimport numpy as cnp

cnp.import_array()


def rls_filter_cy(double[::1] d, double[::1] x_ref, double lam, int N, double delta=1e-2):
    """See :func:`src.filters.rls.rls_filter`. Returns ``(e, w)``."""
    cdef Py_ssize_t n_samples = d.shape[0]
    if x_ref.shape[0] != n_samples:
        raise ValueError("d and x_ref must have the same length")
    if not (0.0 < lam <= 1.0):
        raise ValueError("lam must be in (0, 1]")

    cdef cnp.ndarray[cnp.float64_t, ndim=1] e_arr = np.zeros(n_samples, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] w_arr = np.zeros(N, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] P_arr = np.eye(N, dtype=np.float64) / delta
    cdef cnp.ndarray[cnp.float64_t, ndim=1] pi_arr = np.zeros(N, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] k_arr = np.zeros(N, dtype=np.float64)

    cdef double[::1] e = e_arr
    cdef double[::1] w = w_arr
    cdef double[:, ::1] P = P_arr
    cdef double[::1] pi = pi_arr
    cdef double[::1] k = k_arr

    cdef Py_ssize_t n, i, j
    cdef double y_n, err, denom, inv_lam = 1.0 / lam, tmp

    for n in range(N, n_samples):
        # prediction and error
        y_n = 0.0
        for i in range(N):
            y_n += w[i] * x_ref[n - N + i]
        err = d[n] - y_n
        e[n] = err

        # pi = P @ x ; denom = lam + x @ pi
        denom = lam
        for i in range(N):
            tmp = 0.0
            for j in range(N):
                tmp += P[i, j] * x_ref[n - N + j]
            pi[i] = tmp
            denom += x_ref[n - N + i] * tmp

        # gain and weight update
        for i in range(N):
            k[i] = pi[i] / denom
            w[i] += k[i] * err

        # P = (P - outer(k, pi)) / lam, then symmetrise
        for i in range(N):
            for j in range(N):
                P[i, j] = (P[i, j] - k[i] * pi[j]) * inv_lam
        for i in range(N):
            for j in range(i + 1, N):
                tmp = 0.5 * (P[i, j] + P[j, i])
                P[i, j] = tmp
                P[j, i] = tmp

    return e_arr, w_arr
