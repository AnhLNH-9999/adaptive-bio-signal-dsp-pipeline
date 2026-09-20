"""Least Mean Squares (LMS) adaptive filter -- pure NumPy reference implementation.

Per-sample cost is O(N): one N-length dot product for the prediction and one
N-length update for the weights (2N multiply-accumulates in total).
"""

from __future__ import annotations

import numpy as np


def lms_filter(d: np.ndarray, x_ref: np.ndarray, mu: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Run an N-tap LMS filter and return the error signal and final weights.

    Parameters
    ----------
    d : array_like, shape (n_samples,)
        Observed (noisy) signal, i.e. the desired signal ``d[n]``.
    x_ref : array_like, shape (n_samples,)
        Reference signal correlated with the interference (e.g. a 50 Hz sine).
    mu : float
        Step size (learning rate). Too large diverges, too small converges slowly.
    N : int
        Number of filter taps.

    Returns
    -------
    e : ndarray, shape (n_samples,)
        Error signal ``e[n] = d[n] - y[n]``; this is the cleaned output.
    w : ndarray, shape (N,)
        Final filter weights.
    """
    d = np.asarray(d, dtype=np.float64)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    if d.shape != x_ref.shape:
        raise ValueError(f"d and x_ref must have the same shape, got {d.shape} and {x_ref.shape}")

    n_samples = len(d)
    w = np.zeros(N)
    e = np.zeros(n_samples)

    for n in range(N, n_samples):
        x_window = x_ref[n - N:n]           # most recent N reference samples
        y_n = np.dot(w, x_window)           # filter prediction of the interference
        e[n] = d[n] - y_n                   # what remains is the cleaned signal
        w = w + mu * e[n] * x_window        # gradient-descent weight update

    return e, w
