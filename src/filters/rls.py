"""Recursive Least Squares (RLS) adaptive filter -- pure NumPy reference implementation.

Same call signature as :func:`src.filters.lms.lms_filter` so the two can be
compared on identical inputs. Per-sample cost is O(N^2) because the inverse
correlation matrix ``P`` (N x N) is updated every step.
"""

from __future__ import annotations

import numpy as np


def rls_filter(
    d: np.ndarray,
    x_ref: np.ndarray,
    lam: float,
    N: int,
    delta: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Run an N-tap RLS filter and return the error signal and final weights.

    Parameters
    ----------
    d : array_like, shape (n_samples,)
        Observed (noisy) signal.
    x_ref : array_like, shape (n_samples,)
        Reference signal correlated with the interference.
    lam : float
        Forgetting factor in (0, 1]. Values close to 1 give a long memory and
        low steady-state error; values well below ~0.95 risk covariance
        wind-up when the reference is not persistently exciting.
    N : int
        Number of filter taps.
    delta : float, optional
        Regularisation for the initial inverse correlation matrix
        ``P(0) = I / delta``. Default 1e-2.

    Returns
    -------
    e : ndarray, shape (n_samples,)
        Error signal (cleaned output).
    w : ndarray, shape (N,)
        Final filter weights.
    """
    d = np.asarray(d, dtype=np.float64)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    if d.shape != x_ref.shape:
        raise ValueError(f"d and x_ref must have the same shape, got {d.shape} and {x_ref.shape}")
    if not 0.0 < lam <= 1.0:
        raise ValueError(f"lam must be in (0, 1], got {lam}")

    n_samples = len(d)
    w = np.zeros(N)
    P = np.eye(N) / delta
    e = np.zeros(n_samples)

    for n in range(N, n_samples):
        x_window = x_ref[n - N:n]

        y_n = np.dot(w, x_window)
        e[n] = d[n] - y_n

        pi = P @ x_window                       # O(N^2)
        k = pi / (lam + x_window @ pi)          # gain vector
        w = w + k * e[n]

        P = (P - np.outer(k, pi)) / lam         # O(N^2)
        P = 0.5 * (P + P.T)                     # enforce symmetry for numerical stability

    return e, w
