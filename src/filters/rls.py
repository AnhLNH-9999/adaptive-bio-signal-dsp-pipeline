"""
rls.py — Bộ lọc thích nghi RLS, CÙNG GIAO DIỆN với lms.py thật của bạn
(src/filters/lms.py: lms_filter(d, x_ref, mu, N) -> (e, w)) để so sánh
trực tiếp, công bằng. Đặt file này vào đúng thư mục src/filters/, cạnh
lms.py.
"""
import numpy as np


def rls_filter(d, x_ref, lam, N, delta=1e-2):
    n_samples = len(d)
    w = np.zeros(N)
    P = np.eye(N) / delta
    e = np.zeros(n_samples)

    for n in range(N, n_samples):
        x_window = x_ref[n-N:n]

        y_n = np.dot(w, x_window)
        e[n] = d[n] - y_n

        pi = P @ x_window
        k = pi / (lam + x_window @ pi)

        w = w + k * e[n]

        P = (P - np.outer(k, pi)) / lam
        P = 0.5 * (P + P.T)

    return e, w