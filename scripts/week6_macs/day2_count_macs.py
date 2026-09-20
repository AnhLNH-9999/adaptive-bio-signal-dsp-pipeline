"""Week 6, day 2 -- count the MACs actually executed per iteration of lms_filter / rls_filter."""

import _bootstrap  # noqa: F401

N = 8


def count_macs_lms_actual(N):
    return N + N  # dot(w, x) + weight update


def count_macs_rls_actual(N):
    return (
        N          # y = dot(w, x)
        + N * N    # pi = P @ x
        + N        # denom = lam + x @ pi
        + N        # k = pi / denom   (divisions, counted as MACs)
        + N        # w += k * e
        + N * N    # P = (P - outer(k, pi)) / lam
    )


lms_actual, rls_actual = count_macs_lms_actual(N), count_macs_rls_actual(N)
print(f"Actual count (N={N}):  LMS = {lms_actual} MAC/sample,  RLS = {rls_actual} MAC/sample")
print(f"Theory:                LMS = {2 * N},  RLS = {4 * N * N}")
print(f"LMS matches theory: {lms_actual == 2 * N} | RLS matches theory: {rls_actual == 4 * N * N}")
