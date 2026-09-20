"""Week 6, day 1 -- theoretical MACs per sample and per second (N=8, fs=250 Hz)."""

import _bootstrap  # noqa: F401

from src.utils.eval_utils import count_macs_lms, count_macs_rls

N, fs = 8, 250.0
lms_per_sample, rls_per_sample = 2 * N, 4 * N ** 2

print(f"Formula: LMS = 2N, RLS = 4N^2  (N = {N})")
print(f"  LMS: {lms_per_sample} MAC/sample")
print(f"  RLS: {rls_per_sample} MAC/sample")
print(f"  RLS/LMS (theory) = {rls_per_sample / lms_per_sample:.1f}x")
print(f"\nMACs/second at fs = {fs:.0f} Hz:")
print(f"  LMS: {count_macs_lms(N, fs):,.0f}")
print(f"  RLS: {count_macs_rls(N, fs):,.0f}")
