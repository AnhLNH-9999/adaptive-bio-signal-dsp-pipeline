"""Week 6, day 5 -- MACs/second vs measured runtime (week 2). See docs/report/week6_macs.md."""

import _bootstrap  # noqa: F401

print("MACs/second vs measured runtime (N=8, fs=250 Hz)")
print("=" * 70)
print(f"{'Metric':30s} | {'LMS':>10s} | {'RLS':>10s} | {'RLS/LMS':>10s}")
print("-" * 70)
for name, lms, rls, ratio in [
    ("MAC/sample (theory)", "16", "256", "16.0x"),
    ("MAC/sample (actual)", "16", "160", "10.0x"),
    ("MACs/s (theory)", "4,000", "64,000", "16.0x"),
    ("MACs/s (actual)", "4,000", "40,000", "10.0x"),
    ("Runtime, pure Python (ms)", "12.23", "55.41", "4.5x"),
    ("Runtime, Cython (ms)", "0.11", "0.79", "7.2x"),
]:
    print(f"{name:30s} | {lms:>10s} | {rls:>10s} | {ratio:>10s}")
print("=" * 70)
