"""Week 6, day 3 -- MACs/second table (theory vs actual count), N=8, fs=250 Hz."""

import _bootstrap  # noqa: F401

N, fs = 8, 250.0
rows = [
    ("LMS (theory)", 2 * N),
    ("LMS (actual)", 2 * N),
    ("RLS (theory)", 4 * N * N),
    ("RLS (actual)", 2 * N * N + 4 * N),
]
print(f"{'':16s} | {'MAC/sample':>10s} | {'MACs/s':>12s}")
print("-" * 44)
for name, per_sample in rows:
    print(f"{name:16s} | {per_sample:10d} | {per_sample * fs:12,.0f}")
print(f"\nRLS/LMS (theory): {rows[2][1] / rows[0][1]:.1f}x   RLS/LMS (actual): {rows[3][1] / rows[1][1]:.1f}x")
