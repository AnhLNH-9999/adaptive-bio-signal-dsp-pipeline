"""Latency, SQNR and MACs/second helpers shared by every stage of the pipeline."""

from __future__ import annotations

import time

import numpy as np


def measure_latency(func, *args, n_repeats: int = 50, **kwargs) -> dict[str, float]:
    """Call ``func(*args, **kwargs)`` ``n_repeats`` times and return mean/std wall time in ms."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {"mean_ms": float(np.mean(times)), "std_ms": float(np.std(times))}


def compute_sqnr(reference, test) -> float:
    """Signal-to-Quantisation-Noise Ratio in dB between ``reference`` and ``test``."""
    reference = np.asarray(reference, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean((reference - test) ** 2)
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def count_macs_lms(n_taps: int, fs: float) -> float:
    """Theoretical MACs/second for an LMS filter: ``2N`` per sample."""
    return 2 * n_taps * fs


def count_macs_rls(n_taps: int, fs: float) -> float:
    """Theoretical MACs/second for an RLS filter: ``4N^2`` per sample (order-of-magnitude bound)."""
    return 4 * (n_taps ** 2) * fs
