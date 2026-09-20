"""Discrete Wavelet Transform band-power features for a single EEG channel.

A 5-level ``db4`` decomposition at fs = 250 Hz yields sub-bands that line up
approximately with the classical EEG rhythms:

=====  ==============  ========
Coef   Approx. band    Rhythm
=====  ==============  ========
A5     0 -   4 Hz      delta
D5     4 -   8 Hz      theta
D4     8 -  16 Hz      alpha
D3    16 -  31 Hz      beta
D2    31 -  62 Hz      gamma
D1    62 - 125 Hz      (high-frequency / noise)
=====  ==============  ========

The feature vector is the log of the mean energy in each sub-band, so its
length (6) is fixed regardless of the input window length.
"""

from __future__ import annotations

import numpy as np
import pywt

WAVELET = "db4"
LEVEL = 5
BAND_NAMES = ["delta (A5)", "theta (D5)", "alpha (D4)", "beta (D3)", "gamma (D2)", "D1"]
_EPS = 1e-12


def extract_wavelet_features(signal: np.ndarray, wavelet: str = WAVELET, level: int = LEVEL) -> np.ndarray:
    """Return a fixed-length vector of log band energies for a 1-D signal.

    Parameters
    ----------
    signal : array_like, shape (n_times,)
        One channel of EEG (any length >= 2**level samples recommended).
    wavelet : str, optional
        PyWavelets wavelet name. Default ``"db4"``.
    level : int, optional
        Decomposition depth. Default 5 (gives ``level + 1`` features).

    Returns
    -------
    ndarray, shape (level + 1,)
        ``log(mean(c**2) + eps)`` for ``[A_level, D_level, ..., D_1]``.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = np.array([np.log(np.mean(c ** 2) + _EPS) for c in coeffs])
    return features
