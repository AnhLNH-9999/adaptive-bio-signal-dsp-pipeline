import numpy as np

from src.features.wavelet import BAND_NAMES, extract_wavelet_features


def test_wavelet_no_nan():
    features = extract_wavelet_features(np.random.default_rng(0).standard_normal(1000))
    assert not np.any(np.isnan(features))


def test_wavelet_fixed_length():
    rng = np.random.default_rng(0)
    lengths = {len(extract_wavelet_features(rng.standard_normal(n))) for n in (2000, 1000, 500, 250)}
    assert lengths == {len(BAND_NAMES)}
