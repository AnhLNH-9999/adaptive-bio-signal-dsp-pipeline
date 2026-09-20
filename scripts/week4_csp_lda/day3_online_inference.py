"""Week 4, day 3 -- end-to-end inference: RLS (C3) -> Wavelet + CSP -> LDA, on sliding windows."""

import pickle

import _bootstrap  # noqa: F401
import numpy as np

from src.features.wavelet import extract_wavelet_features
from src.filters.rls import rls_filter

with open(_bootstrap.RESULTS_DIR / "csp_and_split.pkl", "rb") as f:
    d1 = pickle.load(f)
csp, X_test, y_test = d1["csp"], d1["X_test"], d1["y_test"]

with open(_bootstrap.RESULTS_DIR / "lda_model.pkl", "rb") as f:
    d2 = pickle.load(f)
lda, c3_idx = d2["lda"], d2["c3_idx"]


def online_inference(window: np.ndarray, fs: float = 250.0) -> str:
    """Predict the class of one multi-channel window of shape (n_channels, n_times)."""
    c3_signal = window[c3_idx]
    t = np.arange(len(c3_signal)) / fs
    ref_50hz = np.sin(2 * np.pi * 50 * t)
    e_filtered, _ = rls_filter(c3_signal, ref_50hz, lam=0.99, N=4)

    wave_feat = extract_wavelet_features(e_filtered)
    csp_feat = csp.transform(window[np.newaxis, :, :])[0]
    feat_vector = np.hstack([wave_feat, csp_feat]).reshape(1, -1)
    return lda.predict(feat_vector)[0]


if __name__ == "__main__":
    pred = online_inference(X_test[0])
    print(f"Full trial: predicted={pred}, true={y_test[0]}")

    window_len, step = 500, 125  # 2 s window, 0.5 s hop at fs=250 Hz
    trial = X_test[1]
    print(f"\nSliding window (window={window_len}, step={step}):")
    for start in range(0, trial.shape[1] - window_len + 1, step):
        p = online_inference(trial[:, start:start + window_len])
        print(f"  samples {start}-{start + window_len}: predicted = {p}")
    print(f"True label: {y_test[1]}")
