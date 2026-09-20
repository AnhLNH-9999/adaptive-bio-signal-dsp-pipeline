"""Week 3 -- wavelet features on every left/right-hand trial (channel C3), box-plot per band."""

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from src.features.wavelet import BAND_NAMES, extract_wavelet_features

dataset = BNCI2014_001()
paradigm = MotorImagery()
X, y, _ = paradigm.get_data(dataset=dataset, subjects=[1])
print("X shape:", X.shape, " classes:", sorted(set(y)))

raw = list(list(dataset.get_data(subjects=[1])[1].values())[0].values())[0]
c3_idx = raw.ch_names.index("C3")
X_c3 = X[:, c3_idx, :] * 1e6  # volts -> microvolts

mask = (y == "left_hand") | (y == "right_hand")
X_2class, y_2class = X_c3[mask], y[mask]
print("Trials after 2-class filtering:", X_2class.shape[0])

features = np.array([extract_wavelet_features(trial) for trial in X_2class])
print("Feature matrix shape:", features.shape)
assert not np.any(np.isnan(features)), "NaN in features!"

fig, axes = plt.subplots(1, len(BAND_NAMES), figsize=(18, 4))
for i, ax in enumerate(axes):
    ax.boxplot(
        [features[y_2class == "left_hand", i], features[y_2class == "right_hand", i]],
        tick_labels=["left_hand", "right_hand"],
    )
    ax.set_title(BAND_NAMES[i], fontsize=10)
plt.suptitle("Wavelet feature distribution per class (channel C3)")
plt.tight_layout()
out = _bootstrap.RESULTS_DIR / "wavelet_boxplot_classes.png"
plt.savefig(out, dpi=100)
print("Saved", out)
