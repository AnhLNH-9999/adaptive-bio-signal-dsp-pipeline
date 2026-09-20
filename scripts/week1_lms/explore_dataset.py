"""Week 0/1 -- load BNCI2014_001 (BCI Competition IV-2a) with MOABB and inspect it."""

import _bootstrap  # noqa: F401
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

dataset = BNCI2014_001()
paradigm = MotorImagery()
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

print("X shape (n_trials, n_channels, n_times):", X.shape)
print("Classes:", sorted(set(y)))

raw = list(list(dataset.get_data(subjects=[1])[1].values())[0].values())[0]
raw.plot(n_channels=10, duration=5, scalings="auto", block=True)
