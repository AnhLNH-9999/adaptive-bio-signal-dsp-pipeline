"""Week 4, day 1 -- fit CSP on the training split only and persist CSP + split."""

import pickle

import _bootstrap  # noqa: F401
from mne.decoding import CSP
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.model_selection import train_test_split

dataset = BNCI2014_001()
paradigm = MotorImagery()
X, y, _ = paradigm.get_data(dataset=dataset, subjects=[1])
print("X shape (n_trials, n_channels, n_times):", X.shape)

mask = (y == "left_hand") | (y == "right_hand")
X_2class, y_2class = X[mask], y[mask]
print("Trials after 2-class filtering:", X_2class.shape[0])

# split BEFORE fitting CSP to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X_2class, y_2class, test_size=0.3, random_state=42, stratify=y_2class
)
print("Train trials:", X_train.shape[0], "| Test trials:", X_test.shape[0])

csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
csp.fit(X_train, y_train)
print("CSP fitted, filters_ shape:", csp.filters_.shape)

out = _bootstrap.RESULTS_DIR / "csp_and_split.pkl"
with open(out, "wb") as f:
    pickle.dump({"csp": csp, "X_train": X_train, "X_test": X_test,
                 "y_train": y_train, "y_test": y_test}, f)
print("Saved", out)
