"""Week 4, day 2 -- Wavelet (C3) + CSP features -> LDA baseline accuracy."""

import pickle

import _bootstrap  # noqa: F401
import numpy as np
from moabb.datasets import BNCI2014_001
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

from src.features.wavelet import extract_wavelet_features

with open(_bootstrap.RESULTS_DIR / "csp_and_split.pkl", "rb") as f:
    d = pickle.load(f)
csp = d["csp"]
X_train, X_test, y_train, y_test = d["X_train"], d["X_test"], d["y_train"], d["y_test"]

raw = list(list(BNCI2014_001().get_data(subjects=[1])[1].values())[0].values())[0]
c3_idx = raw.ch_names.index("C3")

wave_train = np.array([extract_wavelet_features(trial[c3_idx]) for trial in X_train])
wave_test = np.array([extract_wavelet_features(trial[c3_idx]) for trial in X_test])
print("Wavelet train shape:", wave_train.shape)

csp_train, csp_test = csp.transform(X_train), csp.transform(X_test)  # transform only, no refit
print("CSP train shape:", csp_train.shape)

feat_train = np.hstack([wave_train, csp_train])
feat_test = np.hstack([wave_test, csp_test])
print("Combined feature shape:", feat_train.shape)

lda = LinearDiscriminantAnalysis().fit(feat_train, y_train)
acc = accuracy_score(y_test, lda.predict(feat_test))
print(f"LDA baseline test accuracy: {acc:.4f} ({acc * 100:.1f}%)")

out = _bootstrap.RESULTS_DIR / "lda_model.pkl"
with open(out, "wb") as f:
    pickle.dump({"lda": lda, "c3_idx": c3_idx}, f)
print("Saved", out)
