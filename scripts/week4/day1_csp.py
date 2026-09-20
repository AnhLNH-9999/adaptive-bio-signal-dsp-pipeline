"""
day1_csp.py -- Tuan 4, Ngay 1
Cai dat CSP, huan luyen bo loc khong gian tren tap huan luyen, luu lai.
"""
import sys
import pickle
import numpy as np
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

# --- tai TOAN BO kenh (CSP can nhieu kenh, khac Tuan 3 chi dung C3) ---
dataset = BNCI2014_001()
paradigm = MotorImagery()
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])
print("X shape (n_trials, n_channels, n_times):", X.shape)

# TODO 1: loc ra 2 lop can dung (giong het TODO 1 cua Tuan 3, Ngay 4)
mask = (y == 'left_hand') | (y == 'right_hand')
X_2class = X[mask]
y_2class = y[mask]
print("So trial sau khi loc 2 lop:", X_2class.shape[0])

# --- chia train/test NGAY TU BAY GIO, truoc khi fit CSP (tranh data leakage) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_2class, y_2class, test_size=0.3, random_state=42, stratify=y_2class
)
print("So trial huan luyen:", X_train.shape[0], " | So trial kiem tra:", X_test.shape[0])

# --- fit CSP CHI tren tap huan luyen ---
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
csp.fit(X_train, y_train)
print("CSP fit xong, filters_ shape:", csp.filters_.shape)

# --- luu CSP + bo chia train/test de Ngay 2 dung lai dung y het ---
with open("csp_and_split.pkl", "wb") as f:
    pickle.dump({
        "csp": csp,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }, f)
print("Da luu csp_and_split.pkl")