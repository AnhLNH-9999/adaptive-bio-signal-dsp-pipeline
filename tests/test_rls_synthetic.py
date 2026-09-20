"""
test_rls_synthetic.py -- Ngay 2 / Tuan 2

Kiem thu RLS tren dung tin hieu tong hop da dung o Tuan 1 (sine 10Hz +
nhieu, x_ref la sine sach), thu vai gia tri lambda khac nhau.
Chay tu thu muc goc eeg-dsp-pipeline/ (giong test_lms_synthetic.py).

Dau ra: results/rls_lambda_sweep.png
"""
import sys
sys.path.insert(0, 'src/filters')
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rls import rls_filter

# ---- dung DUNG tin hieu tong hop nhu test_lms.py / test_lms_synthetic.py ----
fs = 250
t = np.arange(0, 2, 1/fs)
freq = 10
np.random.seed(0)
d = 2.0 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
x_ref = np.sin(2 * np.pi * freq * t)

N = 4  # theo NOTES_lms_params.md: N=4 du de loai bo 1 tan so don
lambdas = [0.90, 0.95, 0.99, 0.995, 1.0]

plt.figure(figsize=(8, 5))
print(f"{'lambda':>8} | {'RMS 10% dau':>14} | {'RMS 10% cuoi':>14}")
print("-" * 42)

for lam in lambdas:
    e, w = rls_filter(d, x_ref, lam, N)
    n = len(e)
    rms_first = np.sqrt(np.mean(e[:n//10]**2))
    rms_last = np.sqrt(np.mean(e[-n//10:]**2))
    print(f"{lam:8.3f} | {rms_first:14.4f} | {rms_last:14.4f}")

    # duong cong hoi tu: binh phuong sai so (dB), lam muot nhe de de doc
    err2 = e**2
    win = 5
    kernel = np.ones(win) / win
    smooth = np.convolve(err2, kernel, mode="valid")
    smooth_db = 10 * np.log10(smooth + 1e-12)
    label = f"lambda = {lam}" + ("  (NO! bi phan ky)" if lam == 0.90 else "")
    plt.plot(t[:len(smooth)], smooth_db, label=label)

plt.xlabel("Thoi gian (s)")
plt.ylabel("Sai so binh phuong (dB, da lam muot)")
plt.title("RLS: hoi tu voi cac lambda khac nhau (tin hieu sine 10Hz, N=4)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/rls_lambda_sweep.png", dpi=100)
print("\nDa luu results/rls_lambda_sweep.png")