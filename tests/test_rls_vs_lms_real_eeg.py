import sys
sys.path.insert(0, 'src/filters')
from lms import lms_filter
from rls import rls_filter          # <-- dòng mới thêm
import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001

# --- phần này giữ nguyên y hệt file gốc của bạn ---
dataset = BNCI2014_001()
data = dataset.get_data(subjects=[1])
raw = list(list(data[1].values())[0].values())[0]

fs = raw.info['sfreq']
channel_idx = raw.ch_names.index('C3')
eeg_clean = raw.get_data()[channel_idx]

duration = 10
n_samples = int(duration * fs)
eeg_clean = eeg_clean[:n_samples]
t = np.arange(n_samples) / fs

signal_std = np.std(eeg_clean)
injected_noise = 2 * signal_std * np.sin(2 * np.pi * 50 * t)
eeg_contaminated = eeg_clean + injected_noise
ref_50hz = np.sin(2 * np.pi * 50 * t)

N = 4

# TODO 1: chạy LMS — gọi lms_filter(eeg_contaminated, ref_50hz, mu, N) với mu=0.01
mu = 0.01
e_lms, w_lms = lms_filter(eeg_contaminated, ref_50hz, mu, N)

# TODO 2: chạy RLS — gọi rls_filter(eeg_contaminated, ref_50hz, lam, N) với lam=0.99
lam = 0.99
e_rls, w_rls = rls_filter(eeg_contaminated, ref_50hz, lam, N)

# TODO 3: viết hàm làm mượt bình phương sai số (giống Ngày 2), rồi áp dụng cho cả 2
def smooth_curve(e, win=25):
    err2 = e ** 2 
    kernel = np.ones(win) / win
    return np.convolve(err2, kernel, mode="valid")

curve_lms = smooth_curve(e_lms)
curve_rls = smooth_curve(e_rls)

# TODO 4: vẽ chung 1 biểu đồ, 2 đường
plt.figure(figsize=(8, 5))
plt.plot(curve_lms, label="LMS") 
plt.plot(curve_rls, label="RLS") 
plt.xlabel("Số mẫu")
plt.ylabel("Sai số bình phương (đã làm mượt)")
plt.title("So sánh hội tụ RLS vs LMS trên EEG thật")
plt.legend()
plt.savefig("results/rls_vs_lms_real_eeg.png", dpi=100)
print("Đã lưu results/rls_vs_lms_real_eeg.png")