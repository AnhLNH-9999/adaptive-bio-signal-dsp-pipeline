import sys
sys.path.insert(0, 'src/filters')
from lms import lms_filter
import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001

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

# Dataset nay da duoc notch-filter 50Hz san tu luc thu thap,
# nen tu them nhieu 50Hz da biet truoc de kiem thu bo loc mot cach co kiem soat
signal_std = np.std(eeg_clean)
injected_noise = 2 * signal_std * np.sin(2 * np.pi * 50 * t)
eeg_contaminated = eeg_clean + injected_noise

ref_50hz = np.sin(2 * np.pi * 50 * t)

mu = 0.01
N = 4
e, w = lms_filter(eeg_contaminated, ref_50hz, mu, N)

freqs = np.fft.rfftfreq(n_samples, d=1/fs)
spec_before = np.abs(np.fft.rfft(eeg_contaminated))
spec_after = np.abs(np.fft.rfft(e))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(freqs, spec_before)
axes[0].set_title("Pho truoc loc (co nhieu 50Hz da them)")
axes[0].set_xlim(0, 80)
axes[0].axvline(50, color="r", linestyle="--")

axes[1].plot(freqs, spec_after)
axes[1].set_title("Pho sau loc LMS")
axes[1].set_xlim(0, 80)
axes[1].axvline(50, color="r", linestyle="--")

plt.tight_layout()
plt.savefig("results/lms_real_eeg_spectrum.png", dpi=100)
plt.show()

idx_50 = np.argmin(np.abs(freqs - 50))
print(f"Bien do 50Hz truoc loc: {spec_before[idx_50]:.2f}")
print(f"Bien do 50Hz sau loc: {spec_after[idx_50]:.2f}")