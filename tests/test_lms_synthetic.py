import sys
sys.path.insert(0, 'src/filters')
from lms import lms_filter
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
fs = 250
duration = 2
t = np.arange(0, duration, 1/fs)
n_samples = len(t)

freq = 10
true_interference = 2.0 * np.sin(2 * np.pi * freq * t)
background_noise = 0.1 * np.random.randn(n_samples)

d = true_interference + background_noise
x_ref = np.sin(2 * np.pi * freq * t)

mu = 0.05
N = 4
e, w = lms_filter(d, x_ref, mu, N)

plt.figure(figsize=(10, 4))
plt.plot(t, e)
plt.title("Sai so e[n] theo thoi gian - LMS tren tin hieu tong hop")
plt.xlabel("Thoi gian (s)")
plt.ylabel("e[n]")
plt.axhline(0, color="gray", linewidth=0.5)
plt.savefig("results/lms_synthetic_error.png", dpi=100)
plt.show()

print("Sai so RMS 10% dau:", np.sqrt(np.mean(e[:n_samples//10]**2)))
print("Sai so RMS 10% cuoi:", np.sqrt(np.mean(e[-n_samples//10:]**2)))