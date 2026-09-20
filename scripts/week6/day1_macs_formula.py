"""
day1_macs_formula.py -- Tuan 6, Ngay 1
Cong thuc ly thuyet MACs/mau cho LMS va RLS, ap dung voi N=8 (khop voi
benchmark da do o Tuan 1-2, de Ngay 5 doi chieu duoc).
"""
from eval_utils import count_macs_lms, count_macs_rls

N = 8       # so taps, khop voi benchmark_lms.py / benchmark_all.py Tuan 1-2
fs = 250.0  # Hz, tan so lay mau that cua dataset BNCI2014_001

macs_lms_per_sample = 2 * N
macs_rls_per_sample = 4 * (N ** 2)

print(f"Cong thuc: LMS = 2N, RLS = 4N^2  (N = {N})")
print(f"  LMS: 2 x {N} = {macs_lms_per_sample} MAC/mau")
print(f"  RLS: 4 x {N}^2 = {macs_rls_per_sample} MAC/mau")
print(f"  Ty le RLS/LMS (ly thuyet) = {macs_rls_per_sample/macs_lms_per_sample:.1f}x")

# dung lai DUNG ham da co san tu Tuan 0
macs_lms_per_sec = count_macs_lms(N, fs)
macs_rls_per_sec = count_macs_rls(N, fs)
print(f"\nMACs/giay (dung eval_utils.py Tuan 0):")
print(f"  LMS: {macs_lms_per_sec:,.0f} MACs/s")
print(f"  RLS: {macs_rls_per_sec:,.0f} MACs/s")