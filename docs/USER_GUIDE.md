# User Guide

How to set up, run and configure the pipeline. The weekly write-ups with measured numbers live in
[`docs/report/`](report/).

## 1. Prerequisites

- Python 3.10 – 3.12
- ~1 GB free disk for the BCI Competition IV-2a dataset (downloaded automatically by MOABB into
  `~/mne_data/` on first use)
- A C compiler if you want the Cython filters (Linux: `build-essential`; macOS: Xcode CLT; Windows:
  Microsoft C++ Build Tools). Everything also works without them.
- GPU optional: the week-7 model has < 1 000 parameters and trains on CPU in a few minutes.

## 2. Installation

```bash
git clone https://github.com/AnhLNH-9999/adaptive-bio-signal-dsp-pipeline.git
cd adaptive-bio-signal-dsp-pipeline
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

or with conda: `conda env create -f environment.yml && conda activate eeg-dsp-pipeline`.

Optional Cython build and sanity check:

```bash
python setup.py build_ext --inplace
python -m pytest
```

## 3. Quick demo (no dataset)

```bash
jupyter notebook notebooks/demo.ipynb
```

## 4. Running the pipeline on the real dataset

Scripts are grouped by roadmap week and are meant to be run in order. Each one adds the repository
root to `sys.path` so you can launch it from anywhere; outputs go to `results/`.

| Step | Command | Produces |
| --- | --- | --- |
| Inspect dataset | `python scripts/week1_lms/explore_dataset.py` | interactive MNE plot |
| λ sweep, RLS vs LMS | `python scripts/week2_rls/lambda_sweep.py` | console table |
| Timing benchmark | `python scripts/week2_rls/benchmark_all.py` | console table |
| Wavelet features per class | `python scripts/week3_wavelet/day4_boxplot.py` | `wavelet_boxplot_classes.png` |
| Fit CSP (train split only) | `python scripts/week4_csp_lda/day1_csp.py` | `csp_and_split.pkl` |
| Train LDA baseline | `python scripts/week4_csp_lda/day2_lda.py` | `lda_model.pkl` |
| Sliding-window inference | `python scripts/week4_csp_lda/day3_online_inference.py` | console |
| Q15 LMS run | `python scripts/week5_fixedpoint/day3_run_lms_q15.py` | `q15_lms_output.npz` |
| SQNR | `python scripts/week5_fixedpoint/day4_sqnr.py` | console |
| MACs analysis | `python scripts/week6_macs/day1..5_*.py` | `macs_comparison.png` |
| Train EEGNet-Lite | `python scripts/week7_nn/day2_train_eegnet.py` | `eegnet_float32.pt` |
| QAT 16-bit | `python scripts/week7_nn/day3_train_qat.py` | `eegnet_qat.pt` |
| Evaluate QAT | `python scripts/week7_nn/day4_eval_qat.py` | console |
| NN MACs + latency | `python scripts/week7_nn/day5_ai_macs_latency.py` | console |

## 5. Using the library directly

```python
import numpy as np
from src.filters import lms_filter, rls_filter
from src.features import extract_wavelet_features
from src.fixedpoint import float_to_q15, q15_to_float, lms_filter_q15
from src.utils import compute_sqnr, count_macs_lms, count_macs_rls

fs, t = 250, np.arange(0, 2, 1 / 250)
d = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)   # signal + mains hum
ref = np.sin(2 * np.pi * 50 * t)                                    # reference for the hum

e_lms, w = lms_filter(d, ref, mu=0.01, N=8)
e_rls, w = rls_filter(d, ref, lam=0.99, N=8)
features = extract_wavelet_features(e_rls)                          # 6 log band energies
```

Cython ports (after `python setup.py build_ext --inplace`): `src.filters.lms_cython.lms_filter_cy` and
`src.filters.rls_cython.rls_filter_cy`, identical signatures and results.

## 6. Configuration reference

| Parameter | Meaning | Where | Value used in this project |
| --- | --- | --- | --- |
| `N` | Filter taps | LMS, RLS | 4 (synthetic tests), 8 (benchmarks) |
| `mu` (μ) | LMS step size | `lms_filter` | 0.01 – 0.05 |
| `lam` (λ) | RLS forgetting factor | `rls_filter` | **0.99** (0.90 diverges: covariance wind-up) |
| `delta` | RLS initial `P = I/delta` | `rls_filter` | 1e-2 |
| Q format | Fixed-point format | `src/fixedpoint` | Q15 (int16, 15 fractional bits) |
| Input range | Must be normalised **before** Q15 | `lms_filter_q15` | `d / (max|d| * 1.05)` |
| `wavelet`, `level` | DWT family / depth | `extract_wavelet_features` | `db4`, 5 |
| `n_components` | CSP spatial filters | `day1_csp.py` | 4 |
| `test_size`, `random_state` | Train/test split | `day1_csp.py` | 0.3, 42 |
| `F1, D, F2, kernel_length` | EEGNet-Lite width | `src/models` | 4, 2, 8, 32 |
| `bit_width` | QAT precision | `QuantEEGNetLite` | 16 (Q15-equivalent) |
| `lr`, `epochs`, `batch_size` | Training | `scripts/week7_nn/_data.py` | 1e-3, 100, 16 |

## 7. Troubleshooting

- **SQNR is low (< 10 dB).** The input was not normalised into [−1, 1) before Q15; saturation shows
  up as repeated ±32767 in `e_q15`. See `docs/report/week5_fixedpoint.md`.
- **RLS error grows instead of shrinking.** λ is too small for a single-tone reference
  (covariance wind-up). Use λ ≥ 0.95, or a richer reference signal.
- **LMS does not converge.** μ too large (oscillation) or too small (slow). Halve/double and compare.
- **`ImportError: src.filters.lms_cython`.** Extensions not built; run `python setup.py build_ext --inplace`.
  The scripts fall back to pure Python automatically.
- **`FileNotFoundError: results/csp_and_split.pkl`.** Run week-4 `day1_csp.py` first; the LDA, online
  inference and all week-7 scripts reuse that exact split.
- **MOABB download is slow or fails.** Retry; the dataset is cached under `~/mne_data/` afterwards.
