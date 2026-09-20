# User Guide

How to set up, run, and configure the pipeline. For background, objectives, and evaluation
criteria, see the project proposal; for the day-by-day work plan, see the roadmap doc.

## 1. Prerequisites

- Python 3.10–3.11 (NumPy ≥ 1.24, Cython ≥ 3.0 — pinned in `requirements.txt` / `environment.yml`)
- ~2 GB free disk space (for the BCI Competition IV-2a dataset, once downloaded)
- A C compiler (for Cython) — on Linux/macOS this is usually already present; on Windows, install
  the "Microsoft C++ Build Tools" if `pip install cython` complains at build time.
- Optional: a CUDA-capable GPU speeds up Week 7 (neural-network training) but is not required —
  everything also runs on CPU.

## 2. Installation

**Option A — pip:**

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Option B — conda:**

```bash
conda env create -f environment.yml
conda activate eeg-dsp-pipeline
```

Verify the install:

```bash
python -c "import numpy, scipy, pywt, mne, sklearn, moabb, torch; print('OK')"
```

## 3. Quick demo (no dataset needed)

```bash
jupyter notebook notebooks/demo.ipynb
```

(From the repository root.) Runs the whole pipeline concept (LMS, RLS, Wavelet features, Q15 + SQNR, MACs/second) on a
synthetic signal in a couple of minutes. Start here before touching the real dataset.

## 4. Running the scripts in this repository

All commands below are run **from the repository root** — the scripts do
`sys.path.insert(0, 'src/filters')`, so the working directory matters.

### 4.1 Build the Cython filters (Weeks 1–2)

`benchmarks/benchmark_lms.py` and `benchmarks/benchmark_all.py` import both the pure-Python filters
(`src/filters/lms.py`, `src/filters/rls.py`) and their Cython twins (`lms_cython`, `rls_cython`).
Compile the `.pyx` files in place first:

```bash
cd src/filters
python setup.py build_ext --inplace     # produces lms_cython.*.so / .pyd next to the .pyx
cd ../..
```

The generated `*.c`, `*.so`, `*.pyd` and `build/` are gitignored.

### 4.2 Benchmarks: pure Python vs. Cython

```bash
python benchmarks/benchmark_lms.py    # LMS only, also checks np.allclose(e_python, e_cython)
python benchmarks/benchmark_all.py    # LMS + RLS timing table
```

Both scripts use a fixed seed (`np.random.seed(0)`), 5000 samples, `N = 8` taps, `mu = 0.01`,
`lambda = 0.99`. Filter signatures:

| Function | Signature | Returns |
| --- | --- | --- |
| `lms_filter` / `lms_filter_cy` | `(d, x_ref, mu, N)` | `(e, w)` — error signal, final weights |
| `rls_filter` / `rls_filter_cy` | `(d, x_ref, lam, N)` | `(e, w)` |

### 4.3 Filter tests (Weeks 1–2)

These are plain scripts (not pytest); each saves a plot under `results/`:

| Script | What it does | Output |
| --- | --- | --- |
| `tests/test_lms_synthetic.py` | LMS on a 10 Hz sine + white noise, `mu=0.05`, `N=4`; prints RMS error of first vs. last 10 % | `results/lms_synthetic_error.png` |
| `tests/test_rls_synthetic.py` | Same signal, RLS with `lambda ∈ {0.90, 0.95, 0.99, 0.995, 1.0}`, `N=4` (0.90 diverges on purpose) | `results/rls_lambda_sweep.png` |
| `tests/test_lms_real_eeg.py` | Real EEG (BNCI2014_001, subject 1, channel C3, 10 s) with an injected 50 Hz tone; LMS `mu=0.01`, `N=4`; spectrum before/after | `results/lms_real_eeg_spectrum.png` |
| `tests/test_rls_vs_lms_real_eeg.py` | Same contaminated EEG; LMS (`mu=0.01`) vs. RLS (`lambda=0.99`) smoothed squared-error curves | `results/rls_vs_lms_real_eeg.png` |

```bash
python tests/test_lms_synthetic.py
python tests/test_rls_synthetic.py
python tests/test_lms_real_eeg.py
python tests/test_rls_vs_lms_real_eeg.py
```

The two `*_real_eeg.py` scripts download the dataset through `moabb` on first run. The dataset is
already notch-filtered at 50 Hz, so the scripts inject a known 50 Hz tone (amplitude 2× signal std)
to test the filter in a controlled way.

`src/filters/rls.py` implements `rls_filter(d, x_ref, lam, N, delta=1e-2)` with the same
`(e, w)` interface as `lms_filter`; `delta` sets the initial inverse-correlation matrix
`P = I / delta`, and `P` is re-symmetrised each step for numerical stability.

### 4.4 Week 4 — CSP on the real dataset

```bash
python scripts/week4/day1_csp.py
```

- Loads BNCI2014_001 (BCI Competition IV-2a), subject 1, all channels, via `moabb` — the first run
  downloads the data (~100 MB) into moabb's cache (`~/mne_data` by default).
- Keeps the `left_hand` / `right_hand` classes, splits 70/30 (stratified, `random_state=42`)
  **before** fitting CSP to avoid data leakage, fits `mne.decoding.CSP(n_components=4, log=True)`.
- Writes `csp_and_split.pkl` (CSP object + train/test arrays) in the current directory for the
  Day-2 classifier script to reuse. `*.pkl` is gitignored.

### 4.5 Week 6 — MACs formula

```bash
python scripts/week6/day1_macs_formula.py
```

Prints MACs/sample (`LMS = 2N`, `RLS = 4N²`, N = 8) and MACs/second at `fs = 250 Hz` using
`count_macs_lms` / `count_macs_rls` from `src/filters/eval_utils.py` (Week 0 helper). Run it from
the repository root with `src/filters` importable, e.g.
`PYTHONPATH=src/filters python scripts/week6/day1_macs_formula.py`.

### 4.6 Full pipeline entry points (planned)

```bash
# 1. Download the BCI Competition IV-2a dataset into ./data/
python -m src.data.download_dataset

# 2. Run the classical branch: filter -> Wavelet+CSP -> LDA
python -m src.pipeline.run_classical --config configs/classical_default.yaml

# 3. Convert the LMS filter to fixed-point Q15 and measure SQNR
python -m src.filters.fixed_point --input results/lms_output.npy

# 4. Estimate MACs/second for RLS and LMS
python -m src.filters.macs --taps 32 --fs 250

# 5. Train and quantize the neural-network branch
python -m src.models.train_eegnet --config configs/eegnet_default.yaml
python -m src.models.quantize_qat --checkpoint results/eegnet_float32.pt
```

Each command writes its output (arrays, metrics, plots) under `results/`.

> The module paths above (`src.data`, `src.pipeline`, ...) describe the intended structure from
> the roadmap's Week 0 setup task. Adjust them to match whatever module layout you actually create
> — the important part is having one clearly named entry point per stage.

## 5. Configuration reference

| Parameter | Meaning | Used in | Typical starting value |
| --- | --- | --- | --- |
| `N` (filter order / taps) | Length of the adaptive filter | LMS, RLS | 8 (benchmarks) – 32 |
| `mu` (μ, step size) | LMS learning rate | LMS | 0.01 (benchmarks); 0.001–0.05 |
| `lambda` (λ, forgetting factor) | How quickly RLS "forgets" old samples | RLS | 0.99 (benchmarks); 0.95–0.999 |
| `delta` | Initial value for RLS's inverse covariance matrix (`P = I/delta`) | RLS (`rls.py`) | 1e-2 (default in `rls_filter`) |
| Q-format | Fixed-point format for quantization | `fixed_point.py` | Q15 / Q1.15 (1 sign bit, 15 fractional bits) |
| Normalization range | Range signals are scaled to before quantizing | `fixed_point.py` | `[-1, 1)` — required to avoid overflow |
| `wavelet` | Wavelet family | Wavelet feature extraction | `db4` |
| `level` | Decomposition depth | Wavelet feature extraction | 4 |
| `n_components` | Number of CSP spatial filters to keep | CSP (`day1_csp.py`) | 4 |
| `learning_rate`, `epochs` | Neural-network training | `train_eegnet.py` | 1e-3, 50–100 |
| quantization bits | Target precision for QAT | `quantize_qat.py` | 16 (Q15-equivalent); try 8 if hardware needs it smaller |

## 6. Where outputs go

```
results/
├── lms_output.npy              # filtered signal (float32)
├── lms_q15_output.npy          # filtered signal (Q15-quantized)
├── sqnr_report.json            # SQNR numbers per stage
├── macs_comparison.csv         # RLS vs. LMS vs. NN, MACs/second
├── classical_accuracy.json     # LDA baseline accuracy
├── eegnet_float32.pt           # trained float32 model checkpoint
├── eegnet_qat.pt               # quantized model checkpoint
└── final_comparison_table.csv  # everything combined (Week 8)
```

## 7. Troubleshooting

- **SQNR comes out very low or negative.** Almost always a normalization problem — make sure the
  signal is scaled into `[-1, 1)` *before* calling `float_to_q15`, not after.
- **RLS output looks unstable / blows up.** Lower `lambda` slightly (e.g. 0.999 → 0.98), or
  increase `delta` (a larger initial covariance is more forgiving early on).
- **LMS doesn't converge / error doesn't shrink.** `mu` is usually too large (causes oscillation)
  or too small (converges but very slowly over the recording length). Halve or double it and
  compare.
- **`torch` can't find a GPU.** That's fine — pass `--device cpu` (or leave the device flag unset if
  your script auto-detects); Week 7's dataset is small enough to train on CPU within the roadmap's
  time budget.
- **`pip install cython` fails to build.** You're missing a C compiler; install "build-essential"
  (Linux), Xcode command-line tools (macOS), or the Microsoft C++ Build Tools (Windows), then retry.
- **`ModuleNotFoundError: lms_cython` / `rls_cython`.** The Cython extensions aren't built yet —
  see §4.1 — or you're not running from the repository root.
- **`ModuleNotFoundError: eval_utils`.** Run with `PYTHONPATH=src/filters` (see §4.5).
- **`day1_csp.py` is slow the first time.** It's downloading the dataset; later runs use the cache.
