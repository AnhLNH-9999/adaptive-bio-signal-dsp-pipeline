# User Guide

How to set up, run, and configure the pipeline. For background, objectives, and evaluation
criteria, see the project proposal; for the day-by-day work plan, see the roadmap doc.

## 1. Prerequisites

- Python 3.10–3.11
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
python -c "import numpy, scipy, pywt, mne, torch; print('OK')"
```

## 3. Quick demo (no dataset needed)

```bash
jupyter notebook notebooks/demo.ipynb
```

Runs the whole pipeline concept (LMS, RLS, Wavelet features, Q15 + SQNR, MACs/second) on a
synthetic signal in a couple of minutes. Start here before touching the real dataset.

## 4. Running on the real dataset

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
| `N` (filter order / taps) | Length of the adaptive filter | LMS, RLS | 16–32 |
| `mu` (μ, step size) | LMS learning rate | LMS | 0.001–0.05 (start small, increase if convergence is too slow) |
| `lambda` (λ, forgetting factor) | How quickly RLS "forgets" old samples | RLS | 0.95–0.999 |
| `delta` | Initial value for RLS's inverse covariance matrix | RLS | 1.0 (rarely needs changing) |
| Q-format | Fixed-point format for quantization | `fixed_point.py` | Q15 / Q1.15 (1 sign bit, 15 fractional bits) |
| Normalization range | Range signals are scaled to before quantizing | `fixed_point.py` | `[-1, 1)` — required to avoid overflow |
| `wavelet` | Wavelet family | Wavelet feature extraction | `db4` |
| `level` | Decomposition depth | Wavelet feature extraction | 4 |
| `n_components` | Number of CSP spatial filters to keep | CSP | 4–6 |
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
