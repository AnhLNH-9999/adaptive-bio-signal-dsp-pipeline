# Adaptive Bio-Signal DSP Pipeline with Hardware-Ready Fixed-Point Design

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-green)](CITATION.cff)

Real-time EEG signal-processing pipeline: **RLS/LMS adaptive filtering → Wavelet + CSP feature
extraction → LDA**, a **Q15 fixed-point** port with SQNR and MACs/second analysis for FPGA/MCU
feasibility, and a lightweight **EEGNet-Lite** neural-network branch with quantisation-aware training.

Dataset: BCI Competition IV-2a (`BNCI2014_001`, motor imagery, 22 channels, 250 Hz) via MOABB.

## Table of contents

- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Pipeline overview](#pipeline-overview)
- [Key results](#key-results)
- [Running each stage](#running-each-stage)
- [Tests](#tests)
- [Documentation](#documentation)
- [License](#license)
- [Citation](#citation)

## Quick start

```bash
git clone https://github.com/AnhLNH-9999/adaptive-bio-signal-dsp-pipeline.git
cd adaptive-bio-signal-dsp-pipeline

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# optional: build the Cython filters (~100x faster than pure Python)
python setup.py build_ext --inplace

python -m pytest                     # 13 unit tests, < 1 s
jupyter notebook notebooks/demo.ipynb
```

The demo notebook runs on a **synthetic signal** (no dataset download) and walks through every
stage: LMS, RLS, wavelet features, Q15 + SQNR, MACs/second. For the real dataset see the
[User Guide](docs/USER_GUIDE.md).

## Project structure

```
.
├── src/
│   ├── filters/          # lms.py, rls.py (NumPy) + lms_cython.pyx, rls_cython.pyx
│   ├── features/         # wavelet.py  (db4, level 5 -> 6 log band energies)
│   ├── fixedpoint/       # q15.py, lms_q15.py (integer-only LMS simulation)
│   ├── models/           # eegnet_lite.py, qeegnet_lite.py (Brevitas QAT)
│   └── utils/            # eval_utils.py (latency, SQNR, MACs)
├── scripts/              # one folder per roadmap week, runnable top to bottom
│   ├── week1_lms/ … week7_nn/
├── tests/                # pytest unit tests (LMS, RLS, Cython parity, wavelet, Q15)
├── notebooks/demo.ipynb  # self-contained demo
├── docs/
│   ├── USER_GUIDE.md     # setup, usage, configuration reference
│   └── report/           # weekly write-ups with the measured numbers
├── results/              # generated artefacts (git-ignored)
├── setup.py, pyproject.toml, requirements.txt, environment.yml
└── LICENSE, CITATION.cff
```

## Pipeline overview

| Stage | Module | Script(s) | Week |
| --- | --- | --- | --- |
| Dataset exploration | — | `scripts/week1_lms/explore_dataset.py` | 0–1 |
| LMS / RLS adaptive filtering | `src/filters/` | `scripts/week2_rls/` | 1–2 |
| Wavelet band-power features (C3) | `src/features/wavelet.py` | `scripts/week3_wavelet/` | 3 |
| CSP + LDA baseline, online inference | — | `scripts/week4_csp_lda/` | 4 |
| Q15 fixed-point LMS + SQNR | `src/fixedpoint/` | `scripts/week5_fixedpoint/` | 5 |
| MACs/second analysis | `src/utils/eval_utils.py` | `scripts/week6_macs/` | 6 |
| EEGNet-Lite float32 + QAT 16-bit | `src/models/` | `scripts/week7_nn/` | 7 |

## Key results

| Metric | LMS | RLS | EEGNet-Lite (QAT) |
| --- | --- | --- | --- |
| Runtime, 5000 samples, N = 8 (pure Python) | 12.23 ms | 55.41 ms | — |
| Runtime, same (Cython) | 0.11 ms | 0.79 ms | — |
| MACs/second, N = 8, fs = 250 Hz (theory / counted) | 4 000 / 4 000 | 64 000 / 40 000 | ≈ 760 000 |
| SQNR after Q15 quantisation | 50.3 dB (≥ 30 dB target) | — | > 30 dB (weights) |
| Test accuracy (subject 1, left vs right hand) | LDA baseline 87.4 % | | float32 88.5 % · QAT 96.6 % |

Take-aways: RLS converges in ~250 samples vs. many more for LMS but costs O(N²) per sample; λ < 0.95
triggers covariance wind-up on a single-tone reference. Normalising the input into [−1, 1) before
Q15 raises SQNR from 2.4 dB to 50.3 dB. Theoretical MACs over-state the RLS/LMS gap (16× → 10×
counted → 4.5–7× measured). Full discussion in [`docs/report/`](docs/report/).

## Running each stage

```bash
# Week 2 - filters
python scripts/week2_rls/lambda_sweep.py          # effect of forgetting factor
python scripts/week2_rls/benchmark_all.py         # Python vs Cython timing

# Week 3-4 - real dataset (downloads ~1 GB on first run)
python scripts/week3_wavelet/day4_boxplot.py
python scripts/week4_csp_lda/day1_csp.py
python scripts/week4_csp_lda/day2_lda.py
python scripts/week4_csp_lda/day3_online_inference.py

# Week 5 - fixed point
python scripts/week5_fixedpoint/day3_run_lms_q15.py
python scripts/week5_fixedpoint/day4_sqnr.py

# Week 6 - MACs
python scripts/week6_macs/day1_macs_formula.py    # ... day2 .. day5

# Week 7 - neural network (needs week-4 split in results/)
python scripts/week7_nn/day2_train_eegnet.py
python scripts/week7_nn/day3_train_qat.py
python scripts/week7_nn/day4_eval_qat.py
python scripts/week7_nn/day5_ai_macs_latency.py
```

Every script writes its outputs to `results/`.

## Tests

```bash
python -m pytest -v
```

Covers convergence and NaN-safety of LMS/RLS, bit-exact parity of the Cython ports, fixed-length
wavelet features, Q15 boundary handling, and the ≥ 30 dB SQNR requirement.

## Documentation

- [User Guide](docs/USER_GUIDE.md) — installation, configuration parameters, troubleshooting.
- [Weekly reports](docs/report/) — RLS vs LMS, Q15 fixed-point, MACs/second, NN branch.
- [Demo notebook](notebooks/demo.ipynb).

## License

MIT — see [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff), or:

```bibtex
@software{le2026adaptivebiosignal,
  author  = {Lê, Nguyễn Hoài Anh},
  title   = {Adaptive Bio-Signal DSP Pipeline with Hardware-Ready Fixed-Point Design},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/AnhLNH-9999/adaptive-bio-signal-dsp-pipeline}
}
```
