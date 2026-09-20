# Adaptive Bio-Signal DSP Pipeline with Hardware-Ready Fixed-Point Design

Real-time EEG signal-processing pipeline — RLS/LMS adaptive filtering, Wavelet + CSP feature
extraction, and fixed-point Q15 conversion — with an added lightweight neural-network branch for
behavior prediction, designed from the start with FPGA/microcontroller feasibility in mind.

> Student project (Project 2). See the full proposal for background, objectives, scope, and
> evaluation criteria, and the roadmap doc for the day-by-day 8-week plan.

## Table of contents

- [Quick start](#quick-start)
  - [Run the benchmark / weekly scripts](#run-the-benchmark--weekly-scripts)
- [Project structure](#project-structure)
- [What this pipeline does](#what-this-pipeline-does)
- [Documentation](#documentation)
- [Roadmap / progress](#roadmap--progress)
- [License](#license)
- [Citation](#citation)

## Quick start

```bash
git clone https://github.com/AnhLNH-9999/adaptive-bio-signal-dsp-pipeline.git
cd adaptive-bio-signal-dsp-pipeline

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook notebooks/demo.ipynb
```

The demo notebook runs entirely on a **synthetic signal** (no dataset download needed) and shows
every pipeline stage — LMS/RLS filtering, Wavelet band-power features, Q15 conversion + SQNR,
MACs/second — end to end in a couple of minutes.

To run on the real dataset (BCI Competition IV-2a) instead, see
[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

### Run the benchmark / weekly scripts

All scripts are run **from the repository root** (they add `src/filters` to `sys.path`):

```bash
# Week 1-2: LMS / RLS speed, pure Python vs. Cython
python benchmarks/benchmark_lms.py
python benchmarks/benchmark_all.py

# Week 4: train CSP on BCI Competition IV-2a (downloads ~subject 1 via moabb on first run)
python scripts/week4/day1_csp.py

# Week 6: theoretical MACs/sample and MACs/second for N = 8 taps, fs = 250 Hz
python scripts/week6/day1_macs_formula.py
```

The Cython benchmarks expect the compiled extensions `lms_cython` and `rls_cython` in `src/filters/`
(build with `python setup.py build_ext --inplace`, see the User Guide).

## Project structure

```
.
├── benchmarks/
│   ├── benchmark_lms.py      # LMS: Python thuần vs. Cython (Week 1)
│   └── benchmark_all.py      # LMS + RLS: Python thuần vs. Cython (Week 2)
├── scripts/
│   ├── week4/day1_csp.py     # train CSP on BNCI2014_001 (subject 1), save csp_and_split.pkl
│   └── week6/day1_macs_formula.py  # theoretical MACs/sample for LMS (2N) vs. RLS (4N^2)
├── src/
│   ├── filters/          # lms.py, lms_cython.pyx, rls.py, rls_cython.pyx, fixed-point (Q15), eval_utils.py
│   ├── features/         # Wavelet Transform, CSP, Online Inference Engine
│   └── models/           # neural-network branch (EEGNet-style CNN, QAT)
├── notebooks/
│   └── demo.ipynb        # self-contained demo — fastest way to see it work
├── tests/                # unit tests
├── results/              # generated metrics, plots, comparison tables (gitignored)
├── data/                 # downloaded datasets (gitignored, not committed)
├── docs/
│   └── USER_GUIDE.md     # setup, usage, configuration reference
├── requirements.txt
├── environment.yml
├── LICENSE
└── CITATION.cff
```

## What this pipeline does

| Stage | Module | Roadmap week |
| --- | --- | --- |
| Adaptive filtering (RLS/LMS) | `src/filters/` | Week 1–2 |
| Wavelet + CSP feature extraction | `src/features/` | Week 3–4 |
| Fixed-point Q15 conversion + SQNR | `src/filters/fixed_point.py` | Week 5 |
| MACs/second analysis (RLS vs. LMS) | `src/filters/macs.py` | Week 6 |
| Neural-network branch (behavior prediction) | `src/models/` | Week 7 |
| Consolidated comparison + final report | `results/`, final report | Week 8 |

## Documentation

- **[User Guide](docs/USER_GUIDE.md)** — installation, how to run each stage, configuration
  parameters (filter order, step size, forgetting factor, Q-format, wavelet settings, model config).
- **[Demo notebook](notebooks/demo.ipynb)** — runnable, already-executed walkthrough with plots.
- Full proposal (background, objectives, scope, evaluation criteria) and the detailed weekly
  roadmap are kept alongside the course submission.

## Roadmap / progress

This project follows an 8-week plan (Weeks 0–8: setup, LMS, RLS, Wavelet, CSP, fixed-point, MACs
analysis, neural-network branch, final report). Each week ends with a checkpoint; commits are
tagged accordingly (`week-1`, `week-2`, …) so progress can be reviewed commit by commit.

## License

Licensed under the MIT License — see [`LICENSE`](LICENSE).

## Citation

If you use or build on this code, please cite it — see [`CITATION.cff`](CITATION.cff).
