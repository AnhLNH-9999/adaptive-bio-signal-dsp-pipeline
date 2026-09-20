"""Adaptive Bio-Signal DSP Pipeline.

Top-level package. Sub-packages:

- ``src.filters``     -- LMS / RLS adaptive filters (pure NumPy + Cython)
- ``src.features``    -- Wavelet band-power feature extraction
- ``src.fixedpoint``  -- Q15 fixed-point conversion and Q15 LMS simulation
- ``src.models``      -- EEGNet-Lite (float32) and its QAT variant
- ``src.utils``       -- latency / SQNR / MACs helpers
"""

__version__ = "0.1.0"
