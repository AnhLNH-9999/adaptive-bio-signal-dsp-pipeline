"""Float <-> Q15 (Q1.15) fixed-point conversion.

Q15 is a signed 16-bit format with 1 sign bit and 15 fractional bits, so it
represents values in ``[-1, 32767/32768]`` with a resolution of ``2**-15``.
"""

from __future__ import annotations

import numpy as np

Q15_SCALE = 32768                    # 2**15
Q15_MAX_FLOAT = 32767 / 32768        # largest representable value (0.999969...)
Q15_MIN_FLOAT = -1.0                 # smallest representable value


def float_to_q15(x) -> np.ndarray:
    """Convert a float scalar/array to int16 Q15, saturating outside ``[-1, 1)``.

    Clamping *before* scaling is essential: ``1.0 * 32768 = 32768`` overflows
    int16 (max 32767) and would wrap to a negative number.
    """
    x = np.asarray(x, dtype=np.float64)
    x_clamped = np.clip(x, Q15_MIN_FLOAT, Q15_MAX_FLOAT)
    return np.round(x_clamped * Q15_SCALE).astype(np.int16)


def q15_to_float(q) -> np.ndarray:
    """Convert int16 Q15 values back to float64."""
    return np.asarray(q).astype(np.float64) / Q15_SCALE


if __name__ == "__main__":
    test_vals = np.array([-1.0, 0.0, 0.99999, 1.5, -2.0])
    q = float_to_q15(test_vals)
    back = q15_to_float(q)
    for v, qi, b in zip(test_vals, q, back):
        print(f"{v:8.5f} -> q15={qi:6d} -> back={b:8.5f}")
