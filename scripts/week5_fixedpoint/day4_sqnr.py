"""Week 5 -- SQNR between the float LMS output and the Q15 LMS output."""

import _bootstrap  # noqa: F401
import numpy as np

from src.utils.eval_utils import compute_sqnr

data = np.load(_bootstrap.RESULTS_DIR / "q15_lms_output.npz")
sqnr = compute_sqnr(data["e_float"], data["e_q15_as_float"])
print(f"SQNR (LMS Q15 vs float): {sqnr:.2f} dB")
print(f"Target (proposal): >= 30 dB  ->  {'PASS' if sqnr >= 30 else 'FAIL'}")
