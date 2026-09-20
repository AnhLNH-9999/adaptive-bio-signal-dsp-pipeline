"""Week 7, day 4 -- QAT accuracy + per-layer SQNR (full-precision vs quantised weights)."""

import _bootstrap  # noqa: F401
import numpy as np
import torch
from _data import evaluate, load_split

from src.models.qeegnet_lite import QuantEEGNetLite
from src.utils.eval_utils import compute_sqnr

ckpt = torch.load(_bootstrap.RESULTS_DIR / "eegnet_qat.pt", weights_only=False)
data = load_split()
model = QuantEEGNetLite(n_channels=22, n_times=data["n_times"], n_classes=2)
model.load_state_dict(ckpt["model_state"])
model.eval()

print(f"QAT accuracy: {evaluate(model, data):.4f}")

print("\nPer-layer weight SQNR (full precision vs 16-bit quantised):")
all_full, all_quant = [], []
for name, m in model.named_modules():
    if hasattr(m, "quant_weight"):
        w_full = m.weight.detach().flatten().numpy()
        w_quant = m.quant_weight().value.detach().flatten().numpy()
        all_full.append(w_full)
        all_quant.append(w_quant)
        print(f"  {name:15s}: {compute_sqnr(w_full, w_quant):6.2f} dB")

sqnr_total = compute_sqnr(np.concatenate(all_full), np.concatenate(all_quant))
print(f"\nOverall weight SQNR: {sqnr_total:.2f} dB  (target >= 30 dB -> {'PASS' if sqnr_total >= 30 else 'FAIL'})")
