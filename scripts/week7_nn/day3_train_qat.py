"""Week 7, day 3 -- quantisation-aware training of QuantEEGNetLite (16-bit ~ Q15)."""

import _bootstrap  # noqa: F401
import torch
from _data import evaluate, load_split, train

from src.models.qeegnet_lite import QuantEEGNetLite

data = load_split()
torch.manual_seed(0)
model = train(QuantEEGNetLite(n_channels=22, n_times=data["n_times"], n_classes=2), data)
acc = evaluate(model, data)
print(f"\nFinal QAT (16-bit) test accuracy: {acc:.4f} ({acc * 100:.1f}%)")

out = _bootstrap.RESULTS_DIR / "eegnet_qat.pt"
torch.save({"model_state": model.state_dict(), "mean": data["mean"], "std": data["std"],
            "classes": data["classes"]}, out)
print("Saved", out)
