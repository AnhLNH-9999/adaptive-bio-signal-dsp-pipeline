"""Week 7, day 2 -- train EEGNetLite (float32) on the week-4 split."""

import _bootstrap  # noqa: F401
import torch
from _data import evaluate, load_split, train

from src.models.eegnet_lite import EEGNetLite

data = load_split()
torch.manual_seed(0)
model = train(EEGNetLite(n_channels=22, n_times=data["n_times"], n_classes=2), data)
acc = evaluate(model, data)
print(f"\nFinal float32 test accuracy: {acc:.4f} ({acc * 100:.1f}%)")

out = _bootstrap.RESULTS_DIR / "eegnet_float32.pt"
torch.save({"model_state": model.state_dict(), "mean": data["mean"], "std": data["std"],
            "classes": data["classes"]}, out)
print("Saved", out)
