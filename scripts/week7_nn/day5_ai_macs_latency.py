"""Week 7, day 5 -- MACs per inference (via forward hooks) and latency of the QAT model."""

import _bootstrap  # noqa: F401
import torch

from src.models.qeegnet_lite import QuantEEGNetLite
from src.utils.eval_utils import measure_latency

N_TIMES, FS = 1001, 250.0

ckpt = torch.load(_bootstrap.RESULTS_DIR / "eegnet_qat.pt", weights_only=False)
model = QuantEEGNetLite(n_channels=22, n_times=N_TIMES, n_classes=2)
model.load_state_dict(ckpt["model_state"])
model.eval()

macs_per_layer = {}


def make_hook(name):
    def hook(mod, inp, out):
        if type(mod).__name__ == "QuantConv2d":
            k_h, k_w = mod.kernel_size
            macs = out.shape[1] * out.shape[2] * out.shape[3] * (mod.in_channels // mod.groups) * k_h * k_w
        else:  # QuantLinear
            macs = mod.in_features * mod.out_features
        macs_per_layer[name] = macs
    return hook


hooks = [m.register_forward_hook(make_hook(n)) for n, m in model.named_modules()
         if type(m).__name__ in ("QuantConv2d", "QuantLinear")]
with torch.no_grad():
    model(torch.randn(1, 22, N_TIMES))
for h in hooks:
    h.remove()

print("MACs per layer (one inference on a 1001-sample window):")
for name, macs in macs_per_layer.items():
    print(f"  {name:15s}: {macs:>10,}")
total_macs = sum(macs_per_layer.values())
macs_per_sec = total_macs / (N_TIMES / FS)
print(f"\nTotal MACs/inference: {total_macs:,}")
print(f"MACs/second (continuous): {macs_per_sec:,.0f}")


def infer_one(window):
    with torch.no_grad():
        return model(window.unsqueeze(0)).argmax(1).item()


lat = measure_latency(infer_one, torch.randn(22, N_TIMES), n_repeats=50)
print(f"Inference latency: {lat['mean_ms']:.3f} ms (+/- {lat['std_ms']:.3f})")

print("\n" + "=" * 70)
print(f"{'Algorithm':20s} | {'MACs/s':>14s} | {'Latency (ms)':>13s}")
print("-" * 70)
print(f"{'LMS (N=8)':20s} | {'4,000':>14s} | {'-':>13s}")
print(f"{'RLS (N=8)':20s} | {'40,000':>14s} | {'6.981':>13s}")
print(f"{'EEGNetLite (QAT)':20s} | {macs_per_sec:>14,.0f} | {lat['mean_ms']:>13.3f}")
