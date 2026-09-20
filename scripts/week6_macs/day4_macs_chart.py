"""Week 6, day 4 -- bar chart of MACs/second, LMS vs RLS."""

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt

labels = ["LMS\n(theory)", "LMS\n(actual)", "RLS\n(theory)", "RLS\n(actual)"]
values = [4000, 4000, 64000, 40000]
colors = ["#4C72B0", "#4C72B0", "#C44E52", "#C44E52"]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values, color=colors)
for bar, v in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, v + 1000, f"{v:,.0f}", ha="center", fontsize=9)
plt.ylabel("MACs/second")
plt.title("MACs/second: LMS vs RLS (N=8, fs=250 Hz)")
plt.tight_layout()
out = _bootstrap.RESULTS_DIR / "macs_comparison.png"
plt.savefig(out, dpi=100)
print("Saved", out)
