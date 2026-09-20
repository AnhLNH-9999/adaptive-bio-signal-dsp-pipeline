"""EEGNet-Lite: a compact CNN for EEG classification, after Lawhern et al. (2018).

Deliberately small (a few hundred parameters) to suit ~200 training trials
and the resource-constrained target hardware of this project.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNetLite(nn.Module):
    """Two-block EEGNet variant.

    Block 1 applies a temporal convolution (an FIR-like filter bank) followed
    by a depthwise spatial convolution across all channels. The latter plays the
    same role as CSP in the classical branch, but is learned from data.
    Block 2 is a separable convolution that further reduces dimensionality.
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_times: int = 1001,
        n_classes: int = 2,
        F1: int = 4,
        D: int = 2,
        F2: int = 8,
        kernel_length: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # infer the flattened size instead of hard-coding it
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flatten_size = self.block2(self.block1(dummy)).numel()

        self.classifier = nn.Linear(flatten_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, n_times) -> add a singleton "image channel" dim
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x.flatten(1))


if __name__ == "__main__":
    model = EEGNetLite()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"EEGNetLite parameters: {n_params:,}")
    out = model(torch.randn(8, 22, 1001))
    print(f"Output shape: {tuple(out.shape)}  (expected (8, 2))")
