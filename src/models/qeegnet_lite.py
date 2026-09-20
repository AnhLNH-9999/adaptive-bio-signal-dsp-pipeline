"""Quantisation-aware (QAT) variant of :class:`EEGNetLite` built with Brevitas.

Architecture is identical; standard layers are swapped for their ``Quant*``
counterparts with ``bit_width=16``, i.e. Q15-equivalent precision to match the
fixed-point work on the classical branch. ELU is replaced by ReLU, which
quantises cleanly.
"""

from __future__ import annotations

import brevitas.nn as qnn
import torch
import torch.nn as nn

BIT_WIDTH = 16


class QuantEEGNetLite(nn.Module):
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
        bit_width: int = BIT_WIDTH,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            qnn.QuantConv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2),
                            bias=False, weight_bit_width=bit_width),
            nn.BatchNorm2d(F1),
            qnn.QuantConv2d(F1, F1 * D, (n_channels, 1), groups=F1,
                            bias=False, weight_bit_width=bit_width),
            nn.BatchNorm2d(F1 * D),
            qnn.QuantReLU(bit_width=bit_width),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            qnn.QuantConv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D,
                            bias=False, weight_bit_width=bit_width),
            qnn.QuantConv2d(F1 * D, F2, (1, 1), bias=False, weight_bit_width=bit_width),
            nn.BatchNorm2d(F2),
            qnn.QuantReLU(bit_width=bit_width),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flatten_size = self.block2(self.block1(dummy)).numel()

        self.classifier = qnn.QuantLinear(flatten_size, n_classes, bias=True, weight_bit_width=bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x.flatten(1))
