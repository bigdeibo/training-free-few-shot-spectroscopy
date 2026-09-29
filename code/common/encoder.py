"""The one-dimensional ResNet encoder.

A strided convolution stem followed by three residual stages, global average
pooling and a linear projection to a 128-dimensional embedding, about 0.33M
parameters in total. Spectra enter as `(batch, 1, 512)`.

The encoder is used here in two frozen forms: with weights from the masked
autoencoder, and with random initialisation as a control. `rand_init_encoder`
fixes the seed so that the control is reproducible.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RAND_INIT_SEED = 20260915


class ResBlock1D(nn.Module):
    """Two 9-tap convolutions with a residual connection and optional downsampling."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 9, stride=stride, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 9, padding=4, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm1d(cout))

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + idt)


class ResNet1Encoder(nn.Module):
    """Input `(B, 1, L)`, output `(B, emb_dim)`."""

    def __init__(self, in_len, emb_dim=128, width=32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv1d(1, w, 25, stride=2, padding=12, bias=False),
            nn.BatchNorm1d(w), nn.ReLU())
        self.stage1 = ResBlock1D(w, w)
        self.stage2 = ResBlock1D(w, 2 * w, stride=2)
        self.stage3 = ResBlock1D(2 * w, 4 * w, stride=2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.proj = nn.Linear(4 * w, emb_dim)
        self.out_dim = emb_dim

    def forward(self, x):
        h = self.stem(x)
        h = self.stage3(self.stage2(self.stage1(h)))
        return self.proj(self.head(h))


def count_params(net):
    """Number of trainable parameters."""
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))
