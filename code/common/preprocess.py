"""Spectral preprocessing.

Every spectrum in the main benchmark is SNV-corrected per spectrum, before any
train/test split, so that no cross-sample statistic ever enters the transform.
The Raman arms enter without SNV (Section 2.3 of the paper), and the in-house
pentane/CCl4 task enters as raw absorbance, because per-spectrum demeaning would
erase the additive offset that its Beer-Lambert structure rests on.
"""
import numpy as np


def snv(X):
    """Standard normal variate transform, applied to each spectrum independently."""
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (X - m) / s
