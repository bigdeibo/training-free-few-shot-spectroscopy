"""Spectral preprocessing.

Every spectrum in the main benchmark is SNV-corrected per spectrum, before any
train/test split, so that no cross-sample statistic ever enters the transform.
The Raman arms enter without SNV (Section 2.3 of the paper), and the in-house
pentane/CCl4 task enters as raw absorbance, because per-spectrum demeaning would
erase the additive offset that its Beer-Lambert structure rests on.
"""
import numpy as np


def snv(X):
    """Standard normal variate transform, applied to each spectrum independently.

    Evaluated in single precision, with the guard added to the standard
    deviation instead of special-cased after it, because this is the
    arithmetical path every result CSV here was produced by. The double
    precision form of the same transform is algebraically identical and shifts
    each spectrum by one unit in the last place; that is enough to move a
    two-support-set TabPFN fit, where the in-context regression is close to
    singular, by several thousandths of R-squared. Use this form to reproduce
    the shipped numbers.
    """
    X = np.asarray(X, dtype=np.float32)
    return (X - X.mean(axis=-1, keepdims=True)) / \
        (X.std(axis=-1, keepdims=True) + 1e-8)
