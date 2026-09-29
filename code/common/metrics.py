"""Evaluation metrics.

R2 is the primary one: it is scale-free, so it pools tasks whose targets carry
different units and ranges. R2 below -1 is the paper's "catastrophic failure"
threshold. RMSE and RPD are reported alongside it in the original units.
"""
import numpy as np


def r2_score(pred, y):
    """Coefficient of determination. NaN when the reference has no variance."""
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan


def rmse(pred, y):
    """Root mean squared error, in the target's own units."""
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(y, float)) ** 2)))


def rpd(y, rmse_value):
    """Ratio of performance to deviation: reference standard deviation over RMSE."""
    s = np.std(y, ddof=1)
    return s / rmse_value if rmse_value > 1e-12 else np.inf
