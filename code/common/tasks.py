"""The task registry and the paired evaluation split.

Fifteen quantitative tasks, each one a substance-and-property pair. Every task is
reduced to the same form: SNV-corrected spectra resampled to 512 points, and the
target on its original scale.

`eval_split` is the load-bearing function of the whole comparison. Every method
sees the identical support and test indices for a given (task, K, repetition),
because the split is drawn from a seed that depends on nothing but those three
numbers. Paired tests across methods are therefore valid by construction.
"""
from __future__ import annotations

import numpy as np

L = 512                 # points each spectrum is resampled to
EMB_DIM = 128           # width of the learned encoders' embedding
BASE_SEED = 42
RAMAN_SEED = 66         # the Raman arms draw their splits from a second family

# Task slug -> (dataset key in common.datasets.LOADERS, property name in that loader).
TASKS = {
    "diesel-bp50":               ("diesel", "BP50"),
    "diesel-cetane-number":      ("diesel", "CN"),
    "diesel-density":            ("diesel", "D4052"),
    "diesel-flash-point":        ("diesel", "FLASH"),
    "diesel-freezing-point":     ("diesel", "FREEZE"),
    "diesel-total-aromatics":    ("diesel", "TOTAL"),
    "diesel-viscosity":          ("diesel", "VISC"),
    "corn-moisture":             ("corn", "moisture"),
    "corn-oil":                  ("corn", "oil"),
    "corn-protein":              ("corn", "protein"),
    "corn-starch":               ("corn", "starch"),
    "gasoline-octane":           ("gasoline", "octane"),
    "olive-oil-adulteration":    ("evoo", "adulteration_level"),
    "edible-oil-peroxide-value": ("edible_oil", "peroxide_value"),
    "pentane-ccl4-volume-fraction": ("pentane_ccl4", "phi_pentane"),
}

# The thirteen tasks of the main benchmark. The last two entries of TASKS are not
# in it: the in-house pentane/CCl4 task appears only in the representation
# comparison, and peroxide value is the negative control.
BENCHMARK_TASKS = [
    "diesel-bp50", "diesel-cetane-number", "diesel-density", "diesel-flash-point",
    "diesel-freezing-point", "diesel-total-aromatics", "diesel-viscosity",
    "corn-moisture", "corn-oil", "corn-protein", "corn-starch",
    "gasoline-octane", "olive-oil-adulteration",
]

# Tasks whose spectra are not SNV-corrected (see common/preprocess.py).
RAW_SPECTRA_TASKS = {"pentane-ccl4-volume-fraction"}

_cache: dict = {}


def tasks_except(*excluded):
    """Every task except the named ones.

    Two tasks are dropped from experiments whose protocol they cannot support.
    The in-house pentane/CCl4 series has ten physical samples, so a group-aware
    split cannot supply ten distinct support groups; and peroxide value is the
    negative control, where the whole point is that a selection strategy has
    nothing to select on.
    """
    return [t for t in TASKS if t not in set(excluded)]


def load_task(task):
    """Return `dict(X=(n,512) float32, y=(n,) float64, groups=(n,) or None)`.

    `groups` is set for the three datasets whose spectra are replicates of a
    physical sample: the olive-oil NIR-HSI set (by sample), the edible-oil set
    (by oil) and the in-house pentane/CCl4 set (by bottle). Everywhere else it is
    None and the split is a plain random draw.
    """
    if task in _cache:
        return _cache[task]
    from common.datasets import LOADERS
    from common.preprocess import snv

    ds, attr = TASKS[task]
    d = LOADERS[ds]()
    groups = None
    if ds == "corn":
        X, y = d["instruments"]["m5"][0], d["targets"][attr]
    else:
        X, y = d["X"], d["targets"][attr]
    m = ~np.isnan(y)
    X, y = np.asarray(X, float)[m], np.asarray(y, float)[m]
    if ds in ("evoo", "pentane_ccl4", "edible_oil"):
        groups = np.asarray(d["groups"])[m]

    # Resample onto a common 512-point grid, within each spectrum's own range.
    go = np.linspace(0, 1, X.shape[1])
    gn = np.linspace(0, 1, L)
    X = np.stack([np.interp(gn, go, r) for r in X])
    if task in RAW_SPECTRA_TASKS:
        X = X.astype(np.float32)
    else:
        X = snv(X).astype(np.float32)

    _cache[task] = dict(X=X, y=y, groups=groups)
    return _cache[task]


def eval_split(n, K, rep, seed=BASE_SEED, groups=None):
    """The paired support/test split.

    With `groups=None` the K support indices are a uniform random draw and the
    test set is everything else.

    With `groups` given the split is group-aware: the K support spectra come from
    K distinct groups, one spectrum each, and the test set excludes every member
    of those K groups, replicates included. Without this, near-duplicate
    replicate spectra would straddle the boundary and the olive-oil and
    edible-oil numbers would be meaningless.

    Both branches draw from `seed*1000 + K*100 + rep`, so the two are paired
    across methods in exactly the same way.
    """
    rng = np.random.default_rng(seed * 1000 + K * 100 + rep)
    if groups is None:
        tr = rng.choice(n, K, replace=False)
        te = np.setdiff1d(np.arange(n), tr)
        return tr, te

    groups = np.asarray(groups)
    _, inv = np.unique(groups, return_inverse=True)
    n_groups = int(inv.max()) + 1
    sup_gids = rng.permutation(n_groups)[:K]

    tr = []
    for g in sup_gids:
        members = np.where(inv == g)[0]
        rng.shuffle(members)
        tr.append(int(members[0]))          # one spectrum per support group
    tr = np.array(tr, dtype=int)

    sup_set = set(int(g) for g in sup_gids)
    te = np.where(np.array([int(i) not in sup_set for i in inv]))[0]
    return tr, te


def group_split(groups, K, rep, seed=RAMAN_SEED):
    """Support and test indices when a support unit is a whole group.

    The Raman experiments use this rule rather than `eval_split`: K groups are
    drawn and *every* spectrum of those groups becomes support, with the
    remaining groups forming the test set. A sugar well carries about twenty
    measurements of the same mixture, so K counts groups here and not spectra,
    and the support set is correspondingly larger than K.

    Splits are drawn from a second seed family, disjoint from the one
    `eval_split` uses, so that no Raman deployment shares a draw with a
    near-infrared one.
    """
    rng = np.random.default_rng(seed * 1000 + K * 100 + rep)
    unique = np.array(sorted(set(np.asarray(groups).tolist())))
    chosen = set(rng.choice(unique, size=min(K, len(unique)),
                            replace=False).tolist())
    support = np.array([i for i, g in enumerate(groups) if g in chosen])
    test = np.array([i for i, g in enumerate(groups) if g not in chosen])
    return support, test


def zscore_fit(ys):
    """Support-set mean and standard deviation, the label scaling every method uses."""
    return float(np.mean(ys)), float(np.std(ys) + 1e-8)
