"""Simulated spectra: Beer-Lambert mixing of pure components plus a physical
perturbation chain.

This module produces the simulated half of the pretraining corpus. Its output is
already in the archive as `data/corpus/simulated_spectra.npy`; the code is
shipped so that the archive accounts for how those spectra were made, and so that
`code/corpus_augmentation/run_build_corpus.py` can assemble the full corpus.

Physical model
--------------
1. Beer-Lambert additive mixing: S_mix = sum_i c_i * P_i, c ~ Dirichlet
   (two to four components per spectrum)
2. Baseline drift: low-order polynomial (order 2-3, random coefficients)
3. Multiplicative scatter: a + b * u, where u is the normalized abscissa
   (particle-size and path-length differences)
4. Wavelength shift and stretch: affine perturbation of the abscissa
   x' = x + d + s*(x - x_mid) (instrument wavelength-calibration error)
5. Resolution broadening: optional Gaussian smoothing, random sigma
6. Additive noise: Gaussian, added to meet a target signal-to-noise ratio in dB

Every parameter is drawn from a seed-driven generator, so a run is reproducible.
The pure-component library is built by `build_pure_library` from the NIST
infrared reference collection and from class-mean spectra of the edible-oil and
mayonnaise datasets.
"""
from __future__ import annotations

import numpy as np


def _poly_baseline(rng, n, deg=3, scale=0.3):
    """Random low-order polynomial baseline on a [-1, 1] normalized abscissa."""
    u = np.linspace(-1, 1, n)
    coefs = rng.normal(0, scale, deg + 1)
    coefs[0] = rng.normal(0, scale * 0.5)  # keep the constant term smaller
    return np.polyval(coefs, u)


def perturb(rng, x, shift_pts=3.0, stretch=0.01, smooth_p=0.3,
            baseline_scale=0.15, scatter=(0.9, 1.1, 0.05), snr_db=(30, 60)):
    """Apply the physical perturbation chain to one spectrum.

    x: (L,) spectrum, already resampled to a common length.
    """
    L = len(x)
    y = x.astype(float).copy()
    # 4. Wavelength shift and stretch, by interpolating on a displaced abscissa
    d = rng.uniform(-shift_pts, shift_pts)
    s = rng.uniform(-stretch, stretch)
    idx = np.arange(L)
    new_idx = idx + d + s * (idx - L / 2)
    y = np.interp(idx, new_idx, y, left=y[0], right=y[-1])
    # 5. Resolution broadening
    if rng.random() < smooth_p:
        sigma = rng.uniform(0.5, 2.0)
        k = int(6 * sigma) | 1
        g = np.exp(-0.5 * ((np.arange(k) - k // 2) / sigma) ** 2)
        g /= g.sum()
        y = np.convolve(y, g, mode="same")
    # 3. Multiplicative scatter
    a = rng.uniform(*scatter[:2])
    b = rng.uniform(-scatter[2], scatter[2])
    u = np.linspace(-1, 1, L)
    y = y * (a + b * u)
    # 2. Baseline drift
    y = y + _poly_baseline(rng, L, deg=rng.integers(2, 4),
                           scale=baseline_scale) * (np.abs(y).max() + 1e-8)
    # 6. Additive noise
    snr = rng.uniform(*snr_db)
    sig = np.sqrt(np.mean(y ** 2)) / (10 ** (snr / 20))
    y = y + rng.normal(0, sig, L)
    return y


def build_pure_library(nist_npz=None, class_spectra=None):
    """Build the pure-component spectral library.

    Parameters
    ----------
    nist_npz : Path or None
        The NIST infrared collection, an npz with the object arrays `names`,
        `x` and `y`.
    class_spectra : dict[str, ndarray] or None
        {name: (n_i, L_i)} spectral groups; the mean is taken as that class's
        pure spectrum, as for the seven edible-oil classes.

    Returns dict{name: (L,) spectrum}. Members keep their own lengths and are
    resampled to a common grid at mixing time.
    """
    lib = {}
    if nist_npz is not None:
        z = np.load(nist_npz, allow_pickle=True)
        for name, x, y in zip(z["names"], z["x"], z["y"]):
            if len(y) >= 100 and np.all(np.isfinite(y)):
                lib[f"nist:{name}"] = np.asarray(y, dtype=float)
    if class_spectra:
        for name, X in class_spectra.items():
            if len(X) >= 3:
                lib[f"cls:{name}"] = np.asarray(X, dtype=float).mean(axis=0)
    return lib


def simulate_mixtures(lib, n_samples, L=512, seed=0, k_range=(2, 4),
                      do_perturb=True):
    """Generate simulated mixture spectra from a pure-spectrum library.

    Parameters
    ----------
    lib : dict{name: spectrum}, the output of `build_pure_library`
    n_samples : number of spectra to generate
    L : common output length; each pure spectrum is resampled to L before mixing
    seed : random seed
    k_range : inclusive range of the number of components per spectrum
    do_perturb : whether to apply the perturbation chain above. Clean mixtures
        are available by turning it off, which the simulated-mixture probes do.

    Returns (X_sim (n, L), comps (n, n_lib) mixing ratios, lib_names).
    """
    rng = np.random.default_rng(seed)
    names = sorted(lib)
    P = np.stack([np.interp(np.linspace(0, 1, L),
                            np.linspace(0, 1, len(lib[n])), lib[n])
                  for n in names])  # (n_lib, L)
    # Each component is scaled to [0, 1] so that amplitude differences between
    # sources do not dominate the mixing ratios.
    P = P - P.min(axis=1, keepdims=True)
    P = P / (P.max(axis=1, keepdims=True) + 1e-8)
    n_lib = len(names)
    X = np.zeros((n_samples, L))
    C = np.zeros((n_samples, n_lib))
    for i in range(n_samples):
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        sel = rng.choice(n_lib, size=min(k, n_lib), replace=False)
        w = rng.dirichlet(np.ones(len(sel)))
        C[i, sel] = w
        y = w @ P[sel]
        if do_perturb:
            y = perturb(rng, y)
        X[i] = y
    return X, C, names
