"""Simulate a Raman corpus for the one dataset that starves the corpus rule.

Every representation finding in this work rests on a corpus PCA, and a corpus
PCA needs a corpus. The commercial-gasoline panel has 179 spectra, which is
below the few hundred the corpus-size ablation shows to be enough, and it is the
only task family here in that position. The repair is to simulate spectra from
physics rather than to fit the projection on a few hundred real ones: each
component of a small library contributes Gaussian peaks at the wavenumbers its
reference spectrum reports, a mixture is a non-negative combination of those
components, and a smooth baseline, a scale factor and a little noise finish it.

The concentration vectors are not drawn from a generic prior: every real
spectrum in the panel is fitted by non-negative least squares against the
component basis, and the fitted vectors are resampled with jitter, so the
simulated mixtures occupy the region of composition space the real ones do. No
real spectrum is copied into the corpus and no target value is used anywhere in
the simulation. The composition prior is fitted on the whole real panel, test
spectra included, and that is the one respect in which the real data informs the
corpus; it carries no label information, and it is reported rather than hidden.

The script also reports the fidelity of the result: the principal-angle cosines
between the subspace fitted on the 179 real spectra and the one fitted on real
plus simulated, and the reconstruction error of the simulated spectra under the
real projection.

Input.  `data/simulated-fuel/peaks.csv`, the component peak table, and the
        gasoline panel through `common.datasets.load_fuel_raman`.
Output. `data/simulated-fuel/sim_spectra.npy` and `sim_axis.npy`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.datasets import load_fuel_raman

PEAK_WIDTH = 7.0        # cm^-1, the Gaussian width of one library peak
N_SIMULATED = 400
SEED = 66007
FIDELITY_COMPONENTS = 20


def component_basis(wavenumbers, peaks):
    """One spectrum per library component, from its peak table."""
    components = {}
    for compound, group in peaks.groupby("compound"):
        s = np.zeros(len(wavenumbers))
        for _, row in group.iterrows():
            s += row.intensity * np.exp(
                -0.5 * ((wavenumbers - row.wavenumber) / PEAK_WIDTH) ** 2)
        components[compound] = s
    return components


def simulate(wavenumbers, components, n, rng, real):
    """Mixtures whose composition distribution is fitted to the real spectra.

    Every real spectrum is decomposed against the component basis by
    non-negative least squares plus a constant and a linear baseline term; the
    fitted concentration vectors are then resampled with jitter. Jitter is
    multiplicative on the components already present and small and additive on
    those that are not, so a simulated mixture stays near a real one without
    being a copy of it.
    """
    from scipy.optimize import nnls
    names = list(components)
    B = np.stack([components[k] for k in names])
    A = np.column_stack([B.T, np.ones(len(wavenumbers)),
                         (wavenumbers - wavenumbers.mean()) / 1000.0])
    fits = [nnls(A, x) for x in real]
    concentrations = np.array([f[0][: len(names)] for f in fits])
    residual_fraction = float(np.median([
        f[1] ** 2 / max((x ** 2).sum(), 1e-12) for f, x in zip(fits, real)]))

    out = []
    for _ in range(n):
        c = concentrations[rng.integers(len(concentrations))].copy()
        present = c > 0
        c[present] *= rng.uniform(0.7, 1.3, present.sum())
        c[~present] += np.abs(rng.normal(0, 0.01, (~present).sum())) * c.max()
        base = B.T @ c
        baseline = 0.0
        for _ in range(2):          # two broad bumps, as a fluorescence floor
            mu = rng.uniform(wavenumbers.min(), wavenumbers.max())
            amp = rng.uniform(0.02, 0.10) * base.max()
            sigma = rng.uniform(300, 900)
            baseline += amp * np.exp(
                -0.5 * ((wavenumbers - mu) / sigma) ** 2)
        scale = rng.uniform(0.6, 1.4)
        out.append(scale * (base + baseline)
                   + rng.normal(0, 0.01 * max(base.max(), 1e-9),
                                len(wavenumbers)))
    return np.array(out), names, residual_fraction


def principal_angles(Xa, Xb, k=FIDELITY_COMPONENTS):
    from sklearn.decomposition import PCA
    A = PCA(n_components=k, random_state=0).fit(Xa).components_
    B = PCA(n_components=k, random_state=0).fit(Xb).components_
    return np.linalg.svd(A @ B.T, compute_uv=False)


def main(force=False):
    from common.paths import DATA

    lib = DATA / "simulated-fuel"
    clobbered = [p for p in (lib / "sim_spectra.npy", lib / "sim_axis.npy")
                 if p.exists()]
    if clobbered and not force:
        raise SystemExit(
            "would overwrite the shipped arrays:\n  "
            + "\n  ".join(str(p) for p in clobbered)
            + "\nThe simulation is deterministic, so a rerun should reproduce "
              "them; pass --force if that is what you intend.")

    wavenumbers, real = (lambda d: (np.asarray(d["wavelengths"], dtype=float),
                                    np.asarray(d["X"], dtype=float)))(
        load_fuel_raman("Benchtop"))
    peaks = pd.read_csv(lib / "peaks.csv")

    rng = np.random.default_rng(SEED)
    components = component_basis(wavenumbers, peaks)
    simulated, names, residual_fraction = simulate(
        wavenumbers, components, N_SIMULATED, rng, real=real)
    np.save(lib / "sim_spectra.npy", simulated)
    np.save(lib / "sim_axis.npy", wavenumbers)
    print(f"component library: {len(names)} spectra over {len(wavenumbers)} "
          f"wavenumbers")
    print(f"median NNLS residual fraction against the library: "
          f"{residual_fraction:.3f}")

    from sklearn.decomposition import PCA
    angles = principal_angles(real, np.vstack([real, simulated]))
    projection = PCA(n_components=FIDELITY_COMPONENTS, random_state=0).fit(real)

    def reconstruction_error(X):
        return (((X - projection.inverse_transform(projection.transform(X))) ** 2)
                .sum(1) / (X ** 2).sum(1))

    err_sim = reconstruction_error(simulated)
    err_real = reconstruction_error(real)
    print(f"real n={len(real)}, simulated n={len(simulated)}, "
          f"axis {wavenumbers.min():.0f}-{wavenumbers.max():.0f} cm^-1")
    print(f"top-{FIDELITY_COMPONENTS} principal-angle cosines, real against "
          f"real+simulated: min={angles.min():.3f} mean={angles.mean():.3f}")
    print(f"reconstruction error under the real projection: "
          f"simulated median={np.median(err_sim):.3f} "
          f"p90={np.quantile(err_sim, 0.9):.3f}; "
          f"real median={np.median(err_real):.3f} "
          f"p90={np.quantile(err_real, 0.9):.3f}")


def parse_args(argv=None):
    """The input and output paths are fixed; `--force` is the one option.

    The parser is here so that `--help` prints the module docstring, which
    names the input and the output, instead of starting the simulation.
    """
    parser = argparse.ArgumentParser(
        description="Simulate the augmented fuel Raman corpus.",
        epilog="Inputs and outputs are named in the module docstring.")
    parser.add_argument("--force", action="store_true",
                        help="overwrite the shipped arrays in "
                             "data/simulated-fuel/; without it the script stops "
                             "rather than replace them")
    return parser.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    main(force=_args.force)
