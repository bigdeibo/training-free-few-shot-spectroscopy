"""Assemble the unlabeled corpus that every corpus projection is fitted on.

The paper's representations are principal scores of a projector fitted on 6339
unlabeled spectra, 3339 measured and 3000 simulated. The measured half is drawn
from six public datasets, which this archive does not redistribute, so only the
simulated half ships, as `data/corpus/simulated_spectra.npy`. This script rebuilds
the whole corpus from the simulated half and the public datasets, and the file it
writes is the one the paper's corpus PCA was fitted on.

Both halves go through the same preparation: each spectrum is linearly resampled
to 512 points within its own wavelength range, and nothing else. No scatter
correction is applied to this file, which holds the corpus as the encoders were
pretrained on it.

The simulated half is built by `common.simulate`: a 111-member pure-component
library contributes Beer-Lambert mixtures of two to four components, and a
perturbation chain adds baseline drift, multiplicative scatter, a wavelength
shift and stretch, resolution broadening and noise. The library is the NIST
infrared reference collection plus the class means of the seven edible-oil
classes and the six mayonnaise classes, so those three datasets reach the corpus
twice: once as themselves, and once through the mixture library.

Input.  The six public datasets under `$SPEC_DATA_ROOT`, in the layout
        `data/sources/README.md` gives, including `NIST_IR/nist_ir.npz` and
        `processed/mayonnaise_arrays.npz`.
Output. `data/corpus/spectra.npy`, 6339 x 512, in the row order the paper used:
        the six measured datasets in the order above, then the simulated block.
        With `--simulated-only`, `data/corpus/simulated_spectra.npy` instead,
        which is the file this archive ships and is byte-identical to it.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.datasets import (load_corn, load_diesel, load_evoo, load_gasoline,
                             load_mayonnaise)
from common.simulate import build_pure_library, simulate_mixtures

POINTS = 512
N_SIMULATED = 3000
LIBRARY_SIZE = 111
SEED = 42


def resample(X):
    """Linear interpolation of each row onto `POINTS` points over its own range."""
    n, length = X.shape
    old = np.linspace(0, 1, length)
    new = np.linspace(0, 1, POINTS)
    return np.stack([np.interp(new, old, row) for row in X]).astype(np.float32)


def nist_signature_spectra(path):
    """The NIST infrared collection as a rectangular array over `POINTS` points."""
    z = np.load(path, allow_pickle=True)
    return np.stack([np.interp(np.linspace(0, 1, POINTS),
                               np.linspace(0, 1, len(y)), y)
                     for y in z["y"]]).astype(np.float32)


def measured_blocks(data_root):
    """The six measured blocks, in the order the paper stacks them."""
    blocks = []
    d = load_diesel()
    blocks.append(("swri-diesel", resample(d["X"])))
    g = load_gasoline()
    blocks.append(("kalivas-gasoline", resample(g["X"])))
    c = load_corn()
    for instrument, (Xi, _wl) in c["instruments"].items():
        blocks.append((f"cargill-corn-{instrument}", resample(Xi)))
    e = load_evoo()
    blocks.append(("evoo-nir-hsi", resample(e["X"])))
    m = load_mayonnaise()
    blocks.append(("mayonnaise", resample(m["X"])))
    blocks.append(("nist-ir", nist_signature_spectra(
        Path(data_root) / "NIST_IR" / "nist_ir.npz")))
    return blocks, e, m


def simulated_block(data_root, evoo, mayonnaise):
    """The 3000 simulated spectra and the names of the library that made them."""
    class_spectra = {}
    for cls in sorted(set(evoo["labels"])):
        class_spectra[f"evoo_{cls}"] = evoo["X"][evoo["labels"] == cls]
    for cls in sorted(set(mayonnaise["labels"].astype(str))):
        class_spectra[f"mayo_{cls}"] = mayonnaise["X"][
            mayonnaise["labels"].astype(str) == cls]

    lib = build_pure_library(
        nist_npz=Path(data_root) / "NIST_IR" / "nist_ir.npz",
        class_spectra=class_spectra)
    names = sorted(lib)
    if len(names) != LIBRARY_SIZE:
        raise RuntimeError(
            f"the library has {len(names)} members where the paper's has "
            f"{LIBRARY_SIZE}; check that every public dataset is present and "
            f"complete under the data root")
    X, _compositions, names = simulate_mixtures(
        lib, N_SIMULATED, L=POINTS, seed=SEED, do_perturb=True)
    return X.astype(np.float32), names


def digest(X):
    return hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()[:16]


def main():
    from common.paths import DATA, DATA_ROOT

    args = parse_args()
    if args.simulated_only:
        out = DATA / "corpus" / "simulated_spectra.npy"
        if out.exists() and not args.force:
            raise SystemExit(
                f"{out} already exists. This flag exists to rebuild the array "
                f"the archive ships, and a rerun replaces it. Check the digest "
                f"against data/corpus/recipe.json first, then pass --force if "
                f"you mean to overwrite it.")
        _blocks, evoo, mayonnaise = measured_blocks(DATA_ROOT)
        simulated, names = simulated_block(DATA_ROOT, evoo, mayonnaise)
        np.save(out, simulated)
        print(f"wrote {out.relative_to(DATA.parent)}: {simulated.shape}, "
              f"library {len(names)} members")
        print(f"array digest {digest(simulated)}")
        return

    blocks, evoo, mayonnaise = measured_blocks(DATA_ROOT)
    for name, block in blocks:
        print(f"  measured  {name:<24} {block.shape[0]:>5} spectra")
    simulated, names = simulated_block(DATA_ROOT, evoo, mayonnaise)
    print(f"  simulated {'111-component library':<24} {len(simulated):>5} spectra")

    X = np.vstack([block for _name, block in blocks] + [simulated])
    out = DATA / "corpus" / "spectra.npy"
    np.save(out, X)
    print(f"wrote {out.relative_to(DATA.parent)}: {X.shape}")
    print(f"array digest {digest(X)}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Assemble the unlabeled corpus.",
        epilog="Inputs and outputs are named in the module docstring.")
    parser.add_argument(
        "--simulated-only", action="store_true",
        help="write only the 3000 simulated spectra, the half this archive "
             "ships, and read the public datasets only for their class means")
    parser.add_argument(
        "--force", action="store_true",
        help="with --simulated-only, overwrite the shipped array in "
             "data/corpus/; without it the script stops rather than replace it")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
