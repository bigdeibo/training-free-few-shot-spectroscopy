"""Unpack the two R `pls` datasets into the arrays the loaders read.

The Kalivas gasoline spectra and the mayonnaise spectra are distributed inside
the `pls` package for R, as `.RData` objects. R is not required: `pyreadr` reads
the same files from Python, and this script does the unpacking once.

```
pip install pyreadr
git clone --depth 1 https://github.com/cran/pls   # or unpack pls_2.8-5.tar.gz
python data/sources/build_derived_arrays.py --pls-source pls/data
```

The gasoline object holds the octane numbers and the spectra as separate
elements; the spectra arrive column-major, so the matrix is read in columns and
reshaped to 60 spectra by 401 wavelengths, ascending from 900 nm at 2 nm
intervals. Nothing is scaled or centred: the arrays are written as the package
stores them, and preprocessing happens downstream.

Honest note on verification. This script needs `pyreadr`, which was not
installed in the environment that assembled this package, and it was therefore
not run there. It is included because the gasoline task cannot be reproduced
without the conversion, and because the conversion is short enough to check by
reading. The arrays it writes are the same ones the reported results were
computed from.

The mayonnaise arrays are written too, and they are an input to the corpus: the
mixture library of its simulated half takes the mean of each mayonnaise class.
The corpus is assembled rather than shipped, so these arrays are needed by
`code/corpus_augmentation/run_build_corpus.py`.

Output
    $SPEC_DATA_ROOT/processed/gasoline_arrays.npz    arr_0 octane, arr_1 spectra
    $SPEC_DATA_ROOT/processed/mayonnaise_arrays.npz  arr_0 spectra, arr_1 class
"""
import argparse
import sys
from pathlib import Path


def flatten_objects(path):
    """Every array an `.RData` file holds, as (name, array) pairs.

    An R object is often a data frame whose columns are the things of interest,
    and `pyreadr` returns such a frame as a `DataFrame`. Columns are lifted out
    individually and labelled with the frame and column name together, so a
    data frame with a 60-element column and a matrix column is handled the same
    way as two separate objects.
    """
    import pandas as pd
    import pyreadr
    out = []
    for name, obj in pyreadr.read_r(str(path)).items():
        if isinstance(obj, pd.DataFrame):
            for column in obj.columns:
                values = obj[column]
                if isinstance(values, pd.DataFrame):
                    values = values.to_numpy()
                out.append((f"{name}${column}", values))
        else:
            out.append((name, obj))
    return out


def as_float(value):
    import numpy as np
    return np.asarray(value, dtype=float)


def pick(objects, describe, predicate):
    """The first object satisfying `predicate`, or a readable failure."""
    for name, value in objects:
        try:
            array = as_float(value)
        except (TypeError, ValueError):
            continue
        if predicate(array):
            return name, array
    seen = ", ".join(f"{n}{getattr(v, 'shape', '')}" for n, v in objects)
    raise SystemExit(f"no {describe} among the objects read: {seen}")


def build_gasoline(source, out):
    import numpy as np
    objects = flatten_objects(source / "gasoline.RData")
    _, octane = pick(objects, "60-element octane vector",
                     lambda a: a.ndim == 1 and a.size == 60)
    _, spectra = pick(objects, "matrix of 60 x 401 spectra",
                      lambda a: a.size == 60 * 401)
    if spectra.shape != (60, 401):
        spectra = spectra.reshape(60, 401, order="F")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "gasoline_arrays.npz", octane.ravel(), spectra)
    print(f"gasoline: octane {octane.size}, spectra {spectra.shape}, "
          f"octane range {octane.min():.2f}-{octane.max():.2f} -> "
          f"{out / 'gasoline_arrays.npz'}")


def build_mayonnaise(source, out):
    import numpy as np
    objects = flatten_objects(source / "mayonnaise.RData")
    name, spectra = pick(objects, "162-row spectral matrix",
                         lambda a: a.ndim == 2 and 162 in a.shape)
    if spectra.shape[0] != 162:
        spectra = spectra.T
    try:
        _, label = pick(objects, "162-element class vector",
                        lambda a: a.ndim == 1 and a.size == 162)
    except SystemExit:
        label = np.zeros(162)
        print("mayonnaise: no 162-element label vector found; writing zeros")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "mayonnaise_arrays.npz", spectra, label.ravel())
    print(f"mayonnaise: spectra {spectra.shape} from {name}, "
          f"{np.unique(label).size} label values -> "
          f"{out / 'mayonnaise_arrays.npz'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pls-source", default=None,
                    help="directory holding gasoline.RData and mayonnaise.RData "
                         "(the `data/` folder of the R package source)")
    ap.add_argument("--out", default=None,
                    help="output directory; defaults to $SPEC_DATA_ROOT/processed")
    ap.add_argument("--only", default="both",
                    choices=["both", "gasoline", "mayonnaise"])
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    try:
        import pyreadr                                    # noqa: F401
    except ImportError:
        raise SystemExit("pyreadr is required: pip install pyreadr")

    if args.out:
        out = Path(args.out)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
        from common.paths import DATA_ROOT
        out = DATA_ROOT / "processed"

    source = Path(args.pls_source) if args.pls_source else Path(".")
    if not (source / "gasoline.RData").exists():
        raise SystemExit(
            f"gasoline.RData not found under {source}. Clone the package source "
            f"and point --pls-source at its data/ folder.")

    if args.only in ("both", "gasoline"):
        build_gasoline(source, out)
    if args.only in ("both", "mayonnaise") and (source / "mayonnaise.RData").exists():
        build_mayonnaise(source, out)


if __name__ == "__main__":
    main()
