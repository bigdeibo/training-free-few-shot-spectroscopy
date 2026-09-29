"""Filesystem anchors, and the one place external paths are resolved.

Everything a script reads or writes is reached through this module, so a copy of
the archive can be relocated by editing one file or by setting the matching
environment variable.

  data/public          third-party public datasets, not redistributed here.
                       `SPEC_DATA_ROOT` overrides it. See data/sources/.
  data/                artifacts this work produced, which do ship.
  results/             one CSV per (task, support size, repetition, arm).

One artifact is assembled rather than shipped. The corpus every projection is
fitted on is 3339 measured spectra plus 3000 simulated ones, and only the
simulated half is redistributed, as `CORPUS_SIMULATED`. `CORPUS_SPECTRA` is where
`code/corpus_augmentation/run_build_corpus.py` assembles the whole of it from the
simulated half and the public datasets under the data root.

The TabPFN weights are the exception. Their licence forbids redistribution, so
they are not in the archive and have no default location: `SPEC_TABPFN_MODELS`
must point at a directory holding them, and `tabpfn_weights` raises a readable
error if it does not.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RESULTS = REPO / "results"

DATA_ROOT = Path(os.environ.get("SPEC_DATA_ROOT", DATA / "public"))
ENCODER_CHECKPOINTS = DATA / "encoder-checkpoints"
CORPUS_SIMULATED = DATA / "corpus" / "simulated_spectra.npy"
CORPUS_SPECTRA = DATA / "corpus" / "spectra.npy"

# The three foundation-model generations compared in the paper, under the file
# names the vendor ships.
TABPFN_FILES = {
    "v2.5": "tabpfn-v2.5-regressor-v2.5_default.ckpt",
    "v3": "tabpfn-v3-regressor-v3_default.ckpt",
    "v3.5": "tabpfn-v3.5-20260909.safetensors",
}


def tabpfn_weights(generation):
    """Path to one generation's weight file, or a readable error explaining that
    the weights are not distributed with this archive."""
    env = os.environ.get("SPEC_TABPFN_MODELS")
    if not env:
        raise RuntimeError(
            f"TabPFN weights are not distributed with this archive (the vendor "
            f"licence forbids redistribution). Download them and set "
            f"SPEC_TABPFN_MODELS to the directory holding {TABPFN_FILES[generation]!r}; "
            f"see requirements.txt.")
    path = Path(env) / TABPFN_FILES[generation]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return path
