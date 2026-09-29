"""The same seven arms, on the task where the modalities actually differ.

The regression companion is a null: peroxide value is an oxidation product and
none of the three modalities carries much of it. Oil type is the opposite case.
The nineteen oil classes are visible in different ways to different techniques,
so if concatenating principal scores is all that multimodal fusion needs, the
combined arms should separate classes that no single modality separates.

The pairing, the per-modality PCA, the arms and the splits are exactly those of
the regression companion, so the two tables are paired, not merely adjacent.
Splits are random draws over samples rather than group-aware, the classes being
the unit of interest here rather than the replicate structure.

The head is the classification mode of the same foundation-model checkpoint,
which the v2.5 and v3 checkpoints do not expose; the v3.5 file carries
classification and regression in one.

Output. `results/fusion/three-modal-classification.csv`, columns support_size,
repetition, modality, n_features, accuracy, seconds. Reruns skip rows already
present.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.datasets import load_edible_oil_modalities
from common.features import standardize
from common.io import already_done, append_rows
from common.tasks import RAMAN_SEED

ARMS = ["nir", "mir", "raman", "nir+mir", "nir+raman", "mir+raman",
        "all-three-modalities"]
MODALITY_NAMES = ("nir", "mir", "raman")
N_COMPONENTS = 50
SUPPORT_SIZES = (5, 10, 20)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repetitions", type=int, default=10)
    args = ap.parse_args()

    import torch
    from sklearn.decomposition import PCA
    from common.paths import RESULTS, tabpfn_weights
    from tabpfn import TabPFNClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNClassifier(model_path=str(tabpfn_weights("v3.5")),
                             device=device)
    print(f"[fusion-classification] device={device}", flush=True)

    d = load_edible_oil_modalities()
    y = np.asarray(d["oil_class"], dtype=int)
    print(f"[fusion-classification] n={len(y)}, "
          f"{len(np.unique(y))} classes", flush=True)
    scores = {}
    for name in MODALITY_NAMES:
        X = np.asarray(d[name], dtype=float)
        scores[name] = PCA(n_components=min(N_COMPONENTS, len(X) - 1),
                           random_state=0).fit_transform(X)

    out_file = RESULTS / "fusion" / "three-modal-classification.csv"
    done = already_done(out_file, ("support_size", "repetition", "modality"))
    rows = []
    n = len(y)
    for K in SUPPORT_SIZES:
        for rep in range(args.repetitions):
            rng = np.random.default_rng(RAMAN_SEED * 1000 + K * 100 + rep)
            perm = rng.permutation(n)
            tr, te = perm[:K], perm[K:]
            for arm in ARMS:
                if (K, rep, arm) in done:
                    continue
                used = MODALITY_NAMES if arm == "all-three-modalities" \
                    else tuple(arm.split("+"))
                F = np.concatenate([scores[m] for m in used], axis=1)
                Fs, Fq = standardize(F[tr], F[te])
                t0 = time.time()
                model.fit(Fs, y[tr])
                accuracy = float((model.predict(Fq) == y[te]).mean())
                rows.append(dict(
                    support_size=K, repetition=rep, modality=arm,
                    n_features=int(F.shape[1]), accuracy=round(accuracy, 4),
                    seconds=round(time.time() - t0, 2)))
            print(f"[fusion-classification] K={K} rep={rep} done", flush=True)
    if rows:
        append_rows(out_file, rows)


if __name__ == "__main__":
    main()
