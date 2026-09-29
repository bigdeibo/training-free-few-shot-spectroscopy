"""Does multimodal fusion need an architecture?

Once each modality has been reduced to principal scores, the claim is that
fusion degenerates to concatenation: no fusion layer, no attention, no
architecture of any kind beyond stacking the columns. If that holds, the
representation choice is what buys multimodal behaviour, and the architecture is
not doing any work.

The edible-oil set is the one dataset here measured in three modalities on the
same samples, which is what makes the question answerable at all: near-infrared,
mid-infrared and Raman spectra, one peroxide value per sample. Seven arms are
compared, three single-modality and four combinations, each fitted through a
separate PCA per modality so that a combination is a concatenation of scores
rather than of spectra.

Protocol. Five, ten and twenty support samples, ten repetitions, splits drawn
from the same seed family the Raman experiments use. Support statistics
standardise the concatenated scores and clamp them; labels are z-scored by the
support set. The peroxide value is an oxidation product and the analysis is
reported as a null, so what this table mainly establishes is the pairing and the
protocol; the classification companion carries the modality-complementarity
result.

Output. `results/fusion/three-modal-regression.csv`, columns support_size,
repetition, modality, n_features, r2, seconds. Reruns skip rows already present.
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
from common.metrics import r2_score
from common.tasks import RAMAN_SEED, zscore_fit

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
    from tabpfn import TabPFNRegressor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNRegressor(model_path=str(tabpfn_weights("v3")), device=device)
    print(f"[fusion-regression] device={device}", flush=True)

    d = load_edible_oil_modalities()
    print(f"[fusion-regression] {d['n_dropped']} samples dropped, "
          f"n={len(d['peroxide_value'])}", flush=True)
    y = np.asarray(d["peroxide_value"], dtype=float)
    scores = {}
    for name in MODALITY_NAMES:
        X = np.asarray(d[name], dtype=float)
        scores[name] = PCA(n_components=min(N_COMPONENTS, len(X) - 1),
                           random_state=0).fit_transform(X)

    out_file = RESULTS / "fusion" / "three-modal-regression.csv"
    done = already_done(out_file, ("support_size", "repetition", "modality"))
    rows = []
    n = len(y)
    for K in SUPPORT_SIZES:
        for rep in range(args.repetitions):
            rng = np.random.default_rng(RAMAN_SEED * 1000 + K * 100 + rep)
            perm = rng.permutation(n)
            tr, te = perm[:K], perm[K:]
            ym, ysd = zscore_fit(y[tr])
            for arm in ARMS:
                if (K, rep, arm) in done:
                    continue
                used = MODALITY_NAMES if arm == "all-three-modalities" \
                    else tuple(arm.split("+"))
                F = np.concatenate([scores[m] for m in used], axis=1)
                Fs, Fq = standardize(F[tr], F[te])
                t0 = time.time()
                model.fit(Fs, (y[tr] - ym) / ysd)
                pred = model.predict(Fq) * ysd + ym
                rows.append(dict(
                    support_size=K, repetition=rep, modality=arm,
                    n_features=int(F.shape[1]),
                    r2=round(r2_score(pred, y[te]), 4),
                    seconds=round(time.time() - t0, 2)))
            print(f"[fusion-regression] K={K} rep={rep} done", flush=True)
    if rows:
        append_rows(out_file, rows)


if __name__ == "__main__":
    main()
