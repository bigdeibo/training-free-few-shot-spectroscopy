"""Does the gasoline panel flip once its corpus is large enough?

Gasoline is the one dataset here that sits on the wrong side of the corpus-size
rule. Its panel is 179 spectra, the corpus-size ablation puts saturation at a
few hundred, and the routing rule accordingly sends it to a supervised
projection rather than to the corpus one. The corpus augmentation above lifts it
over that line: 179 real spectra plus 400 simulated ones, with a principal
projection refitted on the combined set.

The test is whether the ordering flips with it. If a corpus fitted on 579
spectra makes the foundation-model head and the ridge head beat variable-
selection partial least squares on this dataset, the rule is doing what it
claims and the original assignment was a corpus-size effect rather than a
property of gasoline. If the ordering does not move, the corpus-size explanation
is wrong here and the honest reading is that the rule has a fidelity wall: the
simulated spectra share the components of gasoline but not its measurement
conditions, and a projection fitted on them does not transfer.

Four properties, three support sizes, ten repetitions. The splits are the ones
the Raman experiments use, drawn over spectra rather than wells because these
are not replicate measurements. Simulated spectra enter the projection fit only;
they never appear in a support or a query set, and no simulated spectrum carries
a target value.

Output. `results/corpus-augmentation/fuel-benchtop-augmented-corpus.csv`,
columns target, support_size, repetition, r2_tabpfn, r2_ridge, r2_pls, seconds.
Reruns skip rows already present.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.baselines import pls_best_components
from common.datasets import load_fuel_raman
from common.features import standardize
from common.io import already_done, append_rows
from common.metrics import r2_score
from common.tasks import group_split, zscore_fit

N_COMPONENTS = 50
SUPPORT_SIZES = (5, 10, 20)
MAX_PLS_COMPONENTS = 10
PLS_FOLDS = 3
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
MIN_TARGET_SAMPLES = 30
MIN_TEST = 10
KEYS = ("target", "support_size", "repetition")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repetitions", type=int, default=10)
    args = ap.parse_args()

    import torch
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from common.paths import DATA, RESULTS, tabpfn_weights
    from tabpfn import TabPFNRegressor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNRegressor(model_path=str(tabpfn_weights("v3")), device=device)
    print(f"[corpus-flip] device={device}", flush=True)

    d = load_fuel_raman("Benchtop")
    X = np.asarray(d["X"], dtype=float)
    groups = np.asarray(d["groups"])
    simulated = np.load(DATA / "simulated-fuel" / "sim_spectra.npy")
    corpus = np.vstack([X, simulated])
    F = PCA(n_components=N_COMPONENTS, random_state=0).fit_transform(corpus)
    Fr = F[: len(X)]
    print(f"[corpus-flip] corpus n={len(corpus)} "
          f"(real {len(X)} + simulated {len(simulated)})", flush=True)

    out_file = RESULTS / "corpus-augmentation" / "fuel-benchtop-augmented-corpus.csv"
    done = already_done(out_file, KEYS)
    rows = []
    for target, values in d["targets"].items():
        usable = ~np.isnan(values)
        if usable.sum() < MIN_TARGET_SAMPLES:
            continue
        Xr, Fu, gu = X[usable], Fr[usable], groups[usable]
        yu = values[usable]
        for K in SUPPORT_SIZES:
            for rep in range(args.repetitions):
                if (target, K, rep) in done:
                    continue
                tr, te = group_split(gu, K, rep)
                if len(te) < MIN_TEST:
                    continue
                ym, ysd = zscore_fit(yu[tr])
                ys = (yu[tr] - ym) / ysd
                Fs, Fq = standardize(Fu[tr], Fu[te])
                t0 = time.time()

                model.fit(Fs, ys)
                r2_tabpfn = r2_score(model.predict(Fq) * ysd + ym, yu[te])
                ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(Fs, ys)
                r2_ridge = r2_score(ridge.predict(Fq) * ysd + ym, yu[te])

                # the partial-least-squares reference sees raw spectra, the
                # scheme the near-infrared arms use
                xm, xsd = Xr[tr].mean(0), Xr[tr].std(0) + 1e-8
                Xs, Xq = (Xr[tr] - xm) / xsd, (Xr[te] - xm) / xsd
                n_folds = min(PLS_FOLDS, len(tr))
                fold_train = len(tr) - int(np.ceil(len(tr) / n_folds))
                nc = pls_best_components(
                    Xs, ys,
                    max_nc=max(1, min(MAX_PLS_COMPONENTS, fold_train - 1,
                                      Xr.shape[1])),
                    folds=n_folds)
                prediction = (PLSRegression(n_components=nc).fit(Xs, ys)
                              .predict(Xq).ravel() * ysd + ym)
                rows.append(dict(
                    target=target, support_size=K, repetition=rep,
                    r2_tabpfn=round(r2_tabpfn, 4),
                    r2_ridge=round(r2_ridge, 4),
                    r2_pls=round(r2_score(prediction, yu[te]), 4),
                    seconds=round(time.time() - t0, 2)))
            print(f"[corpus-flip] {target} K={K} done", flush=True)
    if rows:
        append_rows(out_file, rows)


if __name__ == "__main__":
    main()
