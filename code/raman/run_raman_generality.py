"""Does the representation finding survive a different vibrational modality?

The near-infrared result rests on a representation built from a corpus of
near-infrared spectra. This experiment runs the same recipe on Raman spectra,
where the physics and the instrument are different, to see whether the finding
travels.

Four datasets: sugar mixtures at high and low signal-to-noise ratio, and two
commercial-gasoline panels measured on a benchtop and a handheld instrument. One
change of representation is forced by the data: the corpus is near-infrared, so
a corpus PCA would be extrapolating across modalities, and each dataset gets a
PCA fitted on itself instead. The paper reports that substitution rather than
hiding it, and the corpus-size ablation is what justifies it, since a few
hundred spectra already saturate the subspace.

Protocol. Splits are drawn at group level from a seed family disjoint from the
near-infrared one: the drawn number K counts groups, and every spectrum in a
drawn group becomes support, so a sugar deployment has roughly twenty times K
support spectra and a fuel deployment K. Support statistics standardise the
scores and clamp them; labels are z-scored by the support set. The partial
least-squares reference is fitted on the raw spectra with its components chosen
by inner cross-validation on the support set.

Output. `results/raman/<dataset>.csv`, one row per (target, support size,
repetition), columns target, support_size, repetition, r2_tabpfn, r2_pls,
seconds. Reruns skip rows already present.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.baselines import pls_best_components
from common.datasets import load_fuel_raman, load_sugar_raman
from common.features import standardize
from common.io import already_done, append_rows
from common.metrics import r2_score
from common.tasks import group_split, zscore_fit

DATASETS = {"sugar-high-snr": ("sugar", "High SNR"),
            "sugar-low-snr": ("sugar", "Low SNR"),
            "fuel-benchtop": ("fuel", "Benchtop"),
            "fuel-handheld": ("fuel", "Handheld")}
N_COMPONENTS = 50
SUPPORT_SIZES = (5, 10, 20)
MAX_PLS_COMPONENTS = 10
PLS_FOLDS = 3
MIN_TARGET_SAMPLES = 30
MIN_TEST = 10


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="all",
                    choices=["all"] + sorted(DATASETS))
    ap.add_argument("--repetitions", type=int, default=10)
    args = ap.parse_args()

    import torch
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from common.paths import RESULTS, tabpfn_weights
    from tabpfn import TabPFNRegressor

    names = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNRegressor(model_path=str(tabpfn_weights("v3")), device=device)
    print(f"[raman] device={device}", flush=True)

    for name in names:
        kind, tag = DATASETS[name]
        if kind == "sugar":
            d = load_sugar_raman(tag)
        else:
            d = load_fuel_raman(tag)
        X = np.asarray(d["X"])
        groups = np.asarray(d["groups"])
        F = PCA(n_components=min(N_COMPONENTS, len(X) - 1),
                random_state=0).fit_transform(X)

        out_file = RESULTS / "raman" / f"{name}.csv"
        done = already_done(out_file, ("target", "support_size", "repetition"))
        rows = []
        for target, values in d["targets"].items():
            usable = ~np.isnan(values)
            if usable.sum() < MIN_TARGET_SAMPLES:
                continue
            Xt, Ft, gt, yt_all = X[usable], F[usable], groups[usable], values[usable]
            for K in SUPPORT_SIZES:
                for rep in range(args.repetitions):
                    if (target, K, rep) in done:
                        continue
                    tr, te = group_split(gt, K, rep)
                    if len(te) < MIN_TEST:
                        continue
                    ym, ysd = zscore_fit(yt_all[tr])
                    ys = (yt_all[tr] - ym) / ysd
                    t0 = time.time()

                    Fs, Fq = standardize(Ft[tr], Ft[te])
                    model.fit(Fs, ys)
                    pred_tabpfn = model.predict(Fq) * ysd + ym

                    xm, xsd = Xt[tr].mean(0), Xt[tr].std(0) + 1e-8
                    Xs, Xq = (Xt[tr] - xm) / xsd, (Xt[te] - xm) / xsd
                    n_folds = min(PLS_FOLDS, len(tr))
                    fold_train = len(tr) - int(np.ceil(len(tr) / n_folds))
                    nc = pls_best_components(
                        Xs, ys,
                        max_nc=max(1, min(MAX_PLS_COMPONENTS, fold_train - 1,
                                           Xt.shape[1])),
                        folds=n_folds)
                    pred_pls = PLSRegression(n_components=nc).fit(
                        Xs, ys).predict(Xq).ravel() * ysd + ym

                    rows.append(dict(
                        target=target, support_size=K, repetition=rep,
                        r2_tabpfn=round(r2_score(pred_tabpfn, yt_all[te]), 4),
                        r2_pls=round(r2_score(pred_pls, yt_all[te]), 4),
                        seconds=round(time.time() - t0, 2)))
                print(f"[raman] {name}/{target} K={K} done", flush=True)
            if rows:
                append_rows(out_file, rows)
                done.update((r["target"], r["support_size"], r["repetition"])
                            for r in rows)
                rows = []
        if rows:
            append_rows(out_file, rows)


if __name__ == "__main__":
    main()
