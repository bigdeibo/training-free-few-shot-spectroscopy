"""Is the low-signal-to-noise win the projection, or the head?

On the low-SNR sugar mixtures a partial-least-squares projection beats a PCA one
by a wide margin, and the question is whether that comes from the supervised
projection or from the linear head that follows it. Those two are confounded in
the comparison as usually run, because PLS carries both at once. Separating them
gives four cells on identical splits:

  PCA scores, foundation-model head       the reference recipe
  PCA scores, ridge head                  ridge head alone
  PLS scores, ridge head                  projection alone, since ridge on PLS
                                          scores recovers what PLS itself
                                          predicts and is the check on that
  PLS scores, foundation-model head       projection changed, head held

The third cell is the self-check: ridge on PLS scores should land close to PLS,
and it does. The support set is the only thing any of the four is fitted on: the
projection, its component count, the score standardisation and the label
z-scoring all come from it, and the query set is untransformed except by the
fitted quantities.

Only the low-SNR sugar dataset gets all four cells, because it is the one whose
ordering raised the question. On the other three datasets a single cell is added
back, PCA scores with a ridge head, to test whether that one combination already
reproduces the ordering there. That is why the four output files do not share a
column set.

Output, under `results/raman/`:

  projection-decoupling__sugar-low-snr.csv     all four cells, plus the number
                                               of latent components chosen
  projection-decoupling__sugar-high-snr.csv    the ridge-head cell only
  projection-decoupling__fuel-benchtop.csv     the ridge-head cell only
  projection-decoupling__fuel-handheld.csv     the ridge-head cell only

Reruns skip rows already present.
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

DATASETS = {"sugar-low-snr": ("sugar", "Low SNR"),
            "sugar-high-snr": ("sugar", "High SNR"),
            "fuel-benchtop": ("fuel", "Benchtop"),
            "fuel-handheld": ("fuel", "Handheld")}
FULL_GRID = "sugar-low-snr"
N_COMPONENTS = 50
SUPPORT_SIZES = (5, 10, 20)
MAX_PLS_COMPONENTS = 10
PLS_FOLDS = 3
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
MIN_TARGET_SAMPLES = 30
MIN_TEST = 10
KEYS = ("target", "support_size", "repetition")


def fit_ridge(F, y):
    from sklearn.linear_model import RidgeCV
    return RidgeCV(alphas=RIDGE_ALPHAS).fit(F, y)


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
    print(f"[projection-decoupling] device={device}", flush=True)

    for name in names:
        kind, tag = DATASETS[name]
        d = load_sugar_raman(tag) if kind == "sugar" else load_fuel_raman(tag)
        X = np.asarray(d["X"], dtype=float)
        groups = np.asarray(d["groups"])
        F = PCA(n_components=min(N_COMPONENTS, len(X) - 1),
                random_state=0).fit_transform(X)
        full = name == FULL_GRID

        out_file = RESULTS / "raman" / f"projection-decoupling__{name}.csv"
        done = already_done(out_file, KEYS)
        rows = []
        for target, values in d["targets"].items():
            usable = ~np.isnan(values)
            if usable.sum() < MIN_TARGET_SAMPLES:
                continue
            Xt, Ft, gt = X[usable], F[usable], groups[usable]
            yt_all = values[usable]
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

                    xm, xsd = Xt[tr].mean(0), Xt[tr].std(0) + 1e-8
                    Xs, Xq = (Xt[tr] - xm) / xsd, (Xt[te] - xm) / xsd
                    n_folds = min(PLS_FOLDS, len(tr))
                    fold_train = len(tr) - int(np.ceil(len(tr) / n_folds))
                    nc = pls_best_components(
                        Xs, ys,
                        max_nc=max(1, min(MAX_PLS_COMPONENTS, fold_train - 1,
                                           Xt.shape[1])),
                        folds=n_folds)
                    pls = PLSRegression(n_components=nc).fit(Xs, ys)
                    T_tr, T_q = pls.transform(Xs), pls.transform(Xq)

                    # the ridge head needs the same scale-only standardisation
                    # the other arms get, or the penalty would not be comparable
                    Ts, Tq = standardize(T_tr, T_q)
                    Ps, Pq = standardize(Ft[tr], Ft[te])

                    row = dict(target=target, support_size=K, repetition=rep)
                    if full:
                        row["n_components"] = int(nc)
                        row["r2_pls"] = round(
                            r2_score(pls.predict(Xq).ravel() * ysd + ym,
                                     yt_all[te]), 4)
                        row["r2_pls_scores_ridge"] = round(
                            r2_score(fit_ridge(Ts, ys).predict(Tq) * ysd + ym,
                                     yt_all[te]), 4)
                        model.fit(Ts, ys)
                        row["r2_pls_scores_tabpfn"] = round(
                            r2_score(model.predict(Tq) * ysd + ym,
                                     yt_all[te]), 4)
                    row["r2_pca_scores_ridge"] = round(
                        r2_score(fit_ridge(Ps, ys).predict(Pq) * ysd + ym,
                                 yt_all[te]), 4)
                    row["seconds"] = round(time.time() - t0, 2)
                    rows.append(row)
                print(f"[projection-decoupling] {name}/{target} K={K} done",
                      flush=True)
            if rows:
                append_rows(out_file, rows)
                done.update(tuple(r[k] for k in KEYS) for r in rows)
                rows = []
        if rows:
            append_rows(out_file, rows)


if __name__ == "__main__":
    main()
