"""Where does the training-free advantage end?

The representation comparison runs to twenty support spectra, which is where the
training-free stack looks strongest. This experiment extends the axis to 50 and
100 and puts the partial-least-squares reference beside it at every point, over
the same paired splits.

Two things keep the comparison matched. PLS selects its number of latent
components by inner cross-validation on the support set, never on the query set,
and it is fitted on the raw standardised spectra while the foundation model sees
corpus PCA scores. The task pool is the whole study less the in-house
pentane/CCl4 series, whose ten physical samples cannot supply the large support
sizes; fourteen tasks enter the sweep.

A task is skipped at a support size that would leave five or fewer query
spectra, so the corn series (80 spectra) stops at 50 and the gasoline series
(60) stops at 20.

Output. `results/label-budget/label-budget__<task>.csv`, columns task,
support_size, repetition, r2_tabpfn, seconds_tabpfn, r2_pls, n_components_pls,
seconds_pls. Reruns skip rows already present.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.baselines import pls_best_components
from common.features import corpus_pca, standardize
from common.io import already_done, append_rows
from common.metrics import r2_score
from common.tasks import eval_split, load_task, tasks_except, zscore_fit

SUPPORT_SIZES = (5, 10, 20, 50, 100)
N_COMPONENTS = 100
EXCLUDED = ("pentane-ccl4-volume-fraction",)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="all", help="one task slug, or 'all'")
    ap.add_argument("--repetitions", type=int, default=10)
    args = ap.parse_args()

    import torch
    from common.paths import RESULTS, tabpfn_weights
    from tabpfn import TabPFNRegressor

    tasks = tasks_except(*EXCLUDED) if args.task == "all" else [args.task]
    out_dir = RESULTS / "label-budget"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNRegressor(model_path=str(tabpfn_weights("v3")), device=device)
    print(f"[label-budget] device={device}", flush=True)

    for task in tasks:
        d = load_task(task)
        X = np.asarray(d["X"])
        y = np.asarray(d["y"], dtype=np.float64)
        n = len(y)
        F = corpus_pca(X, N_COMPONENTS)

        out_file = out_dir / f"label-budget__{task}.csv"
        done = already_done(out_file, ("support_size", "repetition"))
        rows = []
        for K in SUPPORT_SIZES:
            if n <= K + 5:
                print(f"[label-budget] {task} K={K}: skipped, n={n}", flush=True)
                continue
            for rep in range(args.repetitions):
                if (K, rep) in done:
                    continue
                tr, te = eval_split(n, K, rep, groups=d.get("groups"))
                if len(te) < 5:
                    continue
                ym, ysd = zscore_fit(y[tr])
                yt, ys = y[te], (y[tr] - ym) / ysd

                t0 = time.time()
                Fs, Fq = standardize(F[tr], F[te])
                model.fit(Fs, ys)
                row = dict(
                    task=task, support_size=K, repetition=rep,
                    r2_tabpfn=round(r2_score(model.predict(Fq) * ysd + ym, yt), 4),
                    seconds_tabpfn=round(time.time() - t0, 2))

                t0 = time.time()
                Xs, Xq = X[tr], X[te]
                xm, xsd = Xs.mean(0), Xs.std(0) + 1e-8
                Xs, Xq = (Xs - xm) / xsd, (Xq - xm) / xsd
                nc = pls_best_components(Xs, ys)
                pred = PLSRegression(n_components=nc).fit(Xs, ys).predict(
                    Xq).ravel() * ysd + ym
                row["r2_pls"] = round(r2_score(pred, yt), 4)
                row["n_components_pls"] = int(nc)
                row["seconds_pls"] = round(time.time() - t0, 2)
                rows.append(row)
                print(f"[label-budget] {task} K={K} rep={rep}: "
                      f"tabpfn={row['r2_tabpfn']:.3f} pls={row['r2_pls']:.3f}",
                      flush=True)
        if rows:
            append_rows(out_file, rows)


if __name__ == "__main__":
    main()
