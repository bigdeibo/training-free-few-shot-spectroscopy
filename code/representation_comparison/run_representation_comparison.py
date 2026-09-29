"""Which representation of a spectrum makes a tabular foundation model work?

One representation arm, one foundation-model generation, and the fifteen tasks
of the study, under the paired split every method in the paper shares.

Twelve representations are compared:

  raw-spectra-512             the SNV spectra as they enter
  corpus-pca-20/50/100/200    principal scores of a PCA fitted on the corpus
  corpus-pca-*-fit-500/1500/3000
                              the same, fitted on a corpus subsample
  wavelet-scattering          second-order Morlet scattering, pooled
  cars-selected               channels kept by CARS on the support set
  random-projections-100      a fixed Gaussian projection, the dimensionality
                              control
  random-init-encoder         untrained 128-d encoder, the architecture control
  simclr-encoder              contrastive encoder, the equal-budget control
  masked-autoencoder          the encoder this work pretrains

Protocol. Support sizes 5, 10 and 20, ten repetitions each. Labels are
z-scored by the support set. Every representation except `raw-spectra-512` and
`cars-selected` is standardised by support statistics and clamped to +/-5.
Support and test indices come from `common.tasks.eval_split`, so they are
identical to those of every other experiment in the paper, group-aware splits
included. CARS picks its channels on the support set alone.

Output. One table per (generation, representation), named
`results/representation-comparison/tabpfn-<generation>__<representation>__<task>.csv`,
with columns task, support_size, repetition, representation, n_features, r2,
rmse, rpd, seconds. Reruns skip rows already present.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import features
from common.baselines import cars_select
from common.io import already_done, append_rows
from common.metrics import r2_score, rpd
from common.tasks import TASKS, eval_split, load_task, zscore_fit

REPRESENTATIONS = [
    "raw-spectra-512",
    "corpus-pca-50", "corpus-pca-20", "corpus-pca-100", "corpus-pca-200",
    "corpus-pca-50-fit-500", "corpus-pca-50-fit-1500", "corpus-pca-50-fit-3000",
    "corpus-pca-100-fit-500", "corpus-pca-100-fit-1500", "corpus-pca-100-fit-3000",
    "wavelet-scattering", "cars-selected", "random-projections-100",
    "random-init-encoder", "simclr-encoder", "masked-autoencoder",
]
GENERATIONS = ["v2.5", "v3", "v3.5"]
SUPPORT_SIZES = (5, 10, 20)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--representation", required=True, choices=REPRESENTATIONS)
    ap.add_argument("--generation", default="v3", choices=GENERATIONS)
    ap.add_argument("--task", default="all", help="one task slug, or 'all'")
    ap.add_argument("--repetitions", type=int, default=10)
    args = ap.parse_args()

    import torch
    from common.paths import RESULTS, tabpfn_weights
    from tabpfn import TabPFNRegressor

    tasks = list(TASKS) if args.task == "all" else [args.task]
    out_dir = RESULTS / "representation-comparison"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNRegressor(model_path=str(tabpfn_weights(args.generation)),
                            device=device)
    print(f"[representation] {args.representation} on foundation model "
          f"{args.generation}, device={device}", flush=True)

    for task in tasks:
        d = load_task(task)
        X = np.asarray(d["X"], dtype=float)
        y = np.asarray(d["y"], dtype=np.float64)
        n = len(y)
        is_std = args.representation in features.STANDARDIZED
        F = X if args.representation == "cars-selected" else \
            features.build(args.representation, X)[0]

        out_file = out_dir / (f"tabpfn-{args.generation}__"
                              f"{args.representation}__{task}.csv")
        done = already_done(out_file, ("support_size", "repetition"))
        rows = []
        for K in SUPPORT_SIZES:
            if n <= K + 5:
                continue
            for rep in range(args.repetitions):
                if (K, rep) in done:
                    continue
                tr, te = eval_split(n, K, rep, groups=d.get("groups"))
                if len(te) == 0:
                    continue
                t0 = time.time()
                ym, ysd = zscore_fit(y[tr])
                if args.representation == "cars-selected":
                    # variable selection sees the support set only
                    mask = cars_select(X[tr], y[tr], seed=66000 + K * 100 + rep)
                    Fs, Fq = F[tr][:, mask], F[te][:, mask]
                elif is_std:
                    Fs, Fq = features.standardize(F[tr], F[te])
                else:
                    Fs, Fq = F[tr], F[te]
                model.fit(Fs, (y[tr] - ym) / ysd)
                pred = model.predict(Fq) * ysd + ym
                yt = y[te]
                err = float(np.sqrt(np.mean((pred - yt) ** 2)))
                rows.append(dict(
                    task=task, support_size=K, repetition=rep,
                    representation=args.representation,
                    n_features=int(Fs.shape[1]),
                    r2=r2_score(pred, yt), rmse=err, rpd=rpd(yt, err),
                    seconds=round(time.time() - t0, 2)))
                print(f"[representation:{args.representation}] {task} K={K} "
                      f"rep={rep}: r2={rows[-1]['r2']:.3f} "
                      f"({rows[-1]['seconds']}s)", flush=True)
        if rows:
            append_rows(out_file, rows)


if __name__ == "__main__":
    main()
