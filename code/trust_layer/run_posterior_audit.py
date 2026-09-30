"""Does a foundation model's posterior stay calibrated out of domain?

The paper's central trust-layer experiment. For each deployment the model is
fitted once on the support set and asked for an eleven-quantile posterior grid
on the query set. The query set is then split in half, by a seed drawn from the
same family as the splits, into a calibration half and an evaluation half, and
three interval constructions are scored on the evaluation half:

  native-posterior                the posterior quantile pair at each level
  split-conformal-absolute        split conformal on |y - posterior median|
  split-conformal-locally-adaptive
                                  split conformal on |y - median| divided by the
                                  native 80% width, so the interval widens where
                                  the posterior itself is unsure

Coverage, mean width and R2 are recorded at nominal levels 50, 68, 80, 90 and
95 per cent, together with a flag for whether that deployment's R2 fell below
-1. The flag is what lets the coverage table and the catastrophic-failure rate
be read against one another.

The representation is corpus PCA (100) throughout, standardised by support
statistics and clamped, with labels z-scored by the support set.

Output.
`results/posterior-audit/posterior-audit__tabpfn-<generation>__<task>.csv`,
columns task, support_size, repetition, method, nominal_coverage,
empirical_coverage, mean_interval_width, r2_evaluation, catastrophic_failure,
seconds. Reruns skip rows already present.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.features import corpus_pca, standardize
from common.io import already_done, append_rows
from common.metrics import r2_score
from common.tasks import TASKS, eval_split, load_task, zscore_fit

GENERATIONS = ["v3", "v3.5"]
N_COMPONENTS = 100
QUANTILES = [0.025, 0.05, 0.10, 0.16, 0.25, 0.50, 0.75, 0.84, 0.90, 0.95, 0.975]
LEVELS = {50: (0.25, 0.75), 68: (0.16, 0.84), 80: (0.10, 0.90),
          90: (0.05, 0.95), 95: (0.025, 0.975)}
MEDIAN = QUANTILES.index(0.50)
WIDTH_LO, WIDTH_HI = QUANTILES.index(0.10), QUANTILES.index(0.90)


def conformal_quantile(scores, alpha):
    """The finite-sample split-conformal quantile of `scores` at level 1 - alpha."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, n) - 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generation", default="v3", choices=GENERATIONS)
    ap.add_argument("--task", default="all", help="one task slug, or 'all'")
    ap.add_argument("--repetitions", type=int, default=10)
    args = ap.parse_args()

    import torch
    from common.paths import RESULTS, tabpfn_weights
    from tabpfn import TabPFNRegressor

    tasks = list(TASKS) if args.task == "all" else [args.task]
    out_dir = RESULTS / "posterior-audit"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNRegressor(model_path=str(tabpfn_weights(args.generation)),
                            device=device)
    print(f"[posterior-audit] foundation model {args.generation}, "
          f"device={device}", flush=True)

    for task in tasks:
        d = load_task(task)
        X = np.asarray(d["X"])
        y = np.asarray(d["y"], dtype=np.float64)
        n = len(y)
        F = corpus_pca(X, N_COMPONENTS)

        out_file = out_dir / (f"posterior-audit__tabpfn-{args.generation}__"
                              f"{task}.csv")
        done = already_done(out_file, ("support_size", "repetition"))
        rows = []
        for K in (5, 10, 20):
            if n <= K + 5:
                continue
            for rep in range(args.repetitions):
                if (K, rep) in done:
                    continue
                tr, te = eval_split(n, K, rep, groups=d.get("groups"))
                if len(te) < 8:                       # two non-empty halves
                    continue
                t0 = time.time()
                ym, ysd = zscore_fit(y[tr])
                Fs, Fq = standardize(F[tr], F[te])
                model.fit(Fs, (y[tr] - ym) / ysd)
                q = np.asarray(model.predict(Fq, output_type="quantiles",
                                             quantiles=QUANTILES)) * ysd + ym
                median = q[MEDIAN]
                yt = y[te]

                rng = np.random.default_rng(66000 + K * 100 + rep)
                perm = rng.permutation(len(te))
                cal, ev = perm[: len(te) // 2], perm[len(te) // 2:]
                width80 = (q[WIDTH_HI] - q[WIDTH_LO]) + 1e-12
                r2_eval = r2_score(median[ev], yt[ev])

                for level, (lo_q, hi_q) in LEVELS.items():
                    alpha = 1 - level / 100
                    q_lo, q_hi = q[QUANTILES.index(lo_q)], q[QUANTILES.index(hi_q)]

                    cov_native = float(((yt[ev] >= q_lo[ev]) &
                                        (yt[ev] <= q_hi[ev])).mean())
                    wid_native = float((q_hi[ev] - q_lo[ev]).mean())

                    absolute = conformal_quantile(
                        np.abs(yt[cal] - median[cal]), alpha)
                    cov_abs = float((np.abs(yt[ev] - median[ev]) <= absolute).mean())

                    local = conformal_quantile(
                        np.abs(yt[cal] - median[cal]) / width80[cal], alpha)
                    cov_loc = float((np.abs(yt[ev] - median[ev])
                                     <= local * width80[ev]).mean())
                    wid_loc = float((2 * local * width80[ev]).mean())

                    for method, coverage, width in (
                            ("native-posterior", cov_native, wid_native),
                            ("split-conformal-absolute", cov_abs, 2 * absolute),
                            ("split-conformal-locally-adaptive", cov_loc, wid_loc)):
                        rows.append(dict(
                            task=task, support_size=K, repetition=rep,
                            method=method, nominal_coverage=level,
                            empirical_coverage=round(coverage, 4),
                            mean_interval_width=round(width, 5),
                            r2_evaluation=round(r2_eval, 4),
                            catastrophic_failure=int(r2_eval < -1),
                            seconds=round(time.time() - t0, 2)))
                print(f"[posterior-audit] {task} K={K} rep={rep}: "
                      f"r2={r2_eval:.3f} native 95%={cov_native:.2f} "
                      f"absolute 95%={cov_abs:.2f} local 95%={cov_loc:.2f} "
                      f"({rows[-1]['seconds']}s)", flush=True)
        if rows:
            append_rows(out_file, rows)


if __name__ == "__main__":
    main()
