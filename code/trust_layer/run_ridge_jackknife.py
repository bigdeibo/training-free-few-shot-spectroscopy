"""The classical comparator: ridge regression with jackknife+ intervals.

Same deployments, same representation, same nominal levels as the posterior
audit, so the two tables are directly comparable. Ridge is the classical
counterpart that has a distribution-free interval guarantee, which is what makes
it the honest competitor to a foundation model's posterior.

Two details differ from the audit on purpose and both matter at these support
sizes.

Scale-only normalisation, no centring, with an unpenalised intercept. Centring
the embeddings by the support mean makes the columns sum to zero, an exact
linear dependency among the K rows; with mean-zero labels and no intercept the
leave-one-out prediction then reproduces the held-out value exactly and the
jackknife+ interval collapses to zero width. Dividing by the support standard
deviation only, and carrying an explicit intercept, breaks that dependency
while leaving the geometry otherwise as the other arms see it.

The penalty is chosen by leave-one-out rather than by generalised
cross-validation. As the penalty goes to zero below K = p the fit interpolates,
the degrees-of-freedom denominator of GCV vanishes, and GCV returns the grid
minimum for every deployment. The closed-form leave-one-out residuals stay
finite there, so their criterion has a real interior minimum.

A calibration half is not needed: jackknife+ needs no hold-out, so coverage and
width are measured on the whole query set.

Output. `results/ridge-jackknife/ridge-jackknife__<task>.csv`, columns task,
support_size, repetition, method, nominal_coverage, empirical_coverage,
mean_interval_width, ridge_penalty, r2_evaluation, catastrophic_failure,
seconds. Reruns skip rows already present.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.features import corpus_pca
from common.io import already_done, append_rows
from common.metrics import r2_score
from common.ridge_uq import jackknife_plus_interval, loo_lambda
from common.tasks import TASKS, eval_split, load_task

N_COMPONENTS = 100
LEVELS = (50, 68, 80, 90, 95)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="all", help="one task slug, or 'all'")
    ap.add_argument("--repetitions", type=int, default=10)
    args = ap.parse_args()

    from common.paths import RESULTS

    tasks = list(TASKS) if args.task == "all" else [args.task]
    out_dir = RESULTS / "ridge-jackknife"

    for task in tasks:
        d = load_task(task)
        X = np.asarray(d["X"])
        y = np.asarray(d["y"], dtype=np.float64)
        n = len(y)
        F = corpus_pca(X, N_COMPONENTS)

        out_file = out_dir / f"ridge-jackknife__{task}.csv"
        done = already_done(out_file, ("support_size", "repetition"))
        rows = []
        for K in (5, 10, 20):
            if n <= K + 5:
                continue
            for rep in range(args.repetitions):
                if (K, rep) in done:
                    continue
                tr, te = eval_split(n, K, rep, groups=d.get("groups"))
                if len(te) < 4:
                    continue
                t0 = time.time()
                sd = F[tr].std(0) + 1e-8
                Fs = np.clip(F[tr] / sd, -5, 5)
                Fq = np.clip(F[te] / sd, -5, 5)
                ys, yt = y[tr], y[te]
                penalty = loo_lambda(Fs, ys)
                r2_eval = None
                for level in LEVELS:
                    lo, hi, info = jackknife_plus_interval(
                        Fs, ys, penalty, Fq, alpha=1 - level / 100)
                    if r2_eval is None:
                        r2_eval = r2_score(info["yhat"], yt)
                    rows.append(dict(
                        task=task, support_size=K, repetition=rep,
                        method="ridge-jackknife-plus", nominal_coverage=level,
                        empirical_coverage=round(
                            float(((yt >= lo) & (yt <= hi)).mean()), 4),
                        mean_interval_width=round(float((hi - lo).mean()), 5),
                        ridge_penalty=penalty,
                        r2_evaluation=round(r2_eval, 4),
                        catastrophic_failure=int(r2_eval < -1),
                        seconds=round(time.time() - t0, 2)))
                print(f"[ridge-jackknife] {task} K={K} rep={rep}: "
                      f"r2={r2_eval:.3f} penalty={penalty:.2g} "
                      f"({rows[-1]['seconds']}s)", flush=True)
        if rows:
            append_rows(out_file, rows)


if __name__ == "__main__":
    main()
