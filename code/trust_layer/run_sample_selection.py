"""Where should the next labelled spectrum come from?

A support set is grown from two spectra to twenty, one batch at a time, and the
stack is evaluated on a fixed held-out set at support sizes 2, 5, 10, 15 and 20.
The candidate pool is 60 per cent of the dataset, split off at group level, and
the held-out 40 per cent is never touched by a selection rule. No rule ever sees
a label from the pool.

Three rules:

  random                  the control
  k-center-diversity      farthest-point sampling in the representation, which
                          needs no model at all
  interval-width          the widest native 80% interval among the candidates,
                          scored in batches at the checkpoints

A fourth rule, the physics gate of `run_physics_gate.py`, was tried first and
dropped: scoring it at every step cost about 1100 s per repetition against
seconds for the other three, and both the query-level null in the gate's own
table and the survival controls argue that the gate belongs at deployment level,
not per query.

The in-house pentane/CCl4 series is excluded (ten physical samples cannot supply
a candidate pool), and so is the peroxide-value control, where the whole point
of the task is that no selection rule has anything to select on.

Output. `results/sample-selection/<task>.csv`, columns task, repetition,
strategy, support_size, r2, rmse, seconds. Reruns skip whole (task, repetition,
strategy) blocks already present.
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
from common.tasks import load_task, tasks_except, zscore_fit

N_COMPONENTS = 100
STRATEGIES = ["random", "k-center-diversity", "interval-width"]
CHECKPOINTS = (2, 5, 10, 15, 20)
SEED_SIZE = 2
CANDIDATE_FRACTION = 0.6
MAX_TEST = 400          # cost control on the olive-oil query set
MAX_POOL_SCORED = 200   # cost control on the interval-width rule
EXCLUDED = ("pentane-ccl4-volume-fraction", "edible-oil-peroxide-value")


def candidate_test_split(n, rep, groups=None, fraction=CANDIDATE_FRACTION):
    """Split into a candidate pool and a held-out set, at group level when the
    dataset has replicate structure.

    The seed family is disjoint from the one `common.tasks.eval_split` uses, so
    the pool here and the support set in every other experiment are different
    draws.
    """
    rng = np.random.default_rng(66 * 1000 + 900 + rep)
    if groups is None:
        perm = rng.permutation(n)
        cut = int(round(fraction * n))
        return perm[:cut], perm[cut:]
    g = np.asarray(groups)
    unique = np.array(sorted(set(g.tolist())))
    perm = rng.permutation(len(unique))
    cut = int(round(fraction * len(unique)))
    chosen = set(unique[perm[:cut]].tolist())
    cand = np.array([i for i in range(n) if g[i] in chosen])
    test = np.array([i for i in range(n) if g[i] not in chosen])
    return cand, test


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="all", help="one task slug, or 'all'")
    ap.add_argument("--repetitions", type=int, default=10)
    args = ap.parse_args()

    import torch
    from common.paths import RESULTS, tabpfn_weights
    from tabpfn import TabPFNRegressor

    tasks = tasks_except(*EXCLUDED) if args.task == "all" else [args.task]
    out_dir = RESULTS / "sample-selection"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNRegressor(model_path=str(tabpfn_weights("v3")), device=device)
    print(f"[sample-selection] device={device}", flush=True)

    for task in tasks:
        d = load_task(task)
        X = np.asarray(d["X"])
        y = np.asarray(d["y"], dtype=np.float64)
        n = len(y)
        F = corpus_pca(X, N_COMPONENTS)

        out_file = out_dir / f"{task}.csv"
        done = already_done(out_file, ("repetition", "strategy"))
        for rep in range(args.repetitions):
            cand, test = candidate_test_split(n, rep, groups=d.get("groups"))
            if len(cand) < 22 or len(test) < 5:
                continue
            if len(test) > MAX_TEST:
                rng_test = np.random.default_rng(77000 + rep)
                test = test[rng_test.choice(len(test), size=MAX_TEST,
                                            replace=False)]
            F_cand, F_test = F[cand], F[test]
            y_test = y[test]

            # Seed support set, drawn once per (task, repetition) and shared by
            # every strategy, so that the arms are paired: their increments are
            # then correlated with the alternatives they are compared against,
            # and a paired test on the differences answers the question the
            # comparison is meant to ask. Drawing the seed independently inside
            # the strategy loop instead leaves the arms unpaired, which silently
            # invalidates any paired statistic computed from the output; at
            # K = SEED_SIZE that defect also produces non-zero differences
            # between arms that hold the same support set by construction.
            seed_support = list(np.random.default_rng(66000 + 900 + rep)
                                .choice(len(cand), size=SEED_SIZE,
                                        replace=False))

            for si, strategy in enumerate(STRATEGIES):
                if (rep, strategy) in done:
                    continue
                t0 = time.time()
                # Per-strategy stream: reproducible, and independent of which
                # strategies have already been evaluated. 7 * si keeps the
                # streams disjoint across arms within a repetition.
                rng = np.random.default_rng(66100 + 900 + rep + 7 * si)
                support = list(seed_support)
                pool = [i for i in range(len(cand)) if i not in support]
                rows = []

                def evaluate(K):
                    ym, ysd = zscore_fit(y[cand[support]])
                    Fs, Fq = standardize(F_cand[support], F_test)
                    model.fit(Fs, (y[cand[support]] - ym) / ysd)
                    pred = model.predict(Fq) * ysd + ym
                    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
                    rows.append(dict(
                        task=task, repetition=rep, strategy=strategy,
                        support_size=K, r2=round(r2_score(pred, y_test), 4),
                        rmse=round(rmse, 5),
                        seconds=round(time.time() - t0, 2)))

                evaluate(SEED_SIZE)
                for next_K in CHECKPOINTS[1:]:
                    needed = min(next_K - len(support), len(pool))
                    if needed <= 0:
                        break
                    if strategy == "random":
                        pick = rng.choice(len(pool), size=needed, replace=False)
                        for i in pick:
                            support.append(pool[i])
                        for i in sorted(pick, reverse=True):
                            pool.pop(i)
                    elif strategy == "k-center-diversity":
                        for _ in range(needed):
                            d2 = ((F_cand[pool][:, None] - F_cand[support][None])
                                  ** 2).sum(-1)
                            nxt = pool[int(d2.min(1).argmax())]
                            support.append(nxt)
                            pool.remove(nxt)
                    else:
                        # interval width, scored in one batch at the checkpoint
                        ym, ysd = zscore_fit(y[cand[support]])
                        Fs, _ = standardize(F_cand[support], F_cand[support])
                        model.fit(Fs, (y[cand[support]] - ym) / ysd)
                        if len(pool) > MAX_POOL_SCORED:
                            sub = rng.choice(len(pool), size=MAX_POOL_SCORED,
                                             replace=False)
                            scored = [pool[i] for i in sub]
                        else:
                            scored = list(pool)
                        _, Fp = standardize(F_cand[support], F_cand[scored])
                        q = np.asarray(model.predict(
                            Fp, output_type="quantiles", quantiles=[0.1, 0.9]))
                        top = np.argsort(q[1] - q[0])[::-1][:needed]
                        for i in top:
                            nxt = scored[i]
                            support.append(nxt)
                            pool.remove(nxt)
                    evaluate(len(support))

                if rows:
                    append_rows(out_file, rows)
                    print(f"[sample-selection] {task} rep={rep} {strategy}: "
                          f"K=20 r2={rows[-1]['r2']:.3f} "
                          f"({sum(r['seconds'] for r in rows):.0f}s)", flush=True)


if __name__ == "__main__":
    main()
