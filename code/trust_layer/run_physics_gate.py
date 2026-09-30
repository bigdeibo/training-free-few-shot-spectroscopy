"""A physics-based deployment gate, and its two survival controls.

The score. Beer-Lambert additivity says the prediction for a mixture should be
the concentration-weighted prediction of its parts. For a query spectrum x, its
same-task support references {x_a} and mixing fractions c in {0.3, 0.5, 0.7},

    residual(x; a, c) = |y(c*x_a + (1-c)*x) - (c*y(x_a) + (1-c)*y(x))| / sd(y)

and the gate score is the mean residual over the references and the fractions.
Because a query that the model cannot extrapolate to breaks the relation, a
high score marks a deployment the model should not be trusted on. The gate is a
property of the spectra and the model, never of the labels, so no label
information enters the score.

Controls. Two are mandatory here and are the reason the table has three AUC
columns rather than one. The `random-projections-100` representation repeats the
whole experiment on a projection with no spectral structure: the gate should
lose its ranking power there, which is what separates a physics score from a
generic one. The random column is a floor. And two task families are expected to
score high throughout, the nonlinearly blending diesel properties (flash point,
viscosity) and the oxidation-product control, which is how the paper establishes
that the gate is domain-selective rather than uniformly informative.

What is measured. Per deployment, the risk-coverage AUC of the gate score
against that of the native 80% interval width; risk is 1 - R2 on the best
fraction of queries kept. Lower is better, since a good score ranks the bad
predictions last. Width is the competitor that needs no physics at all.

Output. `results/physics-gate/physics-gate__<representation>__<task>.csv`,
columns task, support_size, repetition, representation, auc_physics_gate,
auc_interval_width, auc_random_control, mean_physics_gate_score, r2_evaluation,
catastrophic_failure, seconds. Reruns skip rows already present.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.features import corpus_pca, random_projection, standardize
from common.io import already_done, append_rows
from common.metrics import r2_score
from common.tasks import eval_split, load_task, tasks_except, zscore_fit

REPRESENTATIONS = ["corpus-pca-100", "random-projections-100"]
N_COMPONENTS = 100
MIXING_FRACTIONS = (0.3, 0.5, 0.7)
MAX_REFERENCES = 4
QUANTILES = (0.10, 0.50, 0.90)
RISK_FRACTIONS = np.linspace(0.5, 1.0, 11)

# The in-house pentane/CCl4 series has ten physical samples, so a support set of
# ten distinct samples does not exist for it.
EXCLUDED = ("pentane-ccl4-volume-fraction",)


def risk_coverage_auc(score, y, pred, fractions=RISK_FRACTIONS):
    """Area under the risk-coverage curve.

    Queries are ranked by ascending score and the risk 1 - R2 is evaluated on
    the best fraction kept, over fractions from 0.5 to 1.0. A score that puts
    the bad predictions last has a low area.
    """
    order = np.argsort(score)
    risks = []
    for f in fractions:
        keep = order[: max(2, int(round(len(y) * f)))]
        risks.append(1 - r2_score(pred[keep], y[keep]))
    return float(np.trapezoid(risks, fractions))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--representation", default="corpus-pca-100",
                    choices=REPRESENTATIONS)
    ap.add_argument("--task", default="all", help="one task slug, or 'all'")
    ap.add_argument("--repetitions", type=int, default=10)
    ap.add_argument("--max-queries", type=int, default=100,
                    help="cap on queries scored per repetition, since every "
                         "query costs a batch of mixture predictions")
    args = ap.parse_args()

    import torch
    from common.paths import RESULTS, tabpfn_weights
    from tabpfn import TabPFNRegressor

    if args.task == "all":
        tasks = tasks_except(*EXCLUDED)
    else:
        tasks = [args.task]
    out_dir = RESULTS / "physics-gate"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNRegressor(model_path=str(tabpfn_weights("v3")), device=device)
    print(f"[physics-gate] representation={args.representation} "
          f"device={device}", flush=True)

    for task in tasks:
        d = load_task(task)
        X = np.asarray(d["X"])
        y = np.asarray(d["y"], dtype=np.float64)
        n = len(y)
        F = (corpus_pca(X, N_COMPONENTS) if args.representation.startswith("corpus")
             else random_projection(X, N_COMPONENTS))

        out_file = out_dir / (f"physics-gate__{args.representation}__"
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
                if len(te) < 8:
                    continue
                t0 = time.time()
                ym, ysd = zscore_fit(y[tr])
                Fs, _ = standardize(F[tr], F[tr])
                model.fit(Fs, (y[tr] - ym) / ysd)

                # score a seeded subset of the query set: every query costs one
                # batch of mixture predictions, so the cost is linear in queries
                rng = np.random.default_rng(55000 + K * 100 + rep)
                n_query = min(args.max_queries, len(te))
                query_idx = rng.choice(len(te), size=n_query, replace=False)
                Fq, yt = F[te][query_idx], y[te][query_idx]
                Fq, _ = standardize(F[tr], F[te][query_idx])

                q = np.asarray(model.predict(Fq, output_type="quantiles",
                                             quantiles=list(QUANTILES)))
                median = q[1] * ysd + ym
                width80 = (q[2] - q[0]) * ysd

                # additivity residual, measured in feature space. Mixing is
                # exact for a linear projection, so a non-zero residual is the
                # model's, not the projection's.
                n_ref = min(K, MAX_REFERENCES)
                ref = rng.choice(K, size=n_ref, replace=False)
                F_ref = Fs[ref]
                mixes = [c * F_ref[None] + (1 - c) * Fq[:, None]
                         for c in MIXING_FRACTIONS]
                joint = np.concatenate([Fs] + [m.reshape(-1, Fs.shape[1])
                                               for m in mixes])
                predicted = model.predict(joint) * ysd + ym
                predicted_support, predicted_mix = predicted[:K], predicted[K:]
                predicted_ref = predicted_support[ref]

                residual = np.zeros(n_query)
                block = n_query * n_ref
                for i, c in enumerate(MIXING_FRACTIONS):
                    block_i = predicted_mix[i * block:(i + 1) * block]
                    linear = c * predicted_ref[None] + (1 - c) * median[:, None]
                    residual += np.abs(block_i.reshape(n_query, n_ref)
                                       - linear).mean(axis=1)
                gate = residual / len(MIXING_FRACTIONS) / ysd

                r2_eval = r2_score(median, yt)
                rows.append(dict(
                    task=task, support_size=K, repetition=rep,
                    representation=args.representation,
                    auc_physics_gate=round(
                        risk_coverage_auc(gate, yt, median), 4),
                    auc_interval_width=round(
                        risk_coverage_auc(width80, yt, median), 4),
                    auc_random_control=round(
                        risk_coverage_auc(rng.random(n_query), yt, median), 4),
                    mean_physics_gate_score=round(float(gate.mean()), 4),
                    r2_evaluation=round(r2_eval, 4),
                    catastrophic_failure=int(r2_eval < -1),
                    seconds=round(time.time() - t0, 2)))
                print(f"[physics-gate:{args.representation}] {task} K={K} "
                      f"rep={rep}: auc gate={rows[-1]['auc_physics_gate']:.3f} "
                      f"width={rows[-1]['auc_interval_width']:.3f} "
                      f"({rows[-1]['seconds']}s)", flush=True)
        if rows:
            append_rows(out_file, rows)


if __name__ == "__main__":
    main()
