"""How far down in adulteration level can the stack still see a difference?

Quantification on the olive-oil task is near zero at these support sizes, which
is reported as it stands. This analysis asks the weaker and more
deployment-relevant question: at what adulteration level is adulteration
detected at all, as a function of the label budget?

The blank is the pure olive oil in the query set. Thresholds follow the usual
IUPAC convention on the predicted level, blank mean plus 3.3 standard deviations
for the limit of detection and plus 10 for the limit of quantification, and the
detection rate at a level is the fraction of that level's query spectra whose
prediction exceeds the threshold. Splits are group-aware and identical to the
rest of the paper's olive-oil numbers.

Output. `results/detection-limit/detection-limit__olive-oil-adulteration.csv`,
columns support_size, repetition, adulteration_level, n_test_spectra,
detection_rate_at_lod, detection_rate_at_loq, median_prediction, blank_mean,
blank_sd, r2, seconds. Reruns skip rows already present.
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
from common.tasks import eval_split, load_task, zscore_fit

TASK = "olive-oil-adulteration"
N_COMPONENTS = 100
SUPPORT_SIZES = (5, 10, 20)
LEVELS = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 100.0]
LOD_FACTOR, LOQ_FACTOR = 3.3, 10.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repetitions", type=int, default=10)
    args = ap.parse_args()

    import torch
    from common.paths import RESULTS, tabpfn_weights
    from tabpfn import TabPFNRegressor

    out_file = (RESULTS / "detection-limit" /
                "detection-limit__olive-oil-adulteration.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNRegressor(model_path=str(tabpfn_weights("v3")), device=device)
    print(f"[detection-limit] device={device}", flush=True)

    d = load_task(TASK)
    X = np.asarray(d["X"], dtype=float)
    y = np.asarray(d["y"], dtype=float)
    groups = d["groups"]
    F = corpus_pca(X, N_COMPONENTS)

    done = already_done(out_file, ("support_size", "repetition",
                                   "adulteration_level"))
    for K in SUPPORT_SIZES:
        rows = []
        for rep in range(args.repetitions):
            if all((K, rep, level) in done for level in LEVELS):
                continue
            tr, te = eval_split(len(y), K, rep, groups=groups)
            ym, ysd = zscore_fit(y[tr])
            Fs, Fq = standardize(F[tr], F[te])
            t0 = time.time()
            model.fit(Fs, (y[tr] - ym) / ysd)
            pred = model.predict(Fq) * ysd + ym
            yt = y[te]
            blank = pred[yt == 0.0]
            if len(blank) < 5:
                print(f"[detection-limit] K={K} rep={rep}: no blank query "
                      f"spectra, skipped", flush=True)
                continue
            lod = blank.mean() + LOD_FACTOR * blank.std()
            loq = blank.mean() + LOQ_FACTOR * blank.std()
            for level in LEVELS:
                if (K, rep, level) in done:
                    continue
                at_level = yt == level
                if at_level.sum() == 0:
                    continue
                rows.append(dict(
                    support_size=K, repetition=rep, adulteration_level=level,
                    n_test_spectra=int(at_level.sum()),
                    detection_rate_at_lod=float((pred[at_level] > lod).mean()),
                    detection_rate_at_loq=float((pred[at_level] > loq).mean()),
                    median_prediction=float(np.median(pred[at_level])),
                    blank_mean=float(blank.mean()),
                    blank_sd=float(blank.std()),
                    r2=round(r2_score(pred, yt), 4),
                    seconds=round(time.time() - t0, 2)))
            print(f"[detection-limit] K={K} rep={rep} done", flush=True)
        if rows:
            append_rows(out_file, rows)


if __name__ == "__main__":
    main()
