"""Digitise the reference Raman plots into intensity arrays.

The simulated corpus is built from a library of eleven pure components — the
paraffins, aromatics and oxygenates that dominate commercial gasoline. The
library was assembled from published reference Raman plots rather than from
measured standards, and this script is the transcription step: each plot is
reduced to a curve, and the curve is what the peak table in
`data/simulated-fuel/peaks.csv` was read off.

The plots are not redistributed here, so the script cannot run from this
package alone. Point `--source` at a directory of the reference plots and it
regenerates the arrays under `data/simulated-fuel/digitized/`; the digitised
arrays themselves are shipped, and they are what the shipped peak table came
from, so nothing downstream needs this step to be repeated.

Reading the plots. The frame is found as the longest dark horizontal and
vertical runs, because text labels sit outside it. The horizontal calibration
comes from the tick marks below the frame, which are spaced a fixed number of
wavenumbers apart, so the axis is recovered in wavenumbers per pixel without
reading any label. The curve is then sampled column by column, taking the
topmost dark pixel inside the frame, which is the apex where peaks are steep.
The intensity axis runs from zero at the baseline to a hundred at the top of
the frame.

One artefact is left in the arrays as written. The frame begins a little to the
left of the axis origin, so the first few hundred wavenumbers of a digitised
trace extrapolate to negative values, which are not physical. The simulation
reads `data/simulated-fuel/peaks.csv` instead, whose ninety-nine entries are all
inside the plotted range, so the extrapolated tail is never used; a reader who
takes `data/simulated-fuel/digitized/` directly should drop the points below the
axis minimum.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DARK = 128          # a pixel below this counts as curve, frame or label
X_LEFT = 4000.0     # wavenumber at the left frame edge, the usual convention


def frame_bounds(g):
    """Left, right, top and bottom of the plot frame, from its long dark runs."""
    rows_dark = (g < DARK).sum(axis=1)
    cols_dark = (g < DARK).sum(axis=0)
    h = np.where(rows_dark > 0.5 * g.shape[1])[0]
    v = np.where(cols_dark > 0.4 * g.shape[0])[0]
    if len(h) < 2 or len(v) < 2:
        raise RuntimeError("no plot frame found; is this the right image?")
    return v.min(), v.max(), h.min(), h.max()


def wavenumbers_per_pixel(g, left, bottom, tick_spacing):
    """Ticks below the frame give the axis scale in wavenumbers per pixel."""
    strip = g[bottom + 1: bottom + 8, :] < DARK
    projection = strip.sum(axis=0)
    columns = [x for x in range(len(projection)) if projection[x] >= 3]
    ticks = []
    for x in columns:
        if ticks and x - ticks[-1] <= 3:
            ticks[-1] = x          # the same tick, smeared over a few pixels
        else:
            ticks.append(x)
    ticks = [t for t in ticks if t >= left - 2]
    spacings = np.diff(ticks)
    if len(spacings) < 2:
        raise RuntimeError("too few ticks to calibrate the wavenumber axis")
    return float(np.median(spacings)) / tick_spacing


def digitize(path, tick_spacing):
    g = np.asarray(Image.open(path).convert("L")).astype(float)
    left, right, top, bottom = frame_bounds(g)
    per_cm = wavenumbers_per_pixel(g, left, bottom, tick_spacing)
    inner = g[top + 2: bottom - 1, left + 2: right - 1]
    columns, rows = [], []
    for j in range(inner.shape[1]):
        dark = np.where(inner[:, j] < DARK)[0]
        if len(dark):
            columns.append(j)
            rows.append(dark.min())
    columns = np.array(columns) + left + 2
    rows = np.array(rows) + top + 2
    wavenumber = X_LEFT - (columns - left) / per_cm
    intensity = (bottom - rows) / (bottom - top) * 100.0
    order = np.argsort(wavenumber)
    return (wavenumber[order], np.clip(intensity[order], 0, 100),
            (left, right, top, bottom))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=None,
                    help="directory of reference plots; defaults to "
                         "$SPEC_DATA_ROOT/sdbs-raman-plots")
    ap.add_argument("--out", default=None,
                    help="output directory; defaults to "
                         "data/simulated-fuel/digitized")
    ap.add_argument("--tick-spacing", type=float, default=1000.0,
                    help="wavenumbers between major ticks, as printed on the "
                         "plots (default 1000)")
    ap.add_argument("--force", action="store_true",
                    help="redigitise components whose array is already present; "
                         "without it those are left as they are")
    args = ap.parse_args()

    from common.paths import DATA, DATA_ROOT
    source = Path(args.source) if args.source else DATA_ROOT / "sdbs-raman-plots"
    out = Path(args.out) if args.out else DATA / "simulated-fuel" / "digitized"
    if not source.is_dir():
        raise SystemExit(
            f"reference plots not found at {source}. They are third-party "
            f"images and are not redistributed with this package; the "
            f"digitised arrays are shipped, so this step is only needed if you "
            f"want to redo the transcription. Pass --source to point at a "
            f"directory of plots.")
    out.mkdir(parents=True, exist_ok=True)

    plots = sorted(glob.glob(str(source / "*.gif")) +
                   glob.glob(str(source / "*.png")))
    if not plots:
        raise SystemExit(f"no .gif or .png plots under {source}")
    written = skipped = 0
    for path in plots:
        name = Path(path).stem
        target = out / f"{name}.npz"
        if target.exists() and not args.force:
            print(f"{name}: already present, left as it is")
            skipped += 1
            continue
        wavenumber, intensity, frame = digitize(path, args.tick_spacing)
        np.savez(target, wn=wavenumber, y=intensity,
                 frame=np.array(frame))
        written += 1
        print(f"{name}: {len(wavenumber)} points, "
              f"{wavenumber.min():.0f}-{wavenumber.max():.0f} cm^-1, "
              f"peak height {intensity.max():.0f}")
    print(f"{written} components written to {out}"
          + (f", {skipped} already present" if skipped else ""))


if __name__ == "__main__":
    main()
