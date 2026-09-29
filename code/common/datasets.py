"""One loader per dataset, all returning the same dictionary layout.

    X           (n_samples, n_points) spectral matrix
    wavelengths (n_points,) or None
    targets     {property_name: (n_samples,)} values, possibly with NaN
    ids         sample identifiers
    groups      optional; physical-sample identifier for group-aware splitting

Two data roots are in play.

`PACKAGE_DATA` holds the artifacts this work produced: the in-house pentane/CCl4
series, the encoder checkpoints and the derived edible-oil table. They ship with
the archive.

`DATA_ROOT` holds the third-party public datasets. They are not redistributed
here; `data/sources/` lists each one, its licence and a fetch script that places
it at the path the loader below expects. Set the `SPEC_DATA_ROOT` environment
variable to point somewhere else, or edit the default.

Reading these files is the only place the raw public formats are handled; every
quirk noted in a docstring below is a property of the source file, not a choice
made here.

`wavelengths` is None for the two Raman datasets, whose axes are not needed by
anything in this work: the Raman arms enter as they were measured, without the
resampling the near-infrared tasks go through.
"""
import json

import numpy as np
import pandas as pd

from common.paths import DATA as PACKAGE_DATA
from common.paths import DATA_ROOT

# Replicate scans per oil in the edible-oil design; see load_edible_oil.
_EXPECTED_REPS = 3


def _read_eigenvector_csv(path):
    """Parse an Eigenvector CSV export, whose leading rows are metadata."""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            rows.append(line.rstrip("\n").split(","))
    return rows


def _clean_id(s):
    return s.strip().strip('"').strip()


def load_diesel():
    """SWRI diesel NIR, seven property labels.

    The export is two headered CSVs. The spectrum file carries its wavelength
    axis on a row labelled `Axisscale`; the property file is joined to it on the
    sample identifier.
    """
    base = DATA_ROOT / "SWRI_Diesel_NIR"
    spec_rows = _read_eigenvector_csv(base / "diesel_spec.csv")
    prop_rows = _read_eigenvector_csv(base / "diesel_prop.csv")

    axis, data_start = None, None
    for i, r in enumerate(spec_rows):
        if r and _clean_id(r[0]) == "Axisscale":
            axis = np.array([float(x) for x in r[2:] if x.strip()])
            data_start = i + 1
            break
    n_wl = len(axis)
    ids, X = [], []
    for r in spec_rows[data_start:]:
        vals = [x for x in r[2:2 + n_wl]]
        if len(vals) < n_wl or not _clean_id(r[1]):
            continue
        try:
            X.append([float(x) if x.strip() else np.nan for x in vals])
            ids.append(_clean_id(r[1]))
        except ValueError:
            continue
    X = np.array(X, dtype=float)

    header_i = None
    for i, r in enumerate(prop_rows):
        if _clean_id(r[0]) == "Label" and any("CN" == _clean_id(c) for c in r):
            header_i = i
            break
    header = [_clean_id(c) for c in prop_rows[header_i]]
    prop_names = [h for h in header[2:] if h]
    pids, P = [], []
    for r in prop_rows[header_i + 1:]:
        if len(r) < 3 or not _clean_id(r[1]):
            continue
        pids.append(_clean_id(r[1]))
        row = []
        for x in r[2:2 + len(prop_names)]:
            try:
                row.append(float(x.strip()))
            except ValueError:
                row.append(np.nan)
        P.append(row)
    P = np.array(P, dtype=float)

    pos = {pid: j for j, pid in enumerate(pids)}
    targets = {}
    for k, name in enumerate(prop_names):
        t = np.full(len(ids), np.nan)
        for i, sid in enumerate(ids):
            if sid in pos:
                t[i] = P[pos[sid], k]
        targets[name] = t
    return dict(X=X, wavelengths=axis, targets=targets, ids=ids,
                meta=dict(name="SWRI_Diesel_NIR", source="eigenvector.com",
                          desc="Diesel fuel NIR, SWRI"))


def load_gasoline():
    """Kalivas gasoline NIR with octane number.

    60 spectra, 900-1700 nm at 2 nm, unpacked from the R `pls` package into an
    npz. The stored matrix is column-major, hence the Fortran-order reshape when
    the row count does not already match.
    """
    z = np.load(DATA_ROOT / "processed" / "gasoline_arrays.npz")
    octane = z["arr_0"].astype(float)
    nir = z["arr_1"].astype(float)
    if nir.shape[0] != 60:
        nir = nir.reshape(60, 401, order="F")
    wl = 900 + 2 * np.arange(nir.shape[1])
    return dict(X=nir, wavelengths=wl.astype(float), targets=dict(octane=octane),
                ids=[f"gas{i+1}" for i in range(len(octane))],
                meta=dict(name="Kalivas_Gasoline", source="R pls package",
                          desc="Gasoline NIR 900-1700 nm, octane number"))


def load_mayonnaise():
    """Mayonnaise NIR with a six-class oil-type label and no quantitative target.

    Unlabeled for the purposes of this work: it enters the corpus only.
    """
    z = np.load(DATA_ROOT / "processed" / "mayonnaise_arrays.npz")
    nir = z["arr_0"].astype(float)
    wl = 1100 + 4 * np.arange(nir.shape[1], dtype=float)
    return dict(X=nir, wavelengths=wl, labels=z["arr_1"].astype(int), targets={},
                ids=[f"mayo{i+1}" for i in range(nir.shape[0])],
                meta=dict(name="Mayonnaise", source="R pls package",
                          desc="Mayonnaise NIR, 6-class oil type"))


def _dataset_struct_data(struct):
    """Pull the data array and the axis out of an Eigenvector `DataSet` struct.

    The MATLAB struct stores its axis in a nested cell array, and not every cell
    holds a vector of the right length, so the axis is picked by matching length
    against the data.
    """
    s = struct[0, 0] if isinstance(struct, np.ndarray) else struct
    data = np.asarray(getattr(s, "data"), dtype=float)
    axis = np.array([], dtype=float)
    try:
        raw = np.asarray(getattr(s, "axisscale", []), dtype=object).ravel()
        for elem in raw:
            e = (np.asarray(elem, dtype=float).ravel() if not np.isscalar(elem)
                 else np.array([float(elem)]))
            if e.size == data.shape[1]:
                axis = e
                break
    except (ValueError, TypeError):
        pass
    if axis.size != data.shape[1]:
        axis = np.arange(data.shape[1], dtype=float)
    return data, axis


def load_corn():
    """Cargill corn NIR on three instruments, four composition traits.

    The file is an Eigenvector `DataSet` struct and needs `struct_as_record=False`
    to be readable as attributes. The m5 instrument supplies the benchmark tasks;
    mp5 and mp6 are the two instruments deliberately held out of the downstream
    benchmark and used only to build the corpus.
    """
    import scipy.io
    m = scipy.io.loadmat(DATA_ROOT / "Corn_Cargill" / "corn.mat",
                         struct_as_record=False, squeeze_me=False)
    out = {}
    for inst in ["m5", "mp5", "mp6"]:
        X, axis = _dataset_struct_data(m[f"{inst}spec"])
        if axis.size == X.shape[1] and axis[0] < 100:
            axis = 1100 + 2 * np.arange(X.shape[1], dtype=float)
        out[inst] = (X, axis)
    P, _ = _dataset_struct_data(m["propvals"])
    names = ["moisture", "oil", "protein", "starch"]
    targets = {names[i]: P[:, i].astype(float) for i in range(min(4, P.shape[1]))}
    return dict(instruments=out, targets=targets,
                ids=[f"corn{i+1}" for i in range(P.shape[0])],
                meta=dict(name="Cargill_Corn_3inst", source="eigenvector.com",
                          desc="Corn NIR on 3 instruments, four composition traits"))


def load_evoo():
    """EVOO adulteration NIR-HSI, one row per mean spectrum.

    The 1995 spectra come from 641 physical samples; `ids` carries the sample
    identifier that `eval_split` uses to keep replicates on one side of the split.
    """
    df = pd.read_excel(DATA_ROOT / "EVOO_NIR_HSI" / "data" / "Raw_A.xlsx")
    spec_cols = [c for c in df.columns if str(c).replace(".", "").isdigit()]
    X = df[spec_cols].to_numpy(dtype=float)
    wl = np.array([float(c) for c in spec_cols])
    targets = dict(adulteration_level=pd.to_numeric(
        df["adulteration_level"], errors="coerce").to_numpy(dtype=float))
    return dict(X=X, wavelengths=wl, targets=targets,
                labels=df["class_1"].astype(str).to_numpy(),
                ids=df["sample_id"].astype(str).tolist(),
                meta=dict(name="EVOO_NIR_HSI", source="github.com/DNMalavi",
                          desc="EVOO adulteration NIR-HSI mean spectra"))


def load_pentane_ccl4():
    """The in-house FT-NIR n-pentane/CCl4 series, measured for this work.

    Collected on an ABB MB3600 over 800-2500 nm. The target `phi_pentane` is the
    n-pentane volume fraction as prepared (parts per 250). Spectra sharing a
    bottle identifier are replicate scans of one physical sample, so `groups`
    carries the bottle and `eval_split` keeps a bottle whole. The spectra are raw
    absorbance, not SNV-corrected; see `common/preprocess.py`.

    This dataset ships with the package, so it is read from `PACKAGE_DATA`
    rather than from the third-party root.
    """
    z = np.load(PACKAGE_DATA / "in-house-pentane-ccl4" / "pentane_ccl4_FTIR.npz",
                allow_pickle=True)
    return dict(
        X=z["X"].astype(float),
        wavelengths=z["wavelength_nm"].astype(float),
        targets=dict(phi_pentane=z["y"].astype(float)),
        ids=[str(s) for s in z["name"]],
        groups=[str(s) for s in z["sample_id"]],
        rep=z["rep"].astype(int), temp_c=z["temp_c"].astype(int),
        meta=json.loads(str(z["meta"])) if "meta" in z
        else dict(name="pentane_ccl4_FTIR"),
    )


def load_edible_oil():
    """Edible-oil FT-NIR peroxide value (Mendeley 10.17632/ctgg7k4m5g.2).

    The source table `NIR24mm1A.csv` holds 300 rows, being 100 oils measured in
    triplicate, of which two rows are all-NaN padding and belong to the same oil.
    That oil is left with a single scan, which is a degenerate unit for a
    group-aware split: its "remaining replicates are excluded from test"
    guarantee is empty, and whether it lands in the support set or not changes
    the number of test spectra between repetitions. It is dropped, leaving 99
    oils with all three replicates and 297 spectra, and a constant 297 - 3K test
    set. The paper's dataset table reports the source file's own count of 300.

    The axis in the source is in wavenumbers 3799-14999 cm^-1, which spans
    667-2632 nm and so covers every C-H overtone and combination band. It is
    converted to nm here, in ascending order, so that the band masks used
    elsewhere in this work land on the bands they name.

    The wide table (11618 columns) is slow to parse and prone to memory pressure,
    so the derived artifact is cached as an npz and shipped with the package;
    only the small npz is read afterwards.
    """
    cache = PACKAGE_DATA / "derived" / "edible-oil-nir.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return dict(X=z["X"].astype(float), wavelengths=z["wl"].astype(float),
                    targets=dict(peroxide_value=z["y"].astype(float)),
                    ids=[str(s) for s in z["ids"]],
                    groups=[str(s) for s in z["groups"]],
                    meta=dict(name="edible-oil-ft-nir",
                              source="Mendeley 10.17632/ctgg7k4m5g.2",
                              desc="Edible oil FT-NIR 667-2632 nm, peroxide "
                                   "value, 99 oils x 3 replicates"))

    raw = pd.read_csv(DATA_ROOT / "EdibleOil_NIR_MIR" / "NIR24mm1A.csv",
                      header=None, low_memory=False)
    wavenum = np.array([float(x) for x in raw.iloc[0, 2:]], dtype=float)
    nm = 1e7 / wavenum
    body = raw.iloc[1:].reset_index(drop=True)
    cls = pd.to_numeric(body.iloc[:, 0], errors="coerce").to_numpy()
    pv = pd.to_numeric(body.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    spec = body.iloc[:, 2:].to_numpy(dtype=float)
    order = np.argsort(nm)
    nm, spec = nm[order], spec[:, order]

    m = ~np.isnan(pv) & np.isfinite(cls) & ~np.isnan(spec).any(axis=1)
    spec, pv, cls = spec[m], pv[m], cls[m]
    oil_id = np.array([f"c{int(c)}_pv{round(float(v), 3)}" for c, v in zip(cls, pv)])

    _uniq, _cnt = np.unique(oil_id, return_counts=True)
    keep = np.isin(oil_id, list(set(_uniq[_cnt == _EXPECTED_REPS])))
    spec, pv, cls, oil_id = spec[keep], pv[keep], cls[keep], oil_id[keep]
    assert len(np.unique(oil_id)) == 99 and keep.sum() == 297, (
        f"incomplete-oil filter changed: {keep.sum()} spectra over "
        f"{len(np.unique(oil_id))} oils")

    ids = [f"oil{int(c)}_{i}" for i, c in enumerate(cls)]
    np.savez(cache, X=spec.astype(np.float32), wl=nm.astype(np.float32),
             y=pv.astype(np.float32), ids=np.array(ids), groups=np.array(oil_id))
    return dict(X=spec.astype(float), wavelengths=nm.astype(float),
                targets=dict(peroxide_value=pv.astype(float)),
                ids=ids, groups=[str(g) for g in oil_id],
                meta=dict(name="edible-oil-ft-nir",
                          source="Mendeley 10.17632/ctgg7k4m5g.2",
                          desc="Edible oil FT-NIR 667-2632 nm, peroxide value, "
                               "99 oils x 3 replicates"))


def load_sugar_raman(snr="High SNR"):
    """Sugar mixtures by Raman spectroscopy, four sugars quantified together.

    Each well is measured about twenty times, so `groups` carries the well and
    the split keeps a well whole. The target is a volume fraction in microlitres
    of sugar over total volume; the four sugars are quantified separately, each
    on the spectra that carry a value for it.
    """
    import pickle
    base = DATA_ROOT / "sugar_raman" / snr
    with open(base / "data.pkl", "rb") as fh:
        X = pickle.load(fh).astype(float)
    meta = pd.read_csv(base / "metadata.csv")
    sugars = ["Sucrose", "Fructose", "Maltose", "Glucose"]
    fractions = (meta[[f"{s} [ul]" for s in sugars]].to_numpy(dtype=float)
                 / meta["Total Volume [ul]"].to_numpy(dtype=float)[:, None])
    return dict(X=X, wavelengths=None,
                targets={sugars[i]: fractions[:, i] for i in range(4)},
                ids=meta["samp"].astype(str).tolist(),
                groups=meta["samp"].astype(str).to_numpy(),
                meta=dict(name="sugar_raman", source="Georgiev et al., PNAS 2024",
                          desc=f"Sugar mixtures, {snr}"))


FUEL_TARGETS = {"RON": "Research Octane Number",
                "Ethanol": "Ethanol Content (%)",
                "Aromatics": "Aromatics Content",
                "Density": "Density at 15°C"}


def load_fuel_raman(instrument="Benchtop"):
    """Commercial gasoline by Raman spectroscopy, four quality properties.

    The source splits the spectra into train, validation and test parts; the
    three are concatenated, because the paper's own splits are drawn here rather
    than taken from the file. Two properties of the file are repaired on
    reading: a tail of columns is NaN for every row, and a few rows are entirely
    NaN. Both the columns and the rows are dropped.
    """
    base = DATA_ROOT / "fuel_raman" / f"FuelRamanSpectra{instrument}"
    df = pd.concat([pd.read_parquet(base / f"{part}.parquet")
                    for part in ("train", "val", "test")], ignore_index=True)
    spectra_cols = []
    for c in df.columns:
        try:
            float(c)
            spectra_cols.append(c)
        except ValueError:
            continue                    # target columns carry text headers
    X = df[spectra_cols].to_numpy(dtype=float)
    keep = ~np.isnan(X).any(axis=0)
    X = X[:, keep]
    alive = ~np.isnan(X).all(axis=1)
    X, df = X[alive], df[alive].reset_index(drop=True)
    return dict(X=X, wavelengths=np.array([float(c) for c in spectra_cols])[keep],
                targets={k: df[v].to_numpy(dtype=float)
                         for k, v in FUEL_TARGETS.items()},
                ids=[f"fuel{i + 1}" for i in range(len(X))],
                groups=np.arange(len(X)).astype(str),
                meta=dict(name="fuel_raman",
                          source="Voigt et al., Fuel 2019",
                          desc=f"Commercial gasoline, {instrument} Raman"))


def load_edible_oil_modalities():
    """The three modalities of the edible-oil set, paired on the same samples.

    Not a `LOADERS` entry: it returns three spectral matrices rather than one,
    and is used only by the fusion experiment.

    NIR is measured in triplicate, three consecutive rows per sample in
    sample-major order, and is averaged per sample here. MIR and Raman carry one
    row per sample. The three files are row-aligned by construction, and the
    alignment is checked rather than assumed. A sample whose NIR or Raman
    spectrum is entirely NaN is dropped from all three modalities, because no
    fusion arm can be formed for it.
    """
    base = DATA_ROOT / "EdibleOil_NIR_MIR"
    nir = pd.read_csv(base / "NIR24mm1A.csv")
    mir = pd.read_csv(base / "MIR1A.csv")
    raman = pd.read_csv(base / "Raman1A.csv")
    labels = ["Class", "PeroxideValue"]
    nir_rep = nir.drop(columns=labels).to_numpy(dtype=float)
    X_mir = mir.drop(columns=labels).to_numpy(dtype=float)
    X_raman = raman.drop(columns=labels).to_numpy(dtype=float)

    peroxide = mir["PeroxideValue"].to_numpy(dtype=float)
    assert np.allclose(raman["PeroxideValue"].to_numpy(dtype=float), peroxide), \
        "MIR and Raman rows are not aligned"
    assert np.allclose(nir["PeroxideValue"].to_numpy(dtype=float)[::3], peroxide), \
        "NIR replicate rows are not aligned with MIR and Raman"
    oil_class = mir["Class"].to_numpy().astype(int)

    dead = {i // _EXPECTED_REPS for i in
            np.where(np.isnan(nir_rep).any(axis=1))[0]}
    dead |= set(np.where(np.isnan(X_raman).any(axis=1))[0].tolist())
    alive = np.array([i for i in range(len(peroxide)) if i not in dead])

    n_samples = len(peroxide)
    X_nir = nir_rep.reshape(n_samples, _EXPECTED_REPS, -1).mean(axis=1)[alive]
    return dict(nir=X_nir, mir=X_mir[alive], raman=X_raman[alive],
                peroxide_value=peroxide[alive], oil_class=oil_class[alive],
                n_dropped=len(dead),
                meta=dict(name="edible-oil-ft-nir-mir-raman",
                          source="Mendeley 10.17632/ctgg7k4m5g.2"))


LOADERS = dict(diesel=load_diesel, gasoline=load_gasoline,
               mayonnaise=load_mayonnaise, corn=load_corn, evoo=load_evoo,
               pentane_ccl4=load_pentane_ccl4, edible_oil=load_edible_oil,
               sugar_raman=load_sugar_raman, fuel_raman=load_fuel_raman)
