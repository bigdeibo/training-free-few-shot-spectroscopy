"""Retrieve the public datasets into the layout the loaders expect.

Three of the seven task datasets can be fetched without an account and are
fetched here; the rest are listed at the end with the exact command or browser
step, because they need a registration, a git clone, an R package, or a
record-by-record download.

The script is written against each depositor's own machine-readable index rather
than against hard-coded file lists, so it keeps working when a deposit is
re-released with files renamed. The index calls could not be exercised from the
environment that assembled this package, which had no network access; the
Mendeley file identifiers come from the download manifest that accompanied the
data and are the one part known to be exact. Where a fetch fails, the printed
manual steps place the same files by hand.

Usage
    python data/sources/fetch_public_data.py --what all
    python data/sources/fetch_public_data.py --what edible-oil --data-root /path/to/spectra

Nothing is overwritten unless `--force` is given.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

USER_AGENT = "spectroscopy-package/1.0 (data retrieval)"
TIMEOUT = 120
CHUNK = 1 << 20

# ---------------------------------------------------------------- mendeley
# Edible-oil NIR, MIR and Raman, one file identifier per file. The class key is
# small and travels with the spectra.
MENDELEY = {
    "NIR24mm1A.csv":
        "https://data.mendeley.com/public-files/datasets/ctgg7k4m5g/files/"
        "3c449dd2-8699-44db-b6c5-d3c940e181a3/file_downloaded",
    "MIR1A.csv":
        "https://data.mendeley.com/public-files/datasets/ctgg7k4m5g/files/"
        "c9eb515e-4a4e-4882-9b33-88e1d1c7fbf7/file_downloaded",
    "Raman1A.csv":
        "https://data.mendeley.com/public-files/datasets/ctgg7k4m5g/files/"
        "abf49aee-c5d7-4a55-aec4-b59eea61affb/file_downloaded",
    "OilClassKey.csv":
        "https://data.mendeley.com/public-files/datasets/ctgg7k4m5g/files/"
        "dfcd5aa4-9043-4cab-859f-d0bacfc76c97/file_downloaded",
}

# ---------------------------------------------------------------- zenodo
SUGAR_RECORD = "10779223"          # the Raman sugar-mixture measurement data
# ---------------------------------------------------------------- hugging face
FUEL_DATASETS = {"Benchtop": "chlange/FuelRamanSpectraBenchtop",
                 "Handheld": "HTW-KI-Werkstatt/FuelRamanSpectraHandheld"}

MANUAL = """
Four items cannot be fetched by a script.

1. SWRI diesel and Cargill corn, from the Eigenvector Research data archive.
   Both downloads sit behind a free registration.
     https://eigenvector.com/data/SWRI/   -> diesel_spec.csv, diesel_prop.csv
     https://eigenvector.com/data/Corn/   -> corn.mat
   Place them as:
     {root}/SWRI_Diesel_NIR/diesel_spec.csv
     {root}/SWRI_Diesel_NIR/diesel_prop.csv
     {root}/Corn_Cargill/corn.mat

2. EVOO adulteration NIR-HSI, from a git repository.
     git clone https://github.com/DNMalavi/NIR-HSI-ML-for-EVOO-Fraud-Detection
     mkdir -p {root}/EVOO_NIR_HSI/data
     mv NIR-HSI-ML-for-EVOO-Fraud-Detection/Data/Raw_A.xlsx {root}/EVOO_NIR_HSI/data/
   The version used in the paper is commit ac885ed.

3. Kalivas gasoline and mayonnaise, from the R package `pls`.
     pip install pyreadr
     python data/sources/build_derived_arrays.py
   This writes {root}/processed/gasoline_arrays.npz for the gasoline task and
   {root}/processed/mayonnaise_arrays.npz for the corpus mixture library. Both
   are needed: without the mayonnaise arrays the corpus cannot be assembled.

4. The NIST infrared reference collection, one record at a time.
     python data/sources/fetch_nist_ir.py
   This writes {root}/NIST_IR/nist_ir.npz, which the corpus mixture library also
   needs. It takes a few minutes, and the delay between requests is deliberate.

With all four in place, `python code/corpus_augmentation/run_build_corpus.py`
assembles the corpus every projection is fitted on.
"""


def _get(url, binary=True):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    response = urllib.request.urlopen(request, timeout=TIMEOUT)
    return response.read() if binary else response.read().decode("utf-8")


def download(url, dest, force=False):
    """Write one URL to `dest`, unless it is already there."""
    if dest.exists() and not force:
        print(f"  have     {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {dest.name} ...", end="", flush=True)
    try:
        data = _get(url)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print(f" failed ({exc})")
        raise
    dest.write_bytes(data)
    print(f" {len(data) / 1e6:.1f} MB")
    return dest


def fetch_edible_oil(root, force=False):
    """The four Mendeley files, by their published file identifiers."""
    out = root / "EdibleOil_NIR_MIR"
    for name, url in MENDELEY.items():
        download(url, out / name, force=force)
    return out


def fetch_sugar_raman(root, force=False):
    """The Zenodo deposit, unpacked into the two signal-to-noise folders.

    The record holds one archive per scenario rather than loose files, so the
    archive is extracted and the scenario folders are placed by name. The
    deposit's own file names were not verifiable when this script was written,
    hence the printed summary: check it against the page if the layout is
    reported as unexpected.
    """
    out = root / "sugar_raman"
    staging = out / "_download"
    record = json.loads(_get(
        f"https://zenodo.org/api/records/{SUGAR_RECORD}", binary=False))
    files = record.get("files", [])
    if not files:
        raise SystemExit(f"record {SUGAR_RECORD} lists no files")
    for entry in files:
        key = entry.get("key") or entry.get("filename")
        link = (entry.get("links", {}).get("self")
                or entry.get("links", {}).get("download"))
        download(link, staging / key, force=force)
    for archive in sorted(staging.glob("*.zip")):
        target = staging / archive.stem
        if not target.exists():
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)

    def candidate(scenario, needed):
        """The first `needed` file whose directory names this scenario."""
        token = scenario.split()[0].lower()          # "high", "low"
        hits = [p for p in staging.rglob(needed)
                if token in p.parent.name.lower()]
        return hits[0] if hits else None

    placed = []
    for scenario in ("High SNR", "Low SNR"):
        dest = out / scenario
        dest.mkdir(parents=True, exist_ok=True)
        for needed in ("data.pkl", "metadata.csv"):
            found = candidate(scenario, needed)
            if found:
                (dest / needed).write_bytes(found.read_bytes())
                placed.append(dest / needed)
            else:
                print(f"  WARNING  no {needed} for {scenario} in the deposit; "
                      f"place it by hand from {staging}")
    return placed or out


def fetch_fuel_raman(root, force=False):
    """Both gasoline Raman instruments, from their Hugging Face repositories."""
    written = []
    for instrument, repo in FUEL_DATASETS.items():
        out = root / "fuel_raman" / f"FuelRamanSpectra{instrument}"
        index = json.loads(_get(
            f"https://huggingface.co/api/datasets/{repo}", binary=False))
        names = [s.get("rfilename") for s in index.get("siblings", [])]
        wanted = [n for n in names if n.endswith((".parquet", ".csv", ".md"))]
        if not wanted:
            raise SystemExit(f"{repo} lists no data files")
        for name in wanted:
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{name}"
            written.append(download(url, out / name, force=force))
    return written


FETCHERS = {"edible-oil": fetch_edible_oil,
            "sugar-raman": fetch_sugar_raman,
            "fuel-raman": fetch_fuel_raman}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--what", default="all",
                    choices=["all"] + sorted(FETCHERS),
                    help="which dataset to fetch (default: all that can be)")
    ap.add_argument("--data-root", default=None,
                    help="target directory; defaults to $SPEC_DATA_ROOT, or "
                         "the package's data/public/ if that is unset")
    ap.add_argument("--force", action="store_true",
                    help="overwrite files that are already present")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    if args.data_root:
        root = Path(args.data_root)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
        from common.paths import DATA_ROOT
        root = DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    print(f"data root: {root}")

    names = sorted(FETCHERS) if args.what == "all" else [args.what]
    failed = []
    for name in names:
        print(f"[{name}]")
        try:
            FETCHERS[name](root, force=args.force)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                SystemExit) as exc:
            print(f"  could not fetch: {exc}")
            print(f"  place this dataset by hand (see below)")
            failed.append(name)
    print(MANUAL.format(root=root))
    if failed:
        # The manual steps above place the same files by hand, so this is not a
        # fatal error for a reader; the non-zero status is so that a script
        # wrapping this one does not treat a partial fetch as complete.
        print(f"fetched with gaps: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
