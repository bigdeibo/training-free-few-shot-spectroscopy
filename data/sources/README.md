# Public datasets: where they come from and where to put them

This package does not redistribute the public datasets it was evaluated on. Two
artifacts derived from them do ship, and both are named below with the term that
permits them: a reduced arm of the edible-oil collection, and the simulated half
of the corpus. Every dataset itself is obtained from its original source under
its original terms, and this directory holds the source list and the retrieval
scripts for that step.

## Where the loaders look

Every loader in `code/common/datasets.py` reads the public datasets from one
directory, called the data root:

```
$SPEC_DATA_ROOT/            defaults to  <package>/data/public/
```

Set `SPEC_DATA_ROOT` in the environment, or place the datasets under
`data/public/`, which is where they are looked for by default. The directory is
not created here and is not expected to be under version control. The layout
inside it is fixed by the loaders and is given in the tables below.

Two artifacts this work produced are read from the package itself and need no
setup: `data/in-house-pentane-ccl4/`, the Fourier-transform near-infrared
(FT-NIR) series measured for this work, and `data/corpus/simulated_spectra.npy`,
the simulated half of the corpus.

The corpus that every projection is fitted on holds 6339 spectra and does not
ship as a whole. Once the datasets below are in place, run

```
python code/corpus_augmentation/run_build_corpus.py
```

which writes `data/corpus/spectra.npy` from the simulated half plus the measured
half. `data/corpus/recipe.json` records the library, the simulation parameters
and the digests of both halves, so the result can be checked.

## Main datasets

| Dataset | Tasks | Source | Terms | Place under `$SPEC_DATA_ROOT/` |
|---|---|---|---|---|
| SWRI diesel NIR, 784 spectra, 7 reference properties | the seven `diesel-*` tasks | Southwest Research Institute, distributed by the Eigenvector Research data archive, <https://eigenvector.com/data/SWRI/> | No stated licence. **Not redistributed here.** | `SWRI_Diesel_NIR/diesel_spec.csv`, `SWRI_Diesel_NIR/diesel_prop.csv` |
| Cargill corn NIR, 80 samples on three instruments, 4 traits | the four `corn-*` tasks | Cargill, distributed by the Eigenvector Research data archive, <https://eigenvector.com/data/Corn/> | No stated licence. **Not redistributed here.** | `Corn_Cargill/corn.mat` |
| Kalivas gasoline NIR, 60 spectra, octane number | `gasoline-octane` | Kalivas (1997), *Chemom. Intell. Lab. Syst.* 37:255, doi:10.1016/S0169-7439(97)00038-5. Spectra redistributed in the `pls` package for R, version 2.8-5, CRAN | The `pls` package is GPL-2 | `processed/gasoline_arrays.npz`, built by `build_derived_arrays.py` |
| EVOO adulteration NIR-HSI, 1995 spectra from 641 physical samples | `olive-oil-adulteration` | Malavi, Raes and Van Haute (2024), *Curr. Res. Food Sci.* 9:100913, doi:10.1016/j.crfs.2024.100913; data at <https://github.com/DNMalavi/NIR-HSI-ML-for-EVOO-Fraud-Detection> | No stated licence. **Not redistributed here.** | `EVOO_NIR_HSI/data/Raw_A.xlsx` |
| Edible-oil near-infrared (NIR), mid-infrared (MIR) and Raman, 100 oils in triplicate, peroxide value | `edible-oil-peroxide-value` and the three-modal fusion | Gilbraith et al. (2024), Mendeley Data version 2, doi:10.17632/ctgg7k4m5g.2 | CC BY 4.0 | `EdibleOil_NIR_MIR/{NIR24mm1A,MIR1A,Raman1A,OilClassKey}.csv` |
| Raman sugar mixtures, four sugars, two signal-to-noise ratios | the `sugar-*` datasets | Georgiev et al. (2024), *PNAS* 121:e2407439121, doi:10.1073/pnas.2407439121; measurement data at doi:10.5281/zenodo.10779223 | See the deposit | `sugar_raman/High SNR/data.pkl` and `metadata.csv`; the same under `sugar_raman/Low SNR/` |
| Commercial-gasoline Raman, 179 fuels on two instruments | the `fuel-*` datasets | Voigt et al. (2019), *Fuel* 236:829, doi:10.1016/j.fuel.2018.09.006; spectra redistributed by the depositors as Hugging Face datasets `chlange/FuelRamanSpectraBenchtop` and `chlange/FuelRamanSpectraHandheld` | CC BY 4.0 | `fuel_raman/FuelRamanSpectraBenchtop/{train,val,test}.parquet` and the same under `FuelRamanSpectraHandheld/` |

The depositors of the gasoline Raman spectra ask that Legner et al. (2020),
*Energy Fuels* 34:103, doi:10.1021/acs.energyfuels.9b02944, be cited alongside
Voigt et al. whenever the spectra are used. Cite both.

## Corpus-only datasets

Two more datasets enter no task and are used only to build the corpus: the
mixture library of its simulated half draws on both, and the measured half
includes both. They are required for `run_build_corpus.py` and nothing else in
the package reads them.

| Dataset | Role | Source | Terms | Place under `$SPEC_DATA_ROOT/` |
|---|---|---|---|---|
| NIST infrared reference collection, 98 spectra | mixture library of the simulated half, and the `nist-ir` block of the measured half | NIST Chemistry WebBook, <https://webbook.nist.gov/chemistry/>, one JCAMP-DX record per compound | A work of the US Government, no copyright | `NIST_IR/nist_ir.npz`, fetched by `fetch_nist_ir.py` |
| Mayonnaise NIR, 162 spectra, six oil types | mixture library of the simulated half, and the `mayonnaise` block of the measured half | redistributed in the `pls` package for R, version 2.8-5, CRAN | The `pls` package is GPL-2 | `processed/mayonnaise_arrays.npz`, built by `build_derived_arrays.py` |

The edible-oil collection in the table above also serves the mixture library, for
the seven edible-oil classes, in addition to its peroxide-value task.

## Retrieval

```
python data/sources/fetch_public_data.py --what all     # or a subset name
```

The script fetches everything that can be fetched without an account, verifies
what it downloaded, and prints the two steps it cannot do for you: the
Eigenvector archive asks for a free registration before it serves the diesel and
corn files, and the extraction of the gasoline and mayonnaise arrays from the R
`pls` package needs `pyreadr` (see the note in `build_derived_arrays.py`).

The NIST infrared collection has no bulk download, so it is fetched one record at
a time by a second script:

```
python data/sources/fetch_nist_ir.py     # writes $SPEC_DATA_ROOT/NIST_IR/
```

It is limited to the compounds the mixture library needs, waits between
requests, and is resumable.

The EVOO spectra live in a git repository, not in a data deposit:

```
git clone https://github.com/DNMalavi/NIR-HSI-ML-for-EVOO-Fraud-Detection
# then move Data/Raw_A.xlsx to $SPEC_DATA_ROOT/EVOO_NIR_HSI/data/Raw_A.xlsx
```

The version used here is commit `ac885ed`, and within it `Data/SNV_A.csv`, whose
SHA-256 begins `486d18ce31623d2b`. The loader reads `Raw_A.xlsx`, the
unpreprocessed export, and applies its own resampling and scaling; the
preprocessed variants in the same repository, including `SNV_A.csv`, are not
used.

## What is bundled, and why

Three items in `data/` derive from data that came from outside this work, three
more are this work's own output, and one holds no spectra. They are listed
together so that the whole of `data/` is accounted for.

- `data/corpus/simulated_spectra.npy` is the simulated half of the corpus, 3000
  spectra of 512 points. The spectra are generated, not copied, but the mixture
  library they are drawn from is built from the NIST infrared collection and from
  class means of the edible-oil and mayonnaise datasets, so the file derives from
  all three. `data/corpus/recipe.json` names the library and records the digest
  of the array; `code/corpus_augmentation/run_build_corpus.py` rebuilds it.
- `data/derived/edible-oil-nir.npz` is the edible-oil near-infrared arm,
  reduced to the spectra that carry a peroxide value and to the 99 oils
  measured in full triplicate (297 spectra, 11616 points). The source is
  CC BY 4.0 and the reduction is described in `code/common/datasets.py`. The
  loader rebuilds this file from the raw CSV if it is absent, so placing
  `NIR24mm1A.csv` under the data root is enough to regenerate it.
- `data/simulated-fuel/peaks.csv` is a peak table transcribed from published
  reference Raman spectra of eleven pure components; `data/simulated-fuel/digitized/`
  holds the eleven traces the table was read off, one compressed array per
  compound, with keys `wn`, `y` and `frame`. The plots themselves are not
  redistributed; `code/corpus_augmentation/run_digitize_components.py` documents
  how the traces were made from them and can repeat the transcription given a
  directory of the plots. Its docstring also names the one artefact left in the
  traces, an extrapolated tail below the plotted range that the peak table does
  not use.
- `data/simulated-fuel/{sim_spectra.npy,sim_axis.npy}` are simulated, produced
  by `code/corpus_augmentation/run_simulate_corpus.py`. They are the same work's
  output, not third-party data.
- `data/encoder-checkpoints/` holds the two self-supervised encoders whose
  embeddings appear in the representation comparison. They were pretrained on
  the corpus above and carry no third-party data.

And two entries under `data/` that the list above does not reach:

- `data/in-house-pentane-ccl4/` is the FT-NIR series measured for this work, 79
  spectra of n-pentane/CCl4 mixtures from ten bottles, recorded as raw
  absorbance. It is this work's own measurement and carries no third-party data.
- `data/sources/` holds no spectra. It is this file, the source list and the
  retrieval scripts, and it is covered by the code licence rather than by
  `LICENSE-DATA`.

`LICENSE-DATA` covers everything under `data/` that this work produced.

## The foundation model

The three generations of the TabPFN head are the checkpoints the vendor
distributes with the `tabpfn` package. Their licence is non-commercial and
forbids redistribution, so they are not in this package and have no default
location. Install the package, let it cache the checkpoints, and point
`SPEC_TABPFN_MODELS` at the directory holding them:

```
pip install tabpfn
# then, with the checkpoints downloaded into one directory:
export SPEC_TABPFN_MODELS=/path/to/tabpfn-checkpoints
```

The three file names the loaders expect are listed in
`code/common/paths.py`; the generations are the ones the paper calls TabPFN v2.5,
v3 and v3.5.

