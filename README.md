# Training-free few-shot quantitation of NIR and Raman spectra

Code, data and results for the manuscript *Training-free few-shot quantitation
of NIR and Raman spectra: when it works, when to doubt it, and what it costs*
(Tao, Li, Li, Tian and Zhang).

The work asks a narrow question: when a foundation model is asked to quantify a
property from a handful of near-infrared or Raman spectra, with no training on
the target task, what has to be done to the spectra before the model sees them,
and what does the answer cost. It finds that a principal projection fitted on an
unlabeled corpus closes the representation question. Which head wins the
remaining comparison is predicted by the task's linearity and the corpus size,
and the model's own uncertainty cannot be trusted for decisions while an explicit
trust layer can.

## What is here

```
README.md            this file
REPRODUCE.md         every figure and table -> the result file that carries it
GLOSSARY.md          the paper's vocabulary against the file names used here
MANIFEST.md          what is deliberately not in the package, and why
LICENSE              MIT, for the code
LICENSE-DATA         CC BY 4.0, for the data shipped here; the file gives the scope
requirements.txt     dependencies, including how to obtain the model weights
.gitignore           the two directories a reader may create locally
code/                the fifteen experiment scripts, twelve of which produce the
                     results and three the data, and the shared modules they
                     import
data/                the data this work produced: the simulated half of the
                     unlabeled corpus, the in-house spectra, the simulated fuel
                     corpus, the encoder checkpoints, the one reduced public file
                     that ships under its own terms, and the sources list for the
                     public datasets
results/             per-repetition result files, one row per
                     (task, support size, repetition, arm)
```

The scripts under `code/` are the ones that produce the tables of numbers, plus
the three that produce the data. The scripts that draw the figures and typeset
the tables are not included; `REPRODUCE.md` names, for each figure and table,
the result files behind it and the statistic computed from them, so a reader can
rebuild any panel from the files here.

## Start here

If you want to check a number in the paper, read `REPRODUCE.md`. Every number is
in `results/` as shipped, and nothing has to be run or downloaded for that.

If you want to run the experiments, install the dependencies and point two
environment variables at the data:

```
pip install -r requirements.txt
export SPEC_DATA_ROOT=/path/to/public-datasets        # data/sources/README.md
export SPEC_TABPFN_MODELS=/path/to/tabpfn-checkpoints
python code/representation_comparison/run_representation_comparison.py --help
```

The public datasets are not redistributed with this package. The script in
`data/sources/` fetches what can be fetched without an account and lists the
exact steps for the rest. The model weights are not redistributed either, their
licence being non-commercial; `data/sources/README.md` says where to get them.

One step comes first: every corpus projection reads `data/corpus/spectra.npy`,
which is assembled from the public datasets by
`code/corpus_augmentation/run_build_corpus.py`.

The scripts are resumable. Each writes one file per (task, support size,
repetition, arm) and skips rows already present, so an interrupted sweep can be
restarted without redoing work.

## The data

Two kinds of data live in this package and they are licensed separately.

**Produced by this work, shipped under CC BY 4.0.** The simulated half of the
unlabeled corpus (`data/corpus/simulated_spectra.npy`, 3000 spectra), the
in-house n-pentane/CCl4 FT-NIR series (`data/in-house-pentane-ccl4/`, 79 spectra
from ten bottles), the simulated fuel corpus and its component peak table
(`data/simulated-fuel/`), and the two pretrained encoder checkpoints
(`data/encoder-checkpoints/`). The corpus recipe is recorded in
`data/corpus/recipe.json`.

The corpus that a projection is fitted on holds 6339 spectra. The other 3339 are
measured, they come from the six public datasets, and they are not redistributed,
so the corpus is not shipped whole:
`code/corpus_augmentation/run_build_corpus.py` rebuilds all 6339 from the
simulated half and the public datasets, and `recipe.json` records the digests to
check the result against. The simulation draws on the NIST infrared collection
and on class means of the edible-oil and mayonnaise datasets, so those three
reach the corpus twice, once as themselves and once through the mixture library.

**Derived from public data, redistributed only where the terms allow.** One file
in `data/derived/` falls in this class, and it is documented in
`data/sources/README.md` together with the licence that permits it. Everything
else public is fetched, not shipped.

## A note on the numbers

The results are reported as winsorized R², clipped below at −1, aggregated as
medians over repetitions and tested with a paired Wilcoxon signed-rank on
task-repetition units. Comparisons are paired by construction: the support and
query indices for a given task, support size and repetition are drawn from a seed
that depends on those three numbers alone, so any two methods evaluated at the
same cell saw identical spectra. `REPRODUCE.md` gives the seed formulas and the
three task pools.

The paper reports its negative results as findings and this package carries the
files behind them. Adulterant quantitation in olive oil fails below about fifty
reference measurements, and the file is
`results/detection-limit/detection-limit__olive-oil-adulteration.csv`. Peroxide
value is not recovered by any modality, alone or fused:
`results/fusion/three-modal-regression.csv`. The handheld fuel instrument is a
dead zone: pooled over its four targets and ten repetitions the
foundation-model head never reaches a median R² of 0.15, and at five support
spectra every target is negative. Partial least squares is ahead there and is
still weak, at 0.32 at twenty support spectra:
`results/raman/fuel-handheld.csv`. On the main line the foundation model's
margin over ridge on identical features is real but thin:
`results/representation-comparison/` and `results/ridge-jackknife/` carry both
arms on the same splits, and the TabPFN and ridge rows of Table 3 of the paper
are the medians of those two files.

Where a comparison is not matched, the paper says so and `REPRODUCE.md` repeats
it at the point where it matters. The largest such asymmetry is that the
foundation-model head is a tabular model that sees the support and query sets
together in one forward pass, while the partial-least-squares reference is
fitted on the support set alone; the comparison is matched on splits, features
and budgets, and not on that.

## Citing

The manuscript is under review. Until it appears, cite this repository by its
URL. A citation block with the journal reference will be added here when the
paper is published.

## Contact

Purification Equipment Research Institute of CSSC, Handan, China. Questions
about the code or the data are welcome as issues on the repository; a
correspondence address is given in the manuscript.
