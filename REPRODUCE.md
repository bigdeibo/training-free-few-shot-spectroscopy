# Which file carries which number

Every number in the paper comes from one of the CSV files under `results/`, with
the exceptions named under *What is not in `results/`* below. This file is the
index from a figure or a table to the files behind it, with the row filter and
the column names that reproduce it. The scripts that draw the figures and
typeset the tables are not part of this package, so what follows is written to
be enough on its own: a reader who wants a panel can rebuild it from the files
named here, and can re-run the experiment that produced them.

## How a number was produced

Ten result families, written by the scripts in `code/`, one row per
(task, support size, repetition, arm):

| Directory under `results/` | Produced by | Rows | Files |
|---|---|---|---|
| `representation-comparison/` | `code/representation_comparison/run_representation_comparison.py` | 7370 | 257 |
| `posterior-audit/` | `code/trust_layer/run_posterior_audit.py` | 12900 | 30 |
| `ridge-jackknife/` | `code/trust_layer/run_ridge_jackknife.py` | 2150 | 15 |
| `physics-gate/` | `code/trust_layer/run_physics_gate.py` | 510 | 17 |
| `sample-selection/` | `code/trust_layer/run_sample_selection.py` | 975 | 13 |
| `label-budget/` | `code/label_efficiency/run_label_budget.py` | 640 | 14 |
| `raman/` | `code/raman/run_raman_generality.py` and `run_projection_decoupling.py` | 960 | 8 |
| `fusion/` | `code/fusion/run_fusion_regression.py` and `run_fusion_classification.py` | 420 | 2 |
| `detection-limit/` | `code/trust_layer/run_detection_limit.py` | 210 | 1 |
| `corpus-augmentation/` | `code/corpus_augmentation/run_corpus_flip_test.py` | 120 | 1 |

`results/reference-baselines/` is the exception. It holds the partial least
squares, support vector regression and one-dimensional convolutional baselines
at aggregate level, and the scripts that produced it are not in this package.
See the note under Supplementary Section S8.

## What is not in `results/`

Three quantities in the paper are carried by no result file, and they fall into
two classes.

The subspace stability of Supplementary Table S2c and the cumulative explained
variance of the corpus PCA quoted in the same paragraph were computed while
Section S2 was being typeset, from the corpus. The script that computes them is
part of the typesetting chain and is not in this package, so a reader has the
corpus but not these two numbers without recomputing a PCA of it.

The fidelity numbers of Supplementary Figure S1, the median NNLS residual and
the three reconstruction errors, are printed by
`code/corpus_augmentation/run_simulate_corpus.py`. That script ships, and
re-running it against the fuel Raman data reproduces them, but it writes no file
of its own, so they are not in `results/` either.

## Conventions

**Splits.** Support and query indices are drawn from a seed that depends on
nothing but the support size and the repetition number:

```
42 * 1000 + K * 100 + rep          the near-infrared and in-house datasets
66 * 1000 + K * 100 + rep          the Raman datasets and the fusion analysis
```

Two methods evaluated at the same (task, K, rep) therefore see identical
indices, which is what makes the paired tests valid. `code/common/tasks.py`
holds the two split functions. The trust-layer scripts draw their internal
calibration and evaluation splits from their own families, disjoint from these,
and each script's docstring names them.

**Two definitions of `r2_evaluation`.** Both trust-layer files carry a column of
that name, and they do not agree cell for cell. `posterior-audit/` reports R² on
the evaluation half of the conformal split, because the other half is spent on
calibration; `physics-gate/` reports it on the whole test set, because its label
must be fixed before the score is ranked. The paper's gate numbers and failure
rates come from `physics-gate/`, and its coverage and width numbers from
`posterior-audit/`; no printed quantity mixes the two. Do not merge the two
files on `r2_evaluation`.

**Group-aware splits.** Where spectra are replicate measurements of a physical
sample, support spectra are drawn one per sample and every replicate of those
samples is excluded from the query set. This applies to the olive-oil,
edible-oil and in-house datasets, and, in the group form, to the Raman datasets.
The Raman files count samples in `support_size`, not spectra.

**Pools.** Three task pools appear and they are not interchangeable.

- the 13 main benchmark tasks, named in `BENCHMARK_TASKS` in
  `code/common/tasks.py`; the selection experiment ships exactly these
- the 14-task pool, which adds the peroxide-value negative control; this is the
  trust-layer pool of Tables 3 and 4
- the 15-task pool, which adds the in-house pentane/CCl4 task; the
  representation comparison, the posterior audit and ridge with jackknife+ ship
  this one

The in-house task is excluded from the trust layer because its ten physical
samples cannot supply ten distinct support groups, and the negative control is
excluded from the main benchmark because it is a null by construction.

**Winsorization.** No result file stores a clipped value: `r2` and
`r2_evaluation` hold the raw coefficient and run far below −1. The clipping is
applied where a statistic is computed, and to those two columns only, and the
paper writes `winsorized R²` for the result. One statistic must keep the raw
column: the unacceptable-deployment label of Figures 4b and 4c is
`r2_evaluation` < 0 as stored, and the paper describes deployment quality as
unwinsorized for the same reason. Winsorizing the column first would not move
that label, because clipping maps every value at or below −1 onto −1, which is
still negative, so the 28.8 per cent of the trust-layer pool it marks is the
same either way.

**Repetitions.** Ten, from 0 to 9, in every family except `results/sample-selection/`,
which stops at five, running 0 to 4. The `repetition` column says which is which, and one file
that has no such column, `results/reference-baselines/classical-and-deep-baselines.csv`,
is aggregate by construction and records its repetition count in `n_repetitions`.

**Aggregation.** Results are reported as medians over repetitions, never as
means over unpaired aggregates.

## Main figures

### Figure 1 — the stack and the deployment workflow

No data. The figure is a schematic and is drawn from the text of Section 3.

### Figure 2a — the representation comparison

| | |
|---|---|
| Files | `results/representation-comparison/tabpfn-v3__{representation}__{task}.csv` |
| Representations | `raw-spectra-512`, `corpus-pca-50`, `wavelet-scattering`, `cars-selected`, `random-init-encoder`, `simclr-encoder`, `masked-autoencoder` |
| Filter | the 13 main tasks |
| Column | `r2`, winsorized at −1 |
| Unit | one (task, repetition) pair, ten repetitions over thirteen tasks |

The panel shows the seven arms at each support size, with the task-level
distribution and a median marker. The caption's test is a paired Wilcoxon
signed-rank on winsorized R² differences between arms, computed inside each
support size over the 130 task-repetition units. The test is against one arm
named in the caption, and no correction for multiplicity is applied.

The corpus projection this panel compares is the 50-component one. Table 3 and
the rest of the paper operate the 100-component projection, and the two files
are `tabpfn-v3__corpus-pca-50__{task}.csv` and
`tabpfn-v3__corpus-pca-100__{task}.csv`. Do not read the panel's medians out of
the Table 3 file or the reverse: at K = 10 they are 0.35 and 0.38.

### Figure 2b — corpus saturation

| | |
|---|---|
| Files | `results/representation-comparison/tabpfn-v3__{representation}__{task}.csv` |
| Representations | `corpus-pca-100-fit-500`, `-fit-1500`, `-fit-3000`, and `corpus-pca-100` for the full 6339-spectrum corpus |
| Filter | support size 10, the 13 main tasks |
| Column | `r2`, winsorized at −1 |

Median with interquartile range at each corpus size. The projector is refitted
on N spectra drawn from the corpus for the three ablated points; `corpus-pca-100`
uses all 6339. The corpus file does not ship whole:
`code/corpus_augmentation/run_build_corpus.py` completes it from the public
datasets, and `data/sources/README.md` notes the same.

### Figure 3a — the signal-to-noise flip

| | |
|---|---|
| Files | `results/raman/sugar-high-snr.csv`, `results/raman/sugar-low-snr.csv` |
| Columns | `r2_tabpfn`, `r2_pls`, keyed by `target` and `support_size` |
| Unit | one (target, repetition) pair, four sugars over ten repetitions |

One point per sugar and support size, showing the paired difference between the
foundation-model head and the partial-least-squares reference. The caption's
test is a paired Wilcoxon signed-rank over the target-repetition units, n = 40
at each support size.

### Figure 3b — projection and head decoupled

| | |
|---|---|
| File | `results/raman/projection-decoupling__sugar-low-snr.csv` |
| Columns | `r2_pca_scores_ridge`, `r2_pls_scores_ridge`, `r2_pls_scores_tabpfn`, `r2_pls`, `n_components` |
| Filter | the low signal-to-noise sugar dataset |

The four cells of the 2×2 are the four `r2_*` columns. The third, ridge on
partial-least-squares scores, is the self-check: it lands close to `r2_pls`,
which is what a ridge head fitted on the same supervised scores should do. The
other three datasets carry only the `r2_pca_scores_ridge` cell, in the three
other files under `results/raman/`.

### Figure 3c — the routing map

| | |
|---|---|
| Files | `results/label-budget/*.csv` and `results/raman/*.csv` |
| x | median `r2_pls` at support size 20, the task's linearity proxy |
| y | corpus size on a log scale |
| color | whichever head has the higher median |

One point per task that carries both a foundation-model and a
partial-least-squares result under the same split. The linearity proxy is the
median partial-least-squares R² at the largest support size, and the caption
defines it as such.

### Figure 4a — posterior reliability

| | |
|---|---|
| Files | `results/posterior-audit/posterior-audit__tabpfn-v3__{task}.csv` and the same with `tabpfn-v3.5` |
| Columns | `nominal_coverage`, `empirical_coverage`, `method` |
| Levels | 50, 68, 80, 90, 95 |
| Methods | `native-posterior`, `split-conformal-absolute`, and `split-conformal-locally-adaptive`, which the panel does not draw |

Nominal against empirical coverage, one line per method, both generations on the
same axes. Coverage is the mean over repetitions at each nominal level. The
files carry a third method, the locally adaptive conformal variant; the panel
and the paper both report the absolute variant, so a reader rebuilding the panel
must select the two method values named first. The caption's failure threshold
for a deployment is R² < 0, which in the 420-cell pool is 28.8 per cent of
cells.

### Figure 4b — gate against interval width

| | |
|---|---|
| Files | `results/physics-gate/physics-gate__corpus-pca-100__{task}.csv`, joined to `results/posterior-audit/posterior-audit__tabpfn-v3__{task}.csv` |
| Columns | `mean_physics_gate_score` and `r2_evaluation` from the gate file; `mean_interval_width` from the audit file |
| Join key | `task`, `support_size`, `repetition`, with `method` = `native-posterior` and `nominal_coverage` = 90 on the audit side |
| Filter | the 14-task trust-layer pool |

Receiver-operating-characteristic area under the curve as a function of support
size, over the pooled cells of the 14 tasks and ten repetitions at each support
size. The label is an unacceptable deployment, which the caption defines as
`r2_evaluation` < 0; in the pool this is 28.8 per cent of the cells. Both scores
are ranked against that label, the physics-gate score and the native interval
width, and the random control is the same measurement on a shuffled score.

Three cautions about reading the files directly. The gate file does not carry an
interval width: the series in the panel comes from the audit file, which holds
five nominal levels per cell, so the level and the method both have to be fixed
before the join is one-to-one. The gate file's `catastrophic_failure` column uses
the stricter threshold of R² < −1 and is not this label. Its `auc_physics_gate`
and `auc_interval_width` columns are a different quantity again, the trapezoidal
area under the per-cell risk-coverage curve, so neither is the number in the
panel.

### Figure 4c — selection curves

| | |
|---|---|
| File | `results/sample-selection/*.csv` |
| Columns | `strategy`, `support_size`, `r2` |
| Strategies | `random`, `k-center-diversity`, `interval-width` |
| Support sizes | 2, 5, 10, 15, 20; the tests below are quoted at 5, 10 and 20 |

Median winsorized R² against the number of labelled samples acquired, one line
per strategy, with the random line as the reference. This is the one family that
stops at five repetitions, so the comparison in the caption is a paired Wilcoxon
signed-rank of k-center against random over 13 tasks times five repetitions,
n = 65.

The pairing is a property of the data, not of the analysis: within a
(task, repetition) block the three strategies are grown from one shared seed
support set, so their increments are correlated with the alternative they are
tested against. You can check it directly — at `support_size` 2 the three
strategies hold the same support set and every `r2` value is identical across
`random`, `k-center-diversity` and `interval-width`, for each of the 13 tasks
times five repetitions. Run the paired test only on blocks that satisfy this;
if the arms are ever drawn independently, the same test is invalid and the
difference has to be tested unpaired instead.

### Figure 5a — the budget curve

| | |
|---|---|
| File | `results/label-budget/*.csv` |
| Columns | `r2_tabpfn`, `r2_pls`, `support_size` |
| Filter | the 14-task pool |

Pooled median difference in winsorized R² between the stack and the
partial-least-squares reference, with the win rate over paired units, against
the number of labelled samples. Support sizes are 5, 10, 20, 50 and 100.

### Figure 5b — label economics

The same file. Both arms are reduced to a median curve over the 14 tasks, and
the arrow marks the two support sizes at which the curves are equal in median
performance. This interpolation is a property of the median curves and not of
any single task, and the caption says so.

### Figure 6 — deployment showcase

| | |
|---|---|
| Files | `results/physics-gate/physics-gate__corpus-pca-100__{task}.csv` and `results/posterior-audit/posterior-audit__tabpfn-v3__{task}.csv` |
| Join key | `task`, `support_size`, `repetition` |

The gate rejects the lowest-scoring 20 per cent of deployments inside each
support size. The released group is then summarised on three numbers: the
unacceptable rate over all deployments against the rate over the released ones,
the median R² of the released group, and its coverage at a 90 per cent nominal
level under `split-conformal-absolute`. The pool is 14 tasks, three support
sizes and ten repetitions, and the join to the audit file again fixes
`nominal_coverage` = 90 and the method, here the conformal one.

### Figure 7a — three-modal classification

| | |
|---|---|
| File | `results/fusion/three-modal-classification.csv` |
| Columns | `modality`, `support_size`, `accuracy` |
| Arms | `nir`, `mir`, `raman`, `nir+mir`, `nir+raman`, `mir+raman`, `all-three-modalities` |

Median accuracy over ten repetitions, seven arms at each support size. The
caption's test compares the three-modality arm against the best single modality
at each support size, paired by repetition. The comparator is the arm with the
highest median, with the mean breaking a tie, which selects `nir` at 5, `mir`
at 10 and `raman` at 20. Selecting on the median alone lands on `mir` at 5,
where the two medians are equal, and returns 1.000 in place of the caption's
0.500.

### Figure 7b — three-modal regression

| | |
|---|---|
| File | `results/fusion/three-modal-regression.csv` |
| Columns | `modality`, `support_size`, `r2` |

The same seven arms on peroxide value, winsorized at −1. The panel is a null and
is drawn as one: peroxide value is an oxidation product and no arm recovers it,
which is what makes the classification companion the informative half.

## Main tables

**Table 1, datasets and their roles.** No result file. The dataset inventory,
with the source of each dataset, is in `data/sources/README.md`, and the
per-task breakdown with the two Raman datasets is in Supplementary Section S1.
The counts of spectra in the table are the source files' own counts.

**Table 2, trust layer.** One row per support size, six numbers.

| Column | Files | How |
|---|---|---|
| native coverage at 90% | `results/posterior-audit/posterior-audit__tabpfn-v3__{task}.csv` | mean `empirical_coverage` over the 14 tasks and ten repetitions, `method` = `native-posterior`, `nominal_coverage` = 90 |
| conformal coverage at 90% | the same file | `method` = `split-conformal-absolute` |
| width ratio | the same file | per (task, repetition) in the 14-task pool, the conformal 90% `mean_interval_width` divided by the native 90% one; median over the 140 pairs. The ratio is a within-row quantity: it is taken inside one support-size row, and pooling the three rows first gives a different and wrong number |
| gate AUC | `results/physics-gate/physics-gate__corpus-pca-100__{task}.csv` | at each support size, pool the 14 tasks and ten repetitions, label each cell by `r2_evaluation` < 0, and take the receiver-operating-characteristic area under the curve of `mean_physics_gate_score` against that label |
| width AUC | the gate file joined to the audit widths on (task, support size, repetition) | the same pooled area on `mean_interval_width`, taking the `native-posterior` rows at `nominal_coverage` = 90 |
| k-center gain | `results/sample-selection/*.csv` | median paired difference of winsorized `r2` between `k-center-diversity` and `random`, tested with a paired Wilcoxon signed-rank over the 13 tasks times five repetitions, n = 65 |

Two of those rows are easy to get wrong and both were. The width ratio is
measured inside the 14-task pool; leaving the in-house task in moves the first
row from 1.26 to 1.32, because that task carries only the smallest support size
and has the widest intervals. And the audit file holds five nominal levels per
cell, so the join behind the width AUC is one-to-many until the level is fixed:
at 90 per cent the row is 0.542, 0.773, 0.901, and at 80 per cent it is 0.549,
0.775, 0.908. The gate AUC takes no level, since the physics score does not
depend on one.

**Table 3, main comparison.** Three methods, each a median of winsorized R² over
the 13 main tasks and ten repetitions, n = 130.

| Row | Files | Column |
|---|---|---|
| foundation-model head on corpus PCA (100 components) | `results/representation-comparison/tabpfn-v3__corpus-pca-100__{task}.csv` | `r2` |
| ridge on corpus PCA | `results/ridge-jackknife/ridge-jackknife__{task}.csv` | `r2_evaluation`, which is the point prediction and so is the same in all five nominal coverage rows of a cell |
| partial least squares on raw spectra | `results/label-budget/label-budget__{task}.csv` | `r2_pls` |

The negative-control row of the table takes the same three columns for the task
`edible-oil-peroxide-value`. The table's own note gives the pooled medians for
the 14-task pool, which the same files produce by dropping the filter on the
task list.

**Table 4, performance characteristics.** Six rows, each from a different
family.

| Row | Files | How |
|---|---|---|
| accuracy | `results/label-budget/*.csv` | median paired difference `r2_tabpfn` − `r2_pls` at support size 10, winsorized |
| precision | `results/label-budget/*.csv` | per task, the interquartile range of winsorized `r2` over ten repetitions, then the median over the 14 tasks; separately for each arm and each support size |
| calibration | `results/posterior-audit/*.csv` | the native and conformal coverage rows of Table 2, with the width ratio |
| interferences | `results/raman/sugar-high-snr.csv`, `results/raman/sugar-low-snr.csv` | the two panels of Figure 3a |
| sensitivity | `results/label-budget/*.csv` | the pooled crossover of Figure 5a |
| detection limits | `results/detection-limit/detection-limit__olive-oil-adulteration.csv` | `detection_rate_at_lod`, `detection_rate_at_loq` |

The detection-limit row needs its denominator stated, and the table note does:
the rate is the fraction of the query spectra at a given (support size,
repetition, level) whose prediction exceeds the limit, averaged over the ten
repetitions. The limit itself is IUPAC-style, the mean plus 3.3 standard
deviations and the mean plus 10 standard deviations of the predictions on the
unadulterated blanks, which the file carries as `blank_mean` and `blank_sd`.

## Supplementary sections

| Section | Files |
|---|---|
| S1, per-task representation results | `results/representation-comparison/tabpfn-v3__{representation}__{task}.csv`, all 15 tasks of the representation pool |
| S2, the corpus projection and its dimension | `results/representation-comparison/tabpfn-v3__corpus-pca-{20,50,100,200}__{task}.csv` for the dimension sweep, `corpus-pca-100-fit-{500,1500,3000}` and `corpus-pca-100` for the corpus-size sweep; the corpus itself is `data/corpus/recipe.json` together with the half of it that ships, `data/corpus/simulated_spectra.npy`, which `code/corpus_augmentation/run_build_corpus.py` completes from the public datasets |
| S3, the posterior audit per task | `results/posterior-audit/posterior-audit__tabpfn-{v3,v3.5}__{task}.csv` |
| S4, gate selectivity | `results/physics-gate/physics-gate__corpus-pca-100__{task}.csv`, with `physics-gate__random-projections-100__{task}.csv` for the three tasks the random projection control was run on |
| S5, selection strategies | `results/sample-selection/*.csv` |
| S6, ridge with jackknife+ intervals | `results/ridge-jackknife/ridge-jackknife__{task}.csv`, including `ridge_penalty` |
| S7, per-target Raman results | `results/raman/{sugar-high-snr,sugar-low-snr,fuel-benchtop,fuel-handheld}.csv`; Table S7c, the low-SNR decoupling, is `results/raman/projection-decoupling__sugar-low-snr.csv`, the one decoupling file that carries all four arms, the fuel and high-SNR decoupling files carrying the corpus-PCA arm alone |
| S8, fusion and the baselines | `results/fusion/*.csv`, and `results/reference-baselines/classical-and-deep-baselines.csv` |
| S9, the augmented fuel corpus | `results/corpus-augmentation/fuel-benchtop-augmented-corpus.csv`, with `results/raman/fuel-benchtop.csv` for the unaugmented reference |

Section S8 is the one section this package cannot fully regenerate. The baseline
file holds partial least squares, support vector regression and
one-dimensional convolutional results at aggregate level: mean and standard
deviation of R², RMSE and RPD over the repetitions, with the number of
repetitions recorded per row. They were computed under the same split formula
and the same 13 tasks as everything else, which is what makes them comparable at
the aggregate level. The scripts that ran them are not in this package, so the
cells cannot be recomputed from the files here. The convolutional baselines
cover four of the thirteen tasks; the other nine have no convolutional row. The
`n_repetitions` column separates the classical arms, which carry thirty, from
the convolutional arms, which carry ten, matching the repetition counts the
paper assigns to each family. The file is included so that the comparison is
traceable, and this paragraph is the disclosure that it is not reproducible from
this archive alone.

## Re-running an experiment

```
pip install -r requirements.txt
export SPEC_DATA_ROOT=/path/to/public-datasets     # see data/sources/README.md
export SPEC_TABPFN_MODELS=/path/to/tabpfn-checkpoints
python code/corpus_augmentation/run_build_corpus.py   # one step first: see README.md
python code/representation_comparison/run_representation_comparison.py
```

The corpus step comes first because the projection most arms are fitted on is
read from `data/corpus/spectra.npy`, and this archive ships only the simulated
half of that corpus. `run_build_corpus.py` completes it from the public
datasets under `SPEC_DATA_ROOT`. The families that use no corpus projection run
without it.

Each script writes one CSV per (task, support size, repetition, arm) under
`results/`, and each takes an `--help` that lists its own options. The scripts
are resumable: a row already present in the target file is skipped, so an
interrupted run can be restarted. To force a cell to be recomputed, delete its
row from the file or move the file aside.

**A rerun reproduces the shipped tables.** The spectra load in single
precision, and every transform, standardisation and fit here is evaluated in
the precision it is handed: the scripts carry single precision through and
widen nothing to double on the way in. Widening is the same transform, and on
a spectrum the two differ by about one part in a million. It is not negligible
at this support size, because the head is sometimes fitted on two spectra,
where the in-context regression is close to singular and a difference of that
order moves a cell whose R² is far below zero by several hundredths.
The claim was checked by re-running two families and comparing cell by cell:
`results/sample-selection/diesel-cetane-number.csv`, a TabPFN head at support
sizes 2 to 20 over five repetitions, and
`results/ridge-jackknife/ridge-jackknife__diesel-cetane-number.csv`, a
closed-form ridge with jackknife+ intervals over 150 rows. Both reproduce to
the last stored digit; the `seconds` column is wall-clock time and will not.

Every script runs on a GPU when one is visible and falls back to the CPU
otherwise. The `seconds` column of every result file records the wall-clock time
of the cell it belongs to, so the size of a family can be read off the archive
directly. Summed over the rows as shipped, the families fall into two groups:

| Under an hour | | Several hours | |
|---|---|---|---|
| corpus augmentation | 3 min | posterior audit | 17 h |
| ridge with jackknife+ | 8 min | representation comparison | 9 h |
| detection limit | 20 min | Raman | 4 h |
| fusion | 19 min | sample selection | 2 h |
| label budget | 35 min | physics gate | 1 h |

These are the times the recorded runs took, on the single machine that produced
this archive, and they indicate scale without being a benchmark: a different
GPU, or a CPU-only rerun, will move them. The two arms of a paired
comparison are timed separately wherever a file carries both `seconds_tabpfn`
and `seconds_pls`. The encoders are small enough that embedding extraction is
not the bottleneck in any family; the head is what costs.

## Integrity of the rename

The files in this package carry the paper's vocabulary, not the working names
the experiments were run under, and the renaming is documented in
`GLOSSARY.md`. It was applied textually: every CSV was read with all columns as
strings, the declared string columns and the file names were mapped, and the
identical strings were written back. No column was ever parsed as a number, so
no floating-point value could change. The check afterwards was a column-by-column
comparison against the originals: row counts, column counts, and every numeric
column bit-identical, with only the declared string cells differing. The counts
in the table at the top of this file are from that verified build.
