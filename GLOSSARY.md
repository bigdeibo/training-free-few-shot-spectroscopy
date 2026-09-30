# Glossary

Every identifier in this package — directory names, file names, CSV column headers
and CSV cell values — uses the vocabulary of the paper. The project's own working
names never appear here. This file maps the two, so that a reader who finds a name
in the paper can locate the file that carries its numbers, and vice versa.

The mapping was applied mechanically when this package was assembled, and every
file was checked cell by cell afterwards: only the declared string columns
changed, and every numeric column is bit-identical to the run that produced it
(see `REPRODUCE.md`, "Integrity of the rename").

## Task names

The paper's task labels (Supplementary Tables S1, S3) and the slugs used as file
names and as the `task` column.

| Paper label | Slug in this package |
|---|---|
| diesel: BP50 | `diesel-bp50` |
| diesel: cetane number | `diesel-cetane-number` |
| diesel: density | `diesel-density` |
| diesel: flash point | `diesel-flash-point` |
| diesel: freezing point | `diesel-freezing-point` |
| diesel: total aromatics | `diesel-total-aromatics` |
| diesel: viscosity | `diesel-viscosity` |
| corn: moisture | `corn-moisture` |
| corn: oil | `corn-oil` |
| corn: protein | `corn-protein` |
| corn: starch | `corn-starch` |
| gasoline: octane | `gasoline-octane` |
| olive oil: adulteration | `olive-oil-adulteration` |
| edible oils: peroxide value | `edible-oil-peroxide-value` |
| pentane/CCl4: pentane volume fraction (in-house) | `pentane-ccl4-volume-fraction` |

"BP50" is the 50% distillation boiling point, the property the SWRI diesel
benchmark reports under that name. The in-house pentane/CCl4 task is the only
task measured in this work; it appears in the representation comparison and is
excluded from every trust-layer pool.

## Representations (the `representation` column)

| Paper term | Slug |
|---|---|
| raw spectra (512 points) | `raw-spectra-512` |
| corpus PCA (20 / 50 / 100 / 200) | `corpus-pca-20` / `-50` / `-100` / `-200` |
| corpus PCA (100) on a 500- / 1500- / 3000-spectrum corpus | `corpus-pca-100-fit-500` / `-fit-1500` / `-fit-3000` |
| corpus PCA (50) on a 1500-spectrum corpus | `corpus-pca-50-fit-1500` |
| wavelet scattering | `wavelet-scattering` |
| CARS (competitive adaptive reweighted sampling) variable selection | `cars-selected` |
| random-init encoder | `random-init-encoder` |
| random projections (100) | `random-projections-100` |
| SimCLR encoder | `simclr-encoder` |
| masked autoencoder | `masked-autoencoder` |

"Corpus PCA" is a principal-component projection fitted on the 6339-spectrum
unlabeled corpus. The `fit-N` suffix marks the corpus-size ablation of
Supplementary Section S2, where the projector is refitted on N spectra drawn from
that corpus; without a suffix, all 6339 were used. The paper also calls it the
mixed corpus, because 3339 of the 6339 spectra are measured and only 3000 are
simulated. Only the simulated half ships, as `data/corpus/simulated_spectra.npy`;
`code/corpus_augmentation/run_build_corpus.py` assembles the whole of it from
that half and the public datasets. It is not the simulated fuel corpus under
`data/simulated-fuel/`, which is the separate augmentation of Supplementary
Section S9 and carries no corpus projection.

The random-init encoder and the 100-dimensional random projection are two
different controls and are not interchangeable. The first is a convolutional
encoder of the same architecture as the self-supervised ones, left at its random
initialization, and appears in the representation comparison. The second is a
random linear map from spectra to 100 features, and appears only as the control
arm of the physics gate.

## Support units

In the near-infrared and in-house datasets a support spectrum is a sample, so
`support_size` is a count of spectra. In the Raman datasets it is not: a sugar
well is measured about twenty times and the twenty measurements are replicates
of one mixture, so a support unit is a physical sample and all of its spectra
enter the support set together. `support_size` in `results/raman/` therefore
counts samples, and a support set drawn at K = 10 holds roughly two hundred
spectra. The same rule keeps every replicate of a support sample out of the
query set. `code/common/tasks.py` carries the two split functions, `eval_split`
for the first case and `group_split` for the second.

## Foundation-model generations

The `tabpfn-v*` prefix on a result file names the TabPFN checkpoint that produced
the head. These are the generations referred to in the paper as the TabPFN v2.5,
v3 and v3.5 heads. The default generation of the paper is named in every file
that uses it, so no table is ambiguous about what produced it: the posterior
audit, for instance, ships `posterior-audit__tabpfn-v3__*.csv` and
`posterior-audit__tabpfn-v3.5__*.csv` side by side. The checkpoint files
themselves are not redistributed here; `data/sources/README.md` says how to
obtain them, and `code/common/paths.py` says where to put them.

## Trust-layer methods (the `method` column)

| Paper term | Slug |
|---|---|
| native posterior | `native-posterior` |
| split-conformal, absolute | `split-conformal-absolute` |
| split-conformal, locally adaptive | `split-conformal-locally-adaptive` |
| ridge regression with jackknife+ intervals | `ridge-jackknife-plus` |

## Selection strategies (the `strategy` column)

| Paper term | Slug |
|---|---|
| random selection | `random` |
| k-center diversity | `k-center-diversity` |
| uncertainty sampling (native interval width) | `interval-width` |

The physics-gate strategy is not present: it was dropped after smoke calibration
showed a cost of roughly 1100 s per repetition against a budget of seconds for
the other three. See the module docstring in
`code/trust_layer/run_sample_selection.py`.

## Fusion arms (the `modality` column)

Single-modality arms use their modality name in lower case (`nir`, `mir`,
`raman`); multi-modality arms are `+`-joined (`nir+mir`, `nir+raman`,
`mir+raman`); the three-modality arm is `all-three-modalities`.

## Column headers

| Header in this package | Meaning |
|---|---|
| `task` | task slug, from the table above |
| `target` | measured property within a Raman or fusion dataset |
| `representation` | representation slug, from the table above |
| `modality` | fusion arm, from the table above |
| `support_size` | K in the paper: the number of support spectra, except in the Raman files, where a support unit is a physical sample and the column counts groups (see below) |
| `repetition` | repetition index, `rep` in the paper |
| `n_features` | dimension of the representation fed to the head |
| `r2` | coefficient of determination on the query set |
| `r2_evaluation` | coefficient of determination on the evaluation half in `posterior-audit/`, on the whole test set in `physics-gate/` (see REPRODUCE, *Two definitions of `r2_evaluation`*) |
| `rmse` | root-mean-square error |
| `rpd` | ratio of performance to deviation |
| `seconds` | wall-clock time for the cell, in seconds |
| `method` | trust-layer method, from the table above |
| `nominal_coverage` | nominal interval level in per cent |
| `empirical_coverage` | measured coverage at that level |
| `mean_interval_width` | mean interval width at that level |
| `catastrophic_failure` | 1 when `r2_evaluation` < −1 |
| `auc_physics_gate` | risk-coverage AUC of the physics-gate score |
| `auc_interval_width` | risk-coverage AUC of the native interval width |
| `auc_random_control` | risk-coverage AUC of the random control |
| `mean_physics_gate_score` | deployment-mean gate score |
| `strategy` | selection strategy, from the table above |
| `n_components` | number of latent components retained by the projector |
| `r2_tabpfn` | coefficient of determination of the foundation-model head |
| `r2_ridge` | coefficient of determination of the ridge head |
| `r2_pls` | coefficient of determination of the partial-least-squares reference |
| `r2_pca_scores_ridge` | ridge head on corpus principal scores |
| `r2_pls_scores_ridge` | ridge head on partial-least-squares scores |
| `r2_pls_scores_tabpfn` | foundation-model head on partial-least-squares scores |
| `n_components_pls` | latent components chosen for the partial-least-squares reference |
| `seconds_tabpfn`, `seconds_pls` | wall-clock time for each arm of a paired comparison |
| `n_test_spectra` | query spectra scored in one cell |
| `n_test_mean` | mean query-set size over the repetitions of a cell |
| `n_repetitions` | repetitions behind an aggregate row |
| `r2_mean`, `r2_std`, `rmse_mean`, `rmse_std`, `rpd_mean`, `rpd_std` | aggregate of a metric over the repetitions of a cell |
| `adulteration_level` | nominal adulterant fraction of the olive-oil test mixtures, in per cent |
| `median_prediction` | median predicted adulteration level over an evaluation set |
| `blank_mean`, `blank_sd` | mean and standard deviation of the predictions on the unadulterated blank |
| `detection_rate_at_lod` | fraction of query spectra at a level read as detected at the limit of detection |
| `detection_rate_at_loq` | the same at the limit of quantification |
| `ridge_penalty` | ridge penalty selected by leave-one-out |
| `accuracy` | classification accuracy |

## Working names

The project used shorthand of its own while the work was being done, of the kind
every project accumulates: experiment numbers, two- and three-letter codes for
each arm, and abbreviated names for the tasks. None of it appears in this
package and no key to it is published here. Every identifier a reader meets in
`code/`, `data/` and `results/` is either a term from the paper or a plain
English description of what the file holds, and this file maps the two.
