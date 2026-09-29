# What is not in this package, and why

This package ships the result files behind every number in the paper, the code
that produced them, and the data this work created. It does not ship the figures
or tables themselves, the third-party data it was evaluated on, or the
foundation-model weights. Nothing here is withheld to protect the paper's
priority: each exclusion has a stated reason, and each is either licensed
elsewhere, regenerable from what is here, or a matter of what a reader needs.

## Figure and table production

The scripts that draw the figures and typeset the tables are not included, and
neither are their outputs. `REPRODUCE.md` is the substitute. For every panel of
every main figure, every main table and every supplementary section, it names
the result file that carries the numbers, with the filter, column and statistic
computed from them. Any panel can then be rebuilt from the files in `results/`
without those scripts.

One family is the exception, and it is called out in `REPRODUCE.md` as well:
`results/reference-baselines/classical-and-deep-baselines.csv` carries aggregate
rows that were computed from a baseline run outside this package. Its rows are
traceable in the sense that every row names its task, its method, its support
size and its repetition count, but they cannot be regenerated from the archive
alone.

## Third-party data

No public dataset is redistributed in full. Two files here nonetheless carry
material from one, and both are documented in `data/sources/README.md` with the
terms that permit them.

The first is a reduced arm of the edible-oil collection, which is that dataset
recut and is third-party throughout.

The second is the simulated half of the corpus, and it is a mixed case rather
than a third-party file. The spectra are this work's own output; the mixture
library they were drawn from is assembled from the NIST infrared collection and
from class means of the edible-oil and mayonnaise datasets.
`data/corpus/recipe.json` records the library member by member, and
`data/sources/README.md` says which of those sources are third-party and under
what terms.

The rest is not here at all. The datasets behind the thirteen tasks are listed
with their sources, their terms and their placement in
`data/sources/README.md`, and `data/sources/fetch_public_data.py` retrieves what
can be retrieved without an account. The reasons differ by dataset: the
Eigenvector archive serves the diesel and corn files after a free registration;
the edible-oil collection is CC BY 4.0 and could be redistributed, but the full
deposit is 84 MB and only one reduced arm of it is used, so the reduction is
bundled and the deposit is fetched; the sugar Raman and gasoline Raman
collections come from public deposits whose terms permit redistribution but whose
files are large and versioned elsewhere, so the loader takes them from source.

The corpus is a special case, and it is the one place where what is withheld
changes what a reader can do unaided. A projection is fitted on 6339 unlabeled
spectra, 3000 simulated and 3339 measured, and the measured half is drawn from
six public datasets. Only the simulated half ships. The archive therefore cannot
fit a corpus projection out of the box: `run_build_corpus.py` under `code/`
assembles the whole corpus from the shipped half and the public datasets, and the
recipe records the digests of both halves so the result can be checked.

One related item is deliberately absent and worth naming because a reader might
expect it: the Raman peak table in `data/simulated-fuel/peaks.csv` was
transcribed from published reference spectra of pure compounds, and those plots
are third-party. The table is shipped, and the eleven digitised traces the table
was read off ship with it under `data/simulated-fuel/digitized/`. The plots are
not, and `code/corpus_augmentation/run_digitize_components.py` records how the
transcription was made and can repeat it given a directory of the plots.

## Model weights

The foundation-model checkpoints come to 1.1 GB. Their licence is
non-commercial and forbids redistribution, so they are absent, have no default
location and are never fetched automatically. `data/sources/README.md` and
`requirements.txt` say how to obtain them and `code/common/paths.py` says where
to put them. Every result file a model head produced names the generation behind
it, so a reader who lacks the weights can still read the numbers.

The two encoder checkpoints under `data/encoder-checkpoints/` are the opposite
case. They were pretrained by this work on the corpus described in
`data/corpus/recipe.json`, are this work's own output and ship under the data
licence.

## Shared project material

Part of the working directory belonged to a separate manuscript that is not yet
published and that shares provenance with this one, 2.1 GB in all. It is
excluded for two reasons that coincide: it is another paper's material, and its
contents do not belong in this package. The routines this work depends on were
lifted out of it into `code/common/` under neutral names, so nothing here
imports from it and no path in this package points into it.

`results/reference-baselines/` borrows aggregate rows for standard baselines,
partial least squares, support vector regression, a one-dimensional
convolutional network and its augmented variant, that were run on the same
thirteen tasks under the same protocol. These are conventional methods; no
method from that other manuscript appears in any table.

## Working material

Excluded as internal, and of no use to a reader: the manuscript sources
themselves; the figure and table generators and their intermediate files; the
audit and consistency scripts used while writing; the third-party writing tools
under `skills/`; caches, byte-code directories, backup files and LaTeX
intermediates.

The project used shorthand of its own while the work was being done, of the kind
every project accumulates: experiment numbers, short codes for each arm, and
abbreviated task names. None of it survives into this package, in file names, in
column headers or in cell values, and no key to it is published here.
`GLOSSARY.md` records the vocabulary that replaced it.

## What this means for reproducing a result

Two claims in the paper can be checked without running anything: every reported
number is present in `results/` as shipped, and the file that carries it is
named in `REPRODUCE.md`. Re-running an experiment needs three things beyond this
package: the public datasets, which `data/sources/` obtains; the foundation-model
weights, which the reader obtains under the vendor's licence; and compute, whose
scale `REPRODUCE.md` states family by family from the timings the result files
themselves record. `REPRODUCE.md` also gives the command for each family and the
environment variables the loaders read.
