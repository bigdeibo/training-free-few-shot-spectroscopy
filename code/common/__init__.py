"""Shared utilities for the experiments in this package.

Nothing here belongs to one experiment. The nine modules cover what the run
scripts all need:

  paths      where the archive, the public datasets and the weights live
  tasks      the task registry, the paired split, the label scaling
  datasets   one loader per dataset, returning a common dictionary layout
  preprocess the per-spectrum standard normal variate transform
  features   the representations the foundation model is fitted on
  metrics    R2, RMSE and ratio of performance to deviation
  baselines  the PLS reference and the variable-selection variants
  encoder    the one-dimensional ResNet behind the encoder arms
  ridge_uq   closed-form ridge with jackknife+ intervals
  io         result-table writing and resumption

The scripts add `code/` to the path and import these as top-level modules, so
they must be run from a checkout of this archive as

    python code/representation_comparison/run_representation_comparison.py --help

and not from inside `code/common/`.
"""
