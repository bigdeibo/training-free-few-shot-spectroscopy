"""Representations the foundation model is fitted on.

The paper's first question is which representation of a spectrum makes a
tabular foundation model work in the few-shot regime. Twelve answers are
compared; this module builds eleven of them and names the twelfth.

  raw-spectra-512                 the SNV spectra as they enter, 512 points
  corpus-pca-N                    N principal scores of a PCA fitted on the
                                  mixed corpus (data/corpus/), not on the
                                  simulated fuel corpus under data/simulated-fuel/.
                                  The mixed corpus is assembled by
                                  code/corpus_augmentation/run_build_corpus.py:
                                  only its simulated half ships, so run that
                                  first if data/corpus/spectra.npy is absent
  corpus-pca-N-fit-M              the same, fitted on an M-spectrum subsample of
                                  the corpus (the saturation curve)
  wavelet-scattering              second-order Morlet scattering, pooled
  cars-selected                   channels kept by CARS on the support set
  random-projections-100          a fixed random Gaussian projection, the
                                  control that isolates "any 100 dimensions"
  random-init-encoder             128-d embeddings of an untrained encoder
  simclr-encoder                  128-d embeddings of the contrastive encoder
  masked-autoencoder              128-d embeddings of the masked autoencoder

`cars-selected` is absent from `build` because it needs the support labels, so
the caller runs it; see `code/representation_comparison/`.

The two trained encoders ship as checkpoints under data/encoder-checkpoints/.
`random-init-encoder` is reproduced by seeding the same architecture, which is
why the seed below is pinned rather than drawn.
"""
from __future__ import annotations

import numpy as np

from common.paths import CORPUS_SIMULATED, CORPUS_SPECTRA, ENCODER_CHECKPOINTS

EMB_DIM = 128
RANDOM_INIT_SEED = 20260915      # fixes the untrained-encoder control
RANDOM_PROJECTION_SEED = 20260915

# Representations the caller standardises by support statistics; the others
# enter as they are. `raw-spectra-512` is already SNV-corrected per spectrum and
# `cars-selected` is a subset of those channels, so both are left alone.
STANDARDIZED = {
    "wavelet-scattering", "random-projections-100",
    "random-init-encoder", "simclr-encoder", "masked-autoencoder",
}

_pca_cache: dict = {}


def _corpus():
    if not CORPUS_SPECTRA.exists():
        raise FileNotFoundError(
            f"{CORPUS_SPECTRA} is absent. The corpus a projection is fitted on "
            f"is 3339 measured plus 3000 simulated spectra; this archive ships "
            f"only the simulated half, as {CORPUS_SIMULATED.name}. Assemble the "
            f"rest with code/corpus_augmentation/run_build_corpus.py, which "
            f"reads the public datasets under SPEC_DATA_ROOT (see "
            f"data/sources/README.md).")
    return np.load(CORPUS_SPECTRA)


def corpus_pca(X, n_components, fit_on=None):
    """Scores under a PCA fitted on the mixed corpus.

    `fit_on` fits on a seeded subsample of the corpus instead of all of it,
    which is how the corpus-size saturation curve is drawn. The subsample is
    drawn without replacement from a seed that depends only on its size, so the
    same subspace is used for every task at a given size.

    The caller's floating-point precision is carried through the projection.
    The spectra load in single precision and the benchmark projected them as
    they loaded, so evaluating the same projection in double shifts every score
    by about one part in a million relative to the shipped tables, which the
    small-support fits below then amplify. Pass a double-precision array to get
    a double-precision projection.
    """
    key = (n_components, fit_on)
    if key not in _pca_cache:
        from sklearn.decomposition import PCA
        corpus = _corpus()
        if fit_on is not None and fit_on < len(corpus):
            rng = np.random.default_rng(66000 + fit_on)
            corpus = corpus[rng.choice(len(corpus), size=fit_on, replace=False)]
        _pca_cache[key] = PCA(n_components=n_components, random_state=0).fit(corpus)
    X = np.asarray(X)
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(float)
    return _pca_cache[key].transform(X)


def wavelet_scattering(X):
    """Second-order Morlet scattering along the 512-point axis, pooled.

    Five dyadic scales give five first-order and ten second-order coefficients;
    the modulus response of each is averaged over the axis, so one spectrum
    becomes fifteen numbers. Training-free and shift-stable.
    """
    import pywt
    X = np.asarray(X)
    scales = 2.0 ** np.arange(1, 6)
    wavelet = "cmor1.5-1.0"
    first = [np.abs(pywt.cwt(X, [s], wavelet, axis=1)[0][0]) for s in scales]
    feats = [u.mean(axis=1) for u in first]
    for i in range(len(scales)):
        for j in range(i + 1, len(scales)):
            c = pywt.cwt(first[i], [scales[j]], wavelet, axis=1)[0][0]
            feats.append(np.abs(c).mean(axis=1))
    return np.stack(feats, axis=1)


def random_projection(X, n_components=100, seed=RANDOM_PROJECTION_SEED):
    """A fixed Gaussian projection of the 512-point spectra.

    The control for the representation claim: it keeps the dimensionality of
    the PCA arms while discarding every spectral structure.
    """
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((np.asarray(X).shape[1], n_components)) / np.sqrt(
        np.asarray(X).shape[1])
    return np.asarray(X) @ R


def encoder_embeddings(X, weights):
    """128-d embeddings from the one-dimensional ResNet.

    `weights` is one of `random-init-encoder`, `simclr-encoder` or
    `masked-autoencoder`. The untrained control draws its initialisation from a
    pinned seed; the other two load a shipped checkpoint.
    """
    import torch
    from common.encoder import ResNet1Encoder

    X = np.asarray(X, dtype=np.float32)
    net = ResNet1Encoder(X.shape[1], emb_dim=EMB_DIM)
    if weights == "random-init-encoder":
        torch.manual_seed(RANDOM_INIT_SEED)
    else:
        name = {"simclr-encoder": "simclr.pt",
                "masked-autoencoder": "masked-autoencoder.pt"}[weights]
        ck = torch.load(ENCODER_CHECKPOINTS / name, map_location="cpu",
                        weights_only=False)
        net.load_state_dict(ck["enc"] if "enc" in ck else ck)
    net.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    with torch.no_grad():
        Z = net(torch.from_numpy(X)[:, None].to(device)).float().cpu().numpy()
    return Z


def standardize(F_tr, F_te, clip=5.0):
    """Support-set standardisation with a symmetric clamp.

    Every learned or pooled representation is scaled this way before the head
    sees it, so that the head's input scale does not depend on the arm. The
    clamp bounds the influence of a query spectrum far from the support.

    The mean and the spread are taken in the array's own precision, as they were
    in the benchmark. Widening to double first moves the scaled values by a few
    parts in a million, which is not always negligible for a head fitted on a
    handful of spectra.
    """
    F_tr = np.asarray(F_tr)
    F_te = np.asarray(F_te)
    if not np.issubdtype(F_tr.dtype, np.floating):
        F_tr = F_tr.astype(float)
    if not np.issubdtype(F_te.dtype, np.floating):
        F_te = F_te.astype(float)
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-8
    return np.clip((F_tr - mu) / sd, -clip, clip), \
        np.clip((F_te - mu) / sd, -clip, clip)


def build(name, X):
    """Features for every representation that needs no labels.

    `name` is one of the twelve terms listed in the module docstring except
    `cars-selected`. Returns `(features, standardized)`, where the second value
    says whether the caller should standardise before fitting.

    The caller's precision is not widened here. The spectra load in single
    precision and every arm of the benchmark was evaluated on them as loaded;
    the transforms below that need a particular dtype coerce it themselves.
    """
    X = np.asarray(X)
    if name in ("raw-spectra-512", "cars-selected"):
        return X, False
    if name.startswith("corpus-pca-"):
        head, _, tail = name[len("corpus-pca-"):].partition("-fit-")
        return corpus_pca(X, int(head), int(tail) if tail else None), True
    if name == "wavelet-scattering":
        return wavelet_scattering(X), True
    if name == "random-projections-100":
        return random_projection(X), True
    if name.endswith("-encoder"):
        return encoder_embeddings(X, name), True
    raise KeyError(f"unknown representation {name!r}")
