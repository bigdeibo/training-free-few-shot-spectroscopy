"""The classical reference methods used as arms or comparators.

Both routines tune on the support set alone. Variable selection in particular
runs its cross-validation inside `fit`, never on the query set, which is the
anti-leakage rule the whole comparison rests on.
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict


def pls_best_components(X, y, max_nc=15, folds=None):
    """Number of latent components for PLS, chosen by inner cross-validation on
    the support set.

    The default scheme is leave-one-out below 40 samples and five-fold above it.
    The Raman arms pass their own `folds`, which is what caps their grid: with
    three folds and a support set of a few hundred well spectra each fold still
    trains on far more samples than the grid reaches, but the cap keeps the
    comparison with the near-infrared arms on the same footing.
    """
    n = len(y)
    if folds is None:
        cv = KFold(5, shuffle=True, random_state=0) if n > 40 else LeaveOneOut()
    else:
        cv = KFold(folds)
    best_nc, best_rmse = 1, np.inf
    for nc in range(1, min(max_nc, n - 1, X.shape[1]) + 1):
        try:
            pred = cross_val_predict(PLSRegression(n_components=nc), X, y,
                                     cv=cv).ravel()
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            if rmse < best_rmse:
                best_rmse, best_nc = rmse, nc
        except Exception:
            break
    return best_nc


def cars_select(X, y, max_nc=15, n_runs=50, seed=0):
    """Competitive adaptive reweighted sampling (Jiang 2010).

    Each round fits PLS to a Monte Carlo subsample of the support spectra to get
    coefficient magnitudes, keeps a shrinking fraction of the channels under an
    exponentially decreasing retention ratio, and scores the surviving subset by
    cross-validated RMSE. The best subset over all rounds is returned as a
    boolean channel mask. If fewer than two channels survive the mask is
    replaced by the full spectrum, so the worst case degrades to plain PLS
    rather than to an exception.
    """
    n, p = X.shape
    if n < 4 or n_runs < 2:
        return np.ones(p, dtype=bool)
    rng = np.random.default_rng(seed)
    k_edf = np.log(max(p / 2.0, 1.0)) / (n_runs - 1)
    ratio = np.exp(-k_edf * np.arange(n_runs))
    cv = LeaveOneOut() if n <= 40 else KFold(5, shuffle=True, random_state=0)

    # One latent-component count for the whole run, as in the original paper.
    # Reselecting it each round would cost tens of thousands of extra PLS fits
    # per episode on the multivariate tasks.
    A = max(1, min(pls_best_components(X, y, max_nc), n - 1))
    sel = np.ones(p, dtype=bool)
    best_rmsecv, best_sel = np.inf, np.ones(p, dtype=bool)
    for i in range(n_runs):
        m = max(int(round(0.9 * n)), 3)
        idx = rng.choice(n, m, replace=False)
        Xs, ys = X[idx][:, sel], y[idx]
        nc = max(1, min(A, m - 1, int(sel.sum())))
        try:
            w_sel = np.abs(PLSRegression(n_components=nc).fit(Xs, ys).coef_.ravel())
        except Exception:
            w_sel = np.ones(int(sel.sum()))

        n_keep = max(2, int(round(ratio[i] * p)))
        sel_idx = np.where(sel)[0]
        if len(sel_idx) <= n_keep:
            chosen = sel_idx
        else:
            ws_sum = w_sel.sum()
            chosen = rng.choice(sel_idx, size=n_keep, replace=False,
                                p=w_sel / ws_sum if ws_sum > 0 else None)
        sel = np.zeros(p, dtype=bool)
        sel[chosen] = True

        if sel.sum() >= 2:
            nc_cv = max(1, min(A, n - 1, int(sel.sum())))
            try:
                pred = cross_val_predict(PLSRegression(n_components=nc_cv),
                                         X[:, sel], y, cv=cv).ravel()
                rmsecv = float(np.sqrt(np.mean((pred - y) ** 2)))
            except Exception:
                rmsecv = np.inf
        else:
            rmsecv = np.inf
        if rmsecv < best_rmsecv:
            best_rmsecv, best_sel = rmsecv, sel.copy()

    if best_sel.sum() < 2:
        best_sel = np.ones(p, dtype=bool)
    return best_sel
