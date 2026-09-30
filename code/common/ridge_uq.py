"""Closed-form ridge regression with jackknife+ prediction intervals.

The classical uncertainty comparator of Section 2.5. Ridge is fitted in closed
form on the support set, so leave-one-out quantities come out of the hat matrix
without refitting, and the penalty is chosen by leave-one-out on those same
closed-form residuals.

Two details are deliberate and both matter at K <= 20.

Scale-only normalisation, with an unpenalised intercept. Support-mean centring
of the embeddings makes the columns sum to zero, which is an exact linear
dependency among K rows; with mean-zero labels and no intercept the
leave-one-out prediction then reproduces the held-out value exactly and
jackknife+ collapses. Dividing by the support standard deviation only, and
carrying an explicit unpenalised intercept, breaks the dependency while leaving
the geometry otherwise as the other arms see it.

Leave-one-out for the penalty, not generalised cross-validation. As the penalty
goes to zero in the K < p regime the fit interpolates, the degrees-of-freedom
denominator of GCV vanishes and GCV picks the grid minimum, collapsing every
interval to zero width. The closed-form leave-one-out residuals stay finite
there, so their criterion has a genuine interior minimum.

Intervals follow Barber et al., Annals of Statistics 2021, doi:10.1214/20-aos1965.
The guarantee is distribution-free coverage of at least 1-2*alpha.
"""
import numpy as np


def _augment(Phi, lam, fit_intercept):
    """Prepend an unpenalised intercept column when one is asked for."""
    K, p = Phi.shape
    if not fit_intercept:
        return Phi, lam * np.eye(p)
    Phi_aug = np.column_stack([np.ones(K), Phi])
    pen = np.eye(p + 1) * lam
    pen[0, 0] = 0.0
    return Phi_aug, pen


def ridge_solve(Phi, y, lam, fit_intercept=True):
    """Solve min ||Phi w - y||^2 + lam ||w||^2 in closed form.

    Phi is (K, p) support embeddings, scale-only normalised per the note above
    and deliberately *not* mean-centred; y is the support labels. They may be
    given on any linearly rescaled scale, raw physical units or z-scored: with
    the intercept unpenalised, y = s*y' + m gives
    ||Phi w + b - y||^2 + lam||w||^2 = s^2 (||Phi w' + b' - y'||^2 + lam||w'||^2),
    so the selected penalty is scale-invariant. Callers here pass raw labels.
    Returns the coefficients, the inverse of the normal matrix, and the diagonal
    of the hat matrix.
    """
    Phi_aug, pen = _augment(Phi, lam, fit_intercept)
    A = Phi_aug.T @ Phi_aug + pen
    Ainv = np.linalg.inv(A)
    w = Ainv @ Phi_aug.T @ y
    H = Phi_aug @ Ainv @ Phi_aug.T
    return w, Ainv, np.diag(H)


def loo_quantities(Phi, y, lam, fit_intercept=True):
    """Leave-one-out residuals from the hat matrix, with no refitting."""
    w, Ainv, h = ridge_solve(Phi, y, lam, fit_intercept)
    Phi_aug = np.column_stack([np.ones(len(Phi)), Phi]) if fit_intercept else Phi
    r_loo = (y - Phi_aug @ w) / np.clip(1.0 - h, 1e-8, None)
    return w, Ainv, h, r_loo


def loo_lambda(Phi, y, grid=None):
    """Choose the penalty by leave-one-out cross-validation on the closed-form
    residuals. The default selector."""
    if grid is None:
        grid = np.logspace(-4, 1, 26)
    best, best_lam = np.inf, grid[-1]
    for lam in grid:
        _, _, _, r_loo = loo_quantities(Phi, y, lam)
        score = float(np.mean(r_loo ** 2))
        if score < best:
            best, best_lam = score, lam
    return best_lam


def jackknife_plus_interval(Phi, y, lam, Phi_q, alpha, fit_intercept=True):
    """Jackknife+ prediction intervals for the query embeddings `Phi_q`.

    Returns `(lo, hi, info)` on the label scale of the `y` passed in. At K = 5
    the interval endpoints sit on a five-point grid, which is a property of the
    method and is reported as such.
    """
    w, Ainv, h, r_loo = loo_quantities(Phi, y, lam, fit_intercept)
    Phi_a = np.column_stack([np.ones(len(Phi)), Phi]) if fit_intercept else Phi
    Phi_qa = np.column_stack([np.ones(len(Phi_q)), Phi_q]) if fit_intercept else Phi_q

    G = Phi_qa @ Ainv @ Phi_a.T          # (nq, K)
    yhat_q = Phi_qa @ w                  # (nq,)
    # beta_{-i} = w - Ainv phi_i * r_loo[i], so yhat_{-i}(x) = yhat(x) - G[:,i] r_loo[i]
    Y_loo = yhat_q[:, None] - G * r_loo[None, :]
    lo = np.quantile(Y_loo - r_loo[None, :], alpha, axis=1, method="inverted_cdf")
    hi = np.quantile(Y_loo + r_loo[None, :], 1.0 - alpha, axis=1, method="inverted_cdf")
    return lo, hi, dict(yhat=yhat_q, r_loo=r_loo, Y_loo=Y_loo)
