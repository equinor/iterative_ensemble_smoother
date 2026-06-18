"""
Linear regression
-----------------
The purpose of these functions is to learn a sparse mapping from predictors X
to responses y. There are three requirements:
    1. The method must be accurate (a good, predictive model)
    2. The method must be fast (our data is large, with high dimensionality p >> N)
    3. The method must produce sparse results (for interpretability)

>>> import numpy as np
>>> from sklearn.datasets import make_regression
>>> from sklearn.model_selection import train_test_split
>>> (X_full, y_full, coef) = make_regression(n_samples=100, n_features=1000,
...                                          n_informative=10,
...                                          noise=0.3, coef=True,
...                                          random_state=1)
>>> (~np.isclose(coef, 0.0)).sum()
np.int64(10)

>>> X, X_test, y, y_test = train_test_split(X_full, y_full,
...                                         test_size=0.5, random_state=2)

Let us see how many non-zero coefficients sklearn finds:

>>> H_sparse_l1 = linear_l1_regression(U=X, Y=y[:, None])
>>> coef_ = H_sparse_l1.todense().ravel()
>>> (~np.isclose(coef_, 0.0)).sum()
np.int64(54)

Let us check our own Lasso-like boosting algorithm:

>>> H_sparse_ic = linear_boost_ic_regression(U=X, Y=y[:, None])
>>> coef_ = H_sparse_ic.todense().ravel()
>>> (~np.isclose(coef_, 0.0)).sum()
np.int64(5)

Let us check RMSE on the test set for our models:

>>> from sklearn.metrics import mean_squared_error
>>> y_pred = X_test @ H_sparse_l1.todense().ravel()
>>> float(np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)))
135.61...

>>> y_pred = X_test @ H_sparse_ic.todense().ravel()
>>> float(np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)))
116.06...

With the true coefficients, the RMSE is 100x smaller. This highlights
how hard it is to infer coefficients when p >> N.

>>> y_pred = X_test @ coef
>>> float(np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)))
0.24...

Still beat the dummy model though:

>>> y_pred = np.mean(y) * np.ones_like(y_test)
>>> float(np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)))
213.48...
"""

import logging

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.stats import chi2
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def linear_l1_regression(
    U: NDArray[np.floating], Y: NDArray[np.floating]
) -> sp.csc_array:
    """Performs LASSO regression for each response in Y against predictors in
    U, constructing a sparse matrix of regression coefficients.

    The function scales features in U using standard scaling before applying
    LASSO, then re-scales the coefficients to the original scale of U. This
    extracts the effect of each feature in U on each response in Y, ignoring
    intercepts and constant terms.

    Parameters
    ----------
    U : np.ndarray
        2D array of predictors with shape (n, p).
    Y : np.ndarray
        2D array of responses with shape (n, m).

    Returns
    -------
    H_sparse : scipy.sparse.csc_array
        Sparse matrix (m, p) with re-scaled LASSO regression coefficients for
        each response in Y.
    """
    n, p = U.shape  # p: number of features
    n_y, m = Y.shape  # m: number of y responses

    # Assert that the first dimension of U and Y are the same
    if n != n_y:
        raise ValueError("Number of samples in U and Y must be the same")

    log.info("Learning sparse linear map of shape %s", (m, p))

    scaler_u = StandardScaler()
    U_scaled = scaler_u.fit_transform(U)

    scaler_y = StandardScaler()
    Y_scaled = scaler_y.fit_transform(Y)

    # Loop over features
    i_H, j_H, values_H = [], [], []
    for j in range(m):
        y_j = Y_scaled[:, j]

        # Learn individual regularization and fit
        eps = 1e-3
        max_iter = 10000
        model_cv = LassoCV(cv=10, fit_intercept=False, max_iter=max_iter, eps=eps)
        model_cv.fit(U_scaled, y_j)

        # Extract coefficients
        for non_zero_ind in model_cv.coef_.nonzero()[0]:
            i_H.append(j)
            j_H.append(non_zero_ind)
            values_H.append(
                scaler_y.scale_[j]
                * model_cv.coef_[non_zero_ind]
                / scaler_u.scale_[non_zero_ind]
            )

    H_sparse = sp.csc_array(
        (np.array(values_H), (np.array(i_H), np.array(j_H))), shape=(m, p)
    )

    assert H_sparse.shape == (m, p), "Shape of H_sparse must be (m, p)"

    log.info(
        "Density: %.1f%% (%d / %d)", 100 * H_sparse.nnz / (m * p), H_sparse.nnz, m * p
    )

    return H_sparse


def expected_max_chisq(p: int) -> float:
    """Expected maximum of p central chi-square(1) random variables."""

    def dmaxchisq(x: float) -> float:
        return float(1.0 - np.exp(p * chi2.logcdf(x, df=1)))

    expectation, _ = quad(dmaxchisq, 0, np.inf)
    return float(expectation)


def mse(residuals: NDArray[np.floating]) -> float:
    return float(0.5 * np.mean(residuals**2))


def calculate_psi_M(
    x: NDArray[np.floating], y: NDArray[np.floating], beta_estimate: float
) -> tuple[NDArray[np.floating], float]:
    """The psi/score function for mse: 0.5*residual**2."""
    residuals = y - beta_estimate * x
    psi = -residuals * x
    M = -np.mean(x**2)
    return psi, M


def calculate_influence(
    x: NDArray[np.floating], y: NDArray[np.floating], beta_estimate: float
) -> NDArray[np.floating]:
    """The influence of (x, y) on beta_estimate as an mse M-estimator."""
    psi, M = calculate_psi_M(x, y, beta_estimate)
    return psi / M


def boost_linear_regression(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    learning_rate: float = 0.5,
    tol: float = 1e-6,
    max_iter: int = 10000,
    effective_dimension: int | None = None,
) -> NDArray[np.floating]:
    """Boost coefficients of linearly regressing y on standardized X.

    The coefficient selection utilizes information theoretic weighting. The
    stopping criterion utilizes information theoretic loss-reduction.
    """
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    residuals = y.copy()  # residuals = y - X @ coef = y - X @ 0 = y
    residuals_loo = y.copy()

    # A stricter criterion is the loo-adjustment: mse(residuals_loo)-mse
    # (residuals). This converges to TIC. Under certain conditions this is AIC.
    # At worst, we are maximizing squares. See Lunde 2020 Appendix A. This
    #  needs to be adjusted for.
    # The mse_factor adjusts for this.
    # if effective_dimension is None:
    #    effective_dimension = n_features
    # mse_factor = expected_max_chisq(np.ceil(effective_dimension))

    for _ in range(max_iter):
        coef_changes = np.dot(X.T, residuals) / n_samples

        # Could be adjusted for IC -- some features already included
        # The IC would build in additional motivation for sparsity
        feature_evaluation = np.abs(coef_changes)

        # Select feature based on loss criterion
        best_feature = np.argmax(feature_evaluation)
        beta_estimate = coef_changes[best_feature]

        # adjust to loo estimates for coef_change
        # Inlined influence: psi / M
        # where influence = -residuals * x and
        # M = -mean(x^2) = -1 since x standardized
        X_best = X[:, best_feature]
        influence = (residuals - beta_estimate * X_best) * X_best
        beta_estimate_loo = beta_estimate - influence / n_samples

        # residuals_full = residuals - beta_estimate * X[:, best_feature]
        residuals_full_loo = residuals_loo - learning_rate * beta_estimate_loo * X_best

        if mse(residuals_loo) < mse(residuals_full_loo):
            break

        # Check if adding the full weight of the feature would decrease loss
        # if mse(residuals) < mse(residuals_full) + mse_factor * (
        #    mse(residuals_full_loo) - mse(residuals_full)
        # ):
        #    break

        coef_change = beta_estimate * learning_rate

        # Check for convergence
        if np.abs(coef_change) < tol:
            break
        # Update
        residuals -= coef_change * X_best
        coefficients[best_feature] += coef_change

        # loo update
        residuals_loo = residuals_full_loo

    # ensure cutoff values -- very small if data standardized
    # prefer sparsity
    cutoff = 2.0 * learning_rate / np.sqrt(n_samples)  # 2.0: 95% ci-ish
    coefficients[np.abs(coefficients) < cutoff] = 0

    return coefficients


def _fit_single_response_boost(
    j: int,
    U_scaled: NDArray[np.floating],
    y_j: NDArray[np.floating],
    learning_rate: float,
    effective_dimension: int | None,
) -> tuple[int, NDArray[np.integer], NDArray[np.floating]]:
    """Fit boosted regression for a single response column.

    Returns sparse representation (j, nonzero_indices, nonzero_values)
    """
    coefficients_j = boost_linear_regression(
        U_scaled,
        y_j,
        learning_rate=learning_rate,
        effective_dimension=effective_dimension,
    )
    nonzero = coefficients_j.nonzero()[0]
    return j, nonzero, coefficients_j[nonzero]


def linear_boost_ic_regression(
    U: NDArray[np.floating],
    Y: NDArray[np.floating],
    *,
    learning_rate: float = 0.5,
    effective_dimension: int | None = None,
    n_jobs: int = -1,
) -> sp.csc_array:
    """Performs boosted linear regression for each response in Y against
    predictors in U, constructing a sparse matrix of regression coefficients.
    The complexity is tuned with an information theoretic approach.

    The function scales features in U using standard scaling before learning
    the coefficients, then re-scales the coefficients to the original scale of
    U. This extracts the effect of each feature in U on each response in Y,
    ignoring intercepts and constant terms.

    Parameters
    ----------
    U : np.ndarray
        2D array of predictors with shape (n, p).
    Y : np.ndarray
        2D array of responses with shape (n, m).
    n_jobs : int, optional
        Number of parallel jobs. Use -1 for all CPUs. Default is -1.

    Returns
    -------
    H_sparse : scipy.sparse.csc_array
        Sparse matrix (m, p) with re-scaled LASSO regression coefficients for
        each response in Y.
    """
    n, p = U.shape  # p: number of features
    n_y, m = Y.shape  # m: number of y responses

    # Assert that the first dimension of U and Y are the same
    if n != n_y:
        raise ValueError("Number of samples in U and Y must be the same")

    log.info("Learning sparse linear map of shape %s", (m, p))

    scaler_u = StandardScaler()
    U_scaled = scaler_u.fit_transform(U)

    scaler_y = StandardScaler()
    Y_scaled = scaler_y.fit_transform(Y)

    # Fit responses in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_single_response_boost)(
            j=j,
            U_scaled=U_scaled,
            y_j=Y_scaled[:, j],
            learning_rate=learning_rate,
            effective_dimension=effective_dimension,
        )
        for j in range(m)
    )

    # Assemble sparse matrix from results
    i_H, j_H, values_H = [], [], []
    for j, nonzero_indices, nonzero_values in results:
        k = len(nonzero_indices)
        i_H.append(np.full(shape=k, fill_value=j, dtype=np.intp))
        j_H.append(nonzero_indices)
        values_H.append(
            scaler_y.scale_[j] * nonzero_values / scaler_u.scale_[nonzero_indices]
        )

    H_sparse = sp.csc_array(
        (np.concatenate(values_H), (np.concatenate(i_H), np.concatenate(j_H))),
        shape=(m, p),
    )

    # Assert shape of H_sparse
    assert H_sparse.shape == (m, p), "Shape of H_sparse must be (m, p)"

    log.info(
        "Density: %.1f%% (%d / %d)", 100 * H_sparse.nnz / (m * p), H_sparse.nnz, m * p
    )

    return H_sparse


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v"])
