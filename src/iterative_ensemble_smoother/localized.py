# -*- coding: utf-8 -*-

from iterative_ensemble_smoother.esmda import BaseESMDA
from typing import Union

import numpy as np
import numpy.typing as npt

from iterative_ensemble_smoother.utils import sample_mvnormal

from sklearn.linear_model import LassoCV
from sklearn.linear_model import MultiTaskLassoCV


def linear_l1_regression(X, Y):
    """
    Performs LASSO regression for each response in Y against predictors in U,
    constructing a sparse matrix of regression coefficients.

    The function scales features in U using standard scaling before applying
    LASSO, then re-scales the coefficients to the original scale of U. This
    extracts the effect of each feature in U on each response in Y, ignoring
    intercepts and constant terms.

    Parameters
    ----------
    U : np.ndarray
        2D array of predictors with shape (num_variables, ensemble_size).
    Y : np.ndarray
        2D array of responses with shape (num_variables, ensemble_size).


    Examples
    --------
    >>> K = np.array([[-0.2, -0.9],
    ...               [-1.5,  0.2],
    ...               [-0.2,  0.1],
    ...               [ 0.8, -0.4]])
    >>> X = np.array([[ 0.8,  2. , -0.8],
    ...               [ 0.4,  1.7, -0.3]])
    >>> Y = K @ X
    >>> K_est = linear_l1_regression(X, Y)
    >>> np.allclose(K, K_est, atol=0.005)
    True

    """
    # https://github.com/equinor/graphite-maps/blob/main/graphite_maps/linear_regression.py
    # An option is to use MultiTaskLasso too, but not exactly the same objective
    # https://scikit-learn.org/stable/modules/linear_model.html#multi-task-lasso

    # The goal is to solve K @ X = Y for K, or equivalently X.T @ K.T = Y.T
    num_parameters, ensemble_size1 = X.shape
    num_observations, ensemble_size2 = Y.shape

    assert ensemble_size1 == ensemble_size2, "Number of cols in X and Y must be equal"

    # Scale to unit standard deviation. We do this because of numerics in Lasso
    stds_X = np.std(X, axis=1)
    stds_Y = np.std(Y, axis=1)
    X = X / stds_X[:, np.newaxis]
    Y = Y / stds_Y[:, np.newaxis]

    # Loop over features
    coefs = []
    cv = min(5, ensemble_size1)
    alphas = np.logspace(-6, 6, num=17)
    
    # The matrix K in K @ X = Y is built up row by row
    for j, y_j in enumerate(Y): # Loop over rows in Y

        # Learn individual regularization and fit
        model_cv = LassoCV(alphas=alphas, fit_intercept=False, max_iter=10_000, cv=cv)
        model_cv.fit(X.T, y_j)  # model_cv.alpha_
        # TODO: Consider searching smarter for alphas, e.g. momentum or
        # searching within 1-2 orders of magnitude of previous alpha?

        # Scale back the coefficients
        coef_scale = stds_Y[j] / stds_X
        coefs.append(model_cv.coef_ * coef_scale)

    K = np.vstack(coefs)
    assert K.shape == (num_observations, num_parameters)
    return K


class LassoES(BaseESMDA):
    """
    Implement an Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

    The implementation follows :cite:t:`EMERICK2013`.

    Parameters
    ----------
    covariance : np.ndarray
        Either a 1D array of diagonal covariances, or a 2D covariance matrix.
        The shape is either (num_observations,) or (num_observations, num_observations).
        This is C_D in Emerick (2013), and represents observation or measurement
        errors. We observe d from the real world, y from the model g(x), and
        assume that d = y + e, where the error e is multivariate normal with
        covariance given by `covariance`.
    observations : np.ndarray
        1D array of shape (num_observations,) representing real-world observations.
        This is d_obs in Emerick (2013).
    seed : integer or numpy.random._generator.Generator, optional
        A seed or numpy.random._generator.Generator used for random number
        generation. The argument is passed to numpy.random.default_rng().
        The default is None.

    """

    def __init__(
        self,
        covariance: npt.NDArray[np.double],
        observations: npt.NDArray[np.double],
        seed: Union[np.random._generator.Generator, int, None] = None,
    ) -> None:
        """Initialize the instance."""

        super().__init__(covariance=covariance, observations=observations, seed=seed)

    def assimilate(
        self,
        X: npt.NDArray[np.double],
        Y: npt.NDArray[np.double],
        *,
        alpha: float = 1.0,
    ) -> npt.NDArray[np.double]:
        
        # Verify shapes
        _, num_ensemble = X.shape
        num_outputs, num_emsemble2 = Y.shape
        assert (
            num_ensemble == num_emsemble2
        ), "Number of ensemble members in X and Y must match"
        if not np.issubdtype(X.dtype, np.floating):
            raise TypeError("Argument `X` must contain floats")
        if not np.issubdtype(Y.dtype, np.floating):
            raise TypeError("Argument `Y` must contain floats")

        # Center X and Y
        X_center = X - np.mean(X, axis=1, keepdims=True)
        Y_center = Y - np.mean(Y, axis=1, keepdims=True)
        
        # We have R @ R.T / N ~ C_D
        R = np.sqrt(alpha) * sample_mvnormal(
            C_dd_cholesky=self.C_D_L, rng=self.rng, size=num_ensemble
        )
        
        # S = R * sqrt((N - 1) / N), so that S @ S.T / (N-1) = covariance
        S = np.sqrt((num_ensemble - 1) / num_ensemble) * R

        # Compute regularized Kalman gain matrix matrix by solving:
        # (Y + S) K = X
        # The definition of K is:
        # (cov(Y, Y) + covariance) K = cov(X, Y), which is approximately
        # (Y @ Y.T + S @ L.T) K = X @ Y.T, [X and Y are centered] which is approximately
        # (Y + S) @ (Y + S).T @ K = X @ Y.T, which is approximately
        # (Y + S) @ K = X
        # TODO: Write this out properly.
        K = linear_l1_regression(X=(Y_center + S), Y=X_center)

        D = self.perturb_observations(ensemble_size=num_ensemble, alpha=alpha)
        return X + K @ (D - Y)


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
        ]
    )
