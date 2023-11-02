"""
Contains (publicly available, but not officially supported) experimental
features of iterative_ensemble_smoother
"""
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy as sp

from iterative_ensemble_smoother import ESMDA


def correlation_threshold(ensemble_size: int) -> float:
    """Decides whether or not to use user-defined or default threshold.

    Default threshold taken from luo2022,
    Continuous Hyper-parameter OPtimization (CHOP) in an ensemble Kalman filter
    Section 2.3 - Localization in the CHOP problem
    """
    # return 3 / np.sqrt(ensemble_size)
    return 0.33


if __name__ == "__main__":
    # Example showing how to use row scaling
    num_parameters = 20
    num_observations = 15
    num_ensemble = 10

    rng = np.random.default_rng(42)

    X = rng.normal(size=(num_parameters, num_ensemble))
    A = np.exp(np.random.randn(num_observations, num_parameters))
    Y = rng.normal(size=(num_observations, num_ensemble))
    covariance = np.exp(rng.normal(size=num_observations))
    observations = rng.normal(size=num_observations, loc=1)

    # Compute correlation matrix between parameters X and responses Y
    from iterative_ensemble_smoother.esmda_inversion import empirical_cross_covariance

    cov_XY = empirical_cross_covariance(X, Y)
    stds_X = np.std(X, axis=1, ddof=1)
    stds_Y = np.std(Y, axis=1, ddof=1)
    corr_XY = (cov_XY / stds_X[:, np.newaxis]) / stds_Y[np.newaxis, :]

    corr_XY_zero_mask = corr_XY < correlation_threshold(ensemble_size=X.shape[1])

    # Approach 1 - naive approach
    smoother = ESMDA(covariance=covariance, observations=observations, alpha=1, seed=1)
    D = smoother.perturb_observations(size=Y.shape, alpha=1)
    cov_XY[corr_XY_zero_mask] = 0  # Set covariance to zero
    C_DD = empirical_cross_covariance(Y, Y)
    X_posterior = X + cov_XY @ sp.linalg.solve(C_DD + 1 * np.diag(covariance), (D - Y))

    # Approach 2

    corr_XY_bool = corr_XY > correlation_threshold(ensemble_size=X.shape[1])

    print(cov_XY.shape)


class RowScaling:
    # Illustration of how row scaling works, `multiply` is the important part
    # For the actual implementation, which is more involved, see:
    # https://github.com/equinor/ert/blob/84aad3c56e0e52be27b9966322d93dbb85024c1c/src/clib/lib/enkf/row_scaling.cpp
    def __init__(self, alpha=1.0):
        """Alpha is the strength of the update."""
        assert 0 <= alpha <= 1.0
        self.alpha = alpha

    def multiply(self, X, K):
        """Takes a matrix X and a matrix K and performs alpha * X @ K."""
        # This implementation merely mimics how RowScaling::multiply behaves
        # in the C++ code. It mutates the input argument X instead of returning.
        X[:, :] = X @ (K * self.alpha)


def ensemble_smoother_update_step_row_scaling(
    *,
    covariance: npt.NDArray[np.double],
    observations: npt.NDArray[np.double],
    X_with_row_scaling: List[Tuple[npt.NDArray[np.double], RowScaling]],
    Y: npt.NDArray[np.double],
    seed: Union[np.random._generator.Generator, int, None] = None,
    inversion: str = "exact",
    truncation: float = 1.0,
):
    """Perform a single ESMDA update (ES) with row scaling.
    The matrices in X_with_row_scaling WILL BE MUTATED.
    See the ESMDA class for information about input arguments.


    Explanation of row scaling
    --------------------------

    The ESMDA update can be written as:

        X_post = X_prior + X_prior @ K

    where K is a transition matrix. The core of the row scaling approach is that
    for each row i in the matrix X, we apply an update with strength alpha:

        X_post = X_prior + alpha * X_prior @ K
        X_post = X_prior @ (I + alpha * K)

    Clearly 0 <= alpha <= 1 denotes the 'strength' of the update; alpha == 1
    corresponds to a normal smoother update and alpha == 0 corresponds to no
    update. With the per row transformation of X the operation is no longer matrix
    multiplication but the pseudo code looks like:

        for i in rows:
            X_i_post = X_i_prior @ (I + alpha_i * K)

    See also original code:
        https://github.com/equinor/ert/blob/84aad3c56e0e52be27b9966322d93dbb85024c1c/src/clib/lib/enkf/row_scaling.cpp#L51

    """

    # Create ESMDA instance and set alpha=1 => run single assimilation (ES)
    smoother = ESMDA(
        covariance=covariance,
        observations=observations,
        seed=seed,
        inversion=inversion,
        alpha=1,
    )

    # Create transition matrix - common to all parameters in X
    transition_matrix = smoother.compute_transition_matrix(
        Y=Y, alpha=1, truncation=truncation
    )

    # The transition matrix K is a matrix such that
    #     X_posterior = X_prior + X_prior @ K
    # but the C++ code in ERT requires a transition matrix F that obeys
    #     X_posterior = X_prior @ F
    # To accomplish this, we add the identity to the transition matrix in place
    np.fill_diagonal(transition_matrix, transition_matrix.diagonal() + 1)

    # Loop over groups of rows (parameters)
    for X, row_scale in X_with_row_scaling:

        # In the C++ code, multiply() will transform the transition matrix F as
        #    F_new = F * alpha + I * (1 - alpha)
        # but the transition matrix F that we pass below is F := K + I, so:
        #    F_new = (K + I) * alpha + I * (1 - alpha)
        #    F_new = K * alpha + I * alpha + I - I * alpha
        #    F_new = K * alpha + I
        # And the update becomes : X_posterior = X_prior @ F_new
        # The update in the C++ code is equivalent to
        #    X_posterior = X_prior + alpha * X_prior @ K
        # if we had used the original transition matrix K that is returned from
        # ESMDA.compute_transition_matrix
        row_scale.multiply(X, transition_matrix)

    return X_with_row_scaling
