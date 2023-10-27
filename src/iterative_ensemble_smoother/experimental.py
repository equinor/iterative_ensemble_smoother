"""
Contains (publicly available, but not officially supported) experimental
features of iterative_ensemble_smoother
"""
from typing import Union

import numpy as np
import numpy.typing as npt

from iterative_ensemble_smoother import ESMDA


class RowScaling:
    def multiply(self, X, K):
        """Takes a matrix X and a matrix K and performs X @ K."""
        return X @ K


def ensemble_smoother_update_step_row_scaling(
    *,
    covariance: npt.NDArray[np.double],
    observations: npt.NDArray[np.double],
    X_with_row_scaling: list[tuple[npt.NDArray[np.double], RowScaling]],
    Y: npt.NDArray[np.double],
    seed: Union[np.random._generator.Generator, int, None] = None,
    inversion: str = "exact",
    truncation: float = 1.0,
):
    """Perform a single ESMDA update (ES) with row scaling.
    The matrices in X_with_row_scaling WILL BE MUTATED.


    Explanation of row scaling
    --------------------------

    The ESMDA update can be written as:

        X_post = X_prior + X_prior @ K

    where K is a transition matrix. The core of the row scaling approach is that
    for each row i in the matrix X, we apply an update with strength alpha:

        X_post = X_prior + alpha * X_prior @ K

    Clearly 0 <= alpha <= 1 denotes the 'strength' of the update; alpha == 1
    corresponds to a normal smoother update and alpha == 0 corresponds to no
    update. With the per row transformation of X the operation is no longer matrix
    multiplication but the pseudo code looks like:

        for i in rows:
            X_i_post = X_i_prior + alpha * X_i_prior @ K

    See also original code:
        https://github.com/equinor/ert/blob/963f9bc08ebc87374b7ed3403c8ba78c20909ae9/src/clib/lib/enkf/row_scaling.cpp#L51

    """

    # https://github.com/equinor/ert/blob/963f9bc08ebc87374b7ed3403c8ba78c20909ae9/src/clib/lib/enkf/row_scaling.cpp#L51

    # Create ESMDA instance
    # Setting alpha=1 means we run a single data assimilation
    smoother = ESMDA(
        covariance=covariance,
        observations=observations,
        seed=seed,
        inversion=inversion,
        alpha=1,
    )

    # Create transition matrix - common to all parameters in X
    transition_matrix = smoother.compute_transition_matrix(Y=Y, alpha=1, truncation=1.0)

    # Loop over groups of rows (parameters)
    for X, row_scale in X_with_row_scaling:
        X += row_scale.multiply(X, transition_matrix)

    return X_with_row_scaling


if __name__ == "__main__":

    # Example showing how to use row scaling
    num_parameters = 100
    num_observations = 20
    num_ensemble = 10

    rng = np.random.default_rng(42)

    X = rng.normal(size=(num_parameters, num_ensemble))
    Y = rng.normal(size=(num_observations, num_ensemble))
    covariance = np.exp(rng.normal(size=num_observations))
    observations = rng.normal(size=num_observations, loc=1)

    row_groups = [(0,), (1, 2), (4, 5, 6), tuple(range(7, 100))]
    X_with_row_scaling = [(X[idx, :], RowScaling()) for idx in row_groups]
    X_with_row_scaling = X_with_row_scaling.copy()

    X_with_row_scaling_updated = ensemble_smoother_update_step_row_scaling(
        covariance=covariance,
        observations=observations,
        X_with_row_scaling=X_with_row_scaling,
        Y=Y,
        seed=rng,
    )
