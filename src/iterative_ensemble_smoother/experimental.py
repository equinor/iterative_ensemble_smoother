"""
Contains (publicly available, but not officially supported) experimental
features of iterative_ensemble_smoother
"""
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy as sp

from iterative_ensemble_smoother import ESMDA
from iterative_ensemble_smoother.esmda import BaseESMDA
from iterative_ensemble_smoother.esmda_inversion import (
    empirical_cross_covariance,
)


def groupby_indices(X):
    """Yield pairs of (unique_row, indices_of_row).

    Examples
    --------
    >>> X = np.array([[1, 0],
    ...               [1, 0],
    ...               [1, 1],
    ...               [1, 1],
    ...               [1, 0]])
    >>> list(groupby_indices(X))
    [(array([1, 0]), array([0, 1, 4])), (array([1, 1]), array([2, 3]))]

    Another example:
    >>> X = np.array([[1, 2, 3],
    ...               [0, 0, 0],
    ...               [1, 2, 3],
    ...               [1, 1, 1],
    ...               [1, 1, 1],
    ...               [1, 2, 3]])
    >>> list(groupby_indices(X))
    [(array([0, 0, 0]), array([1])), (array([1, 1, 1]), \
array([3, 4])), (array([1, 2, 3]), array([0, 2, 5]))]

    """
    assert X.ndim == 2

    # Code was modified from this answer:
    # https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
    idx_sort = np.lexsort(X.T[::-1, :], axis=0)
    sorted_X = X[idx_sort, :]
    vals, idx_start = np.unique(sorted_X, return_index=True, axis=0)
    res = np.split(idx_sort, idx_start[1:])
    yield from zip(vals, res)


class AdaptiveESMDA(BaseESMDA):
    @staticmethod
    def correlation_threshold(ensemble_size: int) -> float:
        """Return a number that determines whether a correlation is significant.

        Default threshold taken from luo2022,
        Continuous Hyper-parameter OPtimization (CHOP) in an ensemble Kalman filter
        Section 2.3 - Localization in the CHOP problem

        Examples
        --------
        >>> AdaptiveESMDA.correlation_threshold(0)
        1
        >>> AdaptiveESMDA.correlation_threshold(4)
        1
        >>> AdaptiveESMDA.correlation_threshold(9)
        1
        >>> AdaptiveESMDA.correlation_threshold(16)
        0.75
        >>> AdaptiveESMDA.correlation_threshold(36)
        0.5
        """
        return min(1, max(0, 3 / np.sqrt(ensemble_size)))

    @staticmethod
    def compute_cross_covariance_multiplier(
        *,
        alpha: float,
        C_D: npt.NDArray[np.double],
        D: npt.NDArray[np.double],
        Y: npt.NDArray[np.double],
    ) -> npt.NDArray[np.double]:
        """Compute transition matrix T such that:

            X + cov_XY @ transition_matrix

        In the notation of Emerick et al, recall that the update equation is:

            X_posterior = X_prior + cov_XY @ inv(C_DD + alpha * C_D) @ (D - Y)

        This function computes a transition matrix T, defined as:

            T := inv(C_DD + alpha * C_D) @ (D - Y)

        Note that this might not be the most efficient way to compute the update,
        since cov_XY := center(X) @ center(Y) / (N - 1), and we can avoid creating
        this (huge) covariance matrix by performing multiplications right-to-left:

            cov_XY @ inv(C_DD + alpha * C_D) @ (D - Y)
            center(X) @ center(Y) / (N - 1) @ inv(C_DD + alpha * C_D) @ (D - Y)
            center(X) @ [center(Y) / (N - 1) @ inv(C_DD + alpha * C_D) @ (D - Y)]

        The advantage of forming T instead is that if we already have computed
        cov_XY, then we can apply the same T to a reduced number of rows (parameters)
        in cov_XY.
        """

        # TODO: This is re-computed in each call. Can we pre compute it?
        C_DD = empirical_cross_covariance(Y, Y)

        # Arguments for sp.linalg.solve
        solver_kwargs = {
            "overwrite_a": True,
            "overwrite_b": True,
            "assume_a": "pos",  # Assume positive definite matrix (use cholesky)
            "lower": False,  # Only use the upper part while solving
        }

        # Compute T := sp.linalg.inv(C_DD + alpha * C_D) @ (D - Y)
        # by solving the system (C_DD + alpha * C_D) @ T = (D - Y)
        if C_D.ndim == 2:
            # C_D is a covariance matrix
            C_DD += alpha * C_D  # Save memory by mutating
        elif C_D.ndim == 1:
            # C_D is an array, so add it to the diagonal without forming diag(C_D)
            np.fill_diagonal(C_DD, C_DD.diagonal() + alpha * C_D)

        return sp.linalg.solve(C_DD, D - Y, **solver_kwargs)

    def _correlation_matrix(self, cov_XY, X, Y):
        """Compute a correlation matrix given a covariance matrix."""
        assert cov_XY.shape == (X.shape[0], Y.shape[0])

        stds_Y = np.std(Y, axis=1, ddof=1)
        stds_X = np.std(X, axis=1, ddof=1)

        # Compute the correlation matrix from the covariance matrix
        corr_XY = (cov_XY / stds_X[:, np.newaxis]) / stds_Y[np.newaxis, :]

        # Perform checks
        assert corr_XY.max() <= 1
        assert corr_XY.min() >= -1
        return corr_XY

    def assimilate(self, X, Y, D, alpha, correlation_threshold=None, verbose=False):
        """Assimilate data and return an updated ensemble X_posterior.

            X_posterior = smoother.assimilate(X, Y, D, alpha)

        This method first computes the cross-covariance and cross-correlation
        matrices between X and Y. Then it sets correlations that are below the
        threshold to zero. It then loops over parameters in X that are deemed
        significant with respect to Y (based on the threshold), and updates
        these groups together.

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (num_parameters, ensemble_size). Each row
            corresponds to a parameter in the model, and each column corresponds
            to an ensemble member (realization).
        Y : np.ndarray
            2D array of shape (num_observations, ensemble_size), containing
            responses when evaluating the model at X. In other words, Y = g(X),
            where g is the forward model.
        D : np.ndarray
            2D array of shape (num_observations, ensemble_size), containing
            perturbed observations. D = observations + mv_normal(0, covariance),
            and D may be computed by the method `perturb_observations`.
        alpha : float
            The covariance inflation factor. The sequence of alphas should
            obey the equation sum_i (1/alpha_i) = 1. However, this is NOT
            enforced in this method. The user/caller is responsible for this.
        verbose : bool
            Whether to print information.

        Returns
        -------
        X_posterior : np.ndarray
            2D array of shape (num_parameters, ensemble_size).
        """
        assert X.shape[1] == Y.shape[1]
        msg = "`correlation_threshold` must be a callable f(ensemble_size) -> threshold"
        assert callable(correlation_threshold), msg

        # Default correlation threshold function
        if correlation_threshold is None:
            correlation_threshold = self.correlation_threshold

        # Step 1: # Compute cross-correlation between parameters X and responses Y
        # Note: let the number of parameters be n and the number of responses be m.
        # This step requires both O(mn) computation and O(mn) storage, which is
        # larger than the O(n + m^2) computation used in ESMDA, which never explicitly
        # forms the cross-covariance matrix between X and Y. However, if we batch
        # X by parameter group (rows), the storage requirement decreases
        cov_XY = empirical_cross_covariance(X, Y)
        assert cov_XY.shape == (X.shape[0], Y.shape[0])
        corr_XY = self._correlation_matrix(cov_XY, X, Y)
        assert corr_XY.shape == cov_XY.shape

        # Determine which elements in the cross covariance matrix that will
        # be set to zero
        threshold = correlation_threshold(ensemble_size=X.shape[1])
        significant_corr_XY = np.abs(corr_XY) > threshold

        # TODO: memory could be saved by overwriting the input X
        X_out = np.copy(X)
        for (unique_row, indices_of_row) in groupby_indices(significant_corr_XY):

            if verbose:
                print(
                    f"    Assimilating {len(indices_of_row)} parameters"
                    + " with identical correlation thresholds to responses."
                )
                print(
                    "    The parameters are significant wrt "
                    + f"{np.sum(unique_row)} / {len(unique_row)} responses."
                )

            # Get the parameters (X) that have identical significant responses (Y)
            X_subset = X[indices_of_row, :]
            Y_subset = Y[unique_row, :]

            # Compute the update
            cov_XY_mask = np.ix_(indices_of_row, unique_row)
            cov_XY_subset = cov_XY[cov_XY_mask]

            C_D_subset = self.C_D[unique_row]
            D_subset = D[unique_row, :]

            # Compute transition matrix T
            T = self.compute_cross_covariance_multiplier(
                alpha=alpha,
                C_D=C_D_subset,
                D=D_subset,
                Y=Y_subset,
            )
            X_out[indices_of_row, :] = X_subset + cov_XY_subset @ T

        return X_out


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


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
        ]
    )
