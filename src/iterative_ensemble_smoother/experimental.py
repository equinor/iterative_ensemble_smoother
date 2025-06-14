"""
Contains (publicly available, but not officially supported) experimental
features of iterative_ensemble_smoother
"""

import numbers
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore

from iterative_ensemble_smoother import ESMDA
from iterative_ensemble_smoother.esmda import BaseESMDA
from iterative_ensemble_smoother.esmda_inversion import (
    empirical_cross_covariance,
)

T = TypeVar("T")


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
        1.0
        >>> AdaptiveESMDA.correlation_threshold(9)
        1.0
        >>> AdaptiveESMDA.correlation_threshold(16)
        0.75
        >>> AdaptiveESMDA.correlation_threshold(36)
        0.5
        """
        return float(min(1, max(0, 3 / np.sqrt(ensemble_size))))

    @staticmethod
    def compute_cross_covariance_multiplier(
        *,
        alpha: float,
        C_D: npt.NDArray[np.double],
        D: npt.NDArray[np.double],
        Y: npt.NDArray[np.double],
        cov_YY: Optional[npt.NDArray[np.double]] = None,
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
        # Compute cov(Y, Y) if it was not passed to the function.
        # Pre-computation might be faster, since covariance is commutative with
        # respect to indexing, ie, cov(Y[mask, :], YY[mask, :]) = cov(Y, Y)[mask, mask]
        if cov_YY is None:
            C_DD = empirical_cross_covariance(Y, Y)
        else:
            C_DD = cov_YY

        assert C_DD.shape[0] == C_DD.shape[1]
        assert C_DD.shape[0] == Y.shape[0]

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

        return sp.linalg.solve(C_DD, D - Y, **solver_kwargs)  # type: ignore

    def _cov_to_corr_inplace(
        self,
        cov_XY: npt.NDArray[np.double],
        stds_X: npt.NDArray[np.double],
        stds_Y: npt.NDArray[np.double],
    ) -> None:
        """Convert a covariance matrix to a correlation matrix in-place."""

        # Divide each element of cov_XY by the corresponding standard deviations
        cov_XY /= stds_X[:, np.newaxis]
        cov_XY /= stds_Y[np.newaxis, :]

        # Perform checks and clip values to [-1, 1]
        eps = 1e-8
        if not ((cov_XY.max() <= 1 + eps) and (cov_XY.min() >= -1 - eps)):
            msg = "Cross-correlation matrix has entries not in [-1, 1]."
            msg += f"The min and max values are: {cov_XY.min()} and {cov_XY.max()}"
            warnings.warn(msg)

        return np.clip(cov_XY, a_min=-1, a_max=1, out=cov_XY)

    def _corr_to_cov_inplace(
        self,
        corr_XY: npt.NDArray[np.double],
        stds_X: npt.NDArray[np.double],
        stds_Y: npt.NDArray[np.double],
    ) -> None:
        """Convert a correlation matrix to a covariance matrix in-place."""
        # Multiply each element of corr_XY by the corresponding standard deviations
        corr_XY *= stds_X[:, np.newaxis]
        corr_XY *= stds_Y[np.newaxis, :]
        return corr_XY

    def assimilate(
        self,
        *,
        X: npt.NDArray[np.double],
        Y: npt.NDArray[np.double],
        D: npt.NDArray[np.double],
        overwrite: bool = False,
        alpha: float,
        parameter_selection_method: str = "covariance",
        correlation_threshold: Union[Callable[[int], float], float, None] = None,
        cov_YY: Optional[npt.NDArray[np.double]] = None,
        progress_callback: Optional[Callable[[Sequence[T]], Sequence[T]]] = None,
        correlation_callback: Optional[Callable[[npt.NDArray[np.double]], None]] = None,
    ) -> npt.NDArray[np.double]:
        """Assimilate data and return an updated ensemble X_posterior.

        This method updates the ensemble `X` based on observations `D` and
        model responses `Y`. It supports two localization strategies to
        determine which parameters to update:

        1. "covariance": (Default) Uses a statistical correlation threshold to
           identify and update only the parameters that are significantly
           correlated with the model responses.
        2. "regression": Uses a sparse regression method (`linear_boost_ic_regression`)
           to learn a linear map between parameters and responses, updating
           only the parameters with identified non-zero coefficients.

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (num_parameters, ensemble_size).
        Y : np.ndarray
            2D array of shape (num_observations, ensemble_size),
            where Y = g(X).
        D : np.ndarray
            2D array of shape (num_observations, ensemble_size) of
            perturbed observations.
        overwrite: bool
            If True, `X` will be mutated in-place to save memory.
        alpha : float
            The covariance inflation factor.
        parameter_selection_method : str, optional
            The method for localization: "covariance" (default) or "regression".
        correlation_threshold : callable or float or None, optional
            Threshold for the "covariance" method. If None, a default is used.
        cov_YY : np.ndarray or None, optional
            Pre-computed covariance matrix of `Y` to save computation.
        progress_callback : Callable or None, optional
            A callback for progress reporting (e.g., `tqdm.tqdm`).
        correlation_callback : Callable or None, optional
            For "covariance" method, a callback function
            called with the correlation matrix.

        Returns
        -------
        X_posterior : np.ndarray
            The updated 2D array of shape (num_parameters, ensemble_size).
        """
        assert X.shape[1] == Y.shape[1], "X and Y must have the same ensemble size."

        if not overwrite:
            X = np.copy(X)

        if progress_callback is None:

            def progress_callback(x):
                return x  # A simple pass-through function

        # Pre-compute the covariance cov(Y, Y) here if not provided.
        if cov_YY is None:
            cov_YY = empirical_cross_covariance(Y, Y)
        else:
            assert cov_YY.ndim == 2, "'cov_YY' must be a 2D array"
            assert cov_YY.shape == (Y.shape[0], Y.shape[0])

        if parameter_selection_method == "covariance":
            # --- Original method: Covariance with Correlation Thresholding ---
            is_callable = callable(correlation_threshold)
            is_float = (
                isinstance(correlation_threshold, numbers.Real)
                and 0 <= correlation_threshold <= 1
            )
            is_None = correlation_threshold is None
            if not (is_callable or is_float or is_None):
                raise TypeError(
                    "`correlation_threshold` must be a callable or a float in [0, 1]"
                )

            if is_float:
                corr_thresh_val: float = correlation_threshold

                def threshold_func(ensemble_size: int) -> float:
                    return corr_thresh_val

                correlation_threshold = threshold_func
            elif is_None:
                correlation_threshold = self.correlation_threshold

            assert callable(correlation_threshold)

            cov_XY = empirical_cross_covariance(X, Y)
            stds_X = np.std(X, axis=1, ddof=1)
            stds_Y = np.std(Y, axis=1, ddof=1)
            corr_XY = self._cov_to_corr_inplace(np.copy(cov_XY), stds_X, stds_Y)

            threshold = correlation_threshold(X.shape[1])
            significant_corr_XY = np.abs(corr_XY) > threshold

            if correlation_callback is not None:
                correlation_callback(corr_XY)

            significant_rows = np.where(np.any(significant_corr_XY, axis=1))[0]

            for param_num in progress_callback(significant_rows):
                correlated_responses = significant_corr_XY[param_num]
                if not np.any(correlated_responses):
                    continue

                Y_subset = Y[correlated_responses, :]
                cov_XY_mask = np.ix_([param_num], correlated_responses)
                cov_XY_subset = cov_XY[cov_XY_mask]
                cov_YY_mask = np.ix_(correlated_responses, correlated_responses)
                cov_YY_subset = cov_YY[cov_YY_mask]
                C_D_subset = (
                    self.C_D[correlated_responses]
                    if self.C_D.ndim == 1
                    else self.C_D[cov_YY_mask]
                )
                D_subset = D[correlated_responses, :]

                T = self.compute_cross_covariance_multiplier(
                    alpha=alpha,
                    C_D=C_D_subset,
                    D=D_subset,
                    Y=Y_subset,
                    cov_YY=cov_YY_subset,
                )
                X[[param_num], :] += cov_XY_subset @ T

        elif parameter_selection_method == "regression":
            if correlation_threshold is not None or correlation_callback is not None:
                warnings.warn(
                    "'correlation_threshold' and 'correlation_callback' are not used "
                    "with the 'regression' parameter_selection_method."
                )
            try:
                from graphite_maps.linear_regression import linear_boost_ic_regression
            except ImportError as e:
                raise ImportError(
                    "The 'regression' parameter_selection_method"
                    " requires 'graphite_maps'. "
                    "Please install it to use this feature."
                ) from e

            print("Learning sparse linear map H...")
            # How do the parameters linearly influence the observations?
            # For each of the model responses (each column in Y.T),
            # find the best and smallest set of model parameters
            # (from the columns in X.T) that can linearly predict it.
            # This function iterates through each response and performs a special kind
            # of regression called "boosted linear regression".
            # Y.T ≈ X.T @ H
            # The resulting H_sparse matrix is essentially a learned
            # sensitivity or Jacobian matrix.
            # H is a local linear representation of the reservoir simulator
            # like Eclipse of Flow.
            # Each element H[i, j] represents how much observation i changes
            # per unit change in parameter j.
            # It's a linearization of the complex nonlinear flow physics.
            H_sparse = linear_boost_ic_regression(U=X.T, Y=Y.T)
            active_params = np.where(H_sparse.getnnz(axis=0) > 0)[0]
            print(f"Found {len(active_params)} active parameters out of {X.shape[0]}.")

            print("Performing parameter updates...")
            for param_num in progress_callback(active_params):
                correlated_responses = H_sparse[:, param_num].nonzero()[0]

                if len(correlated_responses) == 0:
                    continue

                Y_subset = Y[correlated_responses, :]
                cov_XY_subset = empirical_cross_covariance(X[[param_num], :], Y_subset)
                cov_YY_mask = np.ix_(correlated_responses, correlated_responses)
                cov_YY_subset = cov_YY[cov_YY_mask]
                C_D_subset = (
                    self.C_D[correlated_responses]
                    if self.C_D.ndim == 1
                    else self.C_D[cov_YY_mask]
                )
                D_subset = D[correlated_responses, :]

                T = self.compute_cross_covariance_multiplier(
                    alpha=alpha,
                    C_D=C_D_subset,
                    D=D_subset,
                    Y=Y_subset,
                    cov_YY=cov_YY_subset,
                )
                X[[param_num], :] += cov_XY_subset @ T

        else:
            raise ValueError(
                f"Unknown parameter_selection_method: '{parameter_selection_method}'. "
                "Available methods are 'covariance' and 'regression'."
            )

        return X


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
