"""
Contains (publicly available, but not officially supported) experimental
features of iterative_ensemble_smoother
"""

import logging
import numbers
import warnings
from typing import Callable, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore
from joblib import Parallel, delayed

from iterative_ensemble_smoother.esmda_inversion import (
    empirical_cross_covariance,
)
from iterative_ensemble_smoother.esmda_localized import LocalizedESMDA
from iterative_ensemble_smoother.utils import adjust_for_missing

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AdaptiveESMDA(LocalizedESMDA):
    def assimilate_batch(
        self,
        *,
        X: npt.NDArray[np.double],
        missing: Union[npt.NDArray[np.bool_], None] = None,
        localization_callback: Callable[
            [npt.NDArray[np.double]], npt.NDArray[np.double]
        ]
        | None = None,
    ) -> npt.NDArray[np.double]:
        """Assimilate a batch of parameters against all observations.

        The internal storage used by the class is 2 * ensemble_size * num_observations,
        so a good batch size that is of the same order of magnitude as the internal
        storage is 2 * num_observations. This is only a rough guideline.

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (num_parameters_batch, ensemble_size). Each row
            corresponds to a parameter in the model, and each column corresponds
            to an ensemble member (realization).
        missing : np.ndarray or None
            A boolean 2D array of shape (num_parameters_batch, ensemble_size).
            If an entry is True, then that value is assumed missing. This can
            happen if the ensemble members use different grids, where each
            ensemble member has a slightly different grid layout. If None,
            then all entries are assumed to be valid.
        localization_callback : callable, optional
            A callable that takes as input a Kalman gain 2D array of shape
            (num_parameters_batch, num_observations) and returns a 2D array of
            the same shape. The typical use-case is to associate with each
            parameter and observation a localiation factor between 0 and 1,
            and apply element multiplication. The default is None, which applies
            the identity function (i.e. multiplication with 1 in every entry).

        Returns
        -------
        X_posterior : np.ndarray
            2D array of shape (num_parameters_batch, ensemble_size).

        """
        if not hasattr(self, "D_obs_minus_D"):
            raise Exception("The method `prepare_assmilation` must be called.")
        assert localization_callback is None or callable(localization_callback)
        if not np.issubdtype(X.dtype, np.floating):
            raise TypeError("Argument `X` must contain floats")
        if not (missing is None or np.issubdtype(missing.dtype, np.bool_)):
            raise TypeError("Argument `missing_mask` must contain booleans")

        # The default localization is no localization (identity function)
        if localization_callback is None:

            def localization_callback(
                K: npt.NDArray[np.double],
            ) -> npt.NDArray[np.double]:
                return K

        assert X.ndim == 2
        N_m, N_e = X.shape  # (num_parameters, ensemble_size)
        assert N_e == self.delta_D_inv_cov.shape[0], "Dimension mismatch"

        # Center the parameters, possibly accounting for missing data
        if missing is not None:
            delta_M = adjust_for_missing(X, missing=missing)
        else:
            delta_M = X - np.mean(X, axis=1, keepdims=True)

        # Create Kalman gain of shape (num_parameters_batch, num_observations),
        # then apply the localization callback elementwise
        K = localization_callback(delta_M @ self.delta_D_inv_cov)
        return X + K @ self.D_obs_minus_D

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
        C_DD = empirical_cross_covariance(Y, Y) if cov_YY is None else cov_YY

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
    ) -> npt.NDArray[np.double]:
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
    ) -> npt.NDArray[np.double]:
        """Convert a correlation matrix to a covariance matrix in-place."""
        # Multiply each element of corr_XY by the corresponding standard deviations
        corr_XY *= stds_X[:, np.newaxis]
        corr_XY *= stds_Y[np.newaxis, :]
        return corr_XY

    @staticmethod
    def _update_single_parameter(
        param_num: int,
        correlated_responses_mask: npt.NDArray[np.bool_],
        cov_XY_row: npt.NDArray[np.double],
        Y: npt.NDArray[np.double],
        D: npt.NDArray[np.double],
        C_D: npt.NDArray[np.double],
        cov_YY: npt.NDArray[np.double],
        alpha: float,
        compute_cross_covariance_multiplier: Callable,
    ) -> Tuple[int, npt.NDArray[np.double]]:
        """
        Worker function to compute the update for a single parameter.
        """

        cov_XY_subset = cov_XY_row[correlated_responses_mask].reshape(1, -1)

        Y_subset = Y[correlated_responses_mask, :]
        D_subset = D[correlated_responses_mask, :]

        cov_YY_mask = np.ix_(correlated_responses_mask, correlated_responses_mask)
        cov_YY_subset = cov_YY[cov_YY_mask]

        C_D_subset = (
            C_D[correlated_responses_mask] if C_D.ndim == 1 else C_D[cov_YY_mask]
        )

        T = compute_cross_covariance_multiplier(
            alpha=alpha,
            C_D=C_D_subset,
            D=D_subset,
            Y=Y_subset,
            cov_YY=cov_YY_subset,
        )

        update_vector = cov_XY_subset @ T

        return param_num, update_vector

    def assimilate(
        self,
        *,
        X: npt.NDArray[np.double],
        Y: npt.NDArray[np.double],
        D: npt.NDArray[np.double],
        overwrite: bool = False,
        alpha: float,
        correlation_threshold: Union[Callable[[int], float], float, None] = None,
        cov_YY: Optional[npt.NDArray[np.double]] = None,
        progress_callback: Optional[Callable[[Sequence[T]], Sequence[T]]] = None,
        correlation_callback: Optional[Callable[[npt.NDArray[np.double]], None]] = None,
        n_jobs: int = 1,
    ) -> npt.NDArray[np.double]:
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
        overwrite: bool
            If True, X will be overwritten and mutated.
            If False, the method will not mutate inputs in any way.
            Setting this to True saves memory.
        correlation_threshold : callable or float or None
            Either a callable with signature f(ensemble_size) -> float, or a
            float in the range [0, 1]. Entries in the covariance matrix that
            are lower than the correlation threshold will be set to zero.
            If None, the default 3/sqrt(ensemble_size) is used.
        cov_YY : np.ndarray or None
            A 2D array of shape (num_observations, num_observations) with the
            empirical covariance of Y. If passed, this is not computed in the
            method call, potentially saving time and computation.
        progress_callback : Callable[[Sequence[T]], Sequence[T]] or None
            A callback function that can be used to wrap the iteration over
            parameters for progress reporting.
            It should accept an iterable as input and return an iterable.
            This allows for integration with progress reporting tools like tqdm,
            which can provide visual feedback on the progress of the
            assimilation process.
            If None, no progress reporting is performed.
        correlation_callback : Optional[Callable]
            A callback function that is called with the correlation matrix (2D array)
            as its argument after the correlation matrix computation is complete.
            The callback should handle or process the correlation matrix, such as
            saving or logging it. The callback should not return any value.
        n_jobs : int
            The number of parallel jobs to run. If 1, no parallel processing
            is used. If -1, all available CPU cores are used. Default is 1.

        Returns
        -------
        X_posterior : np.ndarray
            2D array of shape (num_parameters, ensemble_size).
        """
        assert X.shape[1] == Y.shape[1]

        # Check the correlation threshold
        is_callable = callable(correlation_threshold)
        is_float = (
            isinstance(correlation_threshold, numbers.Real)
            and correlation_threshold >= 0
            and correlation_threshold <= 1
        )
        is_None = correlation_threshold is None
        if not (is_callable or is_float or is_None):
            raise TypeError(
                "`correlation_threshold` must be a callable or a float in [0, 1]"
            )

        # Do not overwrite input arguments
        if not overwrite:
            X = np.copy(X)

        # Create `correlation_threshold` if the argument is a float
        if is_float:
            corr_threshold: float = correlation_threshold

            def correlation_threshold(ensemble_size: int) -> float:
                return corr_threshold

        # Default correlation threshold function
        if correlation_threshold is None:
            correlation_threshold = self.correlation_threshold
        assert callable(correlation_threshold), (
            "`correlation_threshold` should be callable"
        )

        if progress_callback is None:

            def progress_callback(x):
                return x  # A simple pass-through function

        # Step 1: # Compute cross-correlation between parameters X and responses Y
        # Note: let the number of parameters be n and the number of responses be m.
        # This step requires both O(mn) computation and O(mn) storage, which is
        # larger than the O(n + m^2) computation used in ESMDA, which never explicitly
        # forms the cross-covariance matrix between X and Y. However, if we batch
        # X by parameter group (rows), the storage requirement decreases
        cov_XY = empirical_cross_covariance(X, Y)
        assert cov_XY.shape == (X.shape[0], Y.shape[0])

        stds_X = np.std(X, axis=1, ddof=1)
        stds_Y = np.std(Y, axis=1, ddof=1)

        corr_XY = self._cov_to_corr_inplace(cov_XY, stds_X, stds_Y)

        # Determine which elements in the cross covariance matrix that will
        # be set to zero
        threshold = correlation_threshold(X.shape[1])
        significant_corr_XY = np.abs(corr_XY) > threshold

        cov_XY = self._corr_to_cov_inplace(corr_XY, stds_X, stds_Y)

        # Pre-compute the covariance cov(Y, Y) here, and index on it later
        if cov_YY is None:
            cov_YY = empirical_cross_covariance(Y, Y)
        else:
            assert cov_YY.ndim == 2, "'cov_YY' must be a 2D array"
            assert cov_YY.shape == (Y.shape[0], Y.shape[0])

        # Identify rows with at least one significant correlation.
        significant_rows = np.any(significant_corr_XY, axis=1)

        # Loop only over rows with significant correlations
        params_to_update = np.where(significant_rows)[0]

        C_D = self.C_D_L**2 if self.C_D_L.ndim == 1 else self.C_D_L.T @ self.C_D_L

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._update_single_parameter)(
                param_num,
                significant_corr_XY[param_num],
                cov_XY[param_num],
                # These are sent in full, which is efficient as they are the same
                # for all workers. Joblib handles this well.
                Y,
                D,
                C_D,
                cov_YY,
                alpha,
                self.compute_cross_covariance_multiplier,
            )
            for param_num in progress_callback(params_to_update)
        )

        for param_num, update_vector in results:
            X[[param_num], :] += update_vector

        if correlation_callback is not None:
            corr_XY = self._cov_to_corr_inplace(cov_XY, stds_X, stds_Y)
            correlation_callback(corr_XY)

        return X


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
        ]
    )
