"""
Adaptive Ensemble Smoother with Multiple Data Assimilation
----------------------------------------------------------

The idea behind AdaptiveESMDA is to use correlations between parameters
and responses to modify the update equation. Recall the equation:

    C_MD @ inv(Y @ Y.T + alpha * C_D) @ (D - Y)

where C_MD is the cross-covariance matrix with shape (parameters, responses).

1. Compute C_MD and normalize it so it becomes a cross-correlation matrix
2. Apply the correlation callback, which returns a boolean mask with shape
   (parameters, responses) indicating which pairs to keep
3. We then loop over every parameter, check what responses have correlation
   to the parameter, and update only those parameters. Suppose for a parameter
   `i` the responses that are correlated have indices `idx`, then we apply:

       C_MD[i, idx] @
       inv(Y[idx, :] @ Y[idx, :].T + alpha * C_D[idx, idx])
       @ (D - Y)[idx, :]

Since forming the full C_MD matrix is often prohibitive, it is often better to
loop over parameter groups. Internally, the class also vectorizes over parameters
that share the same correlation pattern with the responses.
"""

import logging
from typing import Callable, Union

import numpy as np
import numpy.typing as npt
import scipy as sp

from iterative_ensemble_smoother.esmda import BaseESMDA
from iterative_ensemble_smoother.esmda_inversion import invert_subspace
from iterative_ensemble_smoother.utils import (
    clip_correlation_matrix,
    groupby_rows,
    masked_std,
)

logger = logging.getLogger(__name__)


class AdaptiveESMDA(BaseESMDA):
    """
    Hard Thresholding Adaptive ESMDA.

    References
    ----------

    - Adaptive Correlation- and Distance-Based Localization for Iterative
      Ensemble Smoothers in a Coupled Nonlinear Multiscale Model.
      Vossepoel, Femke & Evensen, Geir & Van Leeuwen, Peter Jan. (2025)
      http://doi.org/10.1175/MWR-D-24-0269.1


    Examples
    --------

    A full example where the forward model maps 10 parameters to 3 outputs.
    We will use 100 realizations. First we define the forward model:

    >>> rng = np.random.default_rng(42)
    >>> A = rng.normal(size=(3, 10))
    >>> def forward_model(x):
    ...     return A @ x

    Then we set up the AdaptiveESMDA instance and the prior realizations X:

    >>> covariance = np.ones(3, dtype=float)  # Covariance of the observations / outputs
    >>> observations = np.array([1, 2, 3], dtype=float)  # The observed data
    >>> smoother = AdaptiveESMDA(covariance=covariance,
    ...                          observations=observations, alpha=3, seed=42)
    >>> X = rng.normal(size=(10, 100))

    To assimilate data, we iterate over the assimilation steps in an outer
    loop, then over parameter batches:

    >>> def yield_param_indices():
    ...     yield [1, 2, 3, 4]
    ...     yield [5, 6, 7, 8, 9]
    >>> for iteration in range(smoother.num_assimilations()):
    ...
    ...     Y = np.array([forward_model(x) for x in X.T]).T
    ...
    ...     # Prepare for assimilation
    ...     smoother.prepare_assimilation(Y=Y, truncation=0.99)
    ...
    ...     def func(corr_XY, ensemble_members_per_parameter):
    ...         # Takes an array of shape (params_batch, obs)
    ...         # and an array representing number of non-missing ensemble
    ...         # members per parameter. Returns which pairs (param, obs) to keep.
    ...         return np.ones_like(corr_XY, dtype=np.bool_)
    ...
    ...     for param_idx in yield_param_indices():
    ...         X[param_idx, :] = smoother.assimilate_batch(X=X[param_idx, :])
    ...
    """

    @staticmethod
    def three_over_sqrt_ensemble_members(
        corr_XY: npt.NDArray[np.floating],
        ensemble_members_per_parameter: Union[npt.NDArray[np.int_], int],
    ) -> npt.NDArray[np.bool_]:
        """Use the correlation threshold 3 / sqrt(n). Note that unless
        the number of ensemble members is > 9, all responses are removed
        and no update happens at all.

        This simple thresholding rule is equation (6) in the adaptive localization
        paper: http://doi.org/10.1175/MWR-D-24-0269.1

        The rule is also mentioned in the paper:
            Continuous Hyper-parameter OPtimization (CHOP) in an ensemble Kalman filter
            https://arxiv.org/abs/2206.03050
        """
        threshold = np.clip(
            3 / np.sqrt(ensemble_members_per_parameter), a_min=0.0, a_max=1.0
        )
        # Keep those above threshold
        result: npt.NDArray[np.bool_] = (
            np.abs(corr_XY) > np.atleast_1d(threshold)[:, None]
        )
        return result

    def assimilate_batch(
        self,
        *,
        X: npt.NDArray[np.floating],
        missing: Union[npt.NDArray[np.bool_], None] = None,
        correlation_callback: Callable[
            [npt.NDArray[np.floating], Union[npt.NDArray[np.int_], int]],
            npt.NDArray[np.bool_],
        ]
        | str = "three_over_sqrt_ensemble_members",
        overwrite: bool = False,
    ) -> npt.NDArray[np.floating]:
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
        correlation_callback : callable or string, optional
            A callable with signature
            ``(corr_XY, ensemble_members_per_parameter) -> mask``.
            *corr_XY* is a cross-correlation 2D array of shape
            (num_parameters_batch, num_observations).
            *ensemble_members_per_parameter* is either an int (when there are
            no missing values, equal to the ensemble size) or a 1D int array
            of shape (num_parameters_batch,) giving the number of non-missing
            ensemble members for each parameter.
            The returned array must have the same shape as *corr_XY* and
            represents any kind of correlation thresholding or softening.
        overwrite: bool
            If False (the default), the input arrays will not be overwritten (mutated).
            If True, the method may overwrite the input arrays.

        Returns
        -------
        X_posterior : np.ndarray
            2D array of shape (num_parameters_batch, ensemble_size).

        """
        CALLBACKS = {
            "three_over_sqrt_ensemble_members": self.three_over_sqrt_ensemble_members
        }

        if not overwrite:
            X = X.copy()
        if not hasattr(self, "D_obs_minus_D"):
            raise Exception("The method `prepare_assmilation` must be called.")

        assert X.ndim == 2
        N_m, N_e = X.shape  # (num_parameters, ensemble_size)
        assert N_e == self.delta_DT.shape[0], "Dimension mismatch"

        callback_func: Callable[..., npt.NDArray[np.bool_]]
        if isinstance(correlation_callback, str) and correlation_callback in CALLBACKS:
            callback_func = CALLBACKS[correlation_callback]
        elif callable(correlation_callback):
            callback_func = correlation_callback
        else:
            raise TypeError(
                "`correlation_callback` must be a callable or a "
                f"string in {set(CALLBACKS.keys())}"
            )

        # Compute delta M, which is the centered X matrix
        delta_M = self._compute_delta_M(X=X, missing=missing)

        # Step 1: COMPUTE THE CROSS-COVARIANCE/CORRELATION AND APPLY CALLBACK
        # ===================================================================

        std_Y = np.std(self.delta_DT, axis=0, ddof=1)

        # Deal with potentially missing values in the parameters
        if missing is not None:
            std_X = masked_std(X, missing=missing)
        else:
            std_X = np.std(X, axis=1, ddof=1)

        # Compute cross correlation matrix
        corr_XY = (delta_M @ self.delta_DT) / (N_e - 1)

        # Cross covariance to cross correlation (inplace)
        corr_XY /= std_X[:, None]
        corr_XY /= std_Y[None, :]
        corr_XY = clip_correlation_matrix(corr_XY)

        # Number of ensemble members each entry in corr_XY is based on.
        # The source of missing data is only missing values in X, not Y.
        ensemble_members_per_parameter = (
            N_e if missing is None else np.sum(np.logical_not(missing), axis=1)
        )
        # Apply localization function
        mask_keep = callback_func(corr_XY, ensemble_members_per_parameter)
        logger.debug(f"Percentage of (param, obs) pairs kept: {mask_keep.mean():.1%}")

        # Cross correlation to cross covariance (inplace)
        corr_XY *= std_Y[None, :]
        corr_XY *= std_X[:, None] * (N_e - 1)  # Multiply back

        # Step 2: APPLY UPDATES TO EACH PARAMETER, USING CORRELATED RESPONSES
        # ===================================================================

        alpha = self.alpha[self.iteration]
        delta_D = self.delta_DT.T
        # Loop over observation indices (integer mask) and param idx (bool mask)
        for param_idx, response_idx in groupby_rows(mask_keep):
            # Skip parameters if no responses have correlations
            if not np.any(response_idx):
                continue

            logger.debug(
                f"Assimilating {len(param_idx)} parameters that share "
                "the same correlation structure with observations."
            )

            # Centered responses for this param group
            delta_D_i = delta_D[response_idx, :]

            # Index on the responses in this param group, then factor covariance
            if self.C_D.ndim == 1:
                C_D_L_i = np.sqrt(self.C_D[response_idx])
            else:
                cov_mask = np.ix_(response_idx, response_idx)
                C_D_L_i = sp.linalg.cholesky(self.C_D[cov_mask], lower=False)

            # Compute (Y[idx, :] @ Y[idx, :].T + C_D[idx, idx])^-1
            _, factor1, factor2 = invert_subspace(
                delta_D=delta_D_i,
                C_D_L=C_D_L_i,
                alpha=alpha,
                truncation=self.truncation,
            )

            # Multiply together and store results
            corr_mask = np.ix_(param_idx, response_idx)
            X[param_idx, :] += np.linalg.multi_dot(
                [
                    corr_XY[corr_mask],
                    factor1,
                    factor2,
                    self.D_obs_minus_D[response_idx, :],
                ]
            )

        return X


class TaperedAdaptiveESMDA(BaseESMDA):
    """
    Soft Thresholding Adaptive ESMDA.

    References
    ----------

    - Adaptive Correlation- and Distance-Based Localization for Iterative
      Ensemble Smoothers in a Coupled Nonlinear Multiscale Model.
      Vossepoel, Femke & Evensen, Geir & Van Leeuwen, Peter Jan. (2025)
      http://doi.org/10.1175/MWR-D-24-0269.1


    Examples
    --------

    A full example where the forward model maps 10 parameters to 3 outputs.
    We will use 100 realizations. First we define the forward model:

    >>> rng = np.random.default_rng(42)
    >>> A = rng.normal(size=(3, 10))
    >>> def forward_model(x):
    ...     return A @ x

    Then we set up the TaperedAdaptiveESMDA instance and the prior realizations X:

    >>> covariance = np.ones(3, dtype=float)  # Covariance of the observations
    >>> observations = np.array([1, 2, 3], dtype=float)  # The observed data
    >>> smoother = TaperedAdaptiveESMDA(covariance=covariance,
    ...                                 observations=observations, alpha=3, seed=42)
    >>> X = rng.normal(size=(10, 100))

    To assimilate data, we iterate over the assimilation steps in an outer
    loop, then over parameter batches:

    >>> def yield_param_indices():
    ...     yield [1, 2, 3, 4]
    ...     yield [5, 6, 7, 8, 9]
    >>> for iteration in range(smoother.num_assimilations()):
    ...
    ...     Y = np.array([forward_model(x) for x in X.T]).T
    ...
    ...     # Prepare for assimilation
    ...     smoother.prepare_assimilation(Y=Y, truncation=0.99)
    ...
    ...     for param_idx in yield_param_indices():
    ...         X[param_idx, :] = smoother.assimilate_batch(X=X[param_idx, :])
    ...
    """

    @staticmethod
    def exponential_scale(
        corr_XY: npt.NDArray[np.floating],
        ensemble_members_per_parameter: Union[npt.NDArray[np.int_], int],
    ) -> npt.NDArray[np.floating]:
        """Equation (9) from the paper http://doi.org/10.1175/MWR-D-24-0269.1

        Examples
        --------
        >>> corr_XY = np.linspace(0, 1, num=11)
        >>> corr_XY
        array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
        >>> TaperedAdaptiveESMDA.exponential_scale(corr_XY, 5)
        array([8.        , 8.        , 8.        , 8.        , 2.88908419,
               1.4651216 , 1.04335093, 1.        , 1.        , 1.        ,
               1.        ])
        """
        # The inflation function f(x) = exp(x^2) is a mapping from the domain
        # [0, 1]. We shift and scale the function and impose constraints:
        #  - f(x <= beta * d) = 1
        #  - f(x >= d) = E_max

        # Numbers from the paper http://doi.org/10.1175/MWR-D-24-0269.1
        E_max = 8.0  # Maximum inflation factor
        d = 0.7  # Distance to stop inflating at (reaches E_max)
        beta = 0.5  # beta * d is the distance x where f(x) is at minimum

        dist = 1 - np.abs(corr_XY)  # Correlation distance
        scales = np.ones_like(corr_XY)
        exponent = ((dist - beta * d) / ((1 - beta) * d)) ** 2
        # Note: Using the rule exp(ln(a) * b) = a^b, the calculation is
        # simplified to E_max ** exponent to avoid log/exp operations.
        scales[dist >= d * beta] = np.power(E_max, exponent)[dist >= d * beta]
        scales[dist >= d] = E_max
        return scales

    def assimilate_batch(
        self,
        *,
        X: npt.NDArray[np.floating],
        missing: Union[npt.NDArray[np.bool_], None] = None,
        correlation_callback: Callable[
            [npt.NDArray[np.floating], npt.NDArray[np.int_]], npt.NDArray[np.floating]
        ]
        | str = "exponential_scale",
        overwrite: bool = False,
    ) -> npt.NDArray[np.floating]:
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
        correlation_callback : callable or str, optional
            A callable with signature
            ``(corr_XY, ensemble_members_per_parameter) -> inflation_factors``.
            *corr_XY* is a cross-correlation 2D array of shape
            (num_parameters_batch, num_observations).
            *ensemble_members_per_parameter* is either an int (when there are
            no missing values, equal to the ensemble size) or a 1D int array
            of shape (num_parameters_batch,) giving the number of non-missing
            ensemble members for each parameter.
            The returned array must have the same shape as *corr_XY* and
            each row represents scaling values for the observations.
        overwrite: bool
            If False (the default), the input arrays will not be overwritten (mutated).
            If True, the method may overwrite the input arrays.

        Returns
        -------
        X_posterior : np.ndarray
            2D array of shape (num_parameters_batch, ensemble_size).

        """
        CALLBACKS = {"exponential_scale": self.exponential_scale}

        if not overwrite:
            X = X.copy()
        if not hasattr(self, "D_obs_minus_D"):
            raise Exception("The method `prepare_assmilation` must be called.")

        assert X.ndim == 2
        N_m, N_e = X.shape  # (num_parameters, ensemble_size)
        assert N_e == self.delta_DT.shape[0], "Dimension mismatch"

        callback_func: Callable[
            [npt.NDArray[np.floating], npt.NDArray[np.int_]], npt.NDArray[np.floating]
        ]
        if isinstance(correlation_callback, str) and correlation_callback in CALLBACKS:
            callback_func = CALLBACKS[correlation_callback]
        elif callable(correlation_callback):
            callback_func = correlation_callback
        else:
            raise TypeError(
                "`correlation_callback` must be a callable or a "
                f"string in {set(CALLBACKS.keys())}"
            )

        # Compute delta M, which is the centered X matrix
        delta_M = self._compute_delta_M(X=X, missing=missing)

        # Step 1: COMPUTE THE CROSS-COVARIANCE/CORRELATION AND APPLY CALLBACK
        # ===================================================================

        std_Y = np.std(self.delta_DT, axis=0, ddof=1)

        # Deal with potentially missing values in the parameters
        if missing is not None:
            std_X = masked_std(X, missing=missing)
        else:
            std_X = np.std(X, axis=1, ddof=1)

        # Compute cross correlation matrix
        corr_XY = (delta_M @ self.delta_DT) / (N_e - 1)

        # Cross covariance to cross correlation (inplace)
        corr_XY /= std_X[:, None]
        corr_XY /= std_Y[None, :]
        corr_XY = clip_correlation_matrix(corr_XY)

        # Number of ensemble members each entry in corr_XY is based on.
        # The source of missing data is only missing values in X, not Y.
        ensemble_members_per_parameter = (
            N_e if missing is None else np.sum(np.logical_not(missing), axis=1)
        )
        # Apply localization function
        logger.debug(f"Average correlation: {corr_XY.mean():.1%}")
        inflation_factors = callback_func(corr_XY, ensemble_members_per_parameter)
        logger.debug(f"Average inflation factor: {inflation_factors.mean():.2f}")

        # Cross correlation to cross covariance (inplace)
        corr_XY *= std_Y[None, :]
        corr_XY *= std_X[:, None] * (N_e - 1)  # Multiply back

        # Step 2: APPLY UPDATES TO EACH PARAMETER, USING CORRELATED RESPONSES
        # ===================================================================
        alpha = self.alpha[self.iteration]
        delta_D = self.delta_DT.T

        # Loop over every parameter index i
        for i in range(corr_XY.shape[0]):
            # How much each response should be inflated
            inflation_factor_i = inflation_factors[i, :]

            # Inflate the covariance
            if self.C_D_L.ndim == 1:
                C_D_L_i = self.C_D_L * inflation_factor_i
            else:
                C_D_L_i = (
                    self.C_D_L
                    * inflation_factor_i[:, None]
                    * inflation_factor_i[None, :]
                )

            # Compute (Y[idx, :] @ Y[idx, :].T + C_D[idx, idx])^-1
            _, factor1, factor2 = invert_subspace(
                delta_D=delta_D,
                C_D_L=C_D_L_i,
                alpha=alpha,
                truncation=self.truncation,
            )

            # Multiply together and store results
            X[[i], :] += np.linalg.multi_dot(
                [
                    corr_XY[[i]],
                    factor1,
                    factor2,
                    self.D_obs_minus_D,
                ]
            )

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
