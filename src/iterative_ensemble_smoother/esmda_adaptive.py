import warnings
from typing import Callable, Union

import numpy as np
import numpy.typing as npt

from iterative_ensemble_smoother.esmda import BaseESMDA
from iterative_ensemble_smoother.utils import masked_std


class AdaptiveESMDA(BaseESMDA):
    """
    Adaptive Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

    Reference
    ---------

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

    Then we set up the LocalizedESMDA instance and the prior realizations X:

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
    ...     def func(corr_XY, observations):
    ...         # Takes an array of shape (params_batch, obs)
    ...         # and an array representing number of non-missing observations
    ...         # per parameter. Returns modified correlation matrix.
    ...         return corr_XY
    ...
    ...     for param_idx in yield_param_indices():
    ...         X[param_idx, :] = smoother.assimilate_batch(X=X[param_idx, :],
    ...                           correlation_callback=smoother.three_over_sqrt_n)
    """

    @staticmethod
    def three_over_sqrt_n(
        corr_XY: npt.NDArray[np.double],
        observations: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.double]:
        """Use the correlation threshold 3 / sqrt(n)."""
        threshold = np.clip(3 / np.sqrt(observations), a_min=0.0, a_max=1.0)
        zero_out = np.abs(corr_XY) < threshold[:, None]
        corr_XY[zero_out] = 0
        return corr_XY

    def assimilate_batch(
        self,
        *,
        X: npt.NDArray[np.double],
        missing: Union[npt.NDArray[np.bool_], None] = None,
        correlation_callback: Callable[
            [npt.NDArray[np.double], npt.NDArray[np.int_]], npt.NDArray[np.double]
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
        correlation_callback : callable, optional
            A callable that takes as input a cross-correlation 2D array of shape
            (num_parameters_batch, num_observations) and returns a 2D array of
            the same shape. The returned array represents any kind of correlation
            thresholding or softening.

        Returns
        -------
        X_posterior : np.ndarray
            2D array of shape (num_parameters_batch, ensemble_size).

        """
        if not hasattr(self, "D_obs_minus_D"):
            raise Exception("The method `prepare_assmilation` must be called.")

        assert X.ndim == 2
        N_m, N_e = X.shape  # (num_parameters, ensemble_size)
        assert N_e == self.delta_DT.shape[0], "Dimension mismatch"

        delta_M = self._compute_delta_M(X=X, missing=missing)

        # The default localization is no localization (identity function)
        if correlation_callback is None:

            def correlation_callback(
                corr_XY: npt.NDArray[np.double],
                observations: npt.NDArray[np.int_],
            ) -> npt.NDArray[np.double]:
                return corr_XY

        # Compute cross correlation matrix
        corr_XY = (delta_M @ self.delta_DT) / (N_e - 1)
        std_Y = np.std(self.delta_DT, axis=0, ddof=1)
        if missing is not None:
            std_X = masked_std(X, missing=missing)
        else:
            std_X = np.std(X, axis=1, ddof=1)

        # Cross covariance to cross correlation (inplace)
        corr_XY /= std_X[:, None]
        corr_XY /= std_Y[None, :]
        corr_XY = self._clip_correlation_matrix(corr_XY)

        # Number of observations each entry in corr_XY is based on.
        # The source of missing data is only missing values in X, not Y.
        observations = (
            np.ones(N_m) * N_e
            if missing is None
            else np.sum(np.logical_not(missing), axis=1)
        )
        # Apply localization function
        corr_XY = correlation_callback(corr_XY, observations)

        # Cross correlation to cross covariance (inplace)
        corr_XY *= std_X[:, None]
        corr_XY *= std_Y[None, :]

        # TODO: Slice the product so all-zero rows (params) are not used

        # Return result
        return X + np.linalg.multi_dot(
            [corr_XY, self.term_diag, self.termT, self.D_obs_minus_D]
        )

    @staticmethod
    def _clip_correlation_matrix(
        corr_XY: npt.NDArray[np.double],
    ) -> npt.NDArray[np.double]:
        """Clip correlation matrix to range [-1, 1]."""

        # Perform checks and clip values to [-1, 1]
        eps = 1e-8
        min_value, max_value = corr_XY.min(), corr_XY.max()
        if not ((max_value <= 1 + eps) and (min_value >= -1 - eps)):
            msg = "Cross-correlation matrix has entries not in [-1, 1]."
            msg += f"The min and max values are: {min_value} and {max_value}"
            msg += "Entries will be clipped to the range [-1, 1]."
            warnings.warn(msg)

        return np.clip(corr_XY, a_min=-1, a_max=1, out=corr_XY)


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
        ]
    )
