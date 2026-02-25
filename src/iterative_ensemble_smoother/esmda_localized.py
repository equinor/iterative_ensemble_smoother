"""
Localized ESMDA
---------------

This module implements localized ESMDA, following the paper:

    - "Analysis of the performance of ensemble-based assimilation
       of production and seismic data", Alexandre A. Emerick
       https://doi.org/10.1016/j.petrol.2016.01.029


What is localized ESMDA and what are the performance tradeoffs?
---------------------------------------------------------------

To understand the performance tradeoffs between vanilla ESMDA and localized
ESMDA we must examine the ensemble smoother equation and apply some basic
linear algebra. We assume that the dimensions obey:

    parameters (p) >> observations (o) >> ensemble_realizations (e)

Recall that if we multiply two matrices A and B, and they have shapes
(a, b) and (b, c), then the matrix multiplication uses O(abc) operations.
If we ignore details, the gist of the ensemble smoother update equation is:

    X_post  =   X   +   X       Y^T    [Y Y^T + covar]^{-1} (D_obs - Y)
    (p, e)   (p, e)   (p, e)  (e, o)       (o, o)           (o, e)

The speed and intermediate storage required depends on the order of the matrix
product (as well as other computational tricks like subspace inversion):

- If we go left-to-right, the total cost is O(poe + po^2 + poe) = O(po^2).
- If we go right-to-left, the total cost is O(o^2e + oe^2 + pe^2) = O(pe^2).

Localized ESMDA must form the Kalman gain matrix

    K := X Y^T [Y Y^T + covar]^{-1}

with shape (p, o) in order to apply a localization function elementwise to K,
which determines how parameter i should influence observation j, at entry K_ij.
The localization function is the matrix rho (ρ) in the paper by Emerick.
Forming K has a cost of at least O(poe) (right-to-left), which is not ideal.

Since storing all of K in memory at once can be prohibitive, we first form:

    delta_D_inv_cov := Y^T [Y Y^T + covar]^{-1}

and then update parameters in batches. This works because, given a set of indices
for a batch with size b, the block equations become:

    K[idx, :] = X[idx, :] Y^T [Y Y^T + covar]^{-1} = X[idx, :] delta_D_inv_cov
     (b, o)     (b, e)  (e, o)    (o, o)           =  (b, e)     (e, o)

    X_post[idx, :] = X[idx, :] + K[idx, :] (D_obs - Y)
       (b, e)        (b, e)     (b, o)      (e, o)

In summary localized ESMDA is more computationally expensive than ESMDA,
but we can alleviate the memory requirement by assimilating parameters in batches.

API design
----------

Some notes on design:

- The interface (API) uses human-readable names, but the internals refer to the paper.
- Two main methods are used: `prepare_assimilation()` and `assimilate_batch()`.
- The user is responsible for calling them in the correct order.

Comments
--------

- If `localization_callback` is the identity, LocalizedESMDA is identical to ESMDA.
- The inner loop over parameter blocks saves memory. The result should be the
  same over any possible sequence of parameter blocks.
- Practical issues that are not directly related to ensemble smoothing, such as
  removing inactive realizations, batching the parameters, maintaining grid information
  in order to assess the influence of parameter i on response j, is the caller's
  responsibility.

"""

from typing import Callable, Union

import numpy as np
import numpy.typing as npt

from iterative_ensemble_smoother.esmda import BaseESMDA


class LocalizedESMDA(BaseESMDA):
    """
    Localized Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

    Examples
    --------

    A full example where the forward model maps 10 parameters to 3 outputs.
    We will use 100 realizations. First we define the forward model:

    >>> rng = np.random.default_rng(42)
    >>> A = rng.normal(size=(3, 10))
    >>> def forward_model(x):
    ...     return A @ x

    Then we set up the LocalizedESMDA instance and the prior realizations X:

    >>> covariance = np.ones(3)  # Covariance of the observations / outputs
    >>> observations = np.array([1, 2, 3])  # The observed data
    >>> smoother = LocalizedESMDA(covariance=covariance,
    ...                           observations=observations, alpha=3, seed=42)
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
    ...     def func(K):
    ...         # Takes an array of shape (params_batch, obs)
    ...         # and applies localization to each entry.
    ...         return K # Here we do nothing
    ...
    ...     for param_idx in yield_param_indices():
    ...         X[param_idx, :] = smoother.assimilate_batch(X=X[param_idx, :],
    ...                                                     localization_callback=func)
    """

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
        if not hasattr(self, "delta_DT"):
            raise Exception("The method `prepare_assmilation` must be called.")
        N_m, N_e = X.shape  # (num_parameters, ensemble_size)
        assert N_e == self.delta_DT.shape[0], "Dimension mismatch"
        assert localization_callback is None or callable(localization_callback)

        # In standard ESMDA, we simplify compute the product in a good order
        delta_M = self._compute_delta_M(X=X, missing=missing)

        # The default localization is no localization (identity function)
        if localization_callback is None:

            def localization_callback(
                K: npt.NDArray[np.double],
            ) -> npt.NDArray[np.double]:
                return K

        # Create Kalman gain of shape (num_parameters_batch, num_observations),
        # then apply the localization callback elementwise
        K = localization_callback(
            np.linalg.multi_dot([delta_M, self.delta_DT, self.term_diag, self.termT])
        )
        return X + K @ self.D_obs_minus_D


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
        ]
    )
