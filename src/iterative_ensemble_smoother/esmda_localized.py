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

import numbers
from typing import Callable, Union

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore

from iterative_ensemble_smoother.esmda import BaseESMDA
from iterative_ensemble_smoother.esmda_inversion import (
    normalize_alpha,
    singular_values_to_keep,
)

# =============== Inversion methods ===============


def invert_naive(
    *,
    delta_D: npt.NDArray[np.double],
    C_D_L: npt.NDArray[np.double],
    alpha: float,
    truncation: float,
) -> npt.NDArray[np.double]:
    r"""Naive implementation of the equation:

    (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)

    This function should only be used for testing and verification.
    """
    _, N_e = delta_D.shape  # Number of ensemble members

    covariance = np.diag(C_D_L**2) if C_D_L.ndim == 1 else C_D_L.T @ C_D_L
    return delta_D.T @ np.linalg.inv(
        delta_D @ delta_D.T + alpha * (N_e - 1) * covariance
    )


def invert_subspace(
    *,
    delta_D: npt.NDArray[np.double],
    C_D_L: npt.NDArray[np.double],
    alpha: float,
    truncation: float,
) -> npt.NDArray[np.double]:
    r"""Subspace inversion implementation of the equation:

    (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)

    See the appendix in the 2016 Emerick paper for details:
    https://doi.org/10.1016/j.petrol.2016.01.029
    """
    # Quick verification of shapes
    assert alpha >= 0, "Alpha must be non-negative"

    # Shapes
    N_n, N_e = delta_D.shape

    # If the matrix C_D is 2D, then C_D_L is the (upper) Cholesky factor
    if C_D_L.ndim == 2:
        # Computes G := inv(sqrt(alpha) * C_D_L.T) @ D_delta
        G = sp.linalg.solve_triangular(
            np.sqrt(alpha) * C_D_L, delta_D, lower=False, trans=1
        )

    # If the matrix C_D is 1D, then C_D_L is the square-root of C_D
    else:
        G = delta_D / (np.sqrt(alpha) * C_D_L[:, np.newaxis])

    # Take the SVD and truncate it. N_r is the number of values to keep
    U, w, _ = sp.linalg.svd(G, overwrite_a=True, full_matrices=False)
    idx = singular_values_to_keep(w, truncation=truncation)
    N_r = min(N_n, N_e - 1, idx)  # Number of values in SVD to keep
    U_r, w_r = U[:, :N_r], w[:N_r]

    # Compute the symmetric terms
    if C_D_L.ndim == 2:
        # Computes term := np.linalg.inv(np.sqrt(alpha) * C_D_L) @ U_r @ np.diag(1/w_r)
        term = sp.linalg.solve_triangular(
            np.sqrt(alpha) * C_D_L, (U_r / w_r[np.newaxis, :]), lower=False
        )
    else:
        term = (U_r / w_r[np.newaxis, :]) / (np.sqrt(alpha) * C_D_L)[:, np.newaxis]

    # Diagonal matrix represented as vector
    diag = w_r**2 / (w_r**2 + N_e - 1)
    return np.linalg.multi_dot(  # type: ignore
        [delta_D.T, term * diag, term.T]
    )


class LocalizedESMDA(BaseESMDA):
    """
    Localized Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

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
    alpha : int or 1D np.ndarray, optional
        Multiplicative factor for the covariance.
        If an integer `alpha` is given, an array with length `alpha` and
        elements `alpha` is constructed. If an 1D array is given, it is
        normalized so sum_i 1/alpha_i = 1 and used. The default is 5, which
        corresponds to np.array([5, 5, 5, 5, 5]).
    seed : integer or numpy.random._generator.Generator, optional
        A seed or numpy.random._generator.Generator used for random number
        generation. The argument is passed to numpy.random.default_rng().
        The default is None.

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

    def __init__(
        self,
        *,
        covariance: npt.NDArray[np.double],
        observations: npt.NDArray[np.double],
        alpha: Union[int, npt.NDArray[np.double]] = 5,
        seed: Union[np.random._generator.Generator, int, None] = None,
    ) -> None:
        super().__init__(covariance=covariance, observations=observations, seed=seed)

        if not (
            (isinstance(alpha, np.ndarray) and alpha.ndim == 1)
            or isinstance(alpha, numbers.Integral)
        ):
            raise TypeError("Argument `alpha` must be an integer or a 1D NumPy array.")

        # Alpha can either be an integer (num iterations) or a list of weights
        if isinstance(alpha, np.ndarray) and alpha.ndim == 1:
            self.alpha = normalize_alpha(alpha)
        elif isinstance(alpha, numbers.Integral):
            self.alpha = normalize_alpha(np.ones(alpha))
            assert np.allclose(self.alpha, normalize_alpha(self.alpha))
        else:
            raise TypeError("Alpha must be integer or 1D array.")

    def num_assimilations(self) -> int:
        return len(self.alpha)

    def prepare_assimilation(
        self, *, Y: npt.NDArray[np.double], truncation: float = 0.99
    ) -> None:
        r"""Prepare assimilation of one or several batches of parameters.

        This method call pre-computes everything that is needed to assimilate
        a set of batches once. This saves time since we do not have to repeat
        the same computations for every single batch.

        Parameters
        ----------
        Y : np.ndarray
            2D array of shape (num_observations, ensemble_size), containing
            responses when evaluating the model at X. In other words, Y = g(X),
            where g is the forward model.
        truncation : float
            How large a fraction of the singular values to keep in the inversion
            routine (if the inversion routine supports it). Must be a float in
            the range (0, 1]. A lower number means a more approximate answer and a
            slightly faster computation. The default is 0.99, which is recommended
            by Emerick in the reference paper.

        Returns
        -------
        self
            The instance with mutated state.

        Notes
        -----
        In the equation:

            M + (\delta M)
                (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)
                (D_obs - D)

        This method call corresponds to computing:

            - The next-to-last factors: (\delta D)^T [(\delta D) (\delta D)^T +
                                                      \alpha (N_e - 1) C_D]^(-1)
            - The last factor: (D_obs - D)

        The total internal storage is: 2 * ensemble_size * num_observations,
        since shapes are:

            - The next-to-last factors: (ensemble_size, num_observations)
            - The last factor: (num_observations, ensemble_size)
        """
        assert Y.ndim == 2
        assert 0 < truncation <= 1.0
        if not np.issubdtype(Y.dtype, np.floating):
            raise TypeError("Argument `Y` must contain floats")

        if self.iteration >= self.num_assimilations():
            raise Exception("No more assimilation steps to run.")

        D = Y  # Switch from API notation to paper notation
        N_d, N_e = D.shape  # (num_observations, ensemble_size)
        assert N_d == self.observations.shape[0], "Shape mismatch"
        delta_D = D - np.mean(D, axis=1, keepdims=True)  # Center observations

        # Compute the last factor
        alpha = self.alpha[self.iteration]
        D_obs = self.perturb_observations(ensemble_size=N_e, alpha=alpha)
        self.D_obs_minus_D = D_obs - D

        # Compute parts of the Kalman gain
        self.delta_D_inv_cov = invert_subspace(
            delta_D=delta_D, C_D_L=self.C_D_L, alpha=alpha, truncation=truncation
        )

        self.iteration += 1

    def assimilate_batch(
        self,
        *,
        X: npt.NDArray[np.double],
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

        # The default localization is no localization (identity function)
        if localization_callback is None:

            def localization_callback(
                K: npt.NDArray[np.double],
            ) -> npt.NDArray[np.double]:
                return K

        assert X.ndim == 2
        N_m, N_e = X.shape  # (num_parameters, ensemble_size)
        assert N_e == self.delta_D_inv_cov.shape[0], "Dimension mismatch"

        # Center the parameters
        delta_M = X - np.mean(X, axis=1, keepdims=True)

        # Create Kalman gain of shape (num_parameters_batch, num_observations),
        # then apply the localization callback elementwise
        K = localization_callback(delta_M @ self.delta_D_inv_cov)
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
