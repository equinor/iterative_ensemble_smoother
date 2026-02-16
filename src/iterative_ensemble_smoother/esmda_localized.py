"""
Localized ESMDA
---------------

This module implements localized ESMDA, following the paper:

    - "Analysis of the performance of ensemble-based assimilation
       of production and seismic data", Alexandre A. Emerick

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
The localization function is the matrix rho in the paper by Emerick.
This has a cost of at least O(poe) (right-to-left), which is not ideal.

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
    C_D: npt.NDArray[np.double],
    alpha: float,
    truncation: float,
) -> npt.NDArray[np.double]:
    r"""Naive implementation of the equation:

    (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)

    This function should only be used for testing and verification.
    """
    _, N_e = delta_D.shape  # Number of ensemble members

    covariance = np.diag(C_D) if C_D.ndim == 1 else C_D
    return delta_D.T @ np.linalg.inv(
        delta_D @ delta_D.T + alpha * (N_e - 1) * covariance
    )


def invert_exact(
    *,
    delta_D: npt.NDArray[np.double],
    C_D: npt.NDArray[np.double],
    alpha: float,
    truncation: float,
) -> npt.NDArray[np.double]:
    r"""Not-so-naive implementation of the equation:

    (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)
    """
    _, N_e = delta_D.shape  # Number of ensemble members

    # Equivalent to: delta_D @ delta_D.T, but only computes upper triangular part
    inner = sp.linalg.blas.dsyrk(alpha=1.0, a=delta_D)

    # Add to diagonal
    if C_D.ndim == 1:
        new_diagonal = np.diagonal(inner) + alpha * (N_e - 1) * C_D
        np.fill_diagonal(inner, new_diagonal)
        scaling = np.sqrt(1 / new_diagonal)  # Scaling factor based on diagonal
    else:
        inner += alpha * (N_e - 1) * C_D
        scaling = np.sqrt(1 / np.diag(inner))

    # Scale to correlation-like matrix by scaling rows and columns with diag
    inner *= scaling[:, None]
    inner *= scaling[None, :]

    # Scale RHS
    rhs = delta_D * scaling[:, None]

    # Arguments for sp.linalg.solve
    solver_kwargs = {
        "overwrite_a": True,
        "overwrite_b": True,
        "assume_a": "pos",  # Assume positive definite matrix (use cholesky)
        "lower": True,  # Only use the lower part (upper before transpose) while solving
    }
    # Computes X = delta_D.T @ inner^{-1} by solving a system of equations
    X = sp.linalg.solve(inner.T, rhs, **solver_kwargs)
    X *= scaling[:, None]  # Scale back
    return X.T


def invert_subspace(
    *,
    delta_D: npt.NDArray[np.double],
    C_D: npt.NDArray[np.double],
    alpha: float,
    truncation: float,
) -> npt.NDArray[np.double]:
    r"""Subspace inversion (without rescaling) implementation of the equation:

    (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)

    See the appendix in the 2016 Emerick paper for details:
    https://doi.org/10.1016/j.petrol.2016.01.029
    """
    N_d, N_e = delta_D.shape  # (num_observations, ensemble_size)

    # Take SVD of delta_D, choose singular values and keep the top ones
    U, w, _ = sp.linalg.svd(
        delta_D,
        overwrite_a=False,  # delta_D is used later
        full_matrices=False,
        check_finite=False,
    )
    idx = singular_values_to_keep(w, truncation=truncation)
    N_r = min(N_d, N_e - 1, idx)  # Number of values in SVD to keep
    U_r, w_r = U[:, :N_r], w[:N_r]  # U_r.shape = (num_observations, ~ensemble_size)

    # The matrix F has shape (num_observations, ~ensemble_size), so it is thin.
    # Therefore we compute F.T @ F before taking SVD / EIG.
    F = U_r * (1 / w_r)[np.newaxis, :]  # Factor used next
    if C_D.ndim == 1:
        F = F * np.sqrt(C_D)[:, np.newaxis]
        gram_dsyrk = sp.linalg.blas.dsyrk(alpha * (N_e - 1), F.T, lower=1)
        h_r, Z_r = sp.linalg.eigh(
            gram_dsyrk, overwrite_a=True, check_finite=False, driver="evr", lower=True
        )
    else:
        h_r, Z_r = sp.linalg.eigh(
            alpha * (N_e - 1) * F.T @ C_D @ F,
            overwrite_a=True,
            check_finite=False,
            driver="evr",
            lower=True,
        )

    # This is equation (B.18), and is equivalent to:
    # X = np.diag(S_inv) @ U_r @ np.diag(1/w_r) @ Z_r
    X = U_r @ (Z_r * (1 / w_r)[:, np.newaxis])
    L = 1 / (1 + h_r)

    # This is the fastest approach:
    temp = (delta_D.T @ X) * L[np.newaxis, :]
    return temp @ X.T


def invert_subspace_scaled(
    *,
    delta_D: npt.NDArray[np.double],
    C_D: npt.NDArray[np.double],
    alpha: float,
    truncation: float,
) -> npt.NDArray[np.double]:
    r"""Subspace inversion (with rescaling) implementation of the equation:

    (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)

    See the appendix in the 2016 Emerick paper for details:
    https://doi.org/10.1016/j.petrol.2016.01.029
    """
    N_d, N_e = delta_D.shape  # (num_observations, ensemble_size)

    # Extract diagonals, used to re-scale
    S = np.sqrt(C_D) if C_D.ndim == 1 else np.sqrt(np.diag(C_D))
    S_inv = 1 / S

    # Equation (B.10), which is equivalent to: np.diag(S_inv) @ delta_D
    S_inv_delta_D = delta_D * S_inv[:, np.newaxis]

    # Take SVD and only keep the largest singular values
    U, w, _ = sp.linalg.svd(
        S_inv_delta_D, overwrite_a=True, full_matrices=False, check_finite=False
    )
    idx = singular_values_to_keep(w, truncation=truncation)
    N_r = min(N_d, N_e - 1, idx)  # Number of values in SVD to keep
    U_r, w_r = U[:, :N_r], w[:N_r]  # U_r.shape = (num_observations, ~ensemble_size)

    if C_D.ndim == 1:
        # This is equation (B.13), which can be simplified drastically in the 1D case:
        # R = alpha (N_e - 1) diag(1/w_r) @ U_r.T @ hat(C_D) @ U_r @ diag(1/w_r)
        #  = alpha (N_e - 1) diag(1/w_r) @ U_r.T @ (diag(1/sqrt(C_D))) @ diag(C_D) @
        #    (diag(1/sqrt(C_D))) @ U_r @ diag(1/w_r)
        #  = alpha (N_e - 1) F.T @ F
        # where F = diag(sqrt(C_d)) @ (diag(1/sqrt(C_D))) @ U_r @ diag(1/w_r)
        #         = I @ U_r @ diag(1/w_r)
        #         = U_r * (1/w_r)[np.newaxis,:]
        F = U_r * (1 / w_r)[np.newaxis, :]

        # The matrix F has shape (num_observations, ~ensemble_size), so it is thin.
        # Taking the product first makes it small, and is faster than computing
        # the SVD of F or F.T directly. The code below is equivalent to:
        # Z_r, h_r, _ = sp.linalg.svd(alpha * (N_e - 1) * F.T @ F)
        gram_dsyrk = sp.linalg.blas.dsyrk(alpha * (N_e - 1), F.T, lower=1)
        h_r, Z_r = sp.linalg.eigh(
            gram_dsyrk, overwrite_a=True, check_finite=False, driver="evr", lower=True
        )
    else:
        # Equivalent to:
        # F = np.diag(S_inv) @ U_r @ np.diag(1/w_r)
        F = U_r * (1 / w_r)[np.newaxis, :] * S_inv[:, np.newaxis]
        h_r, Z_r = sp.linalg.eigh(
            alpha * (N_e - 1) * F.T @ C_D @ F,
            overwrite_a=True,
            check_finite=False,
            driver="evr",
            lower=True,
        )

    # This is equation (B.18), and is equivalent to:
    # X = np.diag(S_inv) @ U_r @ np.diag(1/w_r) @ Z_r
    X = (U_r * S_inv[:, np.newaxis]) @ (Z_r * (1 / w_r)[:, np.newaxis])
    L = 1 / (1 + h_r)

    # This is the fastest approach I've found to compute:
    # delta_D.T @ X @ np.diag(L) @ X.T @ delta_D
    temp = (delta_D.T @ X) * L[np.newaxis, :]
    return temp @ X.T


class LocalizedESMDA(BaseESMDA):
    """
    Implement a Localized Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

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
    inversion : str, optional
        Which inversion method to use. The default is "subspace_scaled".
        See the dictionary LocalizedESMDA._inversion_methods for more information.

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
    ...                           observations=observations, alpha=3, seed=42,
    ...                           inversion="subspace_scaled")
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

    # Available inversion methods. The inversion methods all compute
    # (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)
    _inversion_methods = {
        "naive": invert_naive,
        "exact": invert_exact,
        "subspace": invert_subspace,
        "subspace_scaled": invert_subspace_scaled,
    }

    def __init__(
        self,
        *,
        covariance: npt.NDArray[np.double],
        observations: npt.NDArray[np.double],
        alpha: Union[int, npt.NDArray[np.double]] = 5,
        seed: Union[np.random._generator.Generator, int, None] = None,
        inversion: str = "subspace_scaled",
    ) -> None:
        super().__init__(covariance=covariance, observations=observations, seed=seed)

        if not (
            (isinstance(alpha, np.ndarray) and alpha.ndim == 1)
            or isinstance(alpha, numbers.Integral)
        ):
            raise TypeError("Argument `alpha` must be an integer or a 1D NumPy array.")

        if not isinstance(inversion, str):
            raise TypeError(
                "Argument `inversion` must be a string in "
                f"{tuple(self._inversion_methods.keys())}, but got {inversion}"
            )
        if inversion not in self._inversion_methods:
            raise ValueError(
                "Argument `inversion` must be a string in "
                f"{tuple(self._inversion_methods.keys())}, but got {inversion}"
            )

        self.inversion = inversion

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
        inversion_func = self._inversion_methods[self.inversion]
        self.delta_D_inv_cov = inversion_func(
            delta_D=delta_D, C_D=self.C_D, alpha=alpha, truncation=truncation
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
