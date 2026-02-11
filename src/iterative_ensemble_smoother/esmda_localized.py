"""
Localized ESMDA
---------------

This module implements localized ESMDA, following the paper:

    - "Analysis of the performance of ensemble-based assimilation of production and seismic data"
      Alexandre A. Emerick


API design
----------

The interface uses human-readable names, just like ESMDA.
The implementation (local variables in methods) follows the notation in the paper.

The central idea behind localization is to study equation (B.6) and the Kalman gain K:

    M + (\delta M)(\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1) (D_obs - D) =
    M + K (D_obs - D)

The Kalman gain K has shape (parameters, observations) and encodes how much each
observation is influenced by each parameter in every update step of the algorithm.
Given a localization matrix of the same shape with entries in the range [0, 1],
we can "regularize" the kalmain gain:

    K_regularized = K * localization    (elementwise product)


An API might look like:

# Create smoohter instance. Set up all global state used in all iterations
smoother = LocalizedESMDA(covariance, observations, alpha, seed, inversion)
X = np.random.randn(...)

for iteration in range(num_assimilations):

    # Run simulation and keep track of living indices
    Y, living_idx = g(X)

    # Set up all state used for this iteration:
    # (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1) (D_obs - D)
    # With the possibility of dropping dead ensemble members

    smoother.prepare_assimilation(Y, living_idx, truncation=0.99)

    # Loop over parameter blocks and update them
    for param_block_idx in parameter_indicies_generator():

        def localization_callback(K):
            # Logic that ties each parameter index in this block to the observations
            return K * localization

        X[param_block_idx, living_idx] = smoother.assimilate(X=X[param_block_idx, living_idx],
                                                             localization_callback=localization_callback
                                                             )


Comments
--------

- If `localization_callback` is the identity function, LocalizedESMDA is identical to ESMDA.
- The inner loop over parameter blocks saves memory. The result should be the same over any
  possible sequence of parameter blocks.
- The caller is responsible for keeping track of relationships between input parameters and
  observations. For instance, if some points in an input parameter grid are known to be close
  to an observation, the user can create a helper class Grid to keep track of this.

"""

import numbers
from abc import ABC
from typing import Union

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore

from iterative_ensemble_smoother.esmda_inversion import (
    inversion_exact_cholesky,
    inversion_subspace,
    normalize_alpha,
)
from iterative_ensemble_smoother.utils import sample_mvnormal
from iterative_ensemble_smoother.esmda import BaseESMDA
from iterative_ensemble_smoother.esmda_inversion import singular_values_to_keep


def invert_naive(*, delta_D, C_D, alpha, truncation):
    """Naive inversion of the equation:

    (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)
    """
    _, N_e = delta_D.shape  # Number of ensemble members

    covariance = np.diag(C_D) if C_D.ndim == 1 else C_D
    delta_D_inv_cov = delta_D.T @ np.linalg.inv(
        delta_D @ delta_D.T + alpha * (N_e - 1) * covariance
    )
    return delta_D_inv_cov


def invert(*, delta_D, C_D, alpha, truncation):
    """Not-so-naive inversion of the equation:

    (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)
    """
    _, N_e = delta_D.shape  # Number of ensemble members

    # Equivalent to: delta_D @ delta_D.T, but only computes upper triangular part
    inner = sp.linalg.blas.dsyrk(alpha=1.0, a=delta_D)

    # Add to diagonal
    if C_D.ndim == 1:
        np.fill_diagonal(inner, np.diagonal(inner) + alpha * (N_e - 1) * C_D)
    else:
        inner += alpha * (N_e - 1) * C_D

    # Arguments for sp.linalg.solve
    solver_kwargs = {
        "overwrite_a": True,
        "overwrite_b": True,
        "assume_a": "pos",  # Assume positive definite matrix (use cholesky)
        "lower": True,  # Only use the lower part (upper before transpose) while solving
    }

    # Computes X = delta_D.T @ inner^{-1} by solving a system of equations
    return sp.linalg.solve(inner.T, delta_D, **solver_kwargs).T


def invert_subspace(*, delta_D, C_D, alpha, truncation):
    N_d, N_e = delta_D.shape  # (num_observations, ensemble_size)

    # Extract diagonals
    if C_D.ndim == 1:
        S = np.sqrt(C_D)
    else:
        S = np.sqrt(np.diag(C_D))
    S_inv = 1 / S

    # Equation (B.10), which is equivalent to: np.diag(S_inv) @ delta_D
    S_inv_delta_D = delta_D * S_inv[:, np.newaxis]
    assert np.allclose(S_inv_delta_D, np.diag(S_inv) @ delta_D)

    U, w, _ = sp.linalg.svd(S_inv_delta_D, overwrite_a=True, full_matrices=False)
    idx = singular_values_to_keep(w, truncation=truncation)

    # assert np.allclose(VT @ VT.T, np.eye(VT.shape[0]))
    N_r = min(N_d, N_e - 1, idx)  # Number of values in SVD to keep
    U_r, w_r = U[:, :N_r], w[:N_r]  # U_r.shape = (num_observations, ~ensemble_size)

    if C_D.ndim == 1:
        # This is equation (B.13), which can be simplified drastically:
        # R = alpha (N_e - 1) diag(1/w_r) @ U_r.T @ hat(C_D) @ U_r @ diag(1/w_r)
        #  = alpha (N_e - 1) diag(1/w_r) @ U_r.T @ (diag(1/sqrt(C_D))) @ diag(C_D) @ (diag(1/sqrt(C_D))) @ U_r @ diag(1/w_r)
        #  = alpha (N_e - 1) F.T @ F
        # where F = diag(sqrt(C_d)) @ (diag(1/sqrt(C_D))) @ U_r @ diag(1/w_r)
        #         = I @ U_r @ diag(1/w_r)
        #         = U_r * (1/w_r)[np.newaxis,:]
        F = U_r * (1 / w_r)[np.newaxis, :]

        # The matrix F has shape (num_observations, ~ensemble_size), so it is thin.
        # Taking the product first makes it small, and is faster than computing
        # the SVD of F or F.T directly.
        Z_r, h_r, _ = sp.linalg.svd(F.T @ F, full_matrices=False)

    # This is equation (B.18), and is equivalent to:
    # X = np.diag(S_inv) @ U_r @ np.diag(1/w_r) @ Z_r
    X = (U_r * S_inv[:, np.newaxis]) @ (Z_r * (1 / w_r)[:, np.newaxis])
    L = 1 / (1 + h_r)
    return np.linalg.multi_dot([delta_D.T, X * L[np.newaxis], X.T])


class LocalizedESMDA(BaseESMDA):
    # Available inversion methods. The inversion methods all compute
    # (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)
    _inversion_methods = {
        "exact": inversion_exact_cholesky,
        "subspace": inversion_subspace,
    }

    def __init__(
        self,
        *,
        covariance: npt.NDArray[np.double],
        observations: npt.NDArray[np.double],
        alpha: Union[int, npt.NDArray[np.double]] = 5,
        seed: Union[np.random._generator.Generator, int, None] = None,
        inversion: str = "exact",
    ) -> None:
        """Initialize the instance."""

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

        # Store data
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

    def prepare_assmilation(self, *, Y):
        """Prepare assimilation of one or several batches of parameters.

        Parameters
        ----------
        Y : np.ndarray
            2D array of shape (num_observations, ensemble_size), containing
            responses when evaluating the model at X. In other words, Y = g(X),
            where g is the forward model.

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

            - The next-to-last factors: (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)
            - The last factor: (D_obs - D)

        The total internal storage is: 2 * ensemble_size * num_observations, since shapes are:

            - The next-to-last factors: (ensemble_size, num_observations)
            - The last factor: (num_observations, ensemble_size)
        """
        assert Y.ndim == 2
        if self.iteration >= self.num_assimilations():
            raise Exception("No more assimilation steps to run.")

        # We strictly follow the notation from the Appendix in the paper
        N_d, N_e = Y.shape  # (num_observations, ensemble_size)
        delta_D = Y - np.mean(Y, axis=1, keepdims=True)  # Center the observations

        # Compute the last factor
        alpha = self.alpha[iteration]
        D_obs = self.perturb_observations(ensemble_size=N_e, alpha=alpha)
        self.D_obs_minus_D = D_obs - Y

        # Compute the next-to-last factor
        # TODO: Here we must apply a better inversion method
        truncation = 1.0
        self.delta_D_inv_cov = invert_naive(
            delta_D=delta_D, C_D=self.C_D, alpha=alpha, truncation=truncation
        )

        assert np.allclose(
            invert_naive(
                delta_D=delta_D, C_D=self.C_D, alpha=alpha, truncation=truncation
            ),
            invert(delta_D=delta_D, C_D=self.C_D, alpha=alpha, truncation=truncation),
        )

        naive = invert_naive(
            delta_D=delta_D, C_D=self.C_D, alpha=alpha, truncation=truncation
        )
        subspace = invert_subspace(
            delta_D=delta_D, C_D=self.C_D, alpha=alpha, truncation=truncation
        )
        print(np.sqrt(np.mean((naive - subspace) ** 2)))

        assert np.allclose(
            invert_naive(
                delta_D=delta_D, C_D=self.C_D, alpha=alpha, truncation=truncation
            ),
            invert_subspace(
                delta_D=delta_D, C_D=self.C_D, alpha=alpha, truncation=truncation
            ),
        )

        self.iteration += 1
        return self

    def assimilate_batch(self, *, X, localization_callback=None):
        """Assimilate a batch of parameters aginst all observations.

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

        # The default localization is no localization (identity function)
        if localization_callback is None:

            def localization_callback(K):
                return K

        assert X.ndim == 2
        N_m, N_e = X.shape  # (num_parameters, ensemble_size)
        assert N_e == self.delta_D_inv_cov.shape[0], "Dimension mismatch"

        # Center the parameters
        delta_M = X - np.mean(X, axis=1, keepdims=True)

        # Create Kalman gain of shape (num_parameters_batch, ensemble_size),
        # then apply the localization callback elementwise
        K = localization_callback(delta_M @ self.delta_D_inv_cov)
        return X + K @ self.D_obs_minus_D


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    A = rng.normal(size=(3, 10))

    def forward_model(x):
        return A @ x

    # Then we set up the ESMDA instance and the prior realizations X:

    covariance = np.ones(3)  # Covariance of the observations / outputs
    observations = np.array([1, 2, 3])  # The observed data
    smoother = LocalizedESMDA(
        covariance=covariance, observations=observations, alpha=3, seed=42
    )
    X = rng.normal(size=(10, 100))

    # To assimilate data, we iterate over the assimilation steps:

    for iteration in range(smoother.num_assimilations()):
        # Apply the forward model in each realization in the ensemble
        Y = np.array([forward_model(x) for x in X.T]).T

        smoother.prepare_assmilation(Y=Y)

        for batch_idx in [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]]:

            def localization_callback(K):
                return K

            X[batch_idx, :] = smoother.assimilate_batch(
                X=X[batch_idx, :], localization_callback=localization_callback
            )


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
        ]
    )
