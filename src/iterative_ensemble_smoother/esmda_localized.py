"""
Localized ESMDA
---------------

This module implements localized ESMDA, following the paper:

    - "Analysis of the performance of ensemble-based assimilation of production and seismic data"
      Alexandre A. Emerick

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

with shape (p, o) and applies a localization function elementwise to this matrix.
This has a cost of at least O(poe) (right-to-left), which is not ideal.

The disadvantage is the memory and computational requirement, but the advantage
is that we can apply the elementwise localization function, which determines
how parameter i should influence observation j, at entry K_ij.

Since storing K in memory at once is often prohibitive, we actually first form:

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
from typing import Union

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore

from iterative_ensemble_smoother.esmda_inversion import (
    normalize_alpha,
)
from iterative_ensemble_smoother.esmda import BaseESMDA
from iterative_ensemble_smoother.esmda_inversion import singular_values_to_keep


# =============== Inversion methods ===============


def invert_naive(*, delta_D, C_D, alpha, truncation):
    """Naive implementation of the equation:

    (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)

    This function should only be used for testing and verification.
    """
    _, N_e = delta_D.shape  # Number of ensemble members

    covariance = np.diag(C_D) if C_D.ndim == 1 else C_D
    delta_D_inv_cov = delta_D.T @ np.linalg.inv(
        delta_D @ delta_D.T + alpha * (N_e - 1) * covariance
    )
    return delta_D_inv_cov


def invert_exact(*, delta_D, C_D, alpha, truncation):
    """Not-so-naive implementation of the equation:

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
    """Subspace inversion implementation of the equation:

    (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)

    See the appendix in the 2016 Emerick paper for details:
    https://doi.org/10.1016/j.petrol.2016.01.029
    """
    N_d, N_e = delta_D.shape  # (num_observations, ensemble_size)

    # Extract diagonals
    if C_D.ndim == 1:
        S = np.sqrt(C_D)
    else:
        S = np.sqrt(np.diag(C_D))
    S_inv = 1 / S

    # Equation (B.10), which is equivalent to: np.diag(S_inv) @ delta_D
    S_inv_delta_D = delta_D * S_inv[:, np.newaxis]
    # assert np.allclose(S_inv_delta_D, np.diag(S_inv) @ delta_D)

    U, w, _ = sp.linalg.svd(
        S_inv_delta_D, overwrite_a=True, full_matrices=False, check_finite=False
    )
    idx = singular_values_to_keep(w, truncation=truncation)

    # assert np.allclose(VT @ VT.T, np.eye(VT.shape[0]))
    N_r = min(N_d, N_e - 1, idx)  # Number of values in SVD to keep
    U_r, w_r = U[:, :N_r], w[:N_r]  # U_r.shape = (num_observations, ~ensemble_size)

    if C_D.ndim == 1:
        # This is equation (B.13), which can be simplified drastically in the 1D case:
        # R = alpha (N_e - 1) diag(1/w_r) @ U_r.T @ hat(C_D) @ U_r @ diag(1/w_r)
        #  = alpha (N_e - 1) diag(1/w_r) @ U_r.T @ (diag(1/sqrt(C_D))) @ diag(C_D) @ (diag(1/sqrt(C_D))) @ U_r @ diag(1/w_r)
        #  = alpha (N_e - 1) F.T @ F
        # where F = diag(sqrt(C_d)) @ (diag(1/sqrt(C_D))) @ U_r @ diag(1/w_r)
        #         = I @ U_r @ diag(1/w_r)
        #         = U_r * (1/w_r)[np.newaxis,:]
        F = U_r * (1 / w_r)[np.newaxis, :]

        # The matrix F has shape (num_observations, ~ensemble_size), so it is thin.
        # Taking the product first makes it small, and is faster than computing
        # the SVD of F or F.T directly. The code below is equivalent to:
        # Z_r, h_r, _ = sp.linalg.svd(F.T @ F, full_matrices=False)

        gram_dsyrk = sp.linalg.blas.dsyrk(alpha * (N_e - 1), F.T, lower=1)
        h_r, Z_r = sp.linalg.eigh(
            gram_dsyrk, overwrite_a=True, check_finite=False, driver="evr", lower=True
        )
    else:
        # Equivalent to:
        # np.diag(S_inv)  @ U_r @ np.diag(1/w_r)
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

    # This is the fastest approach:
    # print(delta_D.T.shape, X.shape) # (100, 1000) (1000, 99)
    temp = (delta_D.T @ X) * L[np.newaxis, :]
    return temp @ X.T


class LocalizedESMDA(BaseESMDA):
    # Available inversion methods. The inversion methods all compute
    # (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)
    _inversion_methods = {
        "naive": invert_naive,
        "exact": invert_exact,
        "subspace": invert_subspace,
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

    def prepare_assmilation(self, *, Y, truncation=1.0):
        """Prepare assimilation of one or several batches of parameters.

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
            routine. Must be a float in the range (0, 1]. A lower number means
            a more approximate answer and a slightly faster computation.

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
        assert N_d == self.observations.shape[0], "Shape mismatch"
        delta_D = Y - np.mean(Y, axis=1, keepdims=True)  # Center the observations

        # Compute the last factor
        alpha = self.alpha[self.iteration]
        D_obs = self.perturb_observations(ensemble_size=N_e, alpha=alpha)
        self.D_obs_minus_D = D_obs - Y

        # Compute parts of the Kalman gain
        inversion_func = self._inversion_methods[self.inversion]
        self.delta_D_inv_cov = inversion_func(
            delta_D=delta_D, C_D=self.C_D, alpha=alpha, truncation=truncation
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
        print(X.shape, K.shape, self.D_obs_minus_D.shape)
        return X + K @ self.D_obs_minus_D


if __name__ == "__main__":
    from time import perf_counter

    rng = np.random.default_rng(42)
    num_obs = 10
    num_params = 10
    num_realizations = 100

    A = rng.normal(size=(num_obs, num_params))

    def forward_model(x):
        return A @ x

    # Then we set up the ESMDA instance and the prior realizations X:

    covariance = 2.0 ** (
        -np.linspace(0, -10, num_obs)
    )  # Covariance of the observations / outputs
    covariance = np.diag(covariance)
    observations = np.ones(num_obs)  # The observed data
    smoother = LocalizedESMDA(
        covariance=covariance,
        observations=observations,
        alpha=3,
        seed=42,
        inversion="subspace",
    )
    X = rng.normal(size=(num_params, num_realizations))

    # To assimilate data, we iterate over the assimilation steps:

    for iteration in range(smoother.num_assimilations()):
        # Apply the forward model in each realization in the ensemble
        Y = np.array([forward_model(x) for x in X.T]).T

        # break

        st = perf_counter()
        smoother.prepare_assmilation(Y=Y)

        def localization_callback(K):
            return K

        X = smoother.assimilate_batch(X=X, localization_callback=localization_callback)
        print(perf_counter() - st)

        for batch_idx in [[0]]:

            def localization_callback(K):
                return K

            X[batch_idx, :] = smoother.assimilate_batch(
                X=X[batch_idx, :], localization_callback=localization_callback
            )

    from iterative_ensemble_smoother.experimental import DistanceESMDA

    esmda_distance = DistanceESMDA(
        covariance=covariance,
        observations=observations,
        alpha=np.array([1, 2]),
        seed=rng,
    )
    rho = np.ones((X.shape[0], Y.shape[0]))

    st = perf_counter()
    X_posterior = esmda_distance.assimilate(X=X, Y=Y, rho=rho)
    print(perf_counter() - st)

if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
        ]
    )
