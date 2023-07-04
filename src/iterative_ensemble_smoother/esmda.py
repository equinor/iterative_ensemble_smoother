#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

References
----------

Emerick, A.A., Reynolds, A.C. History matching time-lapse seismic data using
the ensemble Kalman filter with multiple data assimilations.
Comput Geosci 16, 639–659 (2012). https://doi.org/10.1007/s10596-012-9275-5

Alexandre A. Emerick, Albert C. Reynolds.
Ensemble smoother with multiple data assimilation.
Computers & Geosciences, Volume 55, 2013, Pages 3-15, ISSN 0098-3004,
https://doi.org/10.1016/j.cageo.2012.03.011.

"""

import numpy as np
import scipy as sp


def empirical_covariance_upper(X):
    """Compute the upper part of the empirical covariance.

    Examples
    --------
    >>> X = np.array([[-2.4, -0.3,  0.7,  0.2,  1.1],
    ...               [-1.5,  0.4, -0.4, -0.9,  1. ],
    ...               [-0.1, -0.4, -0. , -0.5,  1.1]])
    >>> empirical_covariance_upper(X)
    array([[1.873, 0.981, 0.371],
           [0.   , 0.997, 0.392],
           [0.   , 0.   , 0.407]])

    Naive computation:

    >>> empirical_cross_covariance(X, X)
    array([[1.873, 0.981, 0.371],
           [0.981, 0.997, 0.392],
           [0.371, 0.392, 0.407]])
    """
    num_variables, num_observationservations = X.shape
    X = X - np.mean(X, axis=1, keepdims=True)
    # https://www.math.utah.edu/software/lapack/lapack-blas/dsyrk.html
    XXT = sp.linalg.blas.dsyrk(alpha=1, a=X)
    XXT /= num_observationservations - 1
    return XXT


def empirical_cross_covariance(X, Y):
    """Both X and Y have shape (parameters, ensemble_size).

    We use this function instead of np.cov to handle cross-correlation,
    where X and Y have a different number of parameters.

    Examples
    --------
    >>> X = np.array([[-2.4, -0.3,  0.7],
    ...               [ 0.2,  1.1, -1.5]])
    >>> Y = np.array([[ 0.4, -0.4, -0.9],
    ...               [ 1. , -0.1, -0.4],
    ...               [-0. , -0.5,  1.1],
    ...               [-1.8, -1.1,  0.3]])
    >>> empirical_cross_covariance(X, Y)
    array([[-1.035     , -1.15833333,  0.66      ,  1.56333333],
           [ 0.465     ,  0.36166667, -1.08      , -1.09666667]])
    """

    assert X.shape[1] == Y.shape[1], "Ensemble size must be equal"
    X = X - np.mean(X, axis=1, keepdims=True)
    Y = Y - np.mean(Y, axis=1, keepdims=True)
    cov = X @ Y.T / (X.shape[1] - 1)
    assert cov.shape == (X.shape[0], Y.shape[0])
    return cov


def normalize_alpha(alpha):
    """Assure that sum_i (1/alpha_i) = 1

    Examples
    --------
    >>> alpha = np.arange(10) + 1
    >>> np.sum(1/normalize_alpha(alpha))
    1.0
    """
    factor = np.sum(1 / alpha)
    return alpha * factor


# =============================================================================
# INVERSION FUNCTIONS
# =============================================================================
def inversion_naive(*, C_DD, alpha, C_D, D, Y):
    """Naive inversion, used for testing only."""
    # Naive implementation of Equation (3) in Ensemble smoother with multiple...
    return sp.linalg.inv(C_DD + alpha * C_D) @ (D - Y)


def inversion_exact(*, C_DD, alpha, C_D, D, Y):
    solver_kwargs = {"overwrite_a": True, "overwrite_b": True, "assume_a": "pos"}

    # Compute K := sp.linalg.inv(C_DD + alpha * C_D) @ (D - Y)
    if C_D.ndim == 2:
        # C_D is a covariance matrix
        K = sp.linalg.solve(C_DD + alpha * C_D, D - Y, **solver_kwargs)
    elif C_D.ndim == 1:
        # C_D is an array, so add it to the diagonal without forming diag(C_D)
        C_DD.flat[:: num_outputs + 1] += alpha * C_D
        K = sp.linalg.solve(C_DD, D - Y, **solver_kwargs)
    return K


def inversion_lstsq(*, C_DD, alpha, C_D, D, Y):
    ans, *_ = sp.linalg.lstsq(
        C_DD + alpha * C_D, D - Y, overwrite_a=True, overwrite_b=True
    )
    return ans


def inversion_rescaled(*, C_DD, alpha, C_D, D, Y):
    # Cholesky factorize
    C_D_L = sp.linalg.cholesky(C_D * alpha, lower=True)  # Lower triangular cholesky
    C_D_L_inv, _ = sp.linalg.lapack.dtrtri(C_D_L, lower=1)  # Invert lower triangular

    # Form C_tilde
    # sp.linalg.blas.strmm(alpha=1, a=C_D_L_inv, b=C_DD, lower=1)
    C_tilde = C_D_L_inv @ C_DD @ C_D_L_inv.T
    C_tilde.flat[:: C_tilde.shape[0] + 1] += 1  # Add to diagonal

    # Compute SVD (we could do eigenvalues, since C_tilde is symmetric, but it
    # turns out that computing the SVD is faster)

    # On a 1000 x 1000 example, 80% of the time is spent here
    U, s, _ = sp.linalg.svd(C_tilde, overwrite_a=True, full_matrices=False)
    # TODO: truncate SVD

    term = C_D_L_inv.T @ U
    return np.linalg.multi_dot([term / s, term.T, (D - Y)])


def test_inversions(k=10):
    emsemble_members = k

    F = np.random.randn(k, k)
    C_DD = F.T @ F

    E = np.random.randn(k, k)
    C_D = E.T @ E

    alpha = 2

    D = np.random.randn(k, emsemble_members)
    Y = np.random.randn(k, emsemble_members)

    K1 = inversion_naive(C_DD=C_DD, alpha=alpha, C_D=C_D, D=D, Y=Y)
    K2 = inversion_exact(C_DD=C_DD, alpha=alpha, C_D=C_D, D=D, Y=Y)
    K3 = inversion_rescaled(C_DD=C_DD, alpha=alpha, C_D=C_D, D=D, Y=Y)
    K4 = inversion_lstsq(C_DD=C_DD, alpha=alpha, C_D=C_D, D=D, Y=Y)

    assert np.allclose(K1, K2)
    assert np.allclose(K1, K3)
    assert np.allclose(K1, K4)


test_inversions(k=100)

np.random.seed(21)
k = 10

Y = np.random.randn(k, k * 5)
N_n, N_e = Y.shape
D_delta = Y - np.mean(Y, axis=1, keepdims=True)  # Subtract average

E = np.random.randn(k, k)
C_D = E.T @ E
D = np.random.randn(k, k * 5)

alpha = 2

# -------------------------------------------

U, w, _ = sp.linalg.svd(D_delta, overwrite_a=True, full_matrices=True)
clip = min(N_n, N_e - 1)
U_r, w_r = U[:, :clip], w[:clip]

U_r_w_inv = U_r / w_r
X = (N_e - 1) * np.linalg.multi_dot([U_r_w_inv.T, alpha * C_D, U_r_w_inv])

Z, T, _ = sp.linalg.svd(X, overwrite_a=True, full_matrices=True)

term = U_r_w_inv @ Z

C_inv = (term / (1 + T)) @ term.T
K5 = (N_e - 1) * C_inv @ (D - Y)

# -------------------------------------------

C_hat = D_delta @ D_delta.T + (N_e - 1) * alpha * C_D
assert np.allclose(np.linalg.inv(C_hat), C_inv)

K1 = inversion_naive(
    C_DD=empirical_cross_covariance(Y, Y), alpha=alpha, C_D=C_D, D=D, Y=Y
)
assert np.allclose(K1, K5)


1 / 0


class ESMDA:
    def __init__(self, C_D, observations, alpha=None, seed=None, inversion="exact"):
        self.observations = observations
        self.iteration = 0
        self.rng = np.random.default_rng(seed)
        self.inversion = inversion

        # Alpha can either be a number (of iterations) or a list of weights
        if isinstance(alpha, np.ndarray) and alpha.ndim == 1:
            self.alpha = normalize_alpha(alpha)
        elif isinstance(alpha, int):
            self.alpha = np.array([1 / alpha] * alpha)
        else:
            raise TypeError("Alpha must be integer or 1D array.")

        # Only compute the covariance factorization once
        # If it's a full matrix, we gain speedup by only computing cholesky once
        # If it's a diagonal, we gain speedup by never having to compute cholesky
        num_outputs, num_ensemble = observations.shape

        if isinstance(C_D, np.ndarray) and C_D.ndim == 2:
            L = sp.linalg.cholesky(C_D, lower=False)
            cov = sp.stats.Covariance.from_cholesky(L)
        elif isinstance(C_D, np.ndarray) and C_D.ndim == 1:
            assert len(C_D) == num_outputs
            cov = sp.stats.Covariance.from_diagonal(C_D)
        elif isinstance(C_D, float):
            C_D = np.array([C_D] * num_outputs)  # Convert to array
            cov = sp.stats.Covariance.from_diagonal(C_D)

        mean = np.zeros(num_outputs)
        self.mv_normal = sp.stats.multivariate_normal(mean=mean, cov=cov)
        self.C_D = C_D
        assert isinstance(self.C_D, np.ndarray) and self.C_D.ndim in (1, 2)

    def num_assimilations(self):
        return len(self.alpha)

    def assimilate(self, X, Y):
        if self.iteration >= self.num_assimilations():
            raise Exception("No more assimilation steps to run.")

        num_inputs, num_ensemble1 = X.shape
        num_outputs, num_emsemble2 = Y.shape
        assert (
            num_ensemble1 == num_emsemble2
        ), "Number of ensemble members in X and Y must match"

        # Sample from a zero-centered multivariate normal with cov=C_D
        mv_normal_rvs = self.mv_normal.rvs(size=num_ensemble, random_state=self.rng).T

        # Create perturbed observationservations, scaled by alpha
        D = self.observations + self.alpha[self.iteration] * mv_normal_rvs

        # Update the ensemble
        C_MD = empirical_cross_covariance(X, Y)
        C_DD = empirical_covariance_upper(Y)  # Computes upper triangular part only
        C_DD = empirical_cross_covariance(Y, Y)

        if self.inversion == "naive":
            K = inversion_naive(
                C_DD=C_DD, alpha=self.alpha[self.iteration], C_D=self.C_D, D=D, Y=Y
            )
        elif self.inversion == "exact":
            K = inversion_exact(
                C_DD=C_DD, alpha=self.alpha[self.iteration], C_D=self.C_D, D=D, Y=Y
            )

        # Compute K := sp.linalg.inv(C_DD + alpha * C_D) @ (D - Y)
        if self.C_D.ndim == 2:
            # C_D is a covariance matrix
            K = sp.linalg.solve(
                C_DD + self.alpha[self.iteration] * self.C_D,
                D - Y,
                lower=False,  # Only the upper triangular part is used
                overwrite_a=True,
                overwrite_b=True,
                assume_a="pos",
            )
        elif self.C_D.ndim == 1:
            # C_D is an array, so add it to the diagonal without forming diag(C_D)
            C_DD.flat[:: num_outputs + 1] += self.alpha[self.iteration] * self.C_D
            K = sp.linalg.solve(
                C_DD,
                D - Y,
                lower=False,  # Only the upper triangular part is used
                overwrite_a=True,
                overwrite_b=True,
                assume_a="pos",
            )

        # X_posterior = X_current + C_MD @ sp.linalg.inv(C_DD + C_D_alpha) @ (D - Y)
        X_posterior = X_current + C_MD @ K

        self.iteration += 1
        return X_posterior


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    np.random.seed(12)

    # Dimensionality
    num_ensemble = 9
    num_outputs = 2
    num_iputs = 1

    def g(x):
        """Transform a single ensemble member."""
        return np.array([np.sin(x / 2), x]) + 5 + np.random.randn(2, 1) * 0.05

    def G(X):
        """Transform all ensemble members."""
        return np.array([g(x_i) for x_i in X.T]).squeeze().T

    # Prior is N(0, 1)
    X_prior = np.random.randn(num_iputs, num_ensemble) * 1

    # Measurement errors
    C_D = np.eye(num_outputs) * 10
    C_D = np.ones(num_outputs) * 0
    C_D[0] = 0

    # The true inputs and observationservations, a result of running with N(1, 1)
    X_true = np.random.randn(num_iputs, num_ensemble) + 3
    observations = G(X_true)

    # Create ESMDA instance
    esmda = ESMDA(C_D, observations, alpha=5, seed=123)

    X_current = np.copy(X_prior)
    for iteration in range(esmda.num_assimilations()):
        print(f"Iteration number: {iteration + 1}")

        X_posterior = esmda.assimilate(X_current, G(X_current))

        X_current = X_posterior

        # Plot results
        plt.hist(X_prior.ravel(), alpha=0.5, label="prior")
        plt.hist(X_true.ravel(), alpha=0.5, label="true inputs")
        plt.hist(X_current.ravel(), alpha=0.5, label="posterior")
        plt.legend()
        plt.show()

        plt.scatter(*G(X_prior), alpha=0.5, label="G(prior)")
        plt.scatter(*G(X_true), alpha=0.5, label="G(true inputs)")
        plt.scatter(*G(X_current), alpha=0.5, label="G(posterior)")
        plt.legend()
        plt.show()

        time.sleep(0.05)


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-v"])
