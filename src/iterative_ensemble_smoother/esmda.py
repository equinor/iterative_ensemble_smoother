#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


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
    num_variables, num_observations = X.shape
    X = X - np.mean(X, axis=1, keepdims=True)
    XXT = sp.linalg.blas.dsyrk(alpha=1, a=X)
    XXT /= num_observations - 1
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


class ESMDA:
    def __init__(self, C_D, obs, alpha=None, seed=None):
        self.obs = obs
        self.iteration = 0
        self.rng = np.random.default_rng(seed)

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
        num_outputs, num_ensemble = obs.shape

        if isinstance(C_D, np.ndarray) and C_D.ndim == 2:
            L = sp.linalg.cholesky(C_D, lower=False)
            cov = sp.stats.Covariance.from_cholesky(L)
        elif isinstance(C_D, np.ndarray) and C_D.ndim == 1:
            assert len(C_D) == num_outputs
            cov = sp.stats.Covariance.from_diagonal(C_D)
        elif isinstance(C_D, float):
            cov = sp.stats.Covariance.from_diagonal(np.array([C_D] * num_outputs))

        mean = np.zeros(num_outputs)
        self.mv_normal = sp.stats.multivariate_normal(mean=mean, cov=cov)
        self.C_D = C_D

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

        # Inflate the covariance
        # C_D_alpha = self.C_D * self.alpha[self.iteration]

        # Perturb the observation vector
        mv_normal_rvs = self.mv_normal.rvs(size=num_ensemble, random_state=self.rng).T
        D = self.obs + self.alpha[self.iteration] * mv_normal_rvs

        # Update the ensemble
        C_MD = empirical_cross_covariance(X, Y)
        C_DD = empirical_covariance_upper(Y)  # Computes upper triangular part only

        # Compute K := sp.linalg.inv(C_DD + C_D_alpha) @ (D - Y)
        if self.C_D.ndim == 2:
            K = sp.linalg.solve(
                C_DD + self.C_D * self.alpha[self.iteration],
                D - Y,
                lower=False,
                overwrite_a=True,
                overwrite_b=True,
                check_finite=True,
                assume_a="pos",
            )
        elif self.C_D.ndim == 1:
            C_DD.flat[:: num_outputs + 1] += self.C_D * self.alpha[self.iteration]
            K = sp.linalg.solve(
                C_DD,
                D - Y,
                lower=False,
                overwrite_a=True,
                overwrite_b=True,
                check_finite=True,
                assume_a="pos",
            )
        # X_posterior = X_current + C_MD @ sp.linalg.inv(C_DD + C_D_alpha) @ (D - Y)
        X_posterior = X_current + C_MD @ K

        self.iteration += 1
        return X_posterior


if __name__ == "__main__":
    import time

    np.random.seed(12)

    # Dimensionality
    num_ensemble = 999
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
    C_D = np.ones(num_outputs) * 10

    # The true inputs and observations, a result of running with N(1, 1)
    X_true = np.random.randn(num_iputs, num_ensemble) + 3
    obs = G(X_true)

    # Create ESMDA instance
    esmda = ESMDA(C_D, obs, alpha=5, seed=123)

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
