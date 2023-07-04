#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def empirical_covariance(X, Y):
    """Both X and Y have shape (parameters, ensemble_size)

    Examples
    --------
    >>> X = np.arange(9).reshape(3, 3)**0.5
    >>> Y = np.arange(15).reshape(5, 3)**2
    >>> empirical_covariance(X, Y)
    array([[ 1.31658249,  5.55922318,  9.80186386, 14.04450455, 18.28714524],
           [ 0.49870363,  2.01075514,  3.52280665,  5.03485816,  6.54690967],
           [ 0.37667309,  1.51348524,  2.65029738,  3.78710953,  4.92392167]])
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


np.random.seed(12)

# Number of data assimilations
N_a = 5
alpha = normalize_alpha(np.arange(N_a)[::-1] ** 1 + 1)
# alpha = np.array([1])

num_ensemble = 999
num_outputs = 2
num_iputs = 1
M = np.array([[2], [1]])


def g(x):
    return np.array([np.sin(x / 2), x]) + 5 + np.random.randn(2, 1) * 0.05


def G(X):
    return np.array([g(x_i) for x_i in X.T]).squeeze().T

    Y = M @ X
    return Y + np.ones_like(Y) * 10


# Prior is N(0, 1)
X_prior = np.random.randn(num_iputs, num_ensemble) * 1

# Measurement errors
C_D = np.eye(num_outputs) * 0.5

# The true output, a result of running with N(1, 1)
X_true = np.random.randn(num_iputs, num_ensemble) + 3
obs = G(X_true)


X_current = np.copy(X_prior)


class ESMDA:
    def __init__(self, C_D, obs, alpha=None, seed=None):
        self.C_D = C_D
        self.obs = obs
        self.iteration = 0
        self.rng = np.random.default_rng(seed)

        if isinstance(alpha, np.ndarray) and alpha.dim == 1:
            self.alpha = normalize_alpha(alpha)
        elif isinstance(alpha, int):
            self.alpha = np.array([1 / alpha] * alpha)
        else:
            raise TypeError("Alpha must be integer or 1D array.")

    def num_assimilations(self):
        return len(self.alpha)

    def assimilate(self, X, Y):

        num_inputs, num_ensemble1 = X.shape
        num_outputs, num_emsemble2 = Y.shape
        assert (
            num_ensemble1 == num_emsemble2
        ), "Number of ensemble members in X and Y must match"

        # Inflate the covariance
        C_D_alpha = C_D * self.alpha[self.iteration]

        # Perturb the observation vector
        D = (
            self.obs
            + self.rng.multivariate_normal(
                np.zeros(num_outputs), C_D_alpha, size=num_ensemble
            ).T
        )

        # Update the ensemble
        C_MD = empirical_covariance(X, Y)
        C_DD = empirical_covariance(Y, Y)

        # X_posterior = X_current + C_MD @ sp.linalg.inv(C_DD + C_D_alpha) @ (D - Y)
        X_posterior = X_current + C_MD @ sp.linalg.solve(
            C_DD + C_D_alpha,
            D - Y,
            lower=False,
            overwrite_a=True,
            overwrite_b=True,
            check_finite=True,
            assume_a="pos",
        )

        return X_posterior


esmda = ESMDA(C_D, obs, alpha=5, seed=123)


X_current = np.copy(X_prior)
for iteration in range(esmda.num_assimilations()):

    print(f"Iteration number: {iteration + 1}")

    X_posterior = esmda.assimilate(X_current, G(X_current))

    X_current = X_posterior

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
    import time

    time.sleep(0.05)


for i, alpha_i in enumerate(alpha, 1):
    print(f"Iteration number: {i} Alpha: {alpha_i}")

    # Inflate the covariance
    C_D_alpha = C_D * alpha_i

    # Run ensemble
    Y = G(X_current)  # Y is the oservation vector

    # Perturb the observation vector
    D = (
        obs
        + np.random.multivariate_normal(
            np.zeros(num_outputs), C_D_alpha, size=num_ensemble
        ).T
    )

    # Update the ensemble
    C_MD = empirical_covariance(X_current, Y)
    C_DD = empirical_covariance(Y, Y)
    C_D_alpha = C_D * alpha_i

    # print(f"det(C_MD) = {sp.linalg.det(C_MD)}")
    print(f"det(C_DD) = {sp.linalg.det(C_DD)}")
    print(f"det(C_D) = {sp.linalg.det(C_D)}")

    # X_posterior = X_current + C_MD @ sp.linalg.inv(C_DD + C_D_alpha) @ (D - Y)
    X_posterior = X_current + C_MD @ sp.linalg.solve(
        C_DD + C_D_alpha,
        D - Y,
        lower=False,
        overwrite_a=True,
        overwrite_b=True,
        check_finite=True,
        assume_a="pos",
    )

    print((C_DD + C_D_alpha).shape)
    print(D.shape, Y.shape, (D - Y).shape)
    # X_posterior = X_current + C_MD @ sp.linalg.lstsq(C_DD + C_D_alpha, D - Y, overwrite_a=True, overwrite_b=True)

    # X_posterior = X_current + C_MD @ sp.linalg.inv(C_DD + C_D_alpha) @ (D - Y)

    X_current = X_posterior

    print(np.mean(X_prior.ravel()))

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
    import time

    time.sleep(0.05)


# =============================================================================
#
# def objective(X):
#     prior = (X - X_prior).T @ np.linalg.inv(C_M) @ (X - X_prior)
#     likelihood = (G(X) - D).T @ np.linalg.inv(C_D) @ (G(X) - D)
#
#     return np.diag(prior + likelihood)
#
# print("objective at prior", objective(X_prior))
#
#
# =============================================================================


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    # pytest.main(args=[__file__, "--doctest-modules", "-v", "-v"])
