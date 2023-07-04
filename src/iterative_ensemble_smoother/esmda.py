#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble Smoother with Multiple Data Assimilation (ES-MDA)
----------------------------------------------------------



References
----------

Emerick, A.A., Reynolds, A.C. History matching time-lapse seismic data using
the ensemble Kalman filter with multiple data assimilations.
Comput Geosci 16, 639–659 (2012). https://doi.org/10.1007/s10596-012-9275-5

Alexandre A. Emerick, Albert C. Reynolds.
Ensemble smoother with multiple data assimilation.
Computers & Geosciences, Volume 55, 2013, Pages 3-15, ISSN 0098-3004,
https://doi.org/10.1016/j.cageo.2012.03.011.

https://gitlab.com/antoinecollet5/pyesmda

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
    C_D = np.diag(C_D) if C_D.ndim == 1 else C_D
    return sp.linalg.inv(C_DD + alpha * C_D) @ (D - Y)


def inversion_exact(*, C_DD, alpha, C_D, D, Y):
    solver_kwargs = {"overwrite_a": True, "overwrite_b": True, "assume_a": "pos"}

    # Compute K := sp.linalg.inv(C_DD + alpha * C_D) @ (D - Y)
    if C_D.ndim == 2:
        # C_D is a covariance matrix
        K = sp.linalg.solve(C_DD + alpha * C_D, D - Y, **solver_kwargs)
    elif C_D.ndim == 1:
        # C_D is an array, so add it to the diagonal without forming diag(C_D)
        C_DD.flat[:: C_DD.shape[1] + 1] += alpha * C_D
        K = sp.linalg.solve(C_DD, D - Y, **solver_kwargs)
    return K


def inversion_lstsq(*, C_DD, alpha, C_D, D, Y):

    # A covariance matrix was given
    if C_D.ndim == 2:
        lhs = C_DD + alpha * C_D
    # A diagonal covariance matrix was given as a vector
    else:
        lhs = C_DD
        lhs.flat[:: lhs.shape[0] + 1] += alpha * C_D
    ans, *_ = sp.linalg.lstsq(lhs, D - Y, overwrite_a=True, overwrite_b=True)
    return ans


def inversion_rescaled(*, C_DD, alpha, C_D, D, Y):
    """See Appendix A.1 in Emerick et al (2012)"""

    if C_D.ndim == 2:

        # Eqn (57). Cholesky factorize the covariance matrix C_D
        C_D_L = sp.linalg.cholesky(C_D * alpha, lower=True)  # Lower triangular cholesky
        C_D_L_inv, _ = sp.linalg.lapack.dtrtri(
            C_D_L, lower=1
        )  # Invert lower triangular

        # Eqn (59). Form C_tilde
        # sp.linalg.blas.strmm(alpha=1, a=C_D_L_inv, b=C_DD, lower=1)
        C_tilde = C_D_L_inv @ C_DD @ C_D_L_inv.T
        C_tilde.flat[:: C_tilde.shape[0] + 1] += 1  # Add to diagonal

    # When C_D is a diagonal covariance matrix, there is no need to perform
    # the cholesky factorization
    elif C_D.ndim == 1:
        C_D_L_inv = 1 / np.sqrt(C_D * alpha)
        C_tilde = (C_D_L_inv * (C_DD * C_D_L_inv).T).T
        C_tilde.flat[:: C_tilde.shape[0] + 1] += 1  # Add to diagonal

    # Eqn (60). Compute SVD (we could do eigenvalues, since C_tilde is
    # symmetric, but it turns out that computing the SVD is faster)
    # On a 1000 x 1000 example, 80% of the time is spent here
    U, s, _ = sp.linalg.svd(C_tilde, overwrite_a=True, full_matrices=False)
    # TODO: truncate SVD

    # Eqn (61). Compute symmetric term once first, then multiply together and
    # finally multiply with (D - Y)
    term = C_D_L_inv.T @ U if C_D.ndim == 2 else (C_D_L_inv * U.T).T
    return np.linalg.multi_dot([term / s, term.T, (D - Y)])


def inversion_subspace(*, alpha, C_D, D, Y):
    """See Appendix A.2 in Emerick et al (2012)"""

    # N_n is the number of observations
    # N_e is the number of members in the ensemble
    N_n, N_e = Y.shape

    # Subtract the mean of every observation, see Eqn (67)
    D_delta = Y - np.mean(Y, axis=1, keepdims=True)  # Subtract average

    # Eqn (68)
    # TODO: Approximately 50% of the time in the function is spent here
    # consider using randomized svd for further speed gains
    U, w, _ = sp.linalg.svd(D_delta, overwrite_a=True, full_matrices=False)

    # Clip the singular value decomposition
    N_r = min(N_n, N_e - 1)  # Number of values in SVD to keep
    U_r, w_r = U[:, :N_r], w[:N_r]

    # Eqn (70). First compute the symmetric term, then form X
    U_r_w_inv = U_r / w_r
    X = (N_e - 1) * np.linalg.multi_dot([U_r_w_inv.T, alpha * C_D, U_r_w_inv])

    # Eqn (72)
    Z, T, _ = sp.linalg.svd(X, overwrite_a=True, full_matrices=False)

    # Eqn (74).
    # C^+ = (N_e - 1) hat{C}^+
    #     = (N_e - 1) (U / w @ Z) * (1 / (1 + T)) (U / w @ Z)^T
    #     = (N_e - 1) (term) * (1 / (1 + T)) (term)^T
    # and finally we multiiply by (D - Y)
    term = U_r_w_inv @ Z
    return (N_e - 1) * np.linalg.multi_dot([(term / (1 + T)), term.T, (D - Y)])


def inversion_rescaled_subspace(*, alpha, C_D, D, Y):
    """See Appendix A.2 in Emerick et al (2012)"""
    pass


def test_that_all_inversions_all_equal_with_many_ensemble_members(k=10):
    emsemble_members = k + 1

    # Create positive symmetric definite covariance C_D
    E = np.random.randn(k, k)
    C_D = E.T @ E

    # Set alpha to something other than 1 to test that it works
    alpha = 3

    # Create observations
    D = np.random.randn(k, emsemble_members)
    Y = np.random.randn(k, emsemble_members)

    # Compute covariance
    C_DD = empirical_cross_covariance(Y, Y)

    K1 = inversion_naive(C_DD=C_DD, alpha=alpha, C_D=C_D, D=D, Y=Y)
    K2 = inversion_exact(C_DD=C_DD, alpha=alpha, C_D=C_D, D=D, Y=Y)
    K3 = inversion_rescaled(C_DD=C_DD, alpha=alpha, C_D=C_D, D=D, Y=Y)
    K4 = inversion_lstsq(C_DD=C_DD, alpha=alpha, C_D=C_D, D=D, Y=Y)

    # Subspace methods will give the same result as long as ensemble_members > num_outputs
    K5 = inversion_subspace(alpha=alpha, C_D=C_D, D=D, Y=Y)

    assert np.allclose(K1, K2)
    assert np.allclose(K1, K3)
    assert np.allclose(K1, K4)
    assert np.allclose(K1, K5)


def test_that_inversion_methods_work_with_covariance_matrix_and_variance_vector(k=10):
    emsemble_members = k - 1

    E = np.random.randn(k, k)
    C_D = E.T @ E
    C_D = np.diag(np.exp(np.random.randn(k)))  # Diagonal covariance matrix

    # Set alpha to something other than 1 to test that it works
    alpha = 3

    # Create observations
    D = np.random.randn(k, emsemble_members)
    Y = np.random.randn(k, emsemble_members)

    # Compute covariance
    C_DD = empirical_cross_covariance(Y, Y)

    exact_inversion_funcs = [
        inversion_naive,
        inversion_exact,
        inversion_rescaled,
        inversion_lstsq,
    ]

    for func in exact_inversion_funcs:
        result_matrix = func(C_DD=C_DD, alpha=alpha, C_D=C_D, D=D, Y=Y)
        result_vector = func(C_DD=C_DD, alpha=alpha, C_D=np.diag(C_D), D=D, Y=Y)
        assert np.allclose(result_matrix, result_vector)


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
