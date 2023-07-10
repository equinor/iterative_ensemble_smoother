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

from esmda_inversion import (
    normalize_alpha,
    empirical_cross_covariance,
    inversion_exact,
)


class ESMDA:
    _inversion_methods = {"exact": inversion_exact}

    def __init__(self, C_D, observations, alpha=5, seed=None, inversion="exact"):
        """Initialize Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

        The implementation follows the 2012 paper by Emerick et al.

        Parameters
        ----------
        C_D : np.ndarray
            Covariance matrix of outputs of shape (num_outputs, num_ouputs).
        observations : np.ndarray
            2D array of shape (num_inputs, num_ensemble_members).
        alpha : int or 1D np.ndarray, optional
            If an integer `alpha` is given, an array with length `alpha` and
            elements `alpha` is constructed. If an 1D array is given, it is
            normalized so sum_i 1/alpha_i = 1 and used. The default is 5, which
            corresponds to the array numpy.array([5, 5, 5, 5, 5]).
        seed : integer, optional
            A seed used for random number generation. The seed is passed to
            numpy.random.default_rng(). The default is None.
        inversion : str, optional
            Which inversion method to use. The default is "exact".

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        assert inversion in ("exact",)

        self.observations = observations
        self.iteration = 0
        self.rng = np.random.default_rng(seed)
        self.inversion = inversion

        # Alpha can either be a number (of iterations) or a list of weights
        if isinstance(alpha, np.ndarray) and alpha.ndim == 1:
            self.alpha = normalize_alpha(alpha)
        elif isinstance(alpha, int):
            self.alpha = np.array([alpha] * alpha)
            assert np.allclose(self.alpha, normalize_alpha(self.alpha))
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
        """Assimilate data and return an updated ensemble.

        Parameters
        ----------
        X : np.ndarray
            2D array of shape (num_inputs, num_ensemble_members).
        Y : np.ndarray
            2D array of shape (num_ouputs, num_ensemble_members).

        Returns
        -------
        X_posterior : np.ndarray
            2D array of shape (num_inputs, num_ensemble_members).

        """
        if self.iteration >= self.num_assimilations():
            raise Exception("No more assimilation steps to run.")

        num_inputs, num_ensemble = X.shape
        num_outputs, num_emsemble2 = Y.shape
        assert (
            num_ensemble == num_emsemble2
        ), "Number of ensemble members in X and Y must match"

        # Sample from a zero-centered multivariate normal with cov=C_D
        mv_normal_rvs = self.mv_normal.rvs(size=num_ensemble, random_state=self.rng).T

        # Create perturbed observationservations, with C_D scaled by alpha
        # If C_D = L L.T  by the cholesky factorization, then
        # drawing from a zero cented normal is y := L @ z, where z ~ norm(0, 1)
        # Scaling C_D by alpha is equivalent to scaling L with sqrt(alpha)
        D = self.observations + np.sqrt(self.alpha[self.iteration]) * mv_normal_rvs

        # Update the ensemble
        C_MD = empirical_cross_covariance(X, Y)

        if self.inversion == "exact":
            K = inversion_exact(
                alpha=self.alpha[self.iteration], C_D=self.C_D, D=D, Y=Y
            )

        # X_posterior = X_current + C_MD @ K
        # K := sp.linalg.inv(C_DD + C_D_alpha) @ (D - Y)

        # In the typical case where num_outputs >> num_inputs >> ensemble members,
        # multiplying in the order below from the right to the left, i.e.,
        # C_MD @ (inv(C_DD + alpha * C_D) @ (D - Y))
        # is faster than the alternative order
        # (C_MD @ inv(C_DD + alpha * C_D)) @ (D - Y)
        X_posterior = X + C_MD @ K

        self.iteration += 1
        return X_posterior


# =============================================================================
# TESTS
# =============================================================================
import pytest


class TestESMDA:
    @pytest.mark.parametrize(
        "num_ensemble",
        [10, 100, 1000],
    )
    def test_that_alpha_as_integer_and_array_returns_same_result(self, num_ensemble):
        seed = num_ensemble
        np.random.seed(seed)

        num_outputs = 2
        num_iputs = 1

        def g(x):
            """Transform a single ensemble member."""
            return np.array([np.sin(x / 2), x]) + 5

        def G(X):
            """Transform all ensemble members."""
            return np.array([g(x_i) for x_i in X.T]).squeeze().T

        # Prior is N(0, 1)
        X_prior = np.random.randn(num_iputs, num_ensemble)

        # Measurement errors
        C_D = np.eye(num_outputs)

        # The true inputs and observationservations, a result of running with N(1, 1)
        X_true = np.random.randn(num_iputs, num_ensemble) + 1
        observations = G(X_true)

        # Create ESMDA instance and run it
        esmda_integer = ESMDA(C_D, observations, alpha=5, seed=seed)
        X_i_int = np.copy(X_prior)
        for _ in range(esmda_integer.num_assimilations()):
            X_i_int = esmda_integer.assimilate(X_i_int, G(X_i_int))

        # Create another ESMDA instance and run it
        esmda_array = ESMDA(C_D, observations, alpha=np.ones(5), seed=seed)
        X_i_array = np.copy(X_prior)
        for _ in range(esmda_array.num_assimilations()):
            X_i_array = esmda_array.assimilate(X_i_array, G(X_i_array))

        assert np.allclose(X_i_int, X_i_array)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    # =============================================================================
    # RUN AN EXAMPLE
    # =============================================================================

    np.random.seed(12)

    # Dimensionality
    num_ensemble = 999
    num_outputs = 2
    num_iputs = 1

    def g(x):
        """Transform a single ensemble member."""
        # return np.array([x, x]) + 5 + np.random.randn(2, 1) * 0.05
        return np.array([np.sin(x / 2), x]) + 5 + np.random.randn(2, 1) * 0.05

    def G(X):
        """Transform all ensemble members."""
        return np.array([g(x_i) for x_i in X.T]).squeeze().T

    # Prior is N(0, 1)
    X_prior = np.random.randn(num_iputs, num_ensemble) * 1

    # Measurement errors
    C_D = np.eye(num_outputs) * 1

    # The true inputs and observationservations, a result of running with N(1, 1)
    X_true = np.random.randn(num_iputs, num_ensemble) + 3
    observations = G(X_true)

    # Create ESMDA instance
    esmda = ESMDA(C_D, observations, alpha=10, seed=123)

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

        time.sleep(0.0005)


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
            "-v",
            "-k test_that_inversion_methods_work_with_covariance_matrix_and_variance_vector",
        ]
    )
