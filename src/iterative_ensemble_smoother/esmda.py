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

import numbers

from typing import Optional, Union

import numpy as np

import numpy.typing as npt


import scipy as sp  # type: ignore

from esmda_inversion import (  # type: ignore
    normalize_alpha,
    empirical_cross_covariance,
    inversion_exact,
)


class ESMDA:
    _inversion_methods = {"exact": inversion_exact}

    def __init__(
        self,
        C_D: npt.NDArray[np.double],
        observations: npt.NDArray[np.double],
        alpha: Union[int, npt.NDArray[np.double]] = 5,
        seed: Optional[int] = None,
        inversion: str = "exact",
    ) -> None:
        """Initialize Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

        The implementation follows the 2012 paper by Emerick et al.

        Parameters
        ----------
        C_D : np.ndarray
            Covariance matrix of outputs of shape (num_outputs, num_ouputs).
            If a 1D array is passed, it represents a diagonal covariance matrix.
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

        Returns
        -------
        None.

        """
        # Validate inputs
        if not (isinstance(C_D, np.ndarray) and C_D.ndim in (1, 2)):
            raise TypeError("Argument `C_D` must be a NumPy array of dimension 1 or 2.")
            if C_D.ndim == 2 and C_D.shape[0] != C_D.shape[1]:
                raise ValueError("Argument `C_D` must be square if it's 2D.")

        if not (isinstance(observations, np.ndarray) and observations.ndim == 2):
            raise TypeError("Argument `observations` must be a NumPy array of 2.")

        if not observations.shape[0] == C_D.shape[0]:
            raise ValueError("Shapes of `observations` and `C_D` must match.")

        if not (
            (isinstance(alpha, np.ndarray) and alpha.ndim == 1)
            or isinstance(alpha, numbers.Integral)
        ):
            raise TypeError("Argument `alpha` must be an integer or a 1D NumPy array.")

        if not (isinstance(seed, int) or seed is None):
            raise TypeError("Argument `seed` must be an integer.")

        if not isinstance(inversion, str):
            raise TypeError(
                f"Argument `inversion` must be a string in {tuple(self._inversion_methods.keys())}"
            )
        if inversion not in self._inversion_methods.keys():
            raise ValueError(
                f"Argument `inversion` must be a string in {tuple(self._inversion_methods.keys())}"
            )

        # Store data
        self.observations = observations
        self.iteration = 0
        self.rng = np.random.default_rng(seed)
        self.inversion = inversion

        # Alpha can either be a number (of iterations) or a list of weights
        if isinstance(alpha, np.ndarray) and alpha.ndim == 1:
            self.alpha = normalize_alpha(alpha)
        elif isinstance(alpha, numbers.Integral):
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

    def num_assimilations(self) -> int:
        return len(self.alpha)

    def assimilate(
        self,
        X: npt.NDArray[np.double],
        Y: npt.NDArray[np.double],
        ensemble_mask: Optional[npt.NDArray[np.bool_]] = None,
    ) -> npt.NDArray[np.double]:
        """Assimilate data and return an updated ensemble.

        Parameters
        ----------
        X : np.ndarray
            2D array of shape (num_inputs, num_ensemble_members).
        Y : np.ndarray
            2D array of shape (num_ouputs, num_ensemble_members).
        ensemble_mask : np.ndarray
            1D boolean array of length `num_ensemble_members`, describing which
            ensemble members are active. Inactive realizations are ignored.
            Defaults to all active.

        Returns
        -------
        X_posterior : np.ndarray
            2D array of shape (num_inputs, num_ensemble_members).

        """
        if self.iteration >= self.num_assimilations():
            raise Exception("No more assimilation steps to run.")

        # Verify shapes
        num_inputs, num_ensemble = X.shape
        num_outputs, num_emsemble2 = Y.shape
        assert (
            num_ensemble == num_emsemble2
        ), "Number of ensemble members in X and Y must match"
        assert (ensemble_mask is None) or (
            ensemble_mask.ndim == 1 and len(ensemble_mask) == num_ensemble
        )

        if ensemble_mask is None:
            ensemble_mask = np.ones(num_ensemble, dtype=bool)

        # Sample from a zero-centered multivariate normal with cov=C_D
        mv_normal_rvs = self.mv_normal.rvs(
            size=ensemble_mask.sum(), random_state=self.rng
        ).T

        # Line 2 (b) in the description of ES-MDA in the 2013 Emerick paper
        # Create perturbed observationservations, with C_D scaled by alpha
        # If C_D = L L.T  by the cholesky factorization, then
        # drawing from a zero cented normal is y := L @ z, where z ~ norm(0, 1)
        # Scaling C_D by alpha is equivalent to scaling L with sqrt(alpha)
        D = (
            self.observations[:, ensemble_mask]
            + np.sqrt(self.alpha[self.iteration]) * mv_normal_rvs
        )

        # Line 2 (c) in the description of ES-MDA in the 2013 Emerick paper
        # Compute the cross covariance
        C_MD = empirical_cross_covariance(X[:, ensemble_mask], Y[:, ensemble_mask])

        # Choose inversion method, e.g. 'exact'
        inversion_func = self._inversion_methods[self.inversion]
        K = inversion_func(
            alpha=self.alpha[self.iteration], C_D=self.C_D, D=D, Y=Y[:, ensemble_mask]
        )

        # X_posterior = X_current + C_MD @ K
        # K := sp.linalg.inv(C_DD + C_D_alpha) @ (D - Y)

        # In the typical case where num_outputs >> num_inputs >> ensemble members,
        # multiplying in the order below from the right to the left, i.e.,
        # C_MD @ (inv(C_DD + alpha * C_D) @ (D - Y))
        # is faster than the alternative order
        # (C_MD @ inv(C_DD + alpha * C_D)) @ (D - Y)
        X_posterior = np.copy(X)
        X_posterior[:, ensemble_mask] += C_MD @ K

        # X_posterior = X + C_MD @ K

        self.iteration += 1
        return X_posterior


# =============================================================================
# TESTS
# =============================================================================
import pytest


class TestESMDA:
    @pytest.mark.parametrize("num_ensemble", [2**i for i in range(2, 11)])
    def test_that_using_example_mask_only_updates_those_parameters(self, num_ensemble):
        seed = num_ensemble
        rng = np.random.default_rng(seed)
        alpha = rng.choice(np.array([5, 10, 25, 50]))

        num_outputs = 2
        num_iputs = 1

        def g(x):
            """Transform a single ensemble member."""
            return np.array([np.sin(x / 2), x])

        def G(X):
            """Transform all ensemble members."""
            return np.array([g(x_i) for x_i in X.T]).squeeze().T

        # Create an ensemble mask and set half the entries randomly to True
        ensemble_mask = np.zeros(num_ensemble, dtype=bool)
        ensemble_mask[
            rng.choice(num_ensemble, size=num_ensemble // 2, replace=False)
        ] = True

        # Prior is N(0, 1)
        X_prior = rng.normal(size=(num_iputs, num_ensemble))

        # Measurement errors
        C_D = np.eye(num_outputs)

        # The true inputs and observationservations, a result of running with N(1, 1)
        X_true = rng.normal(loc=1, size=(num_iputs, num_ensemble))
        observations = G(X_true)

        # Prepare ESMDA instance running with lower number of ensemble members
        esmda_subset = ESMDA(
            C_D, observations[:, ensemble_mask], alpha=alpha, seed=seed
        )
        X_i_subset = np.copy(X_prior[:, ensemble_mask])

        # Prepare ESMDA instance running with all ensemble members
        esmda_masked = ESMDA(C_D, observations, alpha=alpha, seed=seed)
        X_i = np.copy(X_prior)

        # Run both

        for _ in range(esmda_subset.num_assimilations()):
            X_i_subset = esmda_subset.assimilate(X_i_subset, G(X_i_subset))
            X_i = esmda_masked.assimilate(X_i, G(X_i), ensemble_mask=ensemble_mask)

            assert np.allclose(X_i_subset, X_i[:, ensemble_mask])

    @pytest.mark.parametrize(
        "num_ensemble",
        [10, 100, 1000],
    )
    def test_that_alpha_as_integer_and_array_returns_same_result(self, num_ensemble):
        seed = num_ensemble
        rng = np.random.default_rng(seed)

        num_outputs = 2
        num_iputs = 1

        def g(x):
            """Transform a single ensemble member."""
            return np.array([np.sin(x / 2), x]) + 5

        def G(X):
            """Transform all ensemble members."""
            return np.array([g(x_i) for x_i in X.T]).squeeze().T

        # Prior is N(0, 1)
        X_prior = rng.normal(size=(num_iputs, num_ensemble))

        # Measurement errors
        C_D = np.eye(num_outputs)

        # The true inputs and observationservations, a result of running with N(1, 1)
        X_true = rng.normal(size=(num_iputs, num_ensemble)) + 1
        observations = G(X_true)

        # Create ESMDA instance from an integer `alpha` and run it
        esmda_integer = ESMDA(C_D, observations, alpha=5, seed=seed)
        X_i_int = np.copy(X_prior)
        for _ in range(esmda_integer.num_assimilations()):
            X_i_int = esmda_integer.assimilate(X_i_int, G(X_i_int))

        # Create another ESMDA instance from a vector `alpha` and run it
        esmda_array = ESMDA(C_D, observations, alpha=np.ones(5), seed=seed)
        X_i_array = np.copy(X_prior)
        for _ in range(esmda_array.num_assimilations()):
            X_i_array = esmda_array.assimilate(X_i_array, G(X_i_array))

        # Exactly the same result with equal seeds
        assert np.allclose(X_i_int, X_i_array)


if __name__ == "__main__" and False:
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
        return np.array([np.sin(x / 2), x]) + 5 + np.random.randn(2, 1) * 0.1

    def G(X):
        """Transform all ensemble members."""
        return np.array([g(x_i) for x_i in X.T]).squeeze().T

    # Prior is N(0, 1)
    X_prior = np.random.randn(num_iputs, num_ensemble) * 1

    # Measurement errors
    C_D = np.eye(num_outputs) * 1

    # The true inputs and observationservations, a result of running with N(1, 1)
    X_true = np.random.randn(num_iputs, num_ensemble) + 6
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

        time.sleep(0.05)


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
            "-v",
            # "-k test_that_inversion_methods_work_with_covariance_matrix_and_variance_vector",
        ]
    )
