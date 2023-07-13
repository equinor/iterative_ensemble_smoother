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
from iterative_ensemble_smoother.esmda import ESMDA
import pytest


class TestESMDA:
    @pytest.mark.parametrize("num_ensemble", [2**i for i in range(2, 10)])
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


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "-v",
            # "--durations=10",
        ]
    )
