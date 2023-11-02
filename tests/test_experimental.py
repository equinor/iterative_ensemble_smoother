from copy import deepcopy

import numpy as np
import pytest

from iterative_ensemble_smoother import ESMDA
from iterative_ensemble_smoother.experimental import (
    ensemble_smoother_update_step_row_scaling,
)


class TestRowScaling:
    def test_that_row_scaling_updates_parameters(self):
        class RowScaling:
            # Illustration of how row scaling works, `multiply` is the important part
            # For the actual implementation, which is more involved, see:
            # https://github.com/equinor/ert/blob/84aad3c56e0e52be27b9966322d93dbb85024c1c/src/clib/lib/enkf/row_scaling.cpp
            def __init__(self, alpha=1.0):
                """Alpha is the strength of the update."""
                assert 0 <= alpha <= 1.0
                self.alpha = alpha

            def multiply(self, X, K):
                """Takes a matrix X and a matrix K and performs alpha * X @ K."""
                # This implementation merely mimics how RowScaling::multiply behaves
                # in the C++ code. It mutates the input argument X instead of returning.
                X[:, :] = X @ (K * self.alpha)

        # Example showing how to use row scaling
        num_parameters = 100
        num_observations = 20
        num_ensemble = 10

        rng = np.random.default_rng(42)

        X = rng.normal(size=(num_parameters, num_ensemble))
        Y = rng.normal(size=(num_observations, num_ensemble))
        covariance = np.exp(rng.normal(size=num_observations))
        observations = rng.normal(size=num_observations, loc=1)

        # Split up X into groups of parameters as needed
        row_groups = [(0,), (1, 2), (4, 5, 6), tuple(range(7, 100))]
        X_with_row_scaling = [
            (X[idx, :], RowScaling(alpha=1 / (i + 1)))
            for i, idx in enumerate(row_groups)
        ]
        # Make a copy so we can check that update happened, since input is mutated
        X_before = deepcopy(X_with_row_scaling)

        X_with_row_scaling_updated = ensemble_smoother_update_step_row_scaling(
            covariance=covariance,
            observations=observations,
            X_with_row_scaling=X_with_row_scaling,
            Y=Y,
            seed=rng,
        )

        # Check that an update happened
        assert not np.allclose(X_before[-1][0], X_with_row_scaling_updated[-1][0])

    @pytest.mark.parametrize("inversion", list(ESMDA._inversion_methods.keys()))
    @pytest.mark.parametrize("num_ensemble", [5, 25, 200])
    def test_that_row_scaling_equal_single_ES_update(self, inversion, num_ensemble):
        class RowScaling:
            # Illustration of how row scaling works, `multiply` is the important part
            # For the actual implementation, which is more involved, see:
            # https://github.com/equinor/ert/blob/84aad3c56e0e52be27b9966322d93dbb85024c1c/src/clib/lib/enkf/row_scaling.cpp
            def __init__(self, alpha=1.0):
                """Alpha is the strength of the update."""
                assert 0 <= alpha <= 1.0
                self.alpha = alpha

            def multiply(self, X, K):
                """Takes a matrix X and a matrix K and performs alpha * X @ K."""
                # This implementation merely mimics how RowScaling::multiply behaves
                # in the C++ code. It mutates the input argument X instead of returning.
                X[:, :] = X @ (K * self.alpha)

        # Create data
        num_parameters = 100
        num_observations = 20

        rng = np.random.default_rng(num_ensemble)

        X = rng.normal(size=(num_parameters, num_ensemble))
        Y = rng.normal(size=(num_observations, num_ensemble))
        covariance = np.exp(rng.normal(size=num_observations))
        observations = rng.normal(size=num_observations, loc=1)

        # Split up X into groups of parameters
        row_groups = [tuple(range(17)), tuple(range(17, 100))]
        # When alpha=1 in row scaling, we perform a full update
        X_with_row_scaling = [
            (X[idx, :], RowScaling(alpha=1)) for i, idx in enumerate(row_groups)
        ]

        # Perform an update using row scaling
        X_with_row_scaling_updated = ensemble_smoother_update_step_row_scaling(
            covariance=covariance,
            observations=observations,
            X_with_row_scaling=X_with_row_scaling,
            Y=Y,
            inversion=inversion,
            seed=1,
        )

        # Perform an update using ESMDA API
        smoother = ESMDA(
            covariance=covariance,
            observations=observations,
            alpha=1,
            inversion=inversion,
            seed=1,
        )
        for iteration in range(smoother.num_assimilations()):
            X_posterior = smoother.assimilate(X, Y)

        # The result should be the same
        assert np.allclose(
            X_posterior, np.vstack([X_i for (X_i, _) in X_with_row_scaling_updated])
        )


if __name__ == "__main__":

    pytest.main(
        args=[
            __file__,
            "-v",
        ]
    )
