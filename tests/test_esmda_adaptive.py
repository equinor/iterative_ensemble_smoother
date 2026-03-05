import functools
import time

import numpy as np
import pytest

from iterative_ensemble_smoother import ESMDA
from iterative_ensemble_smoother.esmda_adaptive import AdaptiveESMDA
from iterative_ensemble_smoother.esmda_inversion import (
    normalize_alpha,
)


@pytest.fixture
def linear_problem(request):
    # Seed the problem using indirect parametrization:
    # https://docs.pytest.org/en/latest/example/parametrize.html#indirect-parametrization
    rng = np.random.default_rng(request.param)

    # Create a problem with g(x) = A @ x
    num_parameters = 50
    num_observations = 10
    num_ensemble = 200

    A = np.exp(rng.standard_normal(size=(num_observations, num_parameters)))

    def g(X):
        """Forward model."""
        return A @ X

    # Create observations
    x_true = np.linspace(-1, 1, num=num_parameters)
    observations = g(x_true) + rng.standard_normal(size=num_observations) / 10

    # Initial ensemble and covariance
    X = rng.normal(size=(num_parameters, num_ensemble))
    covariance = rng.triangular(0.1, 1, 1, size=num_observations)
    return X, g, observations, covariance, rng


class TestAdaptiveESMDA:
    @pytest.mark.parametrize(
        "linear_problem",
        range(25),
        indirect=True,
        ids=[f"seed-{i + 1}" for i in range(25)],
    )
    def test_that_adaptive_localization_with_cutoff_1_equals_ensemble_prior(
        self, linear_problem
    ):
        # Create a problem with g(x) = A @ x
        X, g, observations, covariance, _ = linear_problem
        num_params = X.shape[0]
        num_observations = covariance.shape[0]

        # Create adaptive smoother
        smoother = AdaptiveESMDA(
            covariance=covariance, observations=observations, seed=1
        )

        def correlation_callback(corr_XY, observations):
            # cross-correlation matrix contains all correlations,
            # even those deemed insignificant.
            assert corr_XY.shape == (num_params, num_observations)
            return np.zeros_like(corr_XY)  # Cut off all

        X_i = np.copy(X)
        for _ in range(smoother.num_assimilations()):
            smoother.prepare_assimilation(Y=g(X_i))
            smoother.assimilate_batch(X=X_i, correlation_callback=correlation_callback)

        assert np.allclose(X, X_i)

    @pytest.mark.parametrize(
        "linear_problem",
        range(25),
        indirect=True,
        ids=[f"seed-{i + 1}" for i in range(25)],
    )
    def test_that_adaptive_localization_with_cutoff_0_equals_standard_ESMDA_update(
        self, linear_problem
    ):
        # Create a problem with g(x) = A @ x
        X, g, observations, covariance, rng = linear_problem

        missing = rng.random(size=X.shape) > 0.9
        missing[:, :2] = False  # At least two realizations have all params

        # =============================================================================
        # SETUP ESMDA FOR LOCALIZATION AND SOLVE PROBLEM
        # =============================================================================

        smoother = AdaptiveESMDA(
            covariance=covariance, observations=observations, seed=1, alpha=5
        )

        def correlation_callback(corr_XY, observations):
            return corr_XY  # Do nothing

        X_i = np.copy(X)
        for _ in range(smoother.num_assimilations()):
            smoother.prepare_assimilation(Y=g(X_i))
            X_i = smoother.assimilate_batch(
                X=X_i, correlation_callback=correlation_callback, missing=missing
            )

        # =============================================================================
        # VERIFY RESULT AGAINST NORMAL ESMDA ITERATIONS
        # =============================================================================
        smoother = ESMDA(
            covariance=covariance,
            observations=observations,
            alpha=5,
            seed=1,
        )

        X_i2 = np.copy(X)
        for _ in range(smoother.num_assimilations()):
            smoother.prepare_assimilation(Y=g(X_i2))
            X_i2 = smoother.assimilate_batch(X=X_i2, missing=missing)

        assert np.allclose(X_i, X_i2)

    @pytest.mark.skip(reason="Unresolved in algorithm")
    @pytest.mark.parametrize(
        "linear_problem",
        range(25),
        indirect=True,
        ids=[f"seed-{i + 1}" for i in range(25)],
    )
    @pytest.mark.parametrize(
        "cutoffs", [(0, 1e-3), (0.1, 0.2), (0.5, 0.5 + 1e-12), (0.9, 1), (1 - 1e-3, 1)]
    )
    @pytest.mark.parametrize("full_covariance", [True, False])
    def test_that_posterior_generalized_variance_increases_in_cutoff(
        self, linear_problem, cutoffs, full_covariance
    ):
        """This property only holds in the limit as the number of
        ensemble members goes to infinity. As the number of ensemble
        members decrease, this test starts to fail more often."""

        # Create a problem with g(x) = A @ x
        X, g, observations, covariance, _ = linear_problem
        if full_covariance:
            covariance = np.diag(covariance)

        # =============================================================================
        # SETUP ESMDA FOR LOCALIZATION AND SOLVE PROBLEM
        # =============================================================================

        smoother = AdaptiveESMDA(
            covariance=covariance, observations=observations, seed=1, alpha=1
        )

        def correlation_callback(corr_XY, observations_per_parameter, cutoff):
            corr_XY[np.abs(corr_XY) <= cutoff] = 0
            return corr_XY

        X_i = np.copy(X)

        for _ in range(smoother.num_assimilations()):
            # Run forward model
            Y_i = g(X_i)
            smoother.prepare_assimilation(Y=Y_i)

            cutoff_low, cutoff_high = cutoffs
            assert cutoff_low <= cutoff_high

            cb_low = functools.partial(correlation_callback, cutoff=cutoff_low)
            X_i_low_cutoff = smoother.assimilate_batch(
                X=X_i,
                correlation_callback=cb_low,
            )

            cb_high = functools.partial(correlation_callback, cutoff=cutoff_high)
            X_i_high_cutoff = smoother.assimilate_batch(
                X=X_i,
                correlation_callback=cb_high,
            )

            # Compute covariances
            prior_cov = np.cov(X, rowvar=True)
            posterior_cutoff_low_cov = np.cov(X_i_low_cutoff, rowvar=True)
            posterior_cutoff_high_cov = np.cov(X_i_high_cutoff, rowvar=True)
            assert prior_cov.shape == (X.shape[0], X.shape[0])

            # Compute determinants of covariance matrices
            # https://en.wikipedia.org/wiki/Generalized_variance
            # intuitively: large determintant => high covariancce
            #  => larger volume of multivariate normal
            # => less information contained in multivariate normal
            generalized_variance_prior = np.linalg.det(prior_cov)
            generalized_variance_low = np.linalg.det(posterior_cutoff_low_cov)
            generalized_variance_high = np.linalg.det(posterior_cutoff_high_cov)

            # The covariance is positive (semi) definite, so the determinant is >= 0
            assert generalized_variance_low >= 0, f"1 Failed with cutoff={cutoffs}"

            # Assimilating information always leads to more information,
            # which means a smaller lower determinant
            assert generalized_variance_low <= generalized_variance_prior
            assert generalized_variance_high <= generalized_variance_prior

            # A higher threshold means we assimilate less information
            assert generalized_variance_low <= generalized_variance_high, (
                f"2 Failed with cutoff_low={cutoff_low} and cutoff_high={cutoff_high}"
            )

    @pytest.mark.parametrize(
        "linear_problem",
        range(25),
        indirect=True,
        ids=[f"seed-{i + 1}" for i in range(25)],
    )
    def test_that_adaptive_localization_works_with_dying_realizations(
        self,
        linear_problem,
    ):
        """A full worked example of the adaptive localization API, with
        parameter groups and dying realizations (if compute nodes go down).

        This is mainly meant as an example, not a test."""

        # Create a problem with g(x) = A @ x
        X, g, observations, covariance, rng = linear_problem
        num_parameters, num_ensemble = X.shape
        num_observations = len(observations)

        # Split parameters into groups of equal size
        num_groups = 10
        assert num_observations % num_groups == 0, "Num groups must divide parameters"
        parameters_groups = np.array_split(np.arange(num_parameters), num_groups)
        assert len(parameters_groups) == num_groups

        # =============================================================================
        # SETUP ESMDA FOR LOCALIZATION AND SOLVE PROBLEM
        # =============================================================================
        alpha = normalize_alpha(np.array([5, 4, 3, 2, 1]))  # Vector of inflation values
        start_time = time.perf_counter()
        smoother = AdaptiveESMDA(
            covariance=covariance, observations=observations, alpha=alpha, seed=1
        )

        # Simulate realizations that die
        living_mask = rng.choice(
            [True, False], size=(len(alpha), num_ensemble), p=[0.9, 0.1]
        )

        X_i = np.copy(X)
        for i in range(smoother.num_assimilations()):
            print(f"ESMDA iteration {i + 1}")

            # We simulate loss of realizations due to compute clusters going down.
            # Figure out which realizations are still alive:
            alive_mask_i = np.all(living_mask[: i + 1, :], axis=0)
            num_alive = alive_mask_i.sum()
            print(f"  Total realizations still alive: {num_alive} / {num_ensemble}")

            # Run forward model
            Y_i = g(X_i)
            smoother.prepare_assimilation(Y=Y_i[:, alive_mask_i])

            for j, parameter_mask_j in enumerate(parameters_groups, 1):
                print(f"  Updating parameter group {j}/{len(parameters_groups)}")
                # Mask out rows in this parameter group, and columns of realization
                # that are still alive. This step simulates fetching from disk.
                mask = np.ix_(parameter_mask_j, alive_mask_i)
                X_i[mask] = smoother.assimilate_batch(
                    X=X_i[mask], correlation_callback=None
                )

            print()

        print(f"ESMDA with localization - Ran in {time.perf_counter() - start_time} s")

        # =============================================================================
        # VERIFY RESULT AGAINST NORMAL ESMDA ITERATIONS IN ONE BATCH
        # =============================================================================
        start_time = time.perf_counter()
        smoother = ESMDA(
            covariance=covariance,
            observations=observations,
            alpha=alpha,
            seed=1,
        )

        X_i2 = np.copy(X)
        for i in range(smoother.num_assimilations()):
            alive_mask_i = np.all(living_mask[: i + 1, :], axis=0)
            Y_i2 = g(X_i2)
            smoother.prepare_assimilation(Y=Y_i2[:, alive_mask_i])
            X_i2[:, alive_mask_i] = smoother.assimilate_batch(X=X_i2[:, alive_mask_i])

        # For this test to pass, correlation_threshold() should return <= 0
        print(
            "Norm difference between ESMDA with and without localization:",
            np.linalg.norm(X_i - X_i2),
        )
        assert np.allclose(X_i, X_i2, atol=1e-4)

        print(
            f"ESMDA without localization - Ran in {time.perf_counter() - start_time} s"
        )

        print("------------------------------------------")


if __name__ == "__main__":
    pytest.main(
        args=[
            __file__,
            "-v",
            # "-k test_that_adaptive_localization_works_with_dying_realizations"
        ]
    )
