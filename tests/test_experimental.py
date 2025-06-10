import functools
import time
from copy import deepcopy

import numpy as np
import pytest

from iterative_ensemble_smoother import ESMDA
from iterative_ensemble_smoother.esmda_inversion import (
    empirical_cross_covariance,
    normalize_alpha,
)
from iterative_ensemble_smoother.experimental import (
    AdaptiveESMDA,
    DistanceESMDA,
    ensemble_smoother_update_step_row_scaling,
)


@pytest.fixture()
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
    yield X, g, observations, covariance, rng


class TestAdaptiveESMDA:
    @pytest.mark.parametrize(
        "linear_problem",
        range(25),
        indirect=True,
        ids=[f"seed-{i+1}" for i in range(25)],
    )
    def test_that_adaptive_localization_with_cutoff_1_equals_ensemble_prior(
        self, linear_problem
    ):
        # Create a problem with g(x) = A @ x
        X, g, observations, covariance, _ = linear_problem

        # Create adaptive smoother
        smoother = AdaptiveESMDA(
            covariance=covariance, observations=observations, seed=1
        )

        def correlation_callback(corr_matrix):
            # cross-correlation matrix contains all correlations,
            # even those deemed insignificant.
            print(corr_matrix)
            assert corr_matrix.shape[0] == 50
            assert corr_matrix.shape[1] == len(observations)

        X_i = np.copy(X)
        alpha = normalize_alpha(np.ones(5))
        for alpha_i in alpha:
            # Run forward model
            Y_i = g(X_i)

            # Create noise D - common to this ESMDA update
            D_i = smoother.perturb_observations(
                ensemble_size=Y_i.shape[1], alpha=alpha_i
            )

            # Update the relevant parameters and write to X
            X_i = smoother.assimilate(
                X=X_i,
                Y=Y_i,
                D=D_i,
                alpha=alpha_i,
                correlation_threshold=1,
                correlation_callback=correlation_callback,
            )

        assert np.allclose(X, X_i)

    @pytest.mark.parametrize(
        "linear_problem",
        range(25),
        indirect=True,
        ids=[f"seed-{i+1}" for i in range(25)],
    )
    def test_that_adaptive_localization_with_cutoff_0_equals_standard_ESMDA_update(
        self, linear_problem
    ):
        # Create a problem with g(x) = A @ x
        X, g, observations, covariance, _ = linear_problem

        # =============================================================================
        # SETUP ESMDA FOR LOCALIZATION AND SOLVE PROBLEM
        # =============================================================================
        alpha = normalize_alpha(np.ones(5))

        smoother = AdaptiveESMDA(
            covariance=covariance, observations=observations, seed=1
        )

        def correlation_callback(corr_matrix):
            # A correlation threshold of 0 means that all
            # correlations are deemed significant.
            # Therefore, the cross-correlation matrix must
            # include all parameter-resposne pairs.
            assert corr_matrix.shape[0] == X.shape[0]
            assert corr_matrix.shape[1] == len(observations)

        X_i = np.copy(X)
        for _, alpha_i in enumerate(alpha, 1):
            # Run forward model
            Y_i = g(X_i)

            # Create noise D - common to this ESMDA update
            D_i = smoother.perturb_observations(
                ensemble_size=Y_i.shape[1], alpha=alpha_i
            )

            # Update the relevant parameters and write to X
            smoother.assimilate(
                X=X_i,
                Y=Y_i,
                D=D_i,
                overwrite=True,
                alpha=alpha_i,
                correlation_threshold=0,
                correlation_callback=functools.partial(correlation_callback),
            )

        # =============================================================================
        # VERIFY RESULT AGAINST NORMAL ESMDA ITERATIONS
        # =============================================================================
        smoother = ESMDA(
            covariance=covariance,
            observations=observations,
            alpha=alpha,
            seed=1,
        )

        X_i2 = np.copy(X)
        for _ in range(smoother.num_assimilations()):
            X_i2 = smoother.assimilate(X_i2, g(X_i2))

        assert np.allclose(X_i, X_i2)

    @pytest.mark.parametrize(
        "linear_problem",
        range(25),
        indirect=True,
        ids=[f"seed-{i+1}" for i in range(25)],
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
        alpha = normalize_alpha(np.ones(1))

        smoother = AdaptiveESMDA(
            covariance=covariance, observations=observations, seed=1
        )

        X_i = np.copy(X)
        for _, alpha_i in enumerate(alpha, 1):
            # Run forward model
            Y_i = g(X_i)

            # Create noise D - common to this ESMDA update
            D_i = smoother.perturb_observations(
                ensemble_size=Y_i.shape[1], alpha=alpha_i
            )

            cutoff_low, cutoff_high = cutoffs
            assert cutoff_low <= cutoff_high

            X_i_low_cutoff = smoother.assimilate(
                X=X_i,
                Y=Y_i,
                D=D_i,
                alpha=alpha_i,
                correlation_threshold=cutoff_low,
            )
            X_i_high_cutoff = smoother.assimilate(
                X=X_i,
                Y=Y_i,
                D=D_i,
                alpha=alpha_i,
                correlation_threshold=cutoff_high,
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
            assert (
                generalized_variance_low <= generalized_variance_high
            ), f"2 Failed with cutoff_low={cutoff_low} and cutoff_high={cutoff_high}"

    @pytest.mark.parametrize(
        "linear_problem",
        range(25),
        indirect=True,
        ids=[f"seed-{i+1}" for i in range(25)],
    )
    def test_that_adaptive_localization_works_with_dying_realizations(
        self,
        linear_problem,
    ):
        """A full worked example of the adaptive localization API, with
        parameter groups and dying realizations (if compute nodes go down).

        This is mainly meant as an example, not a test. To make it into a test
        worth running, we set the threshold for the correlation to 0 and verify
        that it returns the same result as ESMDA."""

        def zero_correlation_threshold(ensemble_size):
            """Adaptive localization only matches ESMDA when the threshold is <=0."""
            return 0

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
            covariance=covariance, observations=observations, seed=1
        )

        # Simulate realizations that die
        living_mask = rng.choice(
            [True, False], size=(len(alpha), num_ensemble), p=[0.9, 0.1]
        )

        X_i = np.copy(X)
        for i, alpha_i in enumerate(alpha, 1):
            print(f"ESMDA iteration {i} with alpha_i={alpha_i}")

            # Run forward model
            Y_i = g(X_i)

            # We simulate loss of realizations due to compute clusters going down.
            # Figure out which realizations are still alive:
            alive_mask_i = np.all(living_mask[:i, :], axis=0)
            num_alive = alive_mask_i.sum()
            print(f"  Total realizations still alive: {num_alive} / {num_ensemble}")

            # Create noise D - common to this ESMDA update
            D_i = smoother.perturb_observations(ensemble_size=num_alive, alpha=alpha_i)

            # Loop over parameter groups and update
            for j, parameter_mask_j in enumerate(parameters_groups, 1):
                print(f"  Updating parameter group {j}/{len(parameters_groups)}")

                # Mask out rows in this parameter group, and columns of realization
                # that are still alive. This step simulates fetching from disk.
                mask = np.ix_(parameter_mask_j, alive_mask_i)

                # Update the relevant parameters and write to X
                X_i[mask] = smoother.assimilate(
                    X=X_i[mask],
                    Y=Y_i[:, alive_mask_i],
                    D=D_i,
                    alpha=alpha_i,
                    # Pass a function that always returns zero,
                    # no matter what the ensemble size is
                    correlation_threshold=zero_correlation_threshold,
                )

            print()

        print(f"ESMDA with localization - Ran in {time.perf_counter() - start_time} s")

        # =============================================================================
        # VERIFY RESULT AGAINST NORMAL ESMDA ITERATIONS
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
            # Run simulations
            Y_i = g(X_i2)

            # We simulate loss of realizations due to compute clusters going down.
            # Figure out which realizations are still alive:
            alive_mask_i = np.all(living_mask[: i + 1, :], axis=0)
            num_alive = alive_mask_i.sum()

            X_i2[:, alive_mask_i] = smoother.assimilate(
                X_i2[:, alive_mask_i], Y_i[:, alive_mask_i]
            )

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

    @pytest.mark.parametrize(
        "linear_problem",
        range(25),
        indirect=True,
        ids=[f"seed-{i+1}" for i in range(25)],
    )
    def test_that_cov_YY_can_be_computed_outside_of_assimilate(
        self,
        linear_problem,
    ):
        """Cov(Y, Y) may be computed once in each assimilation round.
        This saves time if the user wants to iterate over parameters groups,
        since Cov(Y, Y) is independent of the parameters X, there is no reason
        to compute it more than once.

        Below we do not loop over parameter groups in X, so there is no speed
        gain when passing the covariance cov(Y, Y). This test is just to check
        that the result is the same regardless of whether the user passes
        the covariance matrix or not.
        """

        # Create a problem with g(x) = A @ x
        X, g, observations, covariance, _ = linear_problem
        _, ensemble_size = X.shape

        alpha = normalize_alpha(np.array([5, 4, 3, 2, 1]))  # Vector of inflation values
        smoother = AdaptiveESMDA(
            covariance=covariance, observations=observations, seed=1
        )

        X_i = np.copy(X)
        for i, alpha_i in enumerate(alpha, 1):
            print(f"ESMDA iteration {i} with alpha_i={alpha_i}")

            # Run forward model
            Y_i = g(X_i)

            # Create noise D - common to this ESMDA update
            D_i = smoother.perturb_observations(
                ensemble_size=ensemble_size, alpha=alpha_i
            )

            # Update the parameters without using pre-computed cov_YY
            X_i1 = smoother.assimilate(
                X=X_i,
                Y=Y_i,
                D=D_i,
                alpha=alpha_i,
            )

            # Update the parameters using pre-computed cov_YY
            X_i2 = smoother.assimilate(
                X=X_i,
                Y=Y_i,
                D=D_i,
                alpha=alpha_i,
                cov_YY=empirical_cross_covariance(Y_i, Y_i),
            )

            # Check that the update is the same, whether or not Cov_YY is passed
            assert np.allclose(X_i1, X_i2)

            X_i = X_i1


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
            (X[idx, :], RowScaling(alpha=1)) for _, idx in enumerate(row_groups)
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
        for _ in range(smoother.num_assimilations()):
            X_posterior = smoother.assimilate(X, Y)

        # The result should be the same
        assert np.allclose(
            X_posterior, np.vstack([X_i for (X_i, _) in X_with_row_scaling_updated])
        )


@pytest.mark.parametrize("seed", list(range(9)))
def test_distance_based_localization_on_1D_case_with_single_observation(seed):
    N_m = 100  # Number of model parameters (grid points)
    N_e = 50  # Ensemble size
    j_obs = 50  # Index of the single observation

    alpha_i = 1  # TODO: Implement multiple iterations
    obs_error_var = 0.01  # Variance of observation error

    true_parameters = np.zeros(N_m)

    true_observations = np.array([1.0])

    C_D = np.array([obs_error_var])

    rng = np.random.default_rng(seed)
    X_prior = rng.normal(loc=0.0, scale=0.5, size=(N_m, N_e))

    # Predict observations `Y` using the identity model `g(x) = x`
    # We only observe the state at `j_obs`.
    Y = X_prior[[j_obs], :]

    # Using a simple Gaussian decay for rho
    localization_radius = 10.0
    model_grid = np.arange(N_m)
    distances = np.abs(model_grid - j_obs)
    rho = np.exp(-0.5 * (distances / localization_radius) ** 2).reshape(-1, 1)

    esmda_distance = DistanceESMDA(
        covariance=C_D, observations=true_observations, alpha=alpha_i, seed=rng
    )
    X_posterior = esmda_distance.assimilate(X=X_prior, Y=Y, rho=rho)

    # Find parameters far from the observation where localization weight is near zero
    zero_weight_indices = np.where(rho < 1e-5)[0]
    assert len(zero_weight_indices) > 0

    # Assert that parameters in the zero-weight region have not been updated
    assert np.allclose(
        X_posterior[zero_weight_indices, :], X_prior[zero_weight_indices, :], rtol=1e-2
    )
    # Assert that parameters near the observation *have* been updated
    assert not np.allclose(X_posterior[j_obs, :], X_prior[j_obs, :])

    prior_mean = np.mean(X_prior, axis=1)
    posterior_mean = np.mean(X_posterior, axis=1)

    # The largest update to the mean should occur exactly at the observation point
    update_magnitude = np.abs(posterior_mean - prior_mean)
    assert np.argmax(update_magnitude) == j_obs

    prior_variance = np.var(X_prior, axis=1)
    posterior_variance = np.var(X_posterior, axis=1)

    # The variance of the ensemble should decrease where the update was applied
    assert posterior_variance[j_obs] < prior_variance[j_obs]
    # The variance of the ensemble should NOT change where there was no update
    assert np.allclose(
        posterior_variance[zero_weight_indices], prior_variance[zero_weight_indices]
    )

    esmda = ESMDA(
        covariance=C_D, observations=true_observations, alpha=alpha_i, seed=rng
    )
    X_posterior_global = esmda.assimilate(X=X_prior, Y=Y)
    # The global update MUST change distant parameters due to spurious correlations.
    assert not np.allclose(
        X_posterior_global[zero_weight_indices, :],
        X_prior[zero_weight_indices, :],
    )

    posterior_variance_global = np.var(X_posterior_global, axis=1)
    # The non-localized update should reduce the AVERAGE variance everywhere due to
    # spurious correlations. This test is more robust to random sampling noise
    # than asserting that every single variance must decrease.
    assert np.mean(posterior_variance_global[zero_weight_indices]) < np.mean(
        prior_variance[zero_weight_indices]
    )

    # The localized posterior mean should be a better estimate of the true parameters
    # overall, because it avoids polluting the entire field with noise.
    posterior_mean_global = np.mean(X_posterior_global, axis=1)

    # Calculate Mean Squared Error (MSE) against the true parameters
    mse_localized = np.mean((posterior_mean - true_parameters) ** 2)
    mse_non_localized = np.mean((posterior_mean_global - true_parameters) ** 2)

    # The MSE of the localized result should be significantly smaller.
    assert mse_localized < mse_non_localized


@pytest.mark.parametrize("seed", list(range(9)))
def test_distance_based_localization_on_2D_case_with_single_observation(seed):
    Nx, Ny = 10, 10  # Dimensions of the 2D parameter grid
    N_m = Nx * Ny  # Total number of model parameters
    N_e = 50  # Ensemble size
    x_obs, y_obs = 5, 5  # Index of the single observation in 2D

    alpha_i = 1
    obs_error_var = 0.01

    true_parameters = np.zeros(N_m)
    true_observations = np.array([1.0])
    C_D = np.array([obs_error_var])

    # --- Generate Initial Ensemble and Predictions ---
    rng = np.random.default_rng(seed)
    X_prior = rng.normal(loc=0.0, scale=0.5, size=(N_m, N_e))

    # Convert the 2D observation index to a flat 1D index for slicing
    flat_obs_index = y_obs * Nx + x_obs
    Y = X_prior[[flat_obs_index], :]

    # --- Construct 2D Localization `rho` ---
    localization_radius = 1.0
    xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny))
    distances_2d = np.sqrt((xx - x_obs) ** 2 + (yy - y_obs) ** 2)
    distances = distances_2d.flatten()
    rho = np.exp(-0.5 * (distances / localization_radius) ** 2).reshape(-1, 1)

    # --- Run Assimilations ---
    # Initialize smoothers with the current run's seeded RNG
    esmda_distance = DistanceESMDA(
        covariance=C_D, observations=true_observations, alpha=alpha_i, seed=rng
    )
    X_posterior = esmda_distance.assimilate(X=X_prior, Y=Y, rho=rho)

    esmda_global = ESMDA(
        covariance=C_D, observations=true_observations, alpha=alpha_i, seed=rng
    )
    X_posterior_global = esmda_global.assimilate(X=X_prior, Y=Y)

    # --- Assertions ---
    # Find parameters far from the observation where localization weight is near zero
    zero_weight_indices = np.where(rho < 1e-6)[0]
    assert (
        len(zero_weight_indices) > 0
    ), "No distant points found for testing localization."

    # --- Assertions for Localized Smoother ---
    prior_mean = np.mean(X_prior, axis=1)
    posterior_mean = np.mean(X_posterior, axis=1)
    prior_variance = np.var(X_prior, axis=1)
    posterior_variance = np.var(X_posterior, axis=1)

    # Assert distant parameters are not updated
    assert np.allclose(
        X_posterior[zero_weight_indices, :], X_prior[zero_weight_indices, :], rtol=1e-2
    )
    # Assert parameter at observation IS updated
    assert not np.allclose(X_posterior[flat_obs_index, :], X_prior[flat_obs_index, :])

    # Assert update is maximal at the observation point
    update_magnitude = np.abs(posterior_mean - prior_mean)
    assert np.argmax(update_magnitude) == flat_obs_index

    # Assert variance is reduced at observation, but preserved far away
    assert posterior_variance[flat_obs_index] < prior_variance[flat_obs_index]
    assert np.allclose(
        posterior_variance[zero_weight_indices], prior_variance[zero_weight_indices]
    )

    # --- Comparison Assertions for Global Smoother ---
    posterior_mean_global = np.mean(X_posterior_global, axis=1)
    posterior_variance_global = np.var(X_posterior_global, axis=1)

    # Assert distant parameters ARE updated due to spurious correlations
    assert not np.allclose(
        X_posterior_global[zero_weight_indices, :],
        X_prior[zero_weight_indices, :],
    )

    # Assert AVERAGE variance is reduced everywhere (the sign of ensemble collapse)
    assert np.mean(posterior_variance_global[zero_weight_indices]) < np.mean(
        prior_variance[zero_weight_indices]
    )

    # --- Overall Quality Assertion (MSE) ---
    # Assert that the localized result is a more plausible estimate
    mse_localized = np.mean((posterior_mean - true_parameters) ** 2)
    mse_non_localized = np.mean((posterior_mean_global - true_parameters) ** 2)
    assert mse_localized < mse_non_localized


@pytest.mark.parametrize("seed", list(range(9)))
def test_distance_based_localization_on_3D_case(seed):
    # --- 1. System and Assimilation Setup (3D Grid) ---
    Nx, Ny, Nz = 10, 10, 10  # Dimensions of the 3D parameter grid
    N_m = Nx * Ny * Nz  # Total number of model parameters (now 1000)
    N_e = 50  # Ensemble size
    x_obs, y_obs, z_obs = 5, 5, 5  # Index of the single observation in 3D

    alpha_i = 1
    obs_error_var = 0.01

    true_parameters = np.zeros(N_m)
    true_observations = np.array([1.0])
    C_D = np.array([obs_error_var])

    # --- 2. Generate Initial Ensemble and Predictions ---
    rng = np.random.default_rng(seed)
    X_prior = rng.normal(loc=0.0, scale=0.5, size=(N_m, N_e))

    # Convert the 3D observation index to a flat 1D index for slicing
    flat_obs_index = (z_obs * Nx * Ny) + (y_obs * Nx) + x_obs
    Y = X_prior[[flat_obs_index], :]

    # --- 3. Construct 3D Localization `rho` ---
    # A smaller radius is used to ensure the localization effect is clear
    # and that some weights decay to near-zero within the grid boundaries.
    localization_radius = 1.5

    # Create 3D coordinate grids
    zz, yy, xx = np.meshgrid(np.arange(Nz), np.arange(Ny), np.arange(Nx), indexing="ij")
    # Calculate 3D Euclidean distance from every point to the observation
    distances_3d = np.sqrt((xx - x_obs) ** 2 + (yy - y_obs) ** 2 + (zz - z_obs) ** 2)
    distances = distances_3d.flatten()

    rho = np.exp(-0.5 * (distances / localization_radius) ** 2).reshape(-1, 1)

    # --- 4. Run Assimilations ---
    # Initialize smoothers with the current run's seeded RNG
    esmda_distance = DistanceESMDA(
        covariance=C_D, observations=true_observations, alpha=alpha_i, seed=rng
    )
    X_posterior = esmda_distance.assimilate(X=X_prior, Y=Y, rho=rho)

    esmda_global = ESMDA(
        covariance=C_D, observations=true_observations, alpha=alpha_i, seed=rng
    )
    X_posterior_global = esmda_global.assimilate(X=X_prior, Y=Y)

    # --- 5. Assertions ---
    # Find parameters far from the observation where localization weight is near zero
    zero_weight_indices = np.where(rho < 1e-6)[0]
    assert (
        len(zero_weight_indices) > 0
    ), "No distant points found for testing localization."

    # --- 5a. Assertions for Localized Smoother ---
    prior_mean = np.mean(X_prior, axis=1)
    posterior_mean = np.mean(X_posterior, axis=1)
    prior_variance = np.var(X_prior, axis=1)
    posterior_variance = np.var(X_posterior, axis=1)

    # Assert distant parameters are not updated (within a small tolerance for the tail)
    assert np.allclose(
        X_posterior[zero_weight_indices, :], X_prior[zero_weight_indices, :], atol=1e-5
    )
    # Assert parameter at observation IS updated
    assert not np.allclose(X_posterior[flat_obs_index, :], X_prior[flat_obs_index, :])

    # Assert update is maximal at the observation point
    update_magnitude = np.abs(posterior_mean - prior_mean)
    assert np.argmax(update_magnitude) == flat_obs_index

    # Assert variance is reduced at observation, but preserved far away
    assert posterior_variance[flat_obs_index] < prior_variance[flat_obs_index]
    assert np.allclose(
        posterior_variance[zero_weight_indices], prior_variance[zero_weight_indices]
    )

    # --- 5b. Comparison Assertions for Global Smoother ---
    posterior_mean_global = np.mean(X_posterior_global, axis=1)
    posterior_variance_global = np.var(X_posterior_global, axis=1)

    # Assert distant parameters ARE updated due to spurious correlations
    assert not np.allclose(
        X_posterior_global[zero_weight_indices, :],
        X_prior[zero_weight_indices, :],
    )

    # Assert AVERAGE variance is reduced everywhere (the sign of ensemble collapse)
    assert np.mean(posterior_variance_global[zero_weight_indices]) < np.mean(
        prior_variance[zero_weight_indices]
    )

    # --- 5c. Overall Quality Assertion (MSE) ---
    # Assert that the localized result is a more plausible estimate
    mse_localized = np.mean((posterior_mean - true_parameters) ** 2)
    mse_non_localized = np.mean((posterior_mean_global - true_parameters) ** 2)
    assert mse_localized < mse_non_localized


if __name__ == "__main__":
    pytest.main(
        args=[
            __file__,
            "-v",
        ]
    )
