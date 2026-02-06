import functools
import sys
import time
from copy import deepcopy

import numpy as np
import pytest
from numpy import typing as npt
from numpy.testing import assert_allclose

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
from iterative_ensemble_smoother.utils import (
    calc_max_number_of_layers_per_batch_for_distance_localization,
    calc_rho_for_2d_grid_layer,
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
        ids=[f"seed-{i + 1}" for i in range(25)],
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
        ids=[f"seed-{i + 1}" for i in range(25)],
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

    @pytest.fixture
    def large_linear_problem(self):
        """
        Creates a problem large enough to realistically benchmark parallelization.
        """
        rng = np.random.default_rng(42)  # Use a fixed seed for reproducibility

        num_parameters = 5000
        num_observations = 200
        num_ensemble = 200

        A = np.exp(rng.standard_normal(size=(num_observations, num_parameters)))

        def g(X):
            """Forward model."""
            return A @ X

        x_true = np.linspace(-1, 1, num=num_parameters)
        observations = g(x_true) + rng.standard_normal(size=num_observations) / 10

        X = rng.normal(size=(num_parameters, num_ensemble))
        covariance = rng.triangular(0.1, 1, 1, size=num_observations)
        return X, g, observations, covariance

    def test_parallelization_runtime_comparison(self, large_linear_problem):
        """
        Compares the runtime of the assimilate method with and without parallelization.
        """
        X, g, observations, covariance = large_linear_problem

        # Use a single assimilation step for a clear comparison
        alpha = 1.0

        # --- Setup the smoother and common data ---
        smoother = AdaptiveESMDA(
            covariance=covariance, observations=observations, seed=1
        )
        Y = g(X)
        D = smoother.perturb_observations(ensemble_size=Y.shape[1], alpha=alpha)

        # Make separate copies for each run to ensure a fair start
        X_serial = np.copy(X)
        X_parallel = np.copy(X)

        # --- 1. Serial Execution ---
        print("\n--- Running Serial Benchmark (n_jobs=1) ---")
        start_serial = time.perf_counter()
        X_serial = smoother.assimilate(
            X=X_serial,
            Y=Y,
            D=D,
            alpha=alpha,
            correlation_threshold=0,  # Force the code to update every single parameter.
            n_jobs=1,
        )
        end_serial = time.perf_counter()
        time_serial = end_serial - start_serial
        print(f"Serial execution time: {time_serial:.4f} seconds")

        # --- 2. Parallel Execution ---
        print("\n--- Running Parallel Benchmark (n_jobs=-1) ---")
        start_parallel = time.perf_counter()
        X_parallel = smoother.assimilate(
            X=X_parallel,
            Y=Y,
            D=D,
            alpha=alpha,
            correlation_threshold=0,  # Force the code to update every single parameter.
            n_jobs=-1,
        )
        end_parallel = time.perf_counter()
        time_parallel = end_parallel - start_parallel
        print(f"Parallel execution time: {time_parallel:.4f} seconds")

        # --- 3. Verification and Summary ---
        print("\n--- Benchmark Summary ---")
        # Critical check: ensure both methods produce the same result
        assert np.allclose(X_serial, X_parallel)
        print("✅ Numerical results are identical.")

        if time_parallel > 0:
            speedup = time_serial / time_parallel
            print(f"🚀 Speedup factor: {speedup:.2f}x")

        assert time_serial > time_parallel


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


def assert_tests_for_distance_based_localization(
    X_prior: npt.NDArray[np.float64],
    obs_index_flatten: int,
    rho: npt.NDArray[np.float64],
    true_parameters: npt.NDArray[np.float64],
    X_posterior: npt.NDArray[np.float64],
    X_posterior_global: npt.NDArray[np.float64],
    rho_min: float = 1e-6,
    atol: float = 1e-5,
):
    # Find parameters far from the observation where localization weight is near zero
    zero_weight_indices = np.where(rho < rho_min)[0]
    assert len(zero_weight_indices) > 0, (
        "No distant points found for testing localization."
    )
    # --- Assertions for Localized Smoother ---
    # Assert distant parameters are not updated
    assert np.allclose(
        X_posterior[zero_weight_indices, :], X_prior[zero_weight_indices, :], atol=atol
    )
    # Assert parameter at observation IS updated
    assert not np.allclose(
        X_posterior[obs_index_flatten, :], X_prior[obs_index_flatten, :]
    )

    prior_mean = np.mean(X_prior, axis=1)
    posterior_mean = np.mean(X_posterior, axis=1)
    prior_variance = np.var(X_prior, axis=1)
    posterior_variance = np.var(X_posterior, axis=1)

    # Assert update is maximal at the observation point
    update_magnitude = np.abs(posterior_mean - prior_mean)
    assert np.argmax(update_magnitude) == obs_index_flatten

    # Assert variance is reduced at observation, but preserved far away
    assert posterior_variance[obs_index_flatten] < prior_variance[obs_index_flatten]
    # The variance of the ensemble should NOT change where there was no update
    assert np.allclose(
        posterior_variance[zero_weight_indices], prior_variance[zero_weight_indices]
    )

    # The global update MUST change distant parameters due to spurious correlations.
    assert not np.allclose(
        X_posterior_global[zero_weight_indices, :],
        X_prior[zero_weight_indices, :],
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

    # Calculate Mean Squared Error (MSE) against the true parameters
    mse_localized = np.mean((posterior_mean - true_parameters) ** 2)
    mse_non_localized = np.mean((posterior_mean_global - true_parameters) ** 2)

    # --- Overall Quality Assertion (MSE) ---
    # Assert that the localized result is a more plausible estimate
    assert mse_localized < mse_non_localized


def calculate_rho_1d(
    N_m: int, obs_index: int, localization_radius: float, xinc: float = 1.0
) -> npt.NDArray[np.float64]:
    model_grid = np.arange(N_m) * xinc
    distances = np.abs(model_grid - obs_index * xinc)
    # Ordinary definition of range,
    # rho is approximately 0.05 at localization_radius
    return np.exp(-3.0 * (distances / localization_radius) ** 2).reshape(-1, 1)


def calculate_rho_2d(
    Nx: int, Ny: int, x_obs: int, y_obs: int, localization_radius: float
) -> npt.NDArray[np.float64]:
    xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny))
    distances_2d = np.sqrt((xx - x_obs) ** 2 + (yy - y_obs) ** 2)
    distances = distances_2d.flatten()
    # Ordinary definition of range,
    # rho is approximately 0.05 at localization_radius
    return np.exp(-3.0 * (distances / localization_radius) ** 2).reshape(-1, 1)


def calculate_rho_3d(
    Nx: int,
    Ny: int,
    Nz: int,
    x_obs: int,
    y_obs: int,
    z_obs: int,
    localization_radius: float,
) -> npt.NDArray[np.float64]:
    # Create 3D coordinate grids
    zz, yy, xx = np.meshgrid(np.arange(Nz), np.arange(Ny), np.arange(Nx), indexing="ij")
    # Calculate 3D Euclidean distance from every point to the observation
    distances_3d = np.sqrt((xx - x_obs) ** 2 + (yy - y_obs) ** 2 + (zz - z_obs) ** 2)
    distances = distances_3d.flatten()
    # Ordinary definition of range,
    # rho is approximately 0.05 at localization_radius
    return np.exp(-3.0 * (distances / localization_radius) ** 2).reshape(-1, 1)


def draw_1D_field(
    mean: float,
    stdev: float,
    xinc: float,
    corr_func_name: str,
    corr_range: float,
    nparam: int,
    nreal: int,
    rng,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # Draw prior ensemble of 1D field with nparam. Ensemble size is nreal
    # Returns prior ensemble drawn and covariance matrix

    variance = stdev**2

    # Generate distance matrix
    x_coords = (np.arange(nparam) + 0.5) * xinc
    distances = np.abs(x_coords[:, None] - x_coords[None, :]) / corr_range

    # Compute covariance matrix based on correlation function
    if corr_func_name == "exponential":
        cov_matrix = variance * np.exp(-3.0 * distances)
    elif corr_func_name == "gaussian":
        cov_matrix = variance * np.exp(-3.0 * distances**2)
    else:
        raise ValueError("Unsupported correlation function")

    # Create mean array
    mean_values = np.full((nparam,), mean, dtype=np.float64)

    # Generate random fields with multivariate normal distribution
    fields = rng.multivariate_normal(mean_values, cov_matrix, size=nreal).T
    return fields, cov_matrix


def generate_alpha_vector(nalpha):
    """Generate alpha vector with length nalpha
    where sum of the inverse of alpha is 1.
    """
    # Define alpha[k] = [2**(m-1) + 2**(m-2) + ... + 2**(0)] / 2**(k)
    # such that alpha[k] = (2**m -1)/(2**k)
    # Then sum (1/alpha[k]) for k=0,1,2,..m-1  is 1.0
    # The alpha values are reduced with a factor 1/2 for each iteration
    # Special case: m = 3 gives alpha_vector = [7, 3.5, 1.75]
    return (2**nalpha - 1) / 2 ** np.arange(nalpha)


def predicted_mean_field_values(
    obs_vector: npt.NDArray[np.float64],
    obs_index_vector: npt.NDArray[np.int32],
    field_cov_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculates simple kriging estimate of expected value. This is used
    to check that DL-ESMDA and ordinary ESMDA are close to the kriging estimate
    when number of realizations becomes large.
    """
    # Extract the observed covariance matrix
    # (corresponding to the response covariance matrix)
    # Here simple kriging with 0 observation error is used, and
    # response is equal to the simulated field values in
    # position of the observations.
    obs_cov_matrix = field_cov_matrix[np.ix_(obs_index_vector, obs_index_vector)]

    # Compute the inverse covariance matrix
    inv_cov_matrix = np.linalg.inv(obs_cov_matrix)

    # Extract the covariance matrix between the observations and the full field
    field_obs_cov = field_cov_matrix[obs_index_vector, :]

    # Predicted mean values (broadcast calculation for all parameters at once)
    return field_obs_cov.T @ inv_cov_matrix @ obs_vector


@pytest.mark.parametrize("seed", list(range(9)))
def test_distance_based_localization_on_1D_case_with_single_observation(seed):
    N_m = 100  # Number of model parameters (grid points)
    N_e = 50  # Ensemble size
    j_obs = 50  # Index of the single observation

    alpha_i = 1
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
    localization_radius = 20.0
    rho = calculate_rho_1d(N_m, j_obs, localization_radius)

    esmda_distance = DistanceESMDA(
        covariance=C_D, observations=true_observations, alpha=alpha_i, seed=rng
    )
    X_posterior = esmda_distance.assimilate(X=X_prior, Y=Y, rho=rho)

    esmda = ESMDA(
        covariance=C_D, observations=true_observations, alpha=alpha_i, seed=rng
    )
    X_posterior_global = esmda.assimilate(X=X_prior, Y=Y)
    assert_tests_for_distance_based_localization(
        X_prior,
        j_obs,
        rho,
        true_parameters,
        X_posterior,
        X_posterior_global,
        rho_min=1e-5,
        atol=1e-2,
    )


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
    localization_radius = 2.5
    rho = calculate_rho_2d(Nx, Ny, x_obs, y_obs, localization_radius)

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

    assert_tests_for_distance_based_localization(
        X_prior,
        flat_obs_index,
        rho,
        true_parameters,
        X_posterior,
        X_posterior_global,
        rho_min=1e-6,
        atol=1e-5,
    )


@pytest.mark.parametrize("seed", list(range(9)))
def test_distance_based_localization_on_3D_case_with_single_observation(seed):
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
    localization_radius = 2.5
    rho = calculate_rho_3d(Nx, Ny, Nz, x_obs, y_obs, z_obs, localization_radius)

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

    assert_tests_for_distance_based_localization(
        X_prior,
        flat_obs_index,
        rho,
        true_parameters,
        X_posterior,
        X_posterior_global,
        rho_min=1e-6,
        atol=1e-5,
    )


@pytest.mark.parametrize("seed", list(range(9)))
def test_distance_based_on_1D_case_multiple_data_assimilation(seed):
    N_m = 100  # Number of model parameters (grid points)
    N_e = 50  # Ensemble size
    j_obs = 50  # Index of the single observation
    nalpha = (
        3  # number of different alpha values. They satisfy that sum(1/alpha[i]  = 1
    )
    alpha_vector = generate_alpha_vector(nalpha)
    inverse_alpha_vector = 1.0 / alpha_vector

    # Consistency requirement for alpha
    assert abs(inverse_alpha_vector.sum() - 1.0) < 1e-7

    obs_error_var = 0.01  # Variance of observation error

    true_parameters = np.zeros(N_m)

    true_observations = np.array([1.0])

    C_D = np.array([obs_error_var])

    for iteration in range(len(alpha_vector)):
        alpha_i = np.array([alpha_vector[iteration]])
        if iteration == 0:
            # Draw prior
            rng = np.random.default_rng(seed)
            X_prior = rng.normal(loc=0.0, scale=0.5, size=(N_m, N_e))
            X_previous = X_prior.copy()
            X_previous_global = X_prior.copy()
            X_posterior = X_prior
            X_posterior_global = X_prior
        else:
            X_previous = X_posterior
            X_previous_global = X_posterior_global
        # Predict observations `Y` using the identity model `g(x) = x`
        # We only observe the state at `j_obs`.
        Y = X_previous[[j_obs], :]

        # Using a simple Gaussian decay for rho
        localization_radius = 20.0
        rho = calculate_rho_1d(N_m, j_obs, localization_radius)

        esmda_distance = DistanceESMDA(
            covariance=C_D, observations=true_observations, alpha=alpha_i, seed=rng
        )
        X_posterior = esmda_distance.assimilate(X=X_previous, Y=Y, rho=rho)

        esmda = ESMDA(
            covariance=C_D, observations=true_observations, alpha=alpha_i, seed=rng
        )
        X_posterior_global = esmda.assimilate(X=X_previous_global, Y=Y)
        assert_tests_for_distance_based_localization(
            X_prior,
            j_obs,
            rho,
            true_parameters,
            X_posterior,
            X_posterior_global,
            rho_min=1e-5,
            atol=1e-2,
        )


@pytest.mark.parametrize("seed", list(range(9)))
def test_distance_based_on_1D_case_multiple_data_assimilate_batch(seed):
    """
    This test is almost equal to the one above, but here the function
    assimilate and assimilate_batch are compared to check they give
    same result. The purpose of assimilate_batch is to avoid
    re-calculating steps that is not necessary when running a loop
    over batches of parameters
    """
    N_m = 100  # Number of model parameters (grid points)
    N_e = 50  # Ensemble size
    j_obs = 50  # Index of the single observation
    nalpha = 10  # Number of different sets of alpha values
    number_of_alpha_values = np.arange(1, nalpha, 1, dtype=np.int32)
    for m in number_of_alpha_values:
        alpha_vector = generate_alpha_vector(m)

        obs_error_var = 0.01  # Variance of observation error

        true_parameters = np.zeros(N_m)

        true_observations = np.array([1.0])

        C_D = np.array([obs_error_var])

        for iteration in range(len(alpha_vector)):
            alpha_i = np.array([alpha_vector[iteration]])
            if iteration == 0:
                # Draw prior
                rng = np.random.default_rng(seed)
                X_prior = rng.normal(loc=0.0, scale=0.5, size=(N_m, N_e))
                X_previous = X_prior.copy()
                X_previous_global = X_prior.copy()
                X_posterior = X_prior
                X_posterior_global = X_prior
            else:
                X_previous = X_posterior
                X_previous_global = X_posterior_global
            # Predict observations `Y` using the identity model `g(x) = x`
            # We only observe the state at `j_obs`.
            Y = X_previous[[j_obs], :]

            # Using a simple Gaussian decay for rho
            localization_radius = 20.0
            rho = calculate_rho_1d(N_m, j_obs, localization_radius)

            # Use same seed for both instances of DistanceESMDA
            esmda_distance = DistanceESMDA(
                covariance=C_D, observations=true_observations, alpha=alpha_i, seed=seed
            )
            X_posterior_1 = esmda_distance.assimilate_batch(
                X_batch=X_previous, Y=Y, rho_batch=rho
            )

            esmda_distance = DistanceESMDA(
                covariance=C_D, observations=true_observations, alpha=alpha_i, seed=seed
            )
            X_posterior_2 = esmda_distance.assimilate(X=X_previous, Y=Y, rho=rho)

            diff_X = X_posterior_2 - X_posterior_1
            max_diff = np.max(np.abs(diff_X))
            # Check that the results are equal
            assert max_diff < 1e-9, (
                "The function assimilate and assimilate_batch give different results"
            )

            esmda = ESMDA(
                covariance=C_D, observations=true_observations, alpha=alpha_i, seed=rng
            )

            X_posterior_global = esmda.assimilate(X=X_previous_global, Y=Y)
            assert_tests_for_distance_based_localization(
                X_prior,
                j_obs,
                rho,
                true_parameters,
                X_posterior_1,
                X_posterior_global,
                rho_min=1e-5,
                atol=1e-2,
            )


@pytest.mark.parametrize("seed", list(range(9)))
def test_distance_based_on_1D_case_with_rho_equal_one(seed):
    N_m = 100  # Number of model parameters (grid points)
    N_e = 50  # Ensemble size
    j_obs = 50  # Index of the single observation

    # Run the test over 9 different sets of alpha.
    nalpha = 10
    number_of_alpha_values = np.arange(1, nalpha, 1, dtype=np.int32)
    for m in number_of_alpha_values:
        alpha_vector = generate_alpha_vector(m)
        inverse_alpha_vector = 1.0 / alpha_vector

        # Consistency requirement for alpha
        # Sum of 1/alpha_vector[i] must be 1
        assert abs(inverse_alpha_vector.sum() - 1.0) < 1e-7

        obs_error_var = 0.01  # Variance of observation error
        true_observations = np.array([1.0])
        C_D = np.array([obs_error_var])
        for iteration in range(len(alpha_vector)):
            alpha_i = np.array([alpha_vector[iteration]])
            if iteration == 0:
                # Draw prior
                rng = np.random.default_rng(seed)
                X_prior = rng.normal(loc=0.0, scale=0.5, size=(N_m, N_e))
                X_previous = X_prior.copy()
                X_previous_global = X_prior.copy()
                X_posterior = X_prior
                X_posterior_global = X_prior
            else:
                X_previous = X_posterior
                X_previous_global = X_posterior_global
            # Predict observations `Y` using the identity model `g(x) = x`
            # We only observe the state at `j_obs`.
            Y = X_previous[[j_obs], :]

            # Set rho to 1 (no localization)
            rho = np.ones((N_m, 1), dtype=np.float64)
            # Must use same seed for both DistanceESMDA and ESMDA initialization
            # to be able to compare
            esmda_distance = DistanceESMDA(
                covariance=C_D, observations=true_observations, alpha=alpha_i, seed=seed
            )
            X_posterior = esmda_distance.assimilate(X=X_previous, Y=Y, rho=rho)

            esmda = ESMDA(
                covariance=C_D, observations=true_observations, alpha=alpha_i, seed=seed
            )
            X_posterior_global = esmda.assimilate(X=X_previous_global, Y=Y)

            diff_X = X_posterior - X_posterior_global
            max_diff = np.max(np.abs(diff_X))
            # Check that the results are equal
            assert max_diff < 1e-8, (
                "DistanceESMDA and ESMDA differs when using rho=1."
                f"Difference is: {max_diff}"
            )


@pytest.mark.parametrize(
    (
        "nparam",
        "nreal",
        "xinc",
        "corr_func_name",
        "relative_corr_range",
        "relative_localization_radius",
        "obs_err_std",
        "tolerance",
        "seed",
        "use_localization",
    ),
    [
        (
            2000,
            100,
            50.0,
            "exponential",
            0.05,
            2.5,
            0.001,
            0.17,
            13579,
            True,
        ),
        (
            2000,
            500,
            50.0,
            "exponential",
            0.05,
            2.5,
            0.001,
            0.11,
            987657,
            True,
        ),
        (
            2000,
            40000,
            50.0,
            "exponential",
            0.05,
            2.5,
            0.001,
            0.05,
            82872857,
            True,
        ),
        (
            2000,
            100,
            50.0,
            "exponential",
            0.05,
            2.5,
            0.001,
            0.17,
            13579,
            False,
        ),
        (
            2000,
            500,
            50.0,
            "exponential",
            0.05,
            2.5,
            0.001,
            0.11,
            987657,
            False,
        ),
        (
            2000,
            40000,
            50.0,
            "exponential",
            0.05,
            2.5,
            0.001,
            0.05,
            82872857,
            False,
        ),
    ],
)
def test_distance_based_localization_on_1D_corr_field(
    nparam: int,
    nreal: int,
    xinc: float,
    corr_func_name: str,
    relative_corr_range: float,
    relative_localization_radius: float,
    obs_err_std: float,
    tolerance: float,
    seed: int,
    use_localization: bool,
) -> None:
    # This test will check that Distance based ESMDA of the posterior mean
    # converge to a value close to the theoretical limit for an experiment
    # where observation errors are small and there are no trends in the
    # prior gaussian field. The theorietical limit when using ESMDA
    # should be exactly the same as the simple kriging estimate,
    # but the theoretical limit when using Distancebased localization
    # will depend on the ratio between the spatial correlation length
    # of the prior field and the localization range.
    # The expected result for distance based localization with ESMDA
    # when number of realizations approach a very large number
    # will correspond to gaussian field with a prior with shorter
    # correlation length than what is specified due to the downscaling
    # of the effect of the observations with distance. (Reduced value
    # of Kalman Gain matrix elements)

    # Prior field mean and stdev
    mean = 0.0
    stdev = 1.0

    corr_range = relative_corr_range * nparam * xinc
    obs_vector = np.array([5.0, -5.0, 4.0, 5.0], dtype=np.float64)
    obs_index_vector = np.array(
        [
            int(nparam / 10),
            int(3 * nparam / 10),
            int(6 * nparam / 10),
            int(9 * nparam / 10),
        ],
        dtype=np.int32,
    )
    nobs = obs_index_vector.shape[0]
    alpha = np.array([1.0])
    obs_error_var = obs_err_std**2  # Variance of observation error
    C_D = np.array([obs_error_var] * nobs)

    # Draw prior ensemble with specified spatial correlation function
    rng = np.random.default_rng(seed)
    X_prior, field_cov_matrix = draw_1D_field(
        mean, stdev, xinc, corr_func_name, corr_range, nparam, nreal, rng
    )

    # Calculate theoretical posterior mean
    # assuming prior mean equal 0.
    # Assume forward model is identity  X = g(X)
    # Use simple kriging estimate for theoretical posterior mean
    mean_predicted_field = predicted_mean_field_values(
        obs_vector, obs_index_vector, field_cov_matrix
    )

    # Predict observations `Y` using the identity model `g(x) = x`
    Y = X_prior[obs_index_vector, :]

    # Using a simple Gaussian decay for rho.
    localization_radius = relative_localization_radius * corr_range

    rho_matrix = np.zeros((nparam, nobs), dtype=np.float64)
    if use_localization:
        for i in range(nobs):
            rho = calculate_rho_1d(
                nparam,
                obs_index_vector[i],
                localization_radius,
                xinc=xinc,
            )
            rho_matrix[:, i] = rho[:, 0]
    else:
        rho_matrix[:, :] = 1.0

    esmda_distance = DistanceESMDA(
        covariance=C_D, observations=obs_vector, alpha=alpha, seed=seed
    )
    X_posterior = esmda_distance.assimilate(X=X_prior, Y=Y, rho=rho_matrix)

    esmda = ESMDA(covariance=C_D, observations=obs_vector, alpha=alpha, seed=seed)
    X_posterior_global = esmda.assimilate(X=X_prior, Y=Y)

    # Mean and stdev of ensemble of posterior field
    X_post_mean = X_posterior.mean(axis=1)
    X_post_mean_global = X_posterior_global.mean(axis=1)

    # Difference with theoretical mean based on simple kriging is calculated.
    # Note that ESMDA in this test should approach the kriging estimate
    # which is the theoretical limit of the posterior mean when nreal -> infinity
    # For DL-ESMDA only when localization range is much larger than the spatial
    # correlation length one can expect the posterior mean approaches the
    # simple kriging estimate in this test. But DL-ESMDA will be closer to
    # the simple kriging estimate for practical purposes where nreal is not large.
    # The tolerance specified in this test is specified such that the estimated
    # standard deviation of the differences between DL-ESMDA and simple kriging
    # estimate is less than the tolerances. Note that when nreal increases, the
    # tolerances can be reduced, but will never approach 0 as long as
    # localization range is finite. However, for ESMDA, the tolerance
    # could in theory approach 0 when nreal approach infinity
    X_diff_local_sk = X_post_mean - mean_predicted_field
    X_diff_global_sk = X_post_mean_global - mean_predicted_field
    X_diff_local_abs = np.abs(X_diff_local_sk)
    X_diff_global_abs = np.abs(X_diff_global_sk)
    est_std_diff_global = X_diff_global_abs.std()
    est_std_diff_local = X_diff_local_abs.std()
    X_local_diff = np.max(X_diff_local_abs)
    X_global_diff = np.max(X_diff_global_abs)

    print(f"Number of real: {nreal}")
    print(f"Max difference using DL ESMDA : {X_local_diff}")
    print(f"Max difference using ordinary ESMDA: {X_global_diff}")
    print(
        "Estimated std of difference between DL and simple kriging: "
        f"{est_std_diff_local}"
    )
    print(
        "Estimated std of difference between Global and simple kriging: "
        f"{est_std_diff_global}"
    )
    if use_localization:
        assert est_std_diff_local <= tolerance, (
            "Estimated std of difference between DL "
            f"and simple kriging estimate is {est_std_diff_local} "
            f"is above specified tolerance {tolerance}"
        )
    else:
        # In this case both ESMDA and DL-ESMDA should give the same result
        # since there is no localization (rho=1 for every element in rho)
        assert abs(est_std_diff_global - est_std_diff_local) < 1e-7


@pytest.mark.parametrize(
    ("nx", "ny", "nz", "num_obs", "nreal", "bytes_per_float"),
    [
        (
            100,
            100,
            50,
            100,
            100,
            8,
        ),
        (
            100,
            100,
            100,
            1000,
            100,
            4,
        ),
        (
            750,
            650,
            500,
            1000,
            100,
            4,
        ),
        (
            100,
            100,
            100,
            10000,
            100,
            4,
        ),
        (
            100,
            100,
            100,
            10000,
            500,
            8,
        ),
        (
            250,
            350,
            500,
            1000,
            500,
            4,
        ),
    ],
)
def test_calc_max_number_of_layers_per_batch_for_distance_localization(
    nx: int,
    ny: int,
    nz: int,
    num_obs: int,
    nreal: int,
    bytes_per_float: int,
) -> None:
    # Test for function that estimate max number of grid layers with field parameters
    # that can be update in one update. No expected number of layers can be defined
    # in advance since this depends on the available memory of the computer that run
    # the test.
    max_nlayer_per_batch = (
        calc_max_number_of_layers_per_batch_for_distance_localization(
            nx, ny, nz, num_obs, nreal, bytes_per_float=bytes_per_float
        )
    )

    memory_used = max_nlayer_per_batch * nx * ny * num_obs * 2 * bytes_per_float / 10**9
    if max_nlayer_per_batch == nz:
        print(f"\nAll {nz} grid layers can be updated in one batch.")
    else:
        print(
            f"\nMax number of layers per batch: {max_nlayer_per_batch}. "
            f"Total number of layers: {nz}"
        )
    print(f"Memory per batch used for RHO and K matrix: {memory_used} GB")


@pytest.mark.parametrize(
    ("nx", "ny", "nz", "nobs", "nreal", "min_nbatch"),
    [
        (
            10,
            15,
            100,
            10,
            100,
            4,
        ),
        (
            10,
            15,
            100,
            10,
            10,
            1,
        ),
        (
            100,
            150,
            1,
            10,
            100,
            1,
        ),
        (
            100,
            150,
            1,
            0,
            100,
            1,
        ),
    ],
)
def test_that_batch_handling_for_update_params_3D_does_not_change_values_when_rho_matrix_is_zero(  # noqa: E501
    nx: int,
    ny: int,
    nz: int,
    nobs: int,
    nreal: int,
    min_nbatch: int,
):
    # This test function checks only the batch split part of the
    # member function 'update_params_3D' of the class DistanceESMDA
    # The prior ensemble will only be copied if the rho matrix is zero

    nparam = nx * ny * nz
    # Define a prior ensemble where each field parameter value for each realization
    # is unique. Then it is easy to check that indexing is correct by comparing
    # input with output
    X_prior = np.arange(nparam * nreal, dtype=np.float64).reshape(nparam, nreal)
    observations = np.zeros(nobs, dtype=np.float64)
    obs_var_vector = np.zeros(nobs, dtype=np.float64)
    Y = np.zeros((nobs, nreal), dtype=np.float64)

    # Set the rho matrix to zero matrix
    rho_2D = np.zeros((nx, ny, nobs), dtype=np.float64)

    # Initialize Distance based localization object using the mock version
    # that does not change any values
    alpha = np.array([1.0])
    dl_smoother = DistanceESMDA(
        covariance=obs_var_vector, observations=observations, alpha=alpha, seed=42
    )

    # No update will be done here and the only thing to happen
    # is that input X_prior is returned
    X_post = dl_smoother.update_params(
        X_prior,
        Y,
        rho_2D,
        nz,
        min_nbatch=min_nbatch,
    )

    # Check that X_prior = X_post
    assert X_prior.shape == X_post.shape
    assert_allclose(X_prior, X_post)


def draw_3D_field(
    mean: float,
    stdev: float,
    xinc: float,
    yinc: float,
    zinc: float,
    nx: int,
    ny: int,
    nz: int,
    nreal: int,
    main_corr_range: float,
    perp_corr_range: float,
    vert_corr_range: float,
    start_seed: int = 42,
    corr_func_name: str = "matern32",
    power: float = 1.9,
    azimuth: float = 0.0,
    dip: float = 0.0,
    write_progress: bool = False,
    use_4_byte_float: bool = False,
) -> npt.NDArray[np.float64]:
    # Draw prior ensemble of 3D field. Ensemble size is nreal
    # Returns prior ensemble drawn and covariance matrix

    # Initialize start seed
    import gaussianfft as grf  # noqa

    grf.seed(start_seed)

    # Define spatial correlation function for gaussian fields
    # to be simulated.
    if corr_func_name == "general_exponential":
        variogram = grf.variogram(
            corr_func_name,
            main_corr_range,
            perp_corr_range,
            vert_corr_range,
            azimuth,
            dip,
            power,
        )
    else:
        variogram = grf.variogram(
            corr_func_name,
            main_corr_range,
            perp_corr_range,
            vert_corr_range,
            azimuth,
            dip,
        )

    nparam = nx * ny * nz
    if use_4_byte_float:
        X_prior = np.zeros((nparam, nreal), dtype=np.float32)
    else:
        X_prior = np.zeros((nparam, nreal), dtype=np.float64)
    # Flatten 3D array in F order
    for real_number in range(nreal):
        if write_progress and real_number % 10 == 0:
            print(f"  Sim real nr: {real_number}")
        field_values = grf.simulate(variogram, nx, xinc, ny, yinc, nz, zinc)
        X_prior[:, real_number] = (
            field_values.reshape((nx, ny, nz), order="F").flatten(order="C") * stdev
            + mean
        )

    return X_prior


def draw_random_obs(rng, nobs, nx, ny, nz, obs_err_std):
    nparam = nx * ny * nz
    assert nobs < nparam
    # Define a grid resolution with specified size
    xinc = 50.0
    yinc = 50.0
    zinc = 1.0
    right_handed_grid_indexing = True
    # Draw some observation values (Use same seed every time)
    observations = rng.normal(loc=0.5, scale=0.05, size=nobs)

    # Choose some observation values, errors and positions
    # Draw some position of the observations, ensure no observations at same position
    # since the response values are equal to field values (only one response variable).
    # Multiple response values can have same position.

    # Draw i index, j_index, k_index for grid cell to be used as observed.
    if nobs == 1:
        obs_xpos = np.array([(nx / 2) * xinc])
        obs_ypos = np.array([(ny / 2) * yinc])
        obs_zpos = np.array([(nz / 2) * zinc])
        i_indices = np.array([int(nx / 2)])
        j_indices = np.array([int(ny / 2)])
        k_indices = np.array([int(nz / 2)])
        unique_obs_indices = k_indices + j_indices * nz + i_indices * nz * ny
    else:
        if nobs > nparam:
            raise ValueError(
                "Cannot draw more observations than number of parameters in the field"
            )
        unique_obs_indices = rng.choice(range(nparam), size=nobs, replace=False)
        unique_obs_indices = np.sort(unique_obs_indices)
        i_indices = (unique_obs_indices // (nz * ny)).astype(int)
        j_indices = ((unique_obs_indices % (nz * ny)) // nz).astype(int)
        k_indices = (unique_obs_indices % nz).astype(int)
        if right_handed_grid_indexing:
            # Right-handed grid indexing
            obs_ypos = ((ny - j_indices - 1) + 0.5) * yinc
        else:
            obs_ypos = (j_indices + 0.5) * yinc

        obs_xpos = (i_indices + 0.5) * xinc
        obs_zpos = (k_indices + 0.5) * zinc

    # Set observation error
    obs_var_vector = np.zeros(nobs, dtype=np.float64)
    obs_var_vector[:] = obs_err_std**2
    return (
        observations,
        obs_var_vector,
        obs_xpos,
        obs_ypos,
        obs_zpos,
        i_indices,
        j_indices,
        k_indices,
        unique_obs_indices,
    )


@pytest.mark.skipif(
    sys.version_info >= (3, 13),
    reason="gaussianfft not available on Python >= 3.13",
)
@pytest.mark.parametrize(
    (
        "nx",
        "ny",
        "nz",
        "nobs",
        "nreal",
        "field_mean",
        "field_std",
        "rel_corr_length",
        "obs_err_std",
        "rel_localization_range",
        "case_with_some_zero_variance_field_params",
        "seed",
    ),
    [
        (
            3,
            4,
            5,
            1,
            100,
            0.0,
            1.0,
            0.2,
            0.001,
            0.2,
            False,
            9984356,
        ),
        (
            5,
            4,
            3,
            0,
            100,
            0.0,
            1.0,
            0.1,
            0.01,
            0.1,
            False,
            123456,
        ),
        (
            5,
            4,
            3,
            3,
            100,
            0.0,
            1.0,
            0.1,
            0.01,
            0.1,
            True,
            8765,
        ),
        (
            3,
            5,
            1,
            1,
            100,
            0.0,
            1.0,
            0.3,
            0.001,
            0.3,
            False,
            9984356,
        ),
        (
            3,
            5,
            1,
            1,
            100,
            0.0,
            1.0,
            0.3,
            0.001,
            0.3,
            True,
            9984356,
        ),
    ],
)
def test_update_params_3D(
    snapshot,
    nx: int,
    ny: int,
    nz: int,
    nobs: int,
    nreal: int,
    field_mean: float,
    field_std: float,
    rel_corr_length: float,
    obs_err_std: float,
    rel_localization_range: float,
    case_with_some_zero_variance_field_params: bool,
    seed: int,
):
    # The field parameter is assumed to belong to a box grid
    # with specified nx,ny,nz and grid increments xinc,yinc,zinc
    # The observation position is assumed to be within the same coordinate
    # system as the grid. The grid cell center point coordinates are
    # x[i,j,k] = xinc * (i + 0.5)  i=0,.. nx-1
    # y[i,j,k] = yinc * (j + 0.5)  j=0,.. ny-1
    # z[i,j,k] = zinc * (k + 0.5)  k=0,.. nz-1
    # Z coordinate is not used when calculating RHO matrix, but is used here
    # to define observation values.
    xinc = 50.0
    yinc = 50.0
    zinc = 1.0
    xlength = xinc * nx
    ylength = yinc * ny
    zlength = zinc * nz
    corr_range = max(xlength, ylength) * rel_corr_length
    vert_range = zlength * rel_corr_length
    fraction_of_field_values_with_zero_variance = 0.1

    # Draw prior gaussian fields with spatial correlations
    X_prior = draw_3D_field(
        field_mean,
        field_std,
        xinc,
        yinc,
        zinc,
        nx,
        ny,
        nz,
        nreal,
        corr_range,
        corr_range,
        vert_range,
        seed,
        corr_func_name="gaussian",
    )

    X_prior_3D = X_prior.reshape((nx, ny, nz, nreal))
    rng = np.random.default_rng(seed)
    nparam = nx * ny * nz
    (
        observations,
        obs_var_vector,
        obs_xpos,
        obs_ypos,
        obs_zpos,
        i_indices,
        j_indices,
        k_indices,
        unique_obs_indices,
    ) = draw_random_obs(rng, nobs, nx, ny, nz, obs_err_std)

    # Choose localization range around each obs
    typical_field_size = min(xlength, ylength)
    obs_main_range = np.zeros(nobs, dtype=np.float64)
    obs_main_range[:] = typical_field_size * rel_localization_range
    obs_perp_range = np.zeros(nobs, dtype=np.float64)
    obs_perp_range[:] = typical_field_size * rel_localization_range
    obs_anisotropy_angle = np.zeros(nobs, dtype=np.float64)

    # Calculate rho_for one layer
    rho_2D = calc_rho_for_2d_grid_layer(
        nx,
        ny,
        xinc,
        yinc,
        obs_xpos,
        obs_ypos,
        obs_main_range,
        obs_perp_range,
        obs_anisotropy_angle,
        right_handed_grid_indexing=True,
    )
    # Set responses for each observation equal to the X_prior for simplicity
    # (Forward model is identity Y = X in observation points + small random noise)
    # Note cannot have observations with response with 0 variance
    add_response_variability = rng.normal(loc=0, scale=0.01, size=(nreal, nobs))
    Y = X_prior_3D[i_indices, j_indices, k_indices, :] + add_response_variability.T

    if case_with_some_zero_variance_field_params:
        # Set same field value in all realizations for selected grid cells
        # They should not be updated since the ensemble variance is 0 for those
        # values. The selected grid cells must not be selected as observed.
        nconst_values = int(
            fraction_of_field_values_with_zero_variance * nparam
        )  # Choose a portion of field values to be constant
        unique_indices_const = rng.choice(
            range(nparam), size=nconst_values, replace=False
        )
        print(f"Number of selected grid indices: {len(unique_indices_const)}")
        # Reject indices corresponding to observed grid cells
        unique_indices_const = np.setdiff1d(unique_indices_const, unique_obs_indices)
        # Usually inactive cell values can be set to 0 for all realizations,
        # but choose something else here for visualization purpose
        X_prior[unique_indices_const, :] = 10.0
        print(f"Number of field values with 0 variance: {len(unique_indices_const)}")

    # Initialize Distance based localization object
    alpha = np.array([1.0])
    dl_smoother = DistanceESMDA(
        covariance=obs_var_vector, observations=observations, alpha=alpha, seed=rng
    )

    # Call the function to be tested here
    # Note that no field values with 0 variance is removed
    # from the update calculation in this function,
    # but that is ok since there will be no change of
    # field parameters with 0 variance anyway.
    X_post = dl_smoother.update_params(
        X_prior,
        Y,
        rho_2D,
        nz,
    )
    X_post_3D = X_post.reshape((nx, ny, nz, nreal))
    # Mean and stdev over ensemble of field parameters
    #    X_prior_mean_3D = X_prior_3D.mean(axis=3)
    X_post_mean_3D = X_post_3D.mean(axis=3)
    #    X_prior_stdev_3D = X_prior_3D.std(axis=3)
    X_post_stdev_3D = X_post_3D.std(axis=3)
    # Difference between post and prior mean and stdev
    #    X_diff_mean_3D = X_post_mean_3D - X_prior_mean_3D
    #    X_diff_stdev_3D = X_post_stdev_3D - X_prior_stdev_3D

    # Check results
    #    snapshot.assert_match(str(X_diff_mean_3D) + "\n", "X_diff_mean_3D.txt")
    #    snapshot.assert_match(str(X_diff_stdev_3D) + "\n", "X_diff_stdev_3D.txt")
    snapshot.assert_match(str(X_post_mean_3D) + "\n", "X_post_mean_3D.txt")
    snapshot.assert_match(str(X_post_stdev_3D) + "\n", "X_post_stdev_3D.txt")


if __name__ == "__main__":
    pytest.main(
        args=[
            __file__,
            "-v",
        ]
    )
