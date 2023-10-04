import numpy as np
import pytest
import scipy as sp
from scipy.stats import linregress

import iterative_ensemble_smoother as ies
from iterative_ensemble_smoother import SIES


class TestSIESObjective:
    @pytest.mark.parametrize("seed", list(range(99)))
    @pytest.mark.parametrize("inversion", SIES.inversion_funcs.keys())
    def test_that_sies_objective_function_decreases(self, seed, inversion):
        rng = np.random.default_rng(seed)

        ensemble_size = 50
        num_params = 25
        num_obs = num_params
        X = rng.normal(size=(num_params, ensemble_size))

        C_0 = rng.normal(size=(num_obs, ensemble_size))
        C_1 = rng.normal(size=(num_obs, num_params))
        C_2 = rng.normal(size=(num_obs, num_params))

        def G(X):
            return C_0 + C_1 @ X + 0.1 * C_2 @ X**3

        covariance = 1 + np.exp(rng.normal(size=num_obs))
        observations = rng.normal(np.zeros(num_obs), covariance)

        smoother = SIES(
            parameters=X,
            covariance=covariance,
            observations=observations,
            inversion=inversion,
            seed=rng,
        )

        Y = G(X)
        objective_before = smoother.evaluate_objective(W=smoother.W, Y=Y)

        # One iteration
        X_i = smoother.sies_iteration(Y, 0.5)
        Y_i = G(X_i)
        objective_after = smoother.evaluate_objective(W=smoother.W, Y=Y_i)

        assert objective_after <= objective_before

    @pytest.mark.parametrize("seed", list(range(99)))
    def test_that_sies_objective_function_decreases_with_many_iterations(self, seed):
        rng = np.random.default_rng(seed)

        # Create a problem instance
        ensemble_size = 5
        num_params = 10
        num_obs = num_params
        X = rng.normal(size=(num_params, ensemble_size))

        C_0 = rng.normal(size=(num_obs, ensemble_size))
        C_1 = rng.normal(size=(num_obs, num_params))
        C_2 = rng.normal(size=(num_obs, num_params))

        def G(X):
            # A slightly non-linear function
            return C_0 + C_1 @ X + 0.1 * C_2 @ X**3

        covariance = 1 + np.exp(rng.normal(size=num_obs))
        observations = rng.normal(np.zeros(num_obs), covariance)

        # F = rng.normal(size=(num_obs, num_obs))
        # observation_errors = F.T @ F

        smoother = SIES(
            parameters=X,
            covariance=covariance,
            observations=observations,
        )

        X_i = np.copy(X)

        # Initial evaluation
        Y_i = G(X_i)
        objective_before = smoother.evaluate_objective(W=smoother.W, Y=Y_i)

        # Iterations
        for iteration in range(9):
            # One iteration
            X_i = smoother.sies_iteration(Y_i, 0.01)
            Y_i = G(X_i)

            # Evaluate objective
            objective_after = smoother.evaluate_objective(W=smoother.W, Y=Y_i)

            # Check and update
            assert objective_after <= objective_before
            objective_before = objective_after

    def test_line_search(self):
        """This is a demo (or test) of how line search can be implemented."""
        rng = np.random.default_rng(42)

        ensemble_size = 5
        num_params = 10
        num_obs = num_params
        X = rng.normal(size=(num_params, ensemble_size))

        C_0 = rng.normal(size=(num_obs, ensemble_size))
        C_1 = rng.normal(size=(num_obs, num_params))
        C_2 = rng.normal(size=(num_obs, num_params))

        def G(X):
            return C_0 + C_1 @ X + 0.1 * C_2 @ X**3

        covariance = 1 + np.exp(rng.normal(size=num_obs))
        observations = rng.normal(np.zeros(num_obs), covariance)

        F = rng.normal(size=(num_obs, num_obs))
        covariance = F.T @ F

        smoother = SIES(
            parameters=X,
            covariance=covariance,
            observations=observations,
        )

        def line_search(smoother, g):
            """Demo implementation of line search."""
            BACKTRACK_ITERATIONS = 10
            self = smoother

            # Initial evaluation
            X_i = np.copy(self.X)
            N = self.X.shape[1]  # Ensemble members
            Y_i = g(X_i)

            # Perform a Gauss-Newton iteration
            while True:
                objective_before = self.evaluate_objective(W=self.W, Y=Y_i)

                # Perform a line-search iteration
                for p in range(BACKTRACK_ITERATIONS):
                    step_length = pow(1 / 2, p)  # 1, 0.5, 0.25, 0.125, ...
                    print(f"Step length: {step_length}")

                    # Make a step with the given step length
                    proposed_W = self.propose_W(Y_i, step_length)

                    # Evaluate at the new point
                    proposed_X = self.X + self.X @ proposed_W / np.sqrt(N - 1)
                    proposed_Y = g(proposed_X)
                    objective = self.evaluate_objective(W=proposed_W, Y=proposed_Y)

                    # Accept the step
                    if objective <= objective_before:
                        print(f"Accepting. {objective} <= {objective_before}")
                        # Update variables
                        self.W = proposed_W
                        X_i = proposed_X
                        Y_i = proposed_Y
                        break
                    else:
                        print(f"Rejecting. {objective} > {objective_before}")

                # If no break was triggered in the for loop, we never accepted
                else:
                    msg = (
                        "Terminating. No improvement"
                        + f"after {BACKTRACK_ITERATIONS} iterations."
                    )
                    print(msg)
                    return X_i

                yield X_i

        # Initial evaluation
        objective_before = smoother.evaluate_objective(W=smoother.W, Y=G(smoother.X))

        for X_i in line_search(smoother, G):
            objective = smoother.evaluate_objective(W=smoother.W, Y=G(X_i))
            print(objective_before, objective)


class TestSIESNumerics:
    def test_that_large_responses_are_handled(self):
        # See: https://github.com/equinor/iterative_ensemble_smoother/issues/83
        # Creating response matrix with large outlier that will
        # lead to NaNs.
        parameters = np.array([[1, 2, 3]], dtype=float)
        responses = np.array([[1, 1, 1e12], [1, 10, 100]], dtype=float)
        covariance = np.array([1, 2], dtype=float)
        observations = np.array([10, 20], dtype=float)

        # ============== SUBSPACE EXACT (RAISES) =================

        smoother = SIES(
            parameters=parameters,
            covariance=covariance,
            observations=observations,
            inversion="subspace_exact",
        )

        # Exact inversion does not work
        with pytest.raises(
            np.linalg.LinAlgError,
            match="Matrix is singular.",
        ):
            smoother.sies_iteration(responses, step_length=1.0)

        # ================== DIRECT (WARNS) =====================
        smoother = SIES(
            parameters=parameters,
            covariance=covariance,
            observations=observations,
            inversion="direct",
        )

        with pytest.warns(sp.linalg.LinAlgWarning, match="Ill-conditioned matrix"):
            smoother.sies_iteration(responses, step_length=1.0)

        # ============ SUBSPACE PROJECTED ===============
        smoother = SIES(
            parameters=parameters,
            covariance=covariance,
            observations=observations,
            inversion="subspace_projected",
        )

        # Exact inversion does not work
        smoother.sies_iteration(responses, step_length=1.0)


def test_version_attribute() -> None:
    assert ies.__version__ != "unknown version"
    assert ies.version_tuple != (0, 0, "unknown version", "unknown commit")


class TestSIESTheory:
    @pytest.mark.parametrize("number_of_realizations", [300, 500])
    def test_that_es_update_for_a_linear_model_follows_theory(
        self, number_of_realizations
    ):
        """This test:
        - Tests that the posterior covariance is
          between the prior and maximum likelihood
        - Tests that as we increase belief in the observations,
          we move closer to the maximum likelihood and further from the prior
        """
        rng = np.random.default_rng(42)

        # The following tests follow the
        # posterior properties described in
        # https://ert.readthedocs.io/en/latest/theory/ensemble_based_methods.html#kalman-posterior-properties
        a_true = 1.0
        b_true = 5.0
        number_of_observations = 45

        class LinearModel:
            rng = np.random.default_rng(42)

            def __init__(self, a, b):
                self.a = a
                self.b = b

            @classmethod
            def random(cls):
                a_std = 2.0
                b_std = 2.0
                # Priors with bias
                a_bias = 0.5 * a_std
                b_bias = -0.5 * b_std

                return cls(
                    cls.rng.normal(a_true + a_bias, a_std),
                    cls.rng.normal(b_true + b_bias, b_std),
                )

            def eval(self, x):
                return self.a * x + self.b

        true_model = LinearModel(a_true, b_true)

        ensemble = [LinearModel.random() for _ in range(number_of_realizations)]

        A = np.array(
            [
                [realization.a for realization in ensemble],
                [realization.b for realization in ensemble],
            ]
        )
        mean_prior = np.mean(A, axis=1)

        # We use time as the x-axis and observations are at
        # t=0,1,2...number_of_observations
        times = np.arange(number_of_observations)

        S = np.array([[realization.eval(t) for realization in ensemble] for t in times])

        # When observations != true model, then ml estimates != true parameters.
        # This gives both a more advanced and realistic test. Standard normal
        # N(0,1) noise is added to obtain this. The randomness ensures we are not
        # gaming the test. But the difference could in principle be any non-zero
        # scalar.
        observations = np.array(
            [true_model.eval(t) + rng.standard_normal() for t in times]
        )

        # Leading to fixed Maximum likelihood estimate.
        # It will equal true values when observations are sampled without noise.
        # It will also stay the same over beliefs.
        results = linregress(times, observations)
        maximum_likelihood = np.array([results.slope, results.intercept])

        previous_mean_posterior = mean_prior

        # numerical precision tolerance
        epsilon = 1e-2

        # We iterate with an increased belief in the observations
        for error in [10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01]:
            # An important point here is that we do not iteratively
            # update A, but instead, observations stay the same and
            # we increase our belief in the observations
            # As A is update inplace, we have to reset it.
            A = np.array(
                [
                    [realization.a for realization in ensemble],
                    [realization.b for realization in ensemble],
                ]
            )

            smoother = ies.SIES(
                parameters=A,
                covariance=np.full(observations.shape, error) ** 2,
                observations=observations,
            )

            A_posterior = smoother.sies_iteration(S, step_length=1.0)

            mean_posterior = np.mean(A_posterior, axis=1)

            # All posterior estimates lie between prior and maximum likelihood estimate
            assert (
                np.linalg.norm(mean_posterior - maximum_likelihood)
                - np.linalg.norm(mean_prior - maximum_likelihood)
                < epsilon
            )
            assert (
                np.linalg.norm(mean_prior - mean_posterior)
                - np.linalg.norm(mean_prior - maximum_likelihood)
                < epsilon
            )

            # Posterior parameter estimates improve with increased trust in observations
            assert (
                np.linalg.norm(mean_posterior - maximum_likelihood)
                - np.linalg.norm(previous_mean_posterior - maximum_likelihood)
                < epsilon
            )

            previous_mean_posterior = mean_posterior

        # At strong beliefs, we should be close to the maximum likelihood estimate
        assert np.all(
            np.linalg.norm(previous_mean_posterior - maximum_likelihood) < epsilon
        )

    @pytest.mark.parametrize("seed", list(range(9)))
    @pytest.mark.parametrize("ensemble_size", [100, 200])
    def test_that_posterior_is_between_prior_and_maximum_likelihood(
        self, ensemble_size, seed
    ):
        rng = np.random.default_rng(seed)

        # Problem size
        num_params = 10
        num_responses = 100

        # Create a linear mapping g
        A = rng.normal(size=(num_responses, num_params))

        def g(X):
            return A @ X

        # Inputs and outputs
        # The prior is given by N(0, 1)
        # The maximum likelihood estimate around N(10, 1)

        prior = rng.normal(size=(num_params, ensemble_size))
        x_true = rng.normal(size=num_params) + 10
        observations = A @ x_true + rng.normal(size=(num_responses))

        # Compute the maximum likelihood estimate using linear regression
        x_max_likelihood, *_ = np.linalg.lstsq(A, observations, rcond=None)

        # Iterate over increased belief in observations
        distance_posterior_ml_previous = 1e6
        for standard_deviation in [1e2, 1e1, 1e0, 1e-1, 1e-2]:
            covariance = np.ones(num_responses) * standard_deviation**2

            # Create smoother
            smoother = ies.SIES(
                parameters=prior,
                covariance=covariance,
                observations=observations,
                seed=rng,
            )

            # Run through the model
            responses = g(prior)

            # Only the living simulations get passed
            posterior = smoother.sies_iteration(responses, step_length=0.5)

            posterior_mean = posterior.mean(axis=1)
            prior_mean = prior.mean(axis=1)

            # The posterior should be between the prior and the ML estimate
            #     prior <----- d1 -----> posterior <------ d2 ------> ML
            #           <------------------- d3 -------------------->
            # In other words, for distances d1, d2 and d3: d3 >= d1 and d2 >= d1

            distance_prior_ml = np.linalg.norm(prior_mean - x_max_likelihood)
            distance_prior_posterior = np.linalg.norm(prior_mean - posterior_mean)
            distance_posterior_ml = np.linalg.norm(posterior_mean - x_max_likelihood)

            assert distance_prior_ml >= distance_prior_posterior
            assert distance_prior_ml >= distance_posterior_ml

            # As the observation belief increases, the posterior mean should converge
            # to the maximum likelihood estimate
            assert distance_posterior_ml <= distance_posterior_ml_previous
            distance_posterior_ml_previous = distance_posterior_ml

    @pytest.mark.parametrize("seed", list(range(25)))
    @pytest.mark.parametrize("ensemble_size", [5, 25])
    def test_that_many_small_steps_equals_one_large_step_on_linear_problem(
        self, seed, ensemble_size
    ):
        rng = np.random.default_rng(seed)

        # Problem size
        num_params = 10
        num_responses = 100

        # Create a linear mapping g
        A = rng.normal(size=(num_responses, num_params))

        def g(X):
            return A @ X

        # Inputs and outputs
        parameters = rng.normal(size=(num_params, ensemble_size))
        x_true = np.linspace(-5, 5, num=num_params)
        observations = A @ x_true + rng.normal(size=num_responses)
        covariance = np.ones(num_responses)

        # Smoother - single step with full step length
        smoother = ies.SIES(
            parameters=parameters,
            covariance=covariance,
            observations=observations,
            seed=42,
        )
        X_posterior_ES = smoother.sies_iteration(g(parameters), step_length=1.0)

        # Smoother - several shorter steps
        smoother = ies.SIES(
            parameters=parameters,
            covariance=covariance,
            observations=observations,
            seed=42,
        )
        X_i = np.copy(parameters)
        for iteration in range(18):
            X_i = smoother.sies_iteration(g(X_i), step_length=0.66)

        # The posterior is the same for a linear model
        # This is not true for non-linear models
        assert np.allclose(X_posterior_ES, X_i)

    @pytest.mark.parametrize("seed", list(range(25)))
    def test_that_posterior_covariance_is_smaller_than_prior(self, seed):
        rng = np.random.default_rng(seed)
        covariance_inflation = seed + 1

        # Problem size
        num_params = 10
        num_responses = 100
        ensemble_size = 25

        # Create a linear mapping g
        A = rng.normal(size=(num_responses, num_params))

        def g(X):
            return A @ X

        # Inputs and outputs
        prior = rng.normal(size=(num_params, ensemble_size))
        x_true = np.linspace(-5, 5, num=num_params)
        observations = A @ x_true + rng.normal(size=num_responses)
        covariance = np.ones(num_responses) * covariance_inflation

        # Smoother - single step with full step length
        smoother = ies.SIES(
            parameters=prior,
            covariance=covariance,
            observations=observations,
            seed=seed,
        )
        posterior = smoother.sies_iteration(g(prior), step_length=1.0)

        # We are more certain after assimilating, so the covariance decreases
        assert np.linalg.det(np.cov(posterior)) < np.linalg.det(np.cov(prior))

        # Posterior mean is closer to true value than prior mean
        posterior_mean = posterior.mean(axis=1)
        prior_mean = prior.mean(axis=1)
        assert np.linalg.norm(posterior_mean - x_true) < np.linalg.norm(
            prior_mean - x_true
        )

    @pytest.mark.parametrize("inversion", list(ies.SIES.inversion_funcs.keys()))
    @pytest.mark.parametrize("variance", [1, 2, 5, 9])
    def test_that_update_correctly_multiples_gaussians_in_1D(self, inversion, variance):
        """Test that SIES follows the theory on the Guass-linear case.
        See section "2.3.3 Bayes’ theorem for Gaussian variables" in the book
        "Pattern Recognition and Machine Learning" by Bishop (2006) for more info.
        Also, see Section 8.1.8 in "the matrix cookbook".

        NB! This test is potentially flaky because of finite ensemble size.

        Assume p(x) is N(mu=0, Sigma=2I) and p(y|x) is N(mu=y, Sigma=2I).
        Multiplying these together, we get that p(x|y) is N(mu=y/2, Sigma=I).
        Note that Sigma is a covariance matrix.

        Here we use this property, and assume that the forward model is the identity.
        """
        N = 1000

        rng = np.random.default_rng(variance)

        # X ~ N(0, I * var)
        X = rng.normal(scale=np.sqrt(variance), size=(1, N))

        # Assuming forward model is the identity
        Y = np.copy(X)

        obs_val = 10.0
        covariance = np.ones(1) * variance
        observations = np.ones(1) * obs_val
        smoother = ies.SIES(
            parameters=X, covariance=covariance, observations=observations, seed=rng
        )

        X_posterior = smoother.sies_iteration(
            Y,
            step_length=1.0,
        )

        # Posterior mean should end up between 0 and obs_val (in the middle),
        # when the covariances are equal and the mapping is the identity.
        posterior_mean = X_posterior.mean(axis=1)
        assert np.allclose(posterior_mean, obs_val / 2, rtol=0.1)

        # Posterior covariance
        posterior_variance = 1 / (1 / variance + 1 / variance)
        assert np.allclose(np.var(X_posterior), posterior_variance, rtol=0.1)

    @pytest.mark.parametrize("inversion", list(ies.SIES.inversion_funcs.keys()))
    @pytest.mark.parametrize("variance", [0.25, 0.5, 1, 2, 4])
    def test_that_update_correctly_multiples_gaussians(self, inversion, variance):
        """Test that SIES follows the theory on the Guass-linear case.
        See section "2.3.3 Bayes’ theorem for Gaussian variables" in the book
        "Pattern Recognition and Machine Learning" by Bishop (2006) for more info.
        Also, see Section 8.1.8 in "the matrix cookbook".

        NB! This test is potentially flaky because of finite ensemble size.

        Assume p(x) is N(mu=0, Sigma=2I) and p(y|x) is N(mu=y, Sigma=2I).
        Multiplying these together, we get that p(x|y) is N(mu=y/2, Sigma=I).
        Note that Sigma is a covariance matrix.

        Here we use this property, and assume that the forward model is the identity.
        """
        N = 1000
        nparam = 3

        rng = np.random.default_rng(int(variance * 100))

        # X ~ N(0, I * var)
        X = rng.normal(scale=np.sqrt(variance), size=(nparam, N))

        # Assuming forward model is the identity
        Y = np.copy(X)

        obs_val = 10.0
        covariance = np.ones(nparam) * variance
        observations = np.ones(nparam) * obs_val
        smoother = ies.SIES(
            parameters=X, covariance=covariance, observations=observations, seed=rng
        )

        X_posterior = smoother.sies_iteration(
            Y,
            step_length=1.0,
        )

        # Posterior mean should end up in the middle
        posterior_mean = X_posterior.mean(axis=1)
        assert np.allclose(posterior_mean, obs_val / 2, rtol=0.1)

        theoretical_variance = 1 / (1 / variance + 1 / variance)
        assert np.allclose(
            np.cov(X_posterior), np.eye(nparam) * theoretical_variance, atol=0.33
        )


@pytest.mark.parametrize(
    "ensemble_size,num_params,linear",
    [
        pytest.param(100, 3, True, id="No projection because linear model"),
        pytest.param(
            100,
            3,
            False,
            id="Project because non-linear and num_params < ensemble_size - 1",
        ),
        pytest.param(3, 100, True, id="No projection because linear model"),
        pytest.param(
            3, 100, False, id="No projection because num_params > ensemble_size - 1"
        ),
        pytest.param(
            10, 10, True, id="No projection because num_params > ensemble_size - 1"
        ),
    ],
)
def test_that_global_es_update_is_identical_to_local(ensemble_size, num_params, linear):
    rng = np.random.default_rng(42)

    num_obs = num_params
    X = rng.normal(size=(num_params, ensemble_size))
    Y = X if linear else np.power(X, 2)
    covariance = np.exp(rng.uniform(size=num_obs))
    observations = rng.normal(np.zeros(num_params), covariance)

    # First update a subset or batch of parameters at once
    # and and then loop through each remaining parameter and
    # update it separately.
    # This is meant to emulate a configuration where users
    # want to update, say, a field parameter separately from scalar parameters.

    # A single global update
    smoother = ies.SIES(
        parameters=X, covariance=covariance, observations=observations, seed=1
    )
    X_ES_global = smoother.sies_iteration(Y, step_length=1.0)

    # Update each parameter in turn
    smoother = ies.SIES(
        parameters=X, covariance=covariance, observations=observations, seed=1
    )
    W = smoother.propose_W(Y, step_length=1.0)

    # Create an empty matrix, then fill it in row-by-row
    X_ES_local = np.empty(shape=(num_params, ensemble_size))
    for i in range(X.shape[0]):
        # This is line 9 in algorithm 1, using W
        X_ES_local[i, :] = X[i, :] + X[i, :] @ W / np.sqrt(ensemble_size - 1)

    assert np.allclose(X_ES_global, X_ES_local)


@pytest.mark.parametrize("seed", list(range(10)))
def test_that_ies_runs_with_failed_realizations(seed):
    """When the forward model is a simulation being run in parallel on clusters,
    some of the jobs (realizations) might fail. In that case we know which
    realizations have failed, and we wish to keep assimilating data with the
    remaining realizations.

    We assume that once a realization is "dead", it never wakes up again.
    """

    rng = np.random.default_rng(seed)

    # Problem size
    ensemble_size = 300
    num_params = 20
    num_responses = 200

    # Create a linear mapping g
    A = rng.normal(size=(num_responses, num_params))

    def g(X):
        return A @ X

    # Inputs and outputs
    parameters = rng.normal(size=(num_params, ensemble_size))
    x_true = np.linspace(-5, 5, num=num_params)
    observations = A @ x_true + rng.normal(size=(num_responses))
    covariance = np.ones(num_responses)

    # Create smoother
    smoother = ies.SIES(
        parameters=parameters,
        covariance=covariance,
        observations=observations,
        seed=rng,
    )

    ensemble_mask = np.ones(ensemble_size, dtype=bool)
    X_i = np.copy(parameters)
    for iteration in range(10):
        # Assume we do a model run, but some realizations (ensemble members)
        # fail for some reason. We know which realizations failed.
        # With 10 iterations and p=[0.9, 0.1], around 1/2 make it till the end.
        simulation_ok = rng.choice([True, False], size=ensemble_size, p=[0.95, 0.05])
        ensemble_mask = np.logical_and(ensemble_mask, simulation_ok)

        # Run through the model
        responses = g(X_i)

        # Only the living simulations get passed
        X_i = smoother.sies_iteration(
            responses[:, ensemble_mask], ensemble_mask=ensemble_mask, step_length=0.05
        )

        # The full matrix is returned, but only X_i[:, ensemble_mask]
        assert X_i.shape == parameters.shape

    mean_posterior_all = X_i.mean(axis=1)
    mean_posterior = X_i[:, ensemble_mask].mean(axis=1)

    # Using only the realizations that lived through all iterations is better
    # than using the ones who died along the way
    assert np.linalg.norm(mean_posterior - x_true) < np.linalg.norm(
        mean_posterior_all - x_true
    )

    # The posterior is closer to the true value than the prior
    assert np.linalg.norm(mean_posterior - x_true) < np.linalg.norm(
        parameters.mean(axis=1) - x_true
    )

    # The posterior mean is reasonably close - or at least nothing crazy happens
    assert np.all(np.abs(mean_posterior - x_true) < 3)


@pytest.mark.parametrize("seed", list(range(9)))
def test_that_subspaces_are_still_used_as_original_realizations_fail(seed):
    """When the forward model is a simulation being run parallel on clusters,
    some of the jobs (realizations) might fail.

    Consider the case when the prior is the identity matrix I.
    N = n = 10. In each iteration, another ensemble member "dies."
    After 7 iteations, only the three right-most columns are still alive.
    However, that does NOT mean that the posterior lies in the subspace
    spanned by the three most right-most columns in the prior, since
    in earlier iterations the algorithm is able to explore other subspaces
    before realizations die out.
    """

    rng = np.random.default_rng(seed)

    # Problem size
    ensemble_size = num_params = 10
    num_responses = 100

    # Create a linear mapping g
    A = rng.normal(size=(num_responses, num_params))

    def g(X):
        return A @ X

    # Inputs and outputs
    parameters = np.identity(num_params)  # Identity
    x_true = np.arange(num_params)
    observations = A @ x_true + rng.normal(size=(num_responses))
    covariance = np.ones(num_responses)

    # Create smoother
    smoother = ies.SIES(
        parameters=parameters,
        covariance=covariance,
        observations=observations,
        seed=rng,
    )

    X_i = np.copy(parameters)
    ensemble_mask = np.ones(ensemble_size, dtype=bool)
    for iteration in range(1, 8):
        ensemble_mask[:iteration] = False

        # Only the living simulations get passed
        responses = g(X_i)
        X_i = smoother.sies_iteration(
            responses[:, ensemble_mask], ensemble_mask=ensemble_mask, step_length=0.2
        )

    # First column is the identity, it was never updated
    assert X_i[0, 0] == 1
    assert np.allclose(X_i[1:, 0], 0)

    # Variables keep getting updated
    assert not np.allclose(X_i[1, 1:], X_i[1, 1])
    assert not np.allclose(X_i[2, 2:], X_i[2, 2])
    assert not np.allclose(X_i[3, 3:], X_i[3, 3])

    # If the first 7 columns were not used, then the top 7 rows would be
    # identitally zero, but that is not the case
    assert not np.allclose(X_i[:7, ensemble_mask], 0)


@pytest.mark.parametrize("ensemble_size", [5, 15, 50])
def test_that_full_ensemble_mask_is_equal_to_no_ensemble_mask(ensemble_size):
    """When the forward model is a simulation being run parallel on clusters,
    some of the jobs (realizations) might fail.

    Consider the case when the prior is the identity matrix I.
    N = n = 10. In each iteration, another ensemble member "dies."
    After 7 iteations, only the three right-most columns are still alive.
    However, that does NOT mean that the posterior lies in the subspace
    spanned by the three most right-most columns in the prior, since
    in earlier iterations the algorithm is able to explore other subspaces
    before realizations die out.
    """

    rng = np.random.default_rng(42)

    # Problem size
    num_params = 10
    num_responses = 25

    # Create a linear mapping g
    A = rng.normal(size=(num_responses, num_params))

    def g(X):
        return A @ X

    # Inputs and outputs
    parameters = rng.normal(size=(num_params, ensemble_size))
    x_true = rng.normal(size=(num_params))
    observations = A @ x_true + rng.normal(size=(num_responses))
    covariance = np.ones(num_responses)

    X_i = np.copy(parameters)
    responses = g(X_i)
    ensemble_mask = np.ones(ensemble_size, dtype=bool)

    # Create smoother with the intention of using ensemble mask
    smoother1 = ies.SIES(
        parameters=parameters,
        covariance=covariance,
        observations=observations,
        seed=1,
    )
    assert np.allclose(responses[:, ensemble_mask], responses)
    X_i_mask = smoother1.sies_iteration(
        responses[:, ensemble_mask], ensemble_mask=ensemble_mask, step_length=0.2
    )

    # Create smoother new smoother, since the iteration on the previous one
    # updates the internal state
    smoother2 = ies.SIES(
        parameters=parameters,
        covariance=covariance,
        observations=observations,
        seed=1,
    )
    X_i = smoother2.sies_iteration(responses, step_length=0.2)

    assert np.allclose(X_i_mask, X_i)


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("inversion", list(ies.SIES.inversion_funcs.keys()))
def test_that_diagonal_and_dense_covariance_return_the_same_result(inversion, seed):
    """A 1D array represents the diagonal of a covariance matrix. The user should
    be able to pass either a 2D or 1D representation of a diagonal, and get the same
    results. But a 1D representation is more efficient w.r.t. speed and memory."""
    rng = np.random.default_rng(seed)

    # Create a random problem instance
    ensemble_size = rng.choice([5, 10, 50, 100])
    num_params = rng.choice([5, 10, 50, 100])
    num_obs = rng.choice([5, 10, 50, 100])

    X = rng.normal(size=(num_params, ensemble_size))  # X
    responses = rng.normal(size=(num_obs, ensemble_size))  # Y
    observations = rng.normal(size=num_obs)  # d

    covariance_1D = np.exp(rng.normal(size=num_obs))
    covariance_2D = np.diag(covariance_1D)

    # 1D array of standard deviations
    smoother_diag = ies.SIES(
        parameters=X, covariance=covariance_1D, observations=observations, seed=seed
    )
    assert covariance_1D.ndim == 1
    X_post_diag = smoother_diag.sies_iteration(responses)

    # 2D array of covariances (covariance matrix)
    smoother_covar = ies.SIES(
        parameters=X, covariance=covariance_2D, observations=observations, seed=seed
    )
    assert covariance_2D.ndim == 2
    X_post_covar = smoother_covar.sies_iteration(responses)

    assert np.allclose(X_post_diag, X_post_covar)  # Same result
    assert not np.allclose(X, X_post_covar)  # Update happened


@pytest.mark.parametrize("inversion", list(ies.SIES.inversion_funcs.keys()))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("diagonal", [True, False])
def test_that_float_dtypes_are_preserved(inversion, dtype, diagonal):
    """If every matrix passed is of a certain dtype, then the output
    should also be of the same dtype. 'linalg' does not support float16
    nor float128."""

    rng = np.random.default_rng(42)

    # Create a random problem instance
    ensemble_size = 25
    num_params = 10
    num_obs = 20

    parameters = rng.normal(size=(num_params, ensemble_size))  # X
    responses = rng.normal(size=(num_obs, ensemble_size))  # Y
    observations = rng.normal(size=num_obs)  # d

    covar_factor = rng.normal(size=(num_obs, num_obs))
    covariance = covar_factor.T @ covar_factor
    if diagonal:
        covariance = np.diag(covariance)

    # Convert dtypes
    parameters = parameters.astype(dtype)
    covariance = covariance.astype(dtype)
    observations = observations.astype(dtype)
    responses = responses.astype(dtype)

    smoother = ies.SIES(
        parameters=parameters,
        covariance=covariance,
        observations=observations,
        seed=rng,
        inversion=inversion,
    )

    X_posterior = smoother.sies_iteration(responses)

    assert X_posterior.dtype == dtype


@pytest.mark.limit_memory("70 MB")
def test_memory_usage():
    """Estimate expected memory usage and make sure ES does not waste memory

    # approx. 65
    # Size of input arrays
    nbytes = (
        X.nbytes
        + Y.nbytes
        + observation_errors.nbytes
        + observation_values.nbytes
        + noise.nbytes
    )
    nbytes += noise.nbytes  # Creating E
    nbytes += noise.nbytes  # Creating D
    # scaling response_ensemble
    # (can't scale in-place because response_ensemble is an input argument)
    nbytes += (
        noise.nbytes
    )
    nbytes += 80000 # Omega in C++ (ensemble_size, ensemble_size)
    nbytes += Y.nbytes # Solving for S^T needs Y^T which causes a copy in C++ code
    nbytes += Y.nbytes # Solving for S^T causes both Y^T and S^T to be in memory
    nbytes += Y.nbytes # Creating H in C++
    nbytes /= 1e6
    """
    rng = np.random.default_rng(42)

    ensemble_size = 100
    num_params = 1000
    num_obs = 10000
    X = rng.normal(size=(num_params, ensemble_size))

    Y = rng.normal(size=(num_obs, ensemble_size))

    observation_errors = rng.uniform(size=num_obs)
    observation_values = rng.normal(np.zeros(num_obs), observation_errors)

    smoother = ies.SIES(
        parameters=X,
        covariance=observation_errors,
        observations=observation_values,
    )

    smoother.sies_iteration(Y, 1.0)


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
            #  "-k test_that_update_correctly_multiples_gaussians",
        ]
    )
