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
import pytest

from iterative_ensemble_smoother.esmda import ESMDA
from iterative_ensemble_smoother.esmda_inversion import empirical_cross_covariance
from iterative_ensemble_smoother.sies import SIES


class TestESMDA:
    @pytest.mark.parametrize("num_inputs", [10, 25, 50])
    @pytest.mark.parametrize("num_outputs", [5, 25, 50])
    @pytest.mark.parametrize("sies_inversion", ["direct", "subspace_exact"])
    @pytest.mark.parametrize("seed", list(range(10)))
    def test_that_ESMDA_and_SIES_produce_same_result_with_one_step(
        self, seed, sies_inversion, num_outputs, num_inputs
    ):
        """With a single step (alpha=1), ESMDA = SIES = ES.

        When num_inputs < num_ensemble - 1, then Section (2.4.3) in the SIES
        paper triggers and the result is not identical.
        """
        rng = np.random.default_rng(seed)

        num_ensemble = 10
        alpha = 1

        # Create problem instance
        X = rng.normal(size=(num_inputs, num_ensemble))
        Y = rng.normal(size=(num_outputs, num_ensemble))
        covariance = np.exp(rng.normal(size=num_outputs))
        observations = rng.normal(size=num_outputs, loc=1)

        # Create ESMDA instance and perform one iteration
        esmda = ESMDA(
            covariance, observations, alpha=alpha, seed=seed + 99, inversion="exact"
        )
        X_ESMDA = np.copy(X)

        # Perform one iteration of ESMDA
        X_ESMDA = esmda.assimilate(X_ESMDA, Y)

        # Create SIES instance and perform one iteration
        sies = SIES(
            parameters=X,
            covariance=covariance,
            observations=observations,
            inversion=sies_inversion,
            truncation=1.0,
            seed=seed + 99,
        )

        X_SIES = sies.sies_iteration(responses=Y, step_length=1)

        assert np.allclose(X_ESMDA, X_SIES)

    @pytest.mark.parametrize("seed", list(range(10)))
    @pytest.mark.parametrize("inversion", ["exact", "subspace"])
    def test_that_diagonal_covariance_gives_same_answer_as_dense(self, seed, inversion):
        rng = np.random.default_rng(seed)

        num_outputs = rng.choice([5, 10, 15, 25])
        num_inputs = rng.choice([5, 10, 15, 25])
        num_ensemble = rng.choice([5, 10, 15, 25])
        alpha = rng.choice([1, 3, 5])

        # Prior is N(0, 1)
        X_prior = rng.normal(size=(num_inputs, num_ensemble))
        Y_prior = rng.normal(size=(num_outputs, num_ensemble))

        # Measurement errors
        covariance = np.exp(rng.normal(size=num_outputs))

        # Observations
        observations = rng.normal(size=num_outputs, loc=1)

        esmda = ESMDA(
            covariance, observations, alpha=alpha, seed=seed, inversion=inversion
        )
        X_posterior1 = np.copy(X_prior)
        for _ in range(esmda.num_assimilations()):
            X_posterior1 = esmda.assimilate(X_posterior1, Y_prior)

        esmda = ESMDA(
            np.diag(covariance),
            observations,
            alpha=alpha,
            seed=seed,
            inversion=inversion,
        )
        X_posterior2 = np.copy(X_prior)
        for _ in range(esmda.num_assimilations()):
            X_posterior2 = esmda.assimilate(X_posterior2, Y_prior)

        assert np.allclose(X_posterior1, X_posterior2)

    @pytest.mark.parametrize(
        "num_ensemble",
        [10, 100, 1000],
    )
    def test_that_alpha_as_integer_and_array_returns_same_result(self, num_ensemble):
        seed = num_ensemble
        rng = np.random.default_rng(seed)

        num_outputs = 2
        num_inputs = 1

        def g(x):
            """Transform a single ensemble member."""
            return np.array([np.sin(x / 2), x]) + 5

        def G(X):
            """Transform all ensemble members."""
            return np.array([g(x_i) for x_i in X.T]).squeeze().T

        # Prior is N(0, 1)
        X_prior = rng.normal(size=(num_inputs, num_ensemble))

        # Measurement errors
        covariance = np.eye(num_outputs)

        # The true inputs and observations, a result of running with N(1, 1)
        X_true = rng.normal(size=(num_inputs,)) + 1
        observations = G(X_true)

        # Create ESMDA instance from an integer `alpha` and run it
        esmda_integer = ESMDA(covariance, observations, alpha=5, seed=seed)
        X_i_int = np.copy(X_prior)
        for _ in range(esmda_integer.num_assimilations()):
            X_i_int = esmda_integer.assimilate(X_i_int, G(X_i_int))

        # Create another ESMDA instance from a vector `alpha` and run it
        esmda_array = ESMDA(covariance, observations, alpha=np.ones(5), seed=seed)
        X_i_array = np.copy(X_prior)
        for _ in range(esmda_array.num_assimilations()):
            X_i_array = esmda_array.assimilate(X_i_array, G(X_i_array))

        # Exactly the same result with equal seeds
        assert np.allclose(X_i_int, X_i_array)

    # Linear transformation a x + b
    @pytest.mark.parametrize("a", [-2, 1, 3, 9])
    @pytest.mark.parametrize("b", [-2, 0, 1])
    # Likelihood information
    @pytest.mark.parametrize("X_true", [-2, 5, 10, 15])
    @pytest.mark.parametrize("C_D", [1, 3, 9])
    # Prior information
    @pytest.mark.parametrize("mu", [-1, 0, 6, 12])
    @pytest.mark.parametrize("S", [1, 3, 9])
    def test_that_result_corresponds_with_theory_for_1D_gauss_linear_case(
        self, mu, S, X_true, C_D, a, b
    ):
        # Here we test on a Guass-linear case.
        # The section "2.3.3 Bayes’ theorem for Gaussian variables" in the book
        # Pattern Recognition and Machine Learning by Bishop (2006) states that if
        # p(x)   = N(x | mu, S)
        # p(y|x) = N(y | a * x + b, C_D)
        # then
        # p(x | y) has
        # covariance COV = inv(inv(S) + a * inv(C_D) * a)
        # mean      MEAN = COV (a * inv(C_D) * (y - b) + inv(S) * mu)
        def G(X):
            """Transform all ensemble members."""
            return a * X + b

        # Analytical solution given by Bishop
        def inv(x):
            # Matrix inversion for 1D matrix
            return 1 / x

        COV = inv(inv(S) + a * inv(C_D) * a)
        MEAN = COV * ((a * inv(C_D)) * (G(X_true) - b) + inv(S) * mu)

        num_ensemble = 1_000
        num_inputs = 1

        # Create a random number generator
        parameters = [mu, S, X_true, C_D, a, b]
        seed = abs(sum(p_i * 10**i for i, p_i in enumerate(parameters)))
        rng = np.random.default_rng(seed)

        # Prior is p(x) ~ N(mu, S)
        X_prior = rng.normal(size=(num_inputs, num_ensemble), loc=mu, scale=np.sqrt(S))

        # Create ESMDA instance
        esmda = ESMDA(np.atleast_2d([C_D]), G(np.atleast_1d(X_true)), alpha=1, seed=rng)
        X_i = X_prior
        for _ in range(esmda.num_assimilations()):
            X_i = esmda.assimilate(X_i, G(X_i))

        # Check that analytical solution is close to ESMDA posterior
        # np.isclose(a, b) := abs(`a` - `b`) <= (`atol` + `rtol` * abs(`b`))
        assert np.isclose(X_i.mean(), MEAN, rtol=0.1, atol=0.2)
        assert np.isclose(X_i.var(ddof=1), COV, rtol=0.1, atol=0.1)

    @pytest.mark.parametrize("seed", list(range(100)))
    def test_that_result_corresponds_with_theory_for_gauss_linear_case(self, seed):
        # Here we test on a Guass-linear case.
        # The section "2.3.3 Bayes’ theorem for Gaussian variables" in the book
        # Pattern Recognition and Machine Learning by Bishop (2006) states that if
        # p(x)   = N(x | mu, C_M)
        # p(y|x) = N(y | A x + b, C_D)
        # then
        # p(x | y) has
        # covariance COV = inv(inv(C_M) + A^T @ inv(C_D) @ A)
        # mean      MEAN = COV (A^T * inv(C_D) * (y - b) + inv(C_M) * mu)
        #                = mu + C_M A^T (A C_M A^T + C_D)^-1 A (x - mu)

        # Generate data
        rng = np.random.default_rng(seed)
        num_ensemble = 10_000
        num_inputs, num_outputs = 5, 4

        # Inputs
        mu = rng.normal(size=num_inputs)
        C_M_factor = rng.normal(size=(num_inputs, num_inputs))
        C_M = C_M_factor.T @ C_M_factor + np.eye(num_inputs)

        # Transformation
        A = rng.normal(size=(num_outputs, num_inputs))
        b = rng.normal(size=num_outputs)

        def G(x):
            if x.ndim == 1:
                return A @ x + b
            else:
                return (b.reshape(-1, 1) + (A @ x)).squeeze()

        assert np.allclose(
            G(np.arange(num_inputs)), G(np.arange(num_inputs).reshape(-1, 1)).squeeze()
        )

        # Output covariance
        C_D_factor = rng.normal(size=(num_outputs, num_outputs))
        C_D = C_D_factor.T @ C_D_factor + np.eye(num_outputs)

        # Analytical solution given by Bishop
        X_true = mu + 10
        inv = np.linalg.inv
        COV = inv(inv(C_M) + A.T @ inv(C_D) @ A)
        MEAN = COV @ ((A.T @ inv(C_D)) @ (G(X_true) - b) + inv(C_M) @ mu)
        MEAN2 = mu + C_M @ A.T @ inv(A @ C_M @ A.T + C_D) @ A @ (X_true - mu)
        assert np.allclose(MEAN, MEAN2)  # Both ways to compute mean is equivalent

        # Prior is p(x) ~ N(mu, C_M)
        X_prior = rng.multivariate_normal(mean=mu, cov=C_M, size=num_ensemble).T
        assert G(X_prior).shape == (num_outputs, num_ensemble)

        # Create ESMDA instance
        esmda = ESMDA(C_D, G(X_true), alpha=1, seed=rng)
        X_i = np.copy(X_prior)
        for _ in range(esmda.num_assimilations()):
            X_i = esmda.assimilate(X_i, G(X_i))

        # Check that analytical solution is close to ESMDA posterior
        relative_error_mean = np.linalg.norm(X_i.mean(axis=1) - MEAN) / np.linalg.norm(
            MEAN
        )
        print("relative_error_mean", relative_error_mean)

        assert relative_error_mean < 0.05

        # TODO: The errors in the covariance matrix estimation is always pretty high
        # why is this?
        covariance = empirical_cross_covariance(X_i, X_i)
        relative_error_covariance = np.linalg.norm((covariance - COV)) / np.linalg.norm(
            COV
        )
        print("relative_error_covariance", relative_error_covariance)

        assert relative_error_covariance < 1.2

    # Likelihood information
    @pytest.mark.parametrize("X_true", [5, 10])
    @pytest.mark.parametrize("C_D", [1, 3])
    # Prior information
    @pytest.mark.parametrize("mu", [0, 2, 4])
    @pytest.mark.parametrize("S", [1, 3, 9])
    # Number of iterations
    @pytest.mark.parametrize("alpha", [5, 10, 15])
    def test_that_single_and_multiple_assimil_give_same_res_for_1D_gauss_linear_case(
        self, alpha, mu, S, X_true, C_D
    ):
        # Here we test on a Guass-linear case.
        # The section "2.3.3 Bayes' theorem for Gaussian variables" in the book
        # Pattern Recognition and Machine Learning by Bishop (2006) states that if
        # p(x)   = N(x | mu, S)
        # p(y|x) = N(y | a * x + b, C_D)
        # then
        # p(x | y) has
        # covariance COV = inv(inv(S) + a * inv(C_D) * a)
        # mean      MEAN = COV (a * inv(C_D) * (y - b) + inv(S) * mu)

        def G(X):
            """Transform all ensemble members."""
            a, b = 1, 1
            return a * X + b

        num_ensemble = 10_000
        num_inputs = 1

        # Create a random number generator
        parameters = [alpha, mu, S, X_true, C_D]
        seed = abs(sum(p_i * 10**i for i, p_i in enumerate(parameters)))
        rng = np.random.default_rng(seed)

        # Prior is p(x) ~ N(mu, S)
        X_prior = rng.normal(size=(num_inputs, num_ensemble), loc=mu, scale=np.sqrt(S))

        # Create ESMDA instance with single iteration
        esmda = ESMDA(np.diag([C_D]), G(np.atleast_1d(X_true)), alpha=1, seed=rng)
        X_i_single = np.copy(X_prior)
        for _ in range(esmda.num_assimilations()):
            X_i_single = esmda.assimilate(X_i_single, G(X_i_single))

        # Create ESMDA instance with multiple iterations
        esmda = ESMDA(np.diag([C_D]), G(np.atleast_1d(X_true)), alpha=alpha, seed=rng)
        X_i_multiple = np.copy(X_prior)
        for _ in range(esmda.num_assimilations()):
            X_i_multiple = esmda.assimilate(X_i_multiple, G(X_i_multiple))

        # Check that summary statistics of solutions are close to each other
        # np.isclose(a, b) := abs(`a` - `b`) <= (`atol` + `rtol` * abs(`b`))
        assert np.isclose(X_i_single.mean(), X_i_multiple.mean(), rtol=0.05)
        assert np.isclose(X_i_single.var(), X_i_multiple.var(), rtol=0.1)


class TestESMDAMemory:
    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(42)

        num_outputs = 10_000
        num_inputs = 1_000
        num_ensemble = 100

        # Prior is N(0, 1)
        X_prior = rng.normal(size=(num_inputs, num_ensemble))
        Y_prior = rng.normal(size=(num_outputs, num_ensemble))

        # Measurement errors
        covariance = np.exp(rng.normal(size=num_outputs))

        # Observations
        observations = rng.normal(size=num_outputs, loc=1)

        return X_prior, Y_prior, covariance, observations

    @pytest.mark.limit_memory("138 MB")
    def test_ESMDA_memory_usage_subspace_inversion_without_overwrite(self, setup):
        # TODO: Currently this is a regression test. Work to improve memory usage.

        X_prior, Y_prior, covariance, observations = setup

        # Create ESMDA instance from an integer `alpha` and run it
        esmda = ESMDA(covariance, observations, alpha=1, seed=1, inversion="subspace")

        for _ in range(esmda.num_assimilations()):
            esmda.assimilate(X_prior, Y_prior)

    @pytest.mark.limit_memory("129 MB")
    def test_ESMDA_memory_usage_subspace_inversion_with_overwrite(self, setup):
        # TODO: Currently this is a regression test. Work to improve memory usage.

        X_prior, Y_prior, covariance, observations = setup

        # Create ESMDA instance from an integer `alpha` and run it
        esmda = ESMDA(covariance, observations, alpha=1, seed=1, inversion="subspace")

        for _ in range(esmda.num_assimilations()):
            esmda.assimilate(X_prior, Y_prior, overwrite=True)


@pytest.mark.parametrize("inversion", ["exact", "subspace"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("diagonal", [True, False])
def test_that_float_dtypes_are_preserved(inversion, dtype, diagonal):
    """If every matrix passed is of a certain dtype, then the output
    should also be of the same dtype. 'linalg' does not support float16
    nor float128."""

    rng = np.random.default_rng(42)

    num_outputs = 20
    num_inputs = 10
    num_ensemble = 25

    # Prior is N(0, 1)
    X_prior = rng.normal(size=(num_inputs, num_ensemble))
    Y_prior = rng.normal(size=(num_outputs, num_ensemble))

    # Measurement errors
    covariance = np.exp(rng.normal(size=num_outputs))
    if not diagonal:
        covariance = np.diag(covariance)

    # Observations
    observations = rng.normal(size=num_outputs, loc=1)

    # Convert types
    X_prior = X_prior.astype(dtype)
    Y_prior = Y_prior.astype(dtype)
    covariance = covariance.astype(dtype)
    observations = observations.astype(dtype)

    # Create ESMDA instance from an integer `alpha` and run it
    esmda = ESMDA(covariance, observations, alpha=1, seed=1, inversion=inversion)

    for _ in range(esmda.num_assimilations()):
        X_posterior = esmda.assimilate(X_prior, Y_prior)

    # Check that dtype of returned array matches input dtype
    assert X_posterior.dtype == dtype


@pytest.mark.parametrize("inversion", ESMDA._inversion_methods.keys())
def test_row_by_row_assimilation(inversion):
    # Create problem instance
    rng = np.random.default_rng(42)

    num_outputs = 4
    num_inputs = 5
    num_ensemble = 3

    A = rng.normal(size=(num_outputs, num_inputs))

    def g(X):
        return A @ X

    # Prior is N(0, 1)
    X_prior = rng.normal(size=(num_inputs, num_ensemble))

    covariance = np.exp(rng.normal(size=num_outputs))
    observations = A @ np.linspace(0, 1, num=num_inputs) + rng.normal(
        size=num_outputs, scale=0.01
    )

    # =========== Use the high level API ===========
    smoother = ESMDA(
        covariance=covariance,
        observations=observations,
        alpha=2,
        inversion=inversion,
        seed=1,
    )
    X = np.copy(X_prior)
    for iteration in range(smoother.num_assimilations()):
        X = smoother.assimilate(X, g(X))

    X_posterior_highlevel_API = np.copy(X)

    # =========== Use the low-level level API ===========
    smoother = ESMDA(
        covariance=covariance,
        observations=observations,
        alpha=2,
        inversion=inversion,
        seed=1,
    )
    X = np.copy(X_prior)
    for alpha_i in smoother.alpha:
        K = smoother.compute_transition_matrix(Y=g(X), alpha=alpha_i)

        # Here we could loop over each row in X and multiply by K
        X += X @ K

    X_posterior_lowlevel_API = np.copy(X)

    assert np.allclose(X_posterior_highlevel_API, X_posterior_lowlevel_API)


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "-v",
            "-k test_that_ESMDA_and_SIES_produce_same_result_with_one_step",
        ]
    )
