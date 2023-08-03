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


class TestESMDA:
    @pytest.mark.parametrize("num_ensemble", [2**i for i in range(2, 10)])
    def test_that_using_example_mask_only_updates_those_parameters(self, num_ensemble):
        seed = num_ensemble
        rng = np.random.default_rng(seed)
        alpha = rng.choice(np.array([5, 10, 25, 50]))  # Random alpha

        num_outputs = 2
        num_inputs = 1

        def g(x):
            """Transform a single ensemble member."""
            return np.array([np.sin(x / 2), x])

        def G(X):
            """Transform all ensemble members."""
            return np.array([g(x_i) for x_i in X.T]).squeeze().T

        # Create an ensemble mask and set half the entries randomly to True
        ensemble_mask = np.zeros(num_ensemble, dtype=bool)
        idx = rng.choice(num_ensemble, size=num_ensemble // 2, replace=False)
        ensemble_mask[idx] = True

        # Prior is N(0, 1)
        X_prior = rng.normal(size=(num_inputs, num_ensemble))

        # Measurement errors
        C_D = np.eye(num_outputs)

        # The true inputs and observations, a result of running with X_true = 1
        X_true = np.ones(shape=(num_inputs,))
        observations = G(X_true)

        # Prepare ESMDA instance running with lower number of ensemble members
        esmda_subset = ESMDA(C_D, observations, alpha=alpha, seed=seed)
        X_i_subset = np.copy(X_prior[:, ensemble_mask])

        # Prepare ESMDA instance running with all ensemble members
        esmda_masked = ESMDA(C_D, observations, alpha=alpha, seed=seed)
        X_i = np.copy(X_prior)

        # Run both
        for _ in range(esmda_subset.num_assimilations()):
            X_i_subset = esmda_subset.assimilate(X_i_subset, G(X_i_subset))
            X_i = esmda_masked.assimilate(X_i, G(X_i), ensemble_mask=ensemble_mask)

            # Exactly the same result (seed value for both must be equal )
            assert np.allclose(X_i_subset, X_i[:, ensemble_mask])

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
        C_D = np.eye(num_outputs)

        # The true inputs and observations, a result of running with N(1, 1)
        X_true = rng.normal(size=(num_inputs,)) + 1
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
        inv = lambda x: 1 / x  # Matrix inversion for 1D matrix
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
    def test_that_single_and_multiple_assimilations_achieve_same_result_for_1D_gauss_linear_case(
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
        C_D = np.exp(rng.normal(size=num_outputs))

        # Observations
        observations = rng.normal(size=num_outputs, loc=1)

        return X_prior, Y_prior, C_D, observations

    @pytest.mark.limit_memory("138 MB")
    def test_ESMDA_memory_usage_subspace_inversion(self, setup):
        # TODO: Currently this is a regression test. Work to improve memory usage.

        X_prior, Y_prior, C_D, observations = setup

        # Create ESMDA instance from an integer `alpha` and run it
        esmda = ESMDA(C_D, observations, alpha=1, seed=1, inversion="subspace")

        for _ in range(esmda.num_assimilations()):
            esmda.assimilate(X_prior, Y_prior)


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "-v",
            # "--durations=10",
            #  "-k test_that_result_corresponds_with_theory_for_gauss_linear_case",
            # "--maxfail=1",
        ]
    )

    1 / 0

    import matplotlib.pyplot as plt

    # Here we test on a Guass-linear case.
    # The section "2.3.3 Bayes’ theorem for Gaussian variables" in the book
    # Pattern Recognition and Machine Learning by Bishop (2006) states that if
    # p(x)   = N(x | mu, S)
    # p(y|x) = N(y | a * x + b, C_D)
    # then
    # p(x | y) has
    # covariance COV = inv(inv(C_D) + a * inv(S) * a)
    # mean      MEAN = COV (a * inv(S) * (y - v) + inv(C_D) * mu)
    num_ensemble = 10_000

    # Prior
    mu, S = 0, 2

    # Observations
    X_true, C_D = 10, 10

    # Linear transformation
    a, b = 1, 1

    seed = num_ensemble
    rng = np.random.default_rng(seed)

    num_outputs = 1
    num_inputs = 1

    def g(x):
        """Transform a single ensemble member: a*x + b"""
        return a * x + b

    def G(X):
        """Transform all ensemble members."""
        return np.array([g(x_i) for x_i in X.T]).T

    # Prior is N(0, 1)
    X_prior = rng.normal(size=(num_inputs, num_ensemble), loc=mu, scale=np.sqrt(S))

    # Measurement errors
    C_D = np.diag([C_D])

    # The true inputs and observations, a result of running with X_true
    observations = G(np.atleast_1d(X_true))

    # Create ESMDA instance from an integer `alpha` and run it
    esmda = ESMDA(C_D, observations, alpha=1, seed=rng)
    X_i = np.copy(X_prior)
    print(esmda.alpha)
    for _ in range(esmda.num_assimilations()):
        X_i = esmda.assimilate(X_i, G(X_i))
        print("X_i stats, mean:", X_i.mean(), "var ", X_i.var())

        # Plot results
        plt.hist(X_prior.ravel(), alpha=0.5, label="prior")
        # plt.hist(X_true.ravel(), alpha=0.5, label="true inputs")
        plt.hist(X_i.ravel(), alpha=0.5, label="X_i")
        plt.legend()
        plt.show()

    # Analytical solution
    COV = 1 / (1 / C_D + a**2 / S)
    MEAN = COV * ((a / C_D) * (g(X_true) - b) + (1 / S) * mu)

    print("means", X_i.mean(), MEAN.mean())
    print("covs", X_i.var(ddof=1), COV.mean())
    assert np.allclose(X_i.mean(), MEAN.mean(), rtol=1e-2)
    assert np.allclose(X_i.var(ddof=1), COV.mean(), rtol=1e-2)
