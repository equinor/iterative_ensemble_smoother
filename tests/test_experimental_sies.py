import numpy as np
import scipy as sp
from iterative_ensemble_smoother.experimental_sies import SIES

from iterative_ensemble_smoother.sies_inversion import (
    inversion_naive,
    inversion_exact,
    inversion_direct,
    inversion_direct_corrscale,
)

import pytest


class TestInversions:
    @pytest.mark.parametrize("seed", list(range(10)))
    @pytest.mark.parametrize(
        "func",
        [
            inversion_naive,
            inversion_direct,
            inversion_direct_corrscale,
            inversion_exact,
        ],
    )
    def test_that_inversions_are_equal_diagonal_or_dense_covariance(self, seed, func):
        """Every inversion function should return the same result whether a
        1D array or 2D array is used to represent a diagonal covariance."""
        rng = np.random.default_rng(seed)
        m, N = 100, 10  # Output, realizations
        W = rng.standard_normal(size=(N, N))
        step_length = 0.33
        S = rng.standard_normal(size=(m, N))

        # First create a 1D covariance vector representing the diagonal
        C_dd_1D = 1 + rng.standard_normal(size=m) ** 2
        # Then create the same as a 2D covariance matrix with off-diagonal zeros
        C_dd_2D = np.diag(C_dd_1D)

        # Cholesky of a diagonal is just the square root of each entry
        C_dd_cholesky_1D = np.sqrt(C_dd_1D)
        C_dd_cholesky_2D = np.diag(C_dd_cholesky_1D)

        H = rng.standard_normal(size=(m, N))

        ans_1D = func(
            W=W,
            step_length=step_length,
            S=S,
            C_dd=C_dd_1D,
            H=H,
            C_dd_cholesky=C_dd_cholesky_1D,
        )
        ans_2D = func(
            W=W,
            step_length=step_length,
            S=S,
            C_dd=C_dd_2D,
            H=H,
            C_dd_cholesky=C_dd_cholesky_2D,
        )

        assert np.allclose(ans_1D, ans_2D)

    @pytest.mark.parametrize(
        "func", [inversion_direct, inversion_direct_corrscale, inversion_exact]
    )
    def test_that_exact_inversions_are_all_equal(self, func):
        rng = np.random.default_rng(42)
        m, N = 100, 10  # Output, realizations
        W = rng.standard_normal(size=(N, N))
        step_length = 0.33
        S = rng.standard_normal(size=(m, N))
        C_dd_factor = rng.standard_normal(size=(m, m))
        C_dd = C_dd_factor @ C_dd_factor.T
        C_dd_cholesky = sp.linalg.cholesky(
            C_dd, lower=True
        )  # Lower triangular cholesky
        H = rng.standard_normal(size=(m, N))

        ans_naive = inversion_naive(
            W=W,
            step_length=step_length,
            S=S,
            C_dd=C_dd,
            H=H,
            C_dd_cholesky=C_dd_cholesky,
        )
        ans = func(
            W=W,
            step_length=step_length,
            S=S,
            C_dd=C_dd,
            H=H,
            C_dd_cholesky=C_dd_cholesky,
        )

        assert np.allclose(ans_naive, ans)


@pytest.mark.parametrize("seed", list(range(99)))
def test_that_sies_objective_function_decreases(seed):
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
    )

    Y = G(X)
    objective_before = smoother.evaluate_objective(W=smoother.W, Y=Y)

    # One iteration
    X_i = smoother.sies_iteration(Y, 0.5)
    Y_i = G(X_i)
    objective_after = smoother.evaluate_objective(W=smoother.W, Y=Y_i)

    assert objective_after <= objective_before


@pytest.mark.parametrize("seed", list(range(99)))
def test_that_sies_objective_function_decreases_with_many_iterations(seed):
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


def test_line_search():
    """This is a demo (or test) of how line search can be implemented."""
    rng = np.random.default_rng(4)

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
                print(
                    f"Terminating. No improvement after {BACKTRACK_ITERATIONS} iterations."
                )
                return X_i

            yield X_i

    # Initial evaluation
    objective_before = smoother.evaluate_objective(W=smoother.W, Y=G(smoother.X))

    for X_i in line_search(smoother, G):
        objective = smoother.evaluate_objective(W=smoother.W, Y=G(X_i))
        print(objective_before, objective)


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
    nbytes += (
        noise.nbytes
    )  # scaling response_ensemble (can't scale in-place because response_ensemble is an input argument)
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

    smoother = SIES(
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
            "-v",
            "--maxfail=1",
            # "-k test_that_sies_objective",
        ]
    )