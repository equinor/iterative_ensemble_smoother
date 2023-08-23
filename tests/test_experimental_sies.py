import numpy as np
import scipy as sp
from iterative_ensemble_smoother.experimental_sies import (
    inversion_naive,
    inversion_exact,
    inversion_direct,
    inversion_direct_corrscale,
    SIES,
)

import pytest


class TestInversions:
    @pytest.mark.parametrize(
        "func",
        [
            inversion_naive,
            inversion_direct,
            inversion_direct_corrscale,
            inversion_exact,
        ],
    )
    def test_that_inversions_are_equal_diagonal_or_dense_covariance(self, func):
        rng = np.random.default_rng(42)
        m, N = 100, 10  # Output, realizations
        W = rng.standard_normal(size=(N, N))
        step_length = 0.33
        S = rng.standard_normal(size=(m, N))
        C_dd_1D = 1 + rng.standard_normal(size=m) ** 2
        C_dd_2D = np.diag(C_dd_1D)

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
        param_ensemble=X,
        observation_errors=observation_errors,
        observation_values=observation_values,
    )

    smoother.newton(Y, 1.0)


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-v", "--maxfail=1"])
