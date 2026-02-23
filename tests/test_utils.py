import numpy as np
import pytest

from iterative_ensemble_smoother.utils import adjust_for_missing


@pytest.mark.parametrize("seed", range(100))
def test_adjust_for_missing(seed):
    rng = np.random.default_rng(seed)

    N_e = rng.integers(5, 11)  # Number of realizations / ensemble members
    num_params_x = rng.integers(5, 11)
    num_obs_y = rng.integers(5, 21)

    # Generate random data matrices
    X = rng.standard_normal((num_params_x, N_e))
    Y = rng.standard_normal((num_obs_y, N_e))

    # Build a missing mask for X, ensuring at least 2 non-missing per row
    missing = np.zeros((num_params_x, N_e), dtype=np.bool_)
    for i in range(num_params_x):
        n_missing = rng.integers(0, N_e - 1)  # leave at least 2 observations
        cols = rng.choice(N_e, size=n_missing, replace=False)
        missing[i, cols] = True

    # Compute expected cross-covariance entry-by-entry
    C_expected = np.empty((num_params_x, num_obs_y))
    for i in range(num_params_x):
        obs_mask = ~missing[i]
        x_i = X[i, obs_mask]
        x_centered = x_i - x_i.mean()
        for j in range(num_obs_y):
            y_j = Y[j, obs_mask]
            C_expected[i, j] = x_centered @ y_j / (obs_mask.sum() - 1)

    # Verify that naive matches the function 'adjust_for_missing'
    C_actual = adjust_for_missing(X, missing=missing) @ Y.T / (N_e - 1)
    np.testing.assert_allclose(C_actual, C_expected)


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
        ]
    )
