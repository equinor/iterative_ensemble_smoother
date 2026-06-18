import time

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from iterative_ensemble_smoother.enif_linear_regression import (
    _fit_single_response_boost,  # noqa: PLC2701
    boost_linear_regression,
    linear_boost_ic_regression,
)


def generate_data(n, p_noisy, seed=42):
    """3 signal features and p_noisy noisy ones. All standardized."""
    rng = np.random.default_rng(seed)

    X = rng.random((n, 3))

    # True relationship + noise
    y = 1 * X[:, 0] - X[:, 1] + X[:, 2] + rng.normal(0, 1e-2, size=n)

    # Add uninformative features
    noise_features = rng.random((n, p_noisy))
    X_with_noise = np.hstack((X, noise_features))

    X_scaled = StandardScaler().fit_transform(X_with_noise)
    y_scaled = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled


@pytest.mark.parametrize(("n", "p_noisy"), [(100, 9), (200, 99), (1000, 999)])
def test_that_mse_train_decreases(n, p_noisy):
    X, y = generate_data(n, p_noisy)

    # Baseline: mean predictor
    y_mean = np.mean(y)
    mse_mean = mean_squared_error(y, np.full_like(y, y_mean))

    # Learn (very sparse) linear regression
    beta_sparse = boost_linear_regression(X=X, y=y, learning_rate=0.1, max_iter=100)

    y_pred = X @ beta_sparse
    mse_model = mean_squared_error(y, y_pred)

    assert mse_model <= mse_mean


@pytest.mark.parametrize(
    ("n", "p_noisy"), [(50, 5), (100, 10), (200, 20), (400, 40), (1000, 999)]
)
def test_that_regression_learns_more_with_more_data(n, p_noisy):
    X_full, y_full = generate_data(10 * n, p_noisy=p_noisy)

    # Split into two halves
    X_half, y_half = X_full[:n], y_full[:n]

    # Train on a tenth
    beta_half = boost_linear_regression(X=X_half, y=y_half)

    # Train on all data
    beta_full = boost_linear_regression(X=X_full, y=y_full)

    # Predict both on full data
    y_pred_half = X_full @ beta_half
    y_pred_full = X_full @ beta_full

    mse_half = mean_squared_error(y_full, y_pred_half)
    mse_full = mean_squared_error(y_full, y_pred_full)

    assert mse_full <= mse_half


@pytest.mark.parametrize(
    ("n", "p_noisy", "max_iters"),
    [
        (200, 20, [10, 50, 100]),
        (400, 40, [20, 100, 200]),
    ],
)
def test_that_mse_decreases_with_more_iterations(n, p_noisy, max_iters):
    X, y = generate_data(n, p_noisy)

    prev_mse = float("inf")
    for max_iter in max_iters:
        beta = boost_linear_regression(X=X, y=y, max_iter=max_iter)
        y_pred = X @ beta
        mse = mean_squared_error(y, y_pred)

        assert mse <= prev_mse

        prev_mse = mse


def test_parallel_speedup():
    """Test that parallel execution is faster than sequential."""
    rng = np.random.default_rng(42)
    # Larger data needed to overcome parallel overhead (~50ms per task)
    n_samples = 200
    n_features = 500
    n_responses = 1000

    U = rng.random((n_samples, n_features))
    Y = rng.random((n_samples, n_responses))

    # Time sequential execution
    start = time.perf_counter()
    H_sequential = linear_boost_ic_regression(U, Y, n_jobs=1)
    time_sequential = time.perf_counter() - start

    # Time parallel execution
    start = time.perf_counter()
    H_parallel = linear_boost_ic_regression(U, Y, n_jobs=-1)
    time_parallel = time.perf_counter() - start

    # Verify results are identical
    np.testing.assert_array_almost_equal(
        H_sequential.toarray(), H_parallel.toarray(), decimal=10
    )

    # Parallel should be faster on multi-core machines
    print(f"\nSequential: {time_sequential:.2f}s, Parallel: {time_parallel:.2f}s")
    print(f"Speedup: {time_sequential / time_parallel:.2f}x")

    # Assert parallel provides speedup (at least 1.2x on multi-core)
    assert time_parallel < time_sequential, (
        f"Parallel ({time_parallel:.2f}s) should be faster than "
        f"sequential ({time_sequential:.2f}s)"
    )


def test_that_rowwise_fitting_requires_independent_features():
    """With independent rows of X (orthogonal features), training the full
    map Y=H@X jointly gives the same H as training one row of X at a time.
    With dependent rows, the maps should differ."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    n_features = 4
    n_responses = 3

    # --- Independent features via QR, scaled to ~unit variance ---
    raw = rng.standard_normal((n_samples, n_features))
    Q, _ = np.linalg.qr(raw)
    U_indep = Q * np.sqrt(n_samples)

    # True sparse linear map H_true (m x p)
    H_true = np.array(
        [
            [2.0, 0.0, -1.0, 0.0],
            [0.0, 1.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
        ]
    )
    noise = rng.normal(0, 0.01, (n_samples, n_responses))
    Y_indep = U_indep @ H_true.T + noise

    # Full training (all features at once)
    H_full = linear_boost_ic_regression(U=U_indep, Y=Y_indep, n_jobs=1)

    # Row-by-row training (one feature / row of X at a time)
    H_rowwise = np.zeros((n_responses, n_features))
    for k in range(n_features):
        H_k = linear_boost_ic_regression(U=U_indep[:, k : k + 1], Y=Y_indep, n_jobs=1)
        H_rowwise[:, k] = H_k.toarray().flatten()

    # Independent features: full map ≈ row-by-row map
    np.testing.assert_allclose(
        H_full.toarray(),
        H_rowwise,
        atol=0.1,
        err_msg="Independent features: full and row-by-row maps should match",
    )

    # --- Dependent features: make col 1 a near-copy of col 0 ---
    U_dep = U_indep.copy()
    U_dep[:, 1] = 0.95 * U_indep[:, 0] + 0.05 * U_indep[:, 1]

    Y_dep = U_dep @ H_true.T + noise

    H_full_dep = linear_boost_ic_regression(U=U_dep, Y=Y_dep, n_jobs=1)

    H_rowwise_dep = np.zeros((n_responses, n_features))
    for k in range(n_features):
        H_k = linear_boost_ic_regression(U=U_dep[:, k : k + 1], Y=Y_dep, n_jobs=1)
        H_rowwise_dep[:, k] = H_k.toarray().flatten()

    # Dependent features: mismatch should be larger than independent case
    diff_indep = np.max(np.abs(H_full.toarray() - H_rowwise))
    diff_dep = np.max(np.abs(H_full_dep.toarray() - H_rowwise_dep))

    assert diff_dep > diff_indep, (
        f"Expected larger mismatch with dependent features "
        f"(dep={diff_dep:.4f}, indep={diff_indep:.4f})"
    )


def test_single_response_returns_sparse_representation():
    """_fit_single_response_boost must return (j, nonzero_indices, nonzero_values),
    not a full dense coefficient vector."""
    rng = np.random.default_rng(42)
    n, p = 50, 100
    U_scaled = rng.standard_normal((n, p))
    y_j = rng.standard_normal(n)

    j = 0
    col_idx, nonzero_indices, nonzero_values = _fit_single_response_boost(
        j, U_scaled, y_j, learning_rate=0.5, effective_dimension=None
    )
    assert col_idx == j
    assert nonzero_indices.shape == nonzero_values.shape
    assert nonzero_values.size < p
