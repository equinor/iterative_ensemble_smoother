import networkx as nx
import numpy as np
import pytest
import scipy as sp
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from iterative_ensemble_smoother import enif_precision_estimation as precest


def objective_function(
    C_k: NDArray[np.floating], U: NDArray[np.floating], lambda_l2: float = 1.0
) -> float:
    """
    Objective function for optimizing the affine KR map with standard Gaussian
    reference and l2 regularized dependence.

    Parameters
    ----------
    C_k : np.ndarray
        The current estimate of non-zero elements in row of C-factor of
        associate precision matrix CTC = Prec.
    U : np.ndarray
        The data matrix ordered according to CTC=Prec_u.
    lambda_l2 : float, optional
        The regularization strength for L2 regularization.

    Returns
    -------
    float
        The value of `objective_function`.
    """
    C_k = C_k.copy()

    C_k[-1] = np.exp(C_k[-1])
    Su = U.dot(C_k)
    n, _ = U.shape
    regularization_l2 = 0.5 * lambda_l2 * np.sum(C_k[:-1] ** 2)
    return 0.5 * np.sum(Su**2) - n * np.log(abs(C_k[-1])) + regularization_l2


def gradient(
    C_k: NDArray[np.floating], U: NDArray[np.floating], lambda_l2: float = 1.0
) -> NDArray[np.floating]:
    """
    Gradient of the objective function.

    Parameters
    ----------
    C_k : np.ndarray
        The current estimate of non-zero elements in row of C-factor of
        associate precision matrix CTC = Prec.
    U : np.ndarray
        The data matrix ordered according to CTC=Prec_u.
    lambda_l2 : float, optional
        The regularization strength for L2 regularization.

    Returns
    -------
    np.ndarray
        The gradient of the objective function.
    """
    C_k = C_k.copy()

    n, _ = U.shape
    C_k[-1] = np.exp(C_k[-1])
    prediction = U.dot(C_k)
    grad = U.T.dot(prediction)
    grad[:-1] += lambda_l2 * C_k[:-1]  # Adjust for L2 regularization
    grad[-1] -= n / C_k[-1]  # Adjust for the -log|C_k,k| term
    grad[-1] *= C_k[-1]  # Adjust for log-transform
    return grad


def hessian(
    C_k: NDArray[np.floating], U: NDArray[np.floating], lambda_l2: float = 1.0
) -> NDArray[np.floating]:
    """
    Hessian `objective_function`.

    Parameters
    ----------
    C_k : np.ndarray
        The current estimate of non-zero elements in row of C-factor of
        associate precision matrix CTC = Prec.
    U : np.ndarray
        The data matrix ordered according to CTC=Prec_u.
    lambda_l2 : float, optional
        The regularization strength for L2 regularization.

    Returns
    -------
    np.ndarray
        The Hessian of the objective function.
    """
    C_k = C_k.copy()

    n, _ = U.shape
    H = U.T.dot(U)
    np.fill_diagonal(H[:-1, :-1], H.diagonal()[:-1] + lambda_l2)  # L2-term
    C_k[-1] = np.exp(C_k[-1])  # log-transform
    H[-1, -1] += n / (C_k[-1] ** 2)  # Adjust for the -log|C_k,k| term
    H[-1, -1] *= 2.0 * C_k[-1]  # log-transform adjustment
    return H


def get_precision_data():
    rng = np.random.default_rng(8)
    n = 10  # Size
    density = 0.4

    # Create G indicating sparsity pattern
    G = rng.uniform(size=(n, n)) < (density / 2)
    G = G.T + G
    np.fill_diagonal(G, G.diagonal() + 1)
    G_matrix = (G > 0).astype(int)
    Graph_u = nx.from_scipy_sparse_array(sp.sparse.csc_array(G_matrix))

    # Create data U
    U = rng.normal(size=(999, n))

    return U, Graph_u, G_matrix


@pytest.mark.suitesparse
def test_snapshot_fit_precision_cholesky():
    U, Graph_u, G_matrix = get_precision_data()

    # Estimate precision with fit_precision_cholesky.
    # Cannot use METIS (not reproducible across OSes), use 'natural'
    Prec_est = precest.fit_precision_cholesky(
        U=U, Graph_u=Graph_u, ordering_method="natural"
    )
    Prec_est = Prec_est.todense()

    entries_at_one = Prec_est[G_matrix > 0]
    entries_at_zero = Prec_est[G_matrix == 0]

    desired = np.array([1.03393041, -0.05494438, 1.02092228, 1.00923821])
    np.testing.assert_allclose(entries_at_one[::9], desired, atol=1e-8)

    desired = np.array([0.0, 0.0, 0.0, 0.01763788, -0.03135188, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(entries_at_zero[::9], desired, atol=1e-8)


def test_snapshot_fit_precision_cholesky_approximate():
    U, Graph_u, G_matrix = get_precision_data()
    # Estimate precision with fit_precision_cholesky
    Prec_est = precest.fit_precision_cholesky_approximate(
        U=U, Graph_u=Graph_u, neighbourhood_expansion=2
    )
    Prec_est = Prec_est.todense()

    entries_at_one = Prec_est[G_matrix > 0]
    entries_at_zero = Prec_est[G_matrix == 0]

    desired = np.array([1.03392773, -0.05606645, 1.022626, 1.00883855])
    np.testing.assert_allclose(entries_at_one[::9], desired, atol=1e-8)

    desired = np.array(
        [0.0, -0.04588378, -0.00330536, -0.00124644, -0.0301345, -0.02301515, 0.0, 0.0]
    )
    np.testing.assert_allclose(entries_at_zero[::9], desired, atol=1e-8)


@pytest.mark.suitesparse
@pytest.mark.parametrize("seed", range(99))
def test_precision_cholesky_roundtrip(seed):
    """Starting from a known, sparse precision matrix, we generate data,
    then try to infer the known values from the samples."""

    # Create sparse, pos.def precision matrix
    rng = np.random.default_rng(seed)
    n = 25  # Size
    density = 0.1

    # Create sparse pos def precision matrix
    F = rng.normal(size=(n, n))
    F[rng.uniform(size=(n, n)) > density] = 0
    Prec = F.T @ F + np.eye(n)
    assert np.all(np.linalg.svd(Prec).S > 0), "Pos def"

    G_matrix = (~np.isclose(Prec, 0.0)).astype(int)
    Graph_u = nx.from_scipy_sparse_array(sp.sparse.csc_array(G_matrix))

    Cov = np.linalg.inv(Prec)
    U = rng.multivariate_normal(mean=np.zeros(n), cov=Cov, size=99, method="cholesky")

    # Estimate precision using known structure
    Prec_est = precest.fit_precision_cholesky(
        U=U, Graph_u=Graph_u, ordering_method="amd"
    ).todense()

    RMSE = np.sqrt(np.mean((Prec - Prec_est) ** 2))

    # Estimate the naive way - invert the empirical covariance
    Prec_naive = np.linalg.inv(np.cov(U, rowvar=False))
    RMSE_naive = np.sqrt(np.mean((Prec - Prec_naive) ** 2))

    # Here 0.77 was chosen to make all tests pass, to easier catch
    # regressions. Nothing special about the number. Main idea: beat naive!
    assert RMSE_naive * 0.77 > RMSE


def _report_mean_comparison(name, method_metrics):
    seed_count = len(next(iter(method_metrics.values())))
    summary = [f"{name} ({seed_count} seeds)"]
    mean_rmses = {}
    for method, rmses in method_metrics.items():
        mean_rmses[method] = float(np.mean(rmses))
        summary.append(f"{method}: mean RMSE={mean_rmses[method]:.6f}")
    print(" | ".join(summary))
    return mean_rmses


def _estimate_precisions(samples, positions, graph, ordering_method):
    return {
        "maximin": precest.fit_precision_cholesky_maximin(
            U=samples, grid_points=positions
        ).toarray(),
        "complete": precest.fit_precision_cholesky(
            U=samples, Graph_u=graph, ordering_method=ordering_method
        ).toarray(),
        "approximate": precest.fit_precision_cholesky_approximate(
            U=samples, Graph_u=graph, neighbourhood_expansion=2
        ).toarray(),
    }


def _compare_over_seeds(name, make_case, ordering_method="natural"):
    method_metrics = {method: [] for method in ("maximin", "complete", "approximate")}
    for seed in range(9):
        rng = np.random.default_rng(seed)
        precision_true, covariance, positions, graph = make_case(rng)
        samples = rng.multivariate_normal(
            mean=np.zeros(len(positions)),
            cov=covariance,
            size=99,
            method="cholesky",
        )
        estimates = _estimate_precisions(samples, positions, graph, ordering_method)
        for method, precision_est in estimates.items():
            rmse = np.sqrt(np.mean((precision_true - precision_est) ** 2))
            method_metrics[method].append(rmse)

    return _report_mean_comparison(name, method_metrics)


@pytest.mark.suitesparse
def test_reverse_maximin_vs_cholesky_tridiagonal():
    """Compare the three methods

    1. Complete Cholesky
    2. Incomplete Cholesky
    3. Maximin ordering

    on a 25-node 1D chain using matching one-dimensional coordinates.
    The precistion matrix is tridiagonal."""
    # Create tridiagonal, positive-definite precision
    num_points = 25
    precision_true = (
        2.0 * np.eye(num_points)
        - 0.9 * np.eye(num_points, k=1)
        - 0.9 * np.eye(num_points, k=-1)
    )

    # Compute the true covariance
    covariance = np.linalg.inv(precision_true)

    # Define matching chain coordinates and graph
    positions = np.arange(num_points, dtype=float)[:, None]
    graph = nx.path_graph(num_points)

    # Estimate precision and compare mean RMSE across seeds
    mean_rmses = _compare_over_seeds(
        "tridiagonal",
        lambda _rng: (precision_true, covariance, positions, graph),
    )
    assert mean_rmses["complete"] < mean_rmses["approximate"] < mean_rmses["maximin"]


@pytest.mark.suitesparse
def test_reverse_maximin_vs_cholesky_random_sparse():
    """Compare the three methods

    1. Complete Cholesky
    2. Incomplete Cholesky
    3. Maximin ordering

    on a 2D example with a randomly sparse positive-definite precision matrix.
    A four-neighbour grid graph is supplied to the complete and incomplete
    Cholesky methods. This graph is independent from the random sparsity pattern.
    """
    # Set up the grid and seed-averaged metrics
    side = 25
    num_points = side**2
    density = 0.1
    positions = np.indices((side, side), dtype=float).reshape(2, -1).T
    method_metrics = {method: [] for method in ("maximin", "complete", "approximate")}

    for seed in range(9):
        # Create random sparse, positive-definite precision
        rng = np.random.default_rng(seed)
        random_pattern = np.triu(
            rng.uniform(size=(num_points, num_points)) < density, k=1
        )
        precision_true = rng.normal(size=(num_points, num_points)) * random_pattern
        precision_true = precision_true + precision_true.T
        np.fill_diagonal(precision_true, np.sum(np.abs(precision_true), axis=1) + 1.0)
        assert np.count_nonzero(precision_true) == num_points + 2 * np.count_nonzero(
            random_pattern
        )

        # Define the independent four-neighbour graph
        graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
        grid_pattern = nx.to_numpy_array(graph, dtype=bool)
        np.fill_diagonal(grid_pattern, True)
        assert np.any(precision_true[~grid_pattern] != 0)

        # Generate samples from the true covariance
        covariance = np.linalg.inv(precision_true)
        samples = rng.multivariate_normal(
            mean=np.zeros(num_points), cov=covariance, size=99, method="cholesky"
        )

        # Estimate precision with all three methods
        estimates = _estimate_precisions(
            samples, positions, graph, ordering_method="natural"
        )

        # Collect RMSE for each estimate
        for method, precision_est in estimates.items():
            rmse = np.sqrt(np.mean((precision_true - precision_est) ** 2))
            method_metrics[method].append(rmse)

    # Compare mean RMSE across seeds
    mean_rmses = _report_mean_comparison("random sparse", method_metrics)
    assert mean_rmses["approximate"] < mean_rmses["maximin"] < mean_rmses["complete"]


@pytest.mark.suitesparse
def test_reverse_maxmin_vs_cholesky_exp_decay():
    """Compare the three methods

    1. Complete Cholesky
    2. Incomplete Cholesky
    3. Maximin ordering

    on a 2D example with exponentially decaying covariance. The true precision
    is the dense inverse of this covariance. The four-neighbour grid graph
    is used as a local sparsity approximation.
    """
    # Set up the grid coordinates
    side = 16
    length_scale = 1.0
    positions = np.indices((side, side), dtype=float).reshape(2, -1).T

    # Create the exponentially decaying covariance
    covariance = np.exp(-cdist(positions, positions) / length_scale)

    # Compute the true precision
    precision_true = np.linalg.inv(covariance)

    # Define the local four-neighbour graph
    graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))

    # Estimate precision and compare mean RMSE across seeds
    mean_rmses = _compare_over_seeds(
        "exponential decay",
        lambda _rng: (precision_true, covariance, positions, graph),
    )
    assert mean_rmses["approximate"] < mean_rmses["maximin"] < mean_rmses["complete"]


def test_objective_twice():
    # A regression test: ensure that two calls return the same result.
    rng = np.random.default_rng(42)

    C_k = np.exp(rng.normal(0, 0.1, size=5))
    U = rng.normal(size=(5, 5))

    value1 = objective_function(C_k, U)
    value2 = objective_function(C_k, U)
    np.testing.assert_allclose(value1, value2)

    # Check gradient
    rmse = sp.optimize.check_grad(
        objective_function,
        gradient,
        np.array([1, 2, 3, 4, 4.5]),
        U,
        rng=rng,
    )
    assert rmse <= 0.002


def test_closed_form_matches_iterative_solver():
    """Closed-form row solver agrees with the iterative (L-BFGS-B) solution."""
    rng = np.random.default_rng(0)
    n, n_cols = 200, 5
    U_reduced = rng.normal(size=(n, n_cols))
    lambda_l2 = 2.0 * n_cols

    # Closed-form solution introduced in commit 864f430
    off_diag_cf, diag_cf = precest.solve_row_closed_form(U_reduced, lambda_l2)

    # Iterative reference solution (L-BFGS-B on the log-diagonal parametrisation)
    x0 = np.zeros(n_cols)
    res = minimize(
        fun=objective_function,
        x0=x0,
        args=(U_reduced, lambda_l2),
        method="L-BFGS-B",
        jac=gradient,
        tol=1e-12,
        options={"gtol": 1e-9},
    )
    off_diag_iter = res.x[:-1]
    diag_iter = np.exp(res.x[-1])

    np.testing.assert_allclose(off_diag_cf, off_diag_iter, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(diag_cf, diag_iter, rtol=1e-4)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v"])
