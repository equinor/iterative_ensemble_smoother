import networkx as nx
import numpy as np
import pytest
import scipy as sp
from numpy.typing import NDArray
from scipy.optimize import minimize

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
    U = rng.multivariate_normal(mean=np.zeros(n), cov=Cov, size=99)

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
