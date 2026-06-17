"""
Precision estimation
--------------------

This module contains functions for solving the following problem:
    Given a known sparsity pattern in a precision matrix,
    as well as a data set, how can we estimate the precision matrix values?
"""

import logging
import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse import csc_array, tril
from sksparse.cholmod import cholesky
from tqdm import tqdm

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def reverse_cholesky(
    A: csc_array, *args: object, **kwargs: object
) -> tuple[csc_array, NDArray[np.integer]]:
    """Given a sparse pos. def. matrix A, compute C and P such that
    C.T @ C == P @ A @ P.T, where C is lower-triangular and P is a permutation.

    This differs from the standard Cholesky factorization, which computes
    L @ L.T = P @ A @ P.T, where L is lower-triangular.

    Returns (C: csc_array, permutation_idx: array). The permutation index
    acts like P on the left when indexing the rows. See examples.

    Examples
    --------
    >>> F = sp.random_array(shape=(10, 10), density=0.2, rng=42, format='csc')
    >>> A = F @ F.T
    >>> A.setdiag(10)
    >>> C, permutation_idx = reverse_cholesky(A)
    >>> diff = C.T @ C - A[np.ix_(permutation_idx, permutation_idx)]
    >>> float(np.mean(np.abs(diff)))
    9.7705...e-17

    Or, equivalently (C @ P).T @ (C @ P) = A.
    Here P acts on the right, so we must use its inverse:

    >>> inverse_perm = np.empty_like(permutation_idx)
    >>> inverse_perm[permutation_idx] = np.arange(len(permutation_idx))

    >>> F = C[:, inverse_perm]
    >>> diff = F.T @ F - A
    >>> float(np.mean(np.abs(diff)))
    9.7705...e-17

    The usage of the permutation index (expressing the matrix P), follows the
    convention established by standard Cholesky. This function is a thin
    wrapper around this kind of code (but this function computes C, not L):

    >>> from sksparse.cholmod import cholesky
    >>> factor = cholesky(A)
    >>> L, permutation_idx = factor.L(), factor.P()
    >>> diff = L @ L.T - A[np.ix_(permutation_idx, permutation_idx)]
    >>> float(np.mean(np.abs(diff)))
    1.3323...e-16
    """
    cholesky_factor = cholesky(A, *args, **kwargs)
    L = cholesky_factor.L()
    permutation_idx = cholesky_factor.P()

    # If we ignore the permutation, then we have the relationship:
    #   C = cholesky(A[::-1, ::-1]).T[::-1, ::-1]
    C = L[::-1, ::-1].T
    return C, permutation_idx[::-1]


def solve_row_closed_form(
    U_reduced: np.ndarray,
    lambda_l2: float,
) -> tuple[np.ndarray, float]:
    """
    Closed-form minimizer for one row objective.

    This function computes the exact analytic solution for the row-wise objective
    function, avoiding the need for iterative numerical optimization.
    For the full mathematical derivation of this closed-form solution,
    see ``docs/source/ClosedFormRowSolver.md``.

    Parameters
    ----------
    U_reduced : np.ndarray
        Reduced data matrix with columns corresponding to non-zero entries
        in row ``k`` ordered as ``[..., k]``.
    lambda_l2 : float
        L2 regularization weight on off-diagonal coefficients.

    Returns
    -------
    tuple[np.ndarray, float]
        Off-diagonal coefficients and positive diagonal coefficient.

    Raises
    ------
    ValueError
        If numerical degeneracy prevents a valid positive diagonal estimate.
    """
    n, n_cols = U_reduced.shape
    z = U_reduced[:, -1]

    if n_cols == 1:
        alpha = float(np.dot(z, z))
        if alpha <= 0.0:
            raise ValueError("Degenerate row: non-positive alpha in closed-form solve")
        diag_value = np.sqrt(n / alpha)
        return np.empty(0, dtype=U_reduced.dtype), float(diag_value)

    X = U_reduced[:, :-1]
    gram = X.T @ X
    np.fill_diagonal(gram, np.diag(gram) + lambda_l2)
    rhs = X.T @ z

    beta_tilde = np.linalg.solve(gram, rhs)
    alpha = float(np.dot(z, z) - np.dot(rhs, beta_tilde))
    if alpha <= 0.0:
        raise ValueError("Degenerate row: non-positive alpha in closed-form solve")

    diag_value = np.sqrt(n / alpha)
    off_diag = -diag_value * beta_tilde
    return off_diag, float(diag_value)


def optimize_sparse_affine_kr_map(
    U: NDArray[np.floating],
    G: nx.Graph,
    use_tqdm: bool = True,
) -> csc_array:
    """Optimize the affine Knothe-Rosenblatt (KR) map with standard Gaussian
    reference and l2-regularized dependence using the closed-form row solve.

    Parameters
    ----------
    U : np.ndarray
        The data matrix with shape (samples, parameters)
    G : networkx.Graph
        The graph representing the non-zero structure in C.
        Where C is the Cholesky factor such that C.T @ C = Prev.

    Returns
    -------
    scipy.sparse.csc_array
        The optimized sparse Cholesky factor C of the precision matrix,
        such that C.T @ C = Prec.

    Examples
    --------

    In this example we start with a known precision matrix, generate data,
    then use that data to try to infer the precision matrix again.

    Create precision matrix Prec:

    >>> import numpy as np
    >>> import scipy as sp
    >>> import networkx as nx
    >>> rng = np.random.default_rng(42)
    >>> diagonal = np.array([1, 2, 3, 4], dtype=float)
    >>> Prec = np.diag(diagonal) + np.diag(diagonal[:-1] / 2, k=1)
    >>> Prec += np.diag(diagonal[:-1] / 2, k=-1)
    >>> Prec
    array([[1. , 0.5, 0. , 0. ],
           [0.5, 2. , 1. , 0. ],
           [0. , 1. , 3. , 1.5],
           [0. , 0. , 1.5, 4. ]])

    Sample data U and create corresponding graph G:

    >>> mean = np.ones(Prec.shape[0])
    >>> cov = np.linalg.inv(Prec)
    >>> U = rng.multivariate_normal(mean=mean, cov=cov, size=999)

    The non-zero structure in the Cholesky factor of the Precision:

    >>> G_mat = np.eye(4) + np.diag(np.ones(3), k=-1)
    >>> G_mat = sp.sparse.csc_array(G_mat.astype(int))
    >>> G = nx.from_scipy_sparse_array(G_mat)
    >>> G.edges
    EdgeView([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)])

    Estimate the precision matrix from U and G:

    >>> C = optimize_sparse_affine_kr_map(U=U, G=G, use_tqdm=False).todense()
    >>> Prec_est = (C.T @ C)
    >>> Prec_est.round(2) # Estimated precision matrix
    array([[0.99, 0.46, 0.  , 0.  ],
           [0.46, 2.  , 1.03, 0.  ],
           [0.  , 1.03, 3.19, 1.46],
           [0.  , 0.  , 1.46, 3.69]])

    Demonstrate shift invariance. Notice that we get the same result as above:

    >>> U_shift = U +  np.array([1, 10, 100, 1000])
    >>> C = optimize_sparse_affine_kr_map(U=U_shift, G=G, use_tqdm=False).todense()
    >>> (C.T @ C).round(2)
    array([[0.99, 0.46, 0.  , 0.  ],
           [0.46, 2.  , 1.03, 0.  ],
           [0.  , 1.03, 3.19, 1.46],
           [0.  , 0.  , 1.46, 3.69]])

    Demonstrate affine invariance (shift and scale):

    >>> mu = np.array([5, 2, 3, 1])
    >>> sigma = np.array([1, 2, 4, 8])
    >>> U_scaled = (U + mu) * sigma
    >>> C_scaled = optimize_sparse_affine_kr_map(U=U_scaled, G=G, use_tqdm=False).todense()
    >>> Prec_est_scaled = (C_scaled.T @ C_scaled)
    >>> Prec_est_scaled.round(2)
    array([[0.99, 0.23, 0.  , 0.  ],
           [0.23, 0.5 , 0.13, 0.  ],
           [0.  , 0.13, 0.2 , 0.05],
           [0.  , 0.  , 0.05, 0.06]])

    Notice that this is the same as above:

    >>> Prec_est2 = np.diag(sigma) @ Prec_est_scaled @ np.diag(sigma)
    >>> (Prec_est2).round(2)
    array([[0.99, 0.46, 0.  , 0.  ],
           [0.46, 2.  , 1.03, 0.  ],
           [0.  , 1.03, 3.19, 1.46],
           [0.  , 0.  , 1.46, 3.69]])
    >>> np.allclose(Prec_est, Prec_est2)
    True
    """
    log.info("Starting statistical fitting of precision")

    n, p = U.shape
    assert n > 1

    # Standardize data U to have zero mean, unit variance
    mu = U.mean(axis=0, keepdims=True)
    sigma = U.std(axis=0)
    U_std = (U - mu) / sigma[None, :]

    C_full = sp.lil_array((p, p))  # lil_array for efficient row operations
    loop_function = (
        tqdm(range(p), desc="Learning precision Cholesky factor row-by-row")
        if use_tqdm
        else range(p)
    )

    for k in loop_function:
        non_zero_indices = [j for j in G.neighbors(k) if j < k] + [k]

        # Extract the reduced version of U
        U_reduced = U_std[:, non_zero_indices]

        # Optimization for reduced C_k
        lambda_l2_aic = 2.0 * len(non_zero_indices)
        off_diag_std, diag_value_std = solve_row_closed_form(
            U_reduced=U_reduced,
            lambda_l2=lambda_l2_aic,
        )

        # Unscale the row back to original coordinates
        if len(non_zero_indices) > 1:
            cols_off = non_zero_indices[:-1]
            C_full[k, cols_off] = off_diag_std / sigma[cols_off]
        C_full[k, k] = diag_value_std / sigma[k]

    # Convert to csc_array for efficient storage and arithmetic operations
    return C_full.tocsc()


def fit_precision_cholesky(
    U: NDArray[np.floating],
    Graph_u: nx.Graph,
    *,
    ordering_method: str = "metis",
    use_tqdm: bool = True,
) -> tuple[csc_array, nx.Graph, NDArray[np.integer], csc_array, csc_array]:
    """
    Estimate the precision matrix using Cholesky decomposition.
    An l2-regularized negative log-likelihood is minimized.

    Parameters
    ----------
    U : np.ndarray
        The data matrix with shape (samples, parameters)
    G : networkx.Graph
        Graph representing non-zero structure in the precision matrix.

    Returns
    -------
    scipy.sparse.csc_array
        Estimated precision matrix.
    """
    _, p = U.shape
    assert len(Graph_u.nodes) == p, "nodes in graph equals columns of data"

    # Step 1: Cholesky factor of the dependency graph Graph_u
    # -------------------------------------------------------

    # Create pos. def. matrix with same sparsity structure as Prec
    SPD_Prec = nx.to_scipy_sparse_array(
        Graph_u, weight=None, dtype=np.float64, format="csc"
    )
    # Use Gershgorin circle theorem to ensure positive definite
    # All eigenvalues are in a circle centered at max_degree+1.0
    # and radius < (max_degree+1.0), so guaranteed > 0
    SPD_Prec.setdiag(max(dict(Graph_u.degree()).values()) + 1.0)

    # Compute Cholesky of the sparsity pattern (a symbolic cholesky)
    start = time.perf_counter()
    C_pattern, permutation_idx = reverse_cholesky(
        SPD_Prec, ordering_method=ordering_method
    )
    end = time.perf_counter()
    log.info("Cholesky of precision matrix took %.2f seconds", end - start)
    Graph_C = nx.from_scipy_sparse_array(C_pattern)
    log.info("Parameters in precision: %s\n", tril(SPD_Prec).nnz)
    log.info("Parameters in Cholesky factor: %s", Graph_C.number_of_edges())

    inverse_perm = np.empty_like(permutation_idx)
    inverse_perm[permutation_idx] = np.arange(len(permutation_idx))

    # Step 2: Estimate non-zeros of C, the (reverse) Cholesky factor
    # --------------------------------------------------------------

    # The function 'reverse_cholesky' above has found a C that is a factor
    # of a permuted Graph_u (not the original one), so we must permute U too.
    # Why permute at all? To have as many zeroes as possible in C.
    U_perm = U[:, permutation_idx]
    C = optimize_sparse_affine_kr_map(
        U=U_perm,
        G=Graph_C,
        use_tqdm=use_tqdm,
    )

    # Compute log-determinant of estimate
    prec_logdet = 2.0 * np.sum(np.log(C.diagonal()))
    log.info("Precision has log-determinant: %.3f", prec_logdet)

    # Step 3: Compute precision matrix from C and invert permutation
    # --------------------------------------------------------------

    # Inverse permutation
    inverse_perm = np.empty_like(permutation_idx)
    inverse_perm[permutation_idx] = np.arange(len(permutation_idx))

    # Unwrap C to yield precision (Eqn 73 in paper)
    # Equivalent to: (C.T @ C)[np.ix_(inverse_perm, inverse_perm)]
    C_orig = C[:, inverse_perm]
    return C_orig.T @ C_orig


def fit_precision_cholesky_approximate(
    U: NDArray[np.floating],
    Graph_u: nx.Graph,
    neighbourhood_expansion: int = 2,
    use_tqdm: bool = True,
) -> csc_array:
    """
    Estimate the precision matrix using approximate Cholesky.
    The Cholesky is assumed as sparse as the corresponding precision, with
    sparsity pattern from G, but with increased neighbourhood. This is
    akin to a Vecchia approximation, which is alleviated by the neighbourhood
    expansion.

    No permutation optimisation is performed. It is beneficial if U and G have
    a "sensible" ordering. This may e.g. be that neighbours defined by G are
    close in U.

    Parameters
    ----------
    U : np.ndarray
        The data matrix with shape (samples, parameters)
    G : networkx.Graph
        Graph representing non-zero structure in the precision matrix.
    neighbourhood_expansion: int, optional
        The number of hops to the new neighbourhood set

    Returns
    -------
    scipy.sparse.csc_array
        The optimized sparse Cholesky factor of the precision matrix.
    """
    # The k'th power is a "neighborhood expansion", see docs of nx.power
    G = nx.power(Graph_u, k=neighbourhood_expansion)
    G.add_edges_from((n, n) for n in G.nodes())  # Add edges (1, 1), (2, 2), ...

    C = optimize_sparse_affine_kr_map(
        U=U,
        G=G,
        use_tqdm=use_tqdm,
    )
    Prec_approx = C.T @ C
    return Prec_approx.tocsc()


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v"])


"""Gaussian graphical model with a known (fixed) edge structure.

Estimates a sparse precision (inverse-covariance) matrix whose zero pattern is
*supplied* by a graph, rather than learned from an L1 penalty as in
:class:`sklearn.covariance.GraphicalLasso`.
"""

import warnings

import numpy as np
from scipy import linalg, sparse
from sklearn.covariance import EmpiricalCovariance, empirical_covariance
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array


class StructuredGraphicalModel(EmpiricalCovariance):
    """Gaussian graphical model with a *known* conditional-independence structure.

    Given a fixed undirected graph ``G`` over the ``p`` variables, this estimates
    the Gaussian precision (inverse-covariance) matrix ``Theta`` whose zero
    pattern matches the *missing* edges of ``G``::

        Theta[i, j] == 0   whenever   i != j  and  G[i, j] == 0.

    Diagonal entries are always free (a precision diagonal is never zero), so the
    diagonal of ``G`` is ignored, and ``G`` is symmetrised since the precision is
    symmetric.

    This is the equality-constrained Gaussian maximum-likelihood problem

        maximize_Theta   log det Theta - trace(S @ Theta)
        subject to       Theta[i, j] = 0  for every absent edge (i, j),

    i.e. *covariance selection* / the positive-definite completion of ``S``
    (Dempster, 1972). With ``ridge == 0`` it is solved exactly by the modified
    block coordinate-descent regression of Hastie, Tibshirani & Friedman,
    *The Elements of Statistical Learning*, Algorithm 17.1: each column ``j`` is a
    regression of node ``j`` on its neighbours only, using the *current model
    covariance* ``W11`` as the cross-product matrix instead of the raw ``S11``.

    Parameters
    ----------
    adjacency_matrix : array-like or sparse matrix of shape (n_features, n_features)
        Boolean structure ``G``. A nonzero off-diagonal ``G[i, j]`` marks an edge,
        i.e. a free precision entry ``Theta[i, j]``. Symmetrised internally; the
        diagonal is ignored.

    ridge : float, default=0.0
        L2 (Tikhonov) penalty added to each per-node regression. The block update
        solves ``(W11* + ridge * I) @ beta* = s12*`` instead of
        ``W11* @ beta* = s12*``. Since ``beta = -theta12 / theta22`` are the
        partial-regression coefficients, this shrinks the (nonzero) off-diagonal
        precision entries toward zero and stabilises the column solves when ``S``
        is ill-conditioned. ``ridge == 0`` recovers the exact constrained MLE.

    tol : float, default=1e-4
        Convergence tolerance: iterations stop when the largest absolute change
        in ``covariance_`` between two sweeps falls below ``tol``.

    max_iter : int, default=100
        Maximum number of full coordinate-descent sweeps.

    assume_centered : bool, default=False
        If True, data are assumed zero-mean and not centered before estimating
        the empirical covariance.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated mean (zeros if ``assume_centered``).

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance ``W`` (the structured completion of ``S``).

    precision_ : ndarray of shape (n_features, n_features)
        Estimated sparse precision ``Theta`` respecting the zero pattern of ``G``.
        For ``ridge == 0`` this is exactly ``covariance_ ** -1``; for
        ``ridge > 0`` the two are no longer exact inverses.

    n_iter_ : int
        Number of sweeps run.

    Notes
    -----
    The graph is solved as ``p`` *coupled* regressions, not ``p`` independent
    ones: sharing the model covariance ``W`` across columns is what enforces the
    global constraint and yields the exact completion (ESL, sec. 17.3.1).

    References
    ----------
    .. [1] Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*,
           2nd ed., Algorithm 17.1.
    .. [2] A. Dempster, "Covariance selection", Biometrics, 1972.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_sparse_spd_matrix
    >>> prec = make_sparse_spd_matrix(10, alpha=0.7, random_state=0)
    >>> G = prec != 0
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(np.zeros(10), np.linalg.inv(prec), size=500)
    >>> model = StructuredGraphicalModel(adjacency_matrix=G).fit(X)
    >>> model.precision_.shape
    (10, 10)
    """

    def __init__(
        self,
        adjacency_matrix,
        *,
        ridge=0.0,
        tol=1e-4,
        max_iter=100,
        assume_centered=False,
    ):
        super().__init__(store_precision=True, assume_centered=assume_centered)
        self.adjacency_matrix = adjacency_matrix
        self.ridge = ridge
        self.tol = tol
        self.max_iter = max_iter

    # ------------------------------------------------------------------ utils
    def _neighbourhood(self):
        """Return a symmetric boolean (p, p) adjacency with a zeroed diagonal."""
        G = self.adjacency_matrix
        if sparse.issparse(G):
            G = G.toarray()
        G = np.asarray(G).astype(bool)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise ValueError("adjacency_matrix must be a square (p, p) matrix.")
        G = G | G.T  # precision is symmetric
        np.fill_diagonal(G, False)  # diagonal entries are always free
        return G

    def _column_update(self, W, emp_cov, j, nbr):
        """ESL Algorithm 17.1 step (b)/(c) for column ``j``.

        ``nbr`` are the integer indices of ``j``'s neighbours (always excluding
        ``j``, since the diagonal of ``G`` is zeroed). Because ``beta`` is zero
        off the edge set, only the neighbour block of ``W`` is touched: the
        reduced system is ``q x q`` and ``w12 = W[:, nbr] @ beta`` is O(p*q),
        avoiding any ``(p-1) x (p-1)`` copy.

        Returns the ``q``-vector ``beta`` and the full length-``p`` column
        ``col = W[:, j]`` with its diagonal slot preserved.
        """
        q = nbr.size
        if q == 0:
            col = np.zeros(W.shape[0])
            col[j] = W[j, j]
            return np.empty(0), col
        A = W[np.ix_(nbr, nbr)]  # q x q (advanced indexing -> own copy)
        if self.ridge:
            A[np.diag_indices(q)] += self.ridge
        b = emp_cov[nbr, j]
        try:
            beta = linalg.solve(A, b, assume_a="pos")
        except linalg.LinAlgError:
            beta = linalg.lstsq(A, b)[0]
        col = W[:, nbr] @ beta  # length p; w12 over the off-diagonal rows
        col[j] = W[j, j]  # keep w22 == s22 (diagonal is unpenalised)
        return beta, col

    # ------------------------------------------------------------------- fit
    def fit(self, X, y=None):
        """Fit the model to ``X`` of shape (n_samples, n_features).

        Returns
        -------
        self : object
        """
        X = check_array(X)
        G = self._neighbourhood()
        p = X.shape[1]
        if G.shape[0] != p:
            raise ValueError(
                f"adjacency_matrix has {G.shape[0]} nodes but X has {p} features."
            )

        self.location_ = np.zeros(p) if self.assume_centered else X.mean(0)
        emp_cov = empirical_covariance(X, assume_centered=self.assume_centered)

        # Integer neighbour indices per node (j excluded: G's diagonal is zeroed).
        neighbours = [np.flatnonzero(G[j]) for j in range(p)]

        # --- step 2: block coordinate descent on the model covariance W -----
        W = emp_cov.copy()
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            W_old = W.copy()
            for j in range(p):
                _, col = self._column_update(W, emp_cov, j, neighbours[j])
                W[:, j] = col  # diagonal slot already preserved in col
                W[j, :] = col
            if np.abs(W - W_old).max() < self.tol:
                break
        else:
            warnings.warn(
                f"StructuredGraphicalModel did not converge after "
                f"{self.max_iter} iterations.",
                ConvergenceWarning,
            )

        # --- step 3: recover the structured precision Theta from W ----------
        precision = np.zeros((p, p))
        diag = np.zeros(p)
        for j in range(p):
            nbr = neighbours[j]
            beta, col = self._column_update(W, emp_cov, j, nbr)
            schur = col[nbr] @ beta if nbr.size else 0.0  # w12 @ beta
            theta22 = 1.0 / (emp_cov[j, j] - schur)
            diag[j] = theta22
            precision[nbr, j] = -theta22 * beta  # column j only
        # average the two per-column estimates of each off-diagonal entry
        # (identical when ridge == 0, a symmetric average otherwise)
        precision = 0.5 * (precision + precision.T)
        precision[np.diag_indices(p)] = diag

        self.covariance_ = W
        self.precision_ = precision
        self.n_iter_ = n_iter
        return self


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v"])

    import time

    import numpy as np
    from sklearn.datasets import make_sparse_spd_matrix

    p = 500
    prec = make_sparse_spd_matrix(p, alpha=1 - 8 / p, random_state=0)
    G = prec != 0
    rng = np.random.default_rng(42)
    cov = np.diag(1 / np.diag(prec))
    cov = np.linalg.inv(prec)
    X = rng.multivariate_normal(np.zeros(p), cov, size=100)
    st = time.perf_counter()
    model = StructuredGraphicalModel(adjacency_matrix=G, max_iter=10, ridge=0.001).fit(
        X
    )
    # model.precision_.round(2)

    import networkx as nx
    import scipy

    st = time.perf_counter()
    prec2 = fit_precision_cholesky(
        U=X,
        Graph_u=nx.from_scipy_sparse_array(scipy.sparse.csc_array(G)),
        use_tqdm=False,
    )

    G = prec < 99999

    prec_naive = np.linalg.pinv(np.cov(X, rowvar=False))
