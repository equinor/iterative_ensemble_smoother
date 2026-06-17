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
    prec = make_sparse_spd_matrix(p, alpha=1 - 4 / p, random_state=0)
    G = prec != 0
    rng = np.random.default_rng(42)
    cov = np.diag(1 / np.diag(prec))
    cov = np.linalg.inv(prec)
    X = rng.multivariate_normal(np.zeros(p), cov, size=35)
    st = time.perf_counter()
    model = StructuredGraphicalModel(adjacency_matrix=G, max_iter=500, ridge=0.1).fit(X)
    # model.precision_.round(2)

    G = prec < 99

    prec_naive = np.linalg.pinv(np.cov(X, rowvar=False))
