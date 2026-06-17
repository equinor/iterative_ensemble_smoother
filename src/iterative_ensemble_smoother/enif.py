import logging
from typing import Literal

import networkx as nx
import numpy as np
import scipy as sp
from numpy.typing import NDArray
from scipy.sparse import diags_array, sparray
from scipy.sparse.linalg import bicgstab
from tqdm import tqdm

from iterative_ensemble_smoother import enif_linear_regression as lr
from iterative_ensemble_smoother.enif_precision_estimation import (
    fit_precision_cholesky,
)
from iterative_ensemble_smoother.enif_utils import generate_gaussian_noise

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class EnIF:
    """Initialize an Ensemble Information Filter (EnIF).

    The filter is parametrized by the prior precision of the state `u` (or a
    graph specifying its sparsity), the precision of the observation noise,
    and optionally the linear map `H`. Anything left as `None` at initialization
    can be learned from data via `fit`.

    Parameters
    ----------
    Prec_u : sparray, optional
        Prior precision matrix of the state, shape (params, params). If omitted,
        it is estimated from data using `Graph_u` as the sparsity pattern.
    Graph_u : nx.Graph, optional
        Conditional-independence graph on the `params` state components,
        defining the sparsity of `Prec_u`. Required when `Prec_u` is not
        provided.
    Prec_eps : sparray
        Precision matrix of the observation noise, shape (responses, responses).
    H : sparray, optional
        Linear observation operator mapping state to responses, shape
        (responses, params). If omitted, it is estimated from data by `fit`.
    """

    @staticmethod
    def _validate_sparse_2d(name: str, value: sparray, square: bool = False) -> None:
        if not (isinstance(value, sp.sparse.sparray) and value.ndim == 2):
            raise TypeError(f"`{name}` must be a 2D sparse array")
        if square and value.shape[0] != value.shape[1]:
            raise ValueError(f"`{name}` must be a square 2D sparse array")

    def __init__(
        self,
        *,
        Prec_u: sparray | None = None,
        Graph_u: nx.Graph | None = None,
        Prec_eps: sparray,
        H: sparray | None = None,
    ) -> None:
        assert Prec_u is not None or Graph_u is not None, (
            "Provide either Prec_u or Graph_u"
        )

        if Prec_u is not None:
            self._validate_sparse_2d("Prec_u", Prec_u, square=True)
        if H is not None:
            self._validate_sparse_2d("H", H)
        self._validate_sparse_2d("Prec_eps", Prec_eps, square=True)

        self.Prec_u = Prec_u
        self.Graph_u = Graph_u
        self.Prec_eps = Prec_eps
        self.H = H
        self.unexplained_variance: NDArray[np.floating] | None = None

    def fit(
        self,
        U: NDArray[np.floating],
        Y: NDArray[np.floating] | None = None,
        learning_algorithm: Literal["LASSO", "influence-boost"] = "LASSO",
        ordering_method: str = "metis",
    ) -> None:
        """Fit the prior precision of `u` and, optionally the mapping `H`.

        If `Prec_u` was not supplied at construction, it is estimated from `U`
        using the sparsity pattern of `Graph_u`. If `Y` is supplied and `H` was
        not set at construction, a sparse linear map `H` is learned from `U`
        to `Y` and the per-response residual variance is stored on the
        instance. Already-provided quantities are kept as-is.

        Parameters
        ----------
        U : ndarray of shape (realizations, parameters)
            Prior ensemble: `n` realizations of the `p`-dimensional state.
        Y : ndarray of shape (realizations, responses), optional
            Response ensemble used to learn `H`. Must be omitted if `H` was
            provided at construction.
        learning_algorithm : {"LASSO", "influence-boost"}, default="LASSO"
            Estimator used to fit `H`. Ignored when `Y` is not provided.
        ordering_method : str, default="metis"
            Fill-reducing ordering passed to the Cholesky factorization when
            estimating `Prec_u`.
        """

        if self.Prec_u is None:
            self.fit_precision(
                U,
                ordering_method=ordering_method,
            )
        else:
            log.info("Precision u exists. Use `fit_precision` to refit if necessary")
        if Y is not None:
            assert self.H is None, "Y should not be provided if H exists"
            self.fit_H(
                U=U,
                Y=Y,
                learning_algorithm=learning_algorithm,
            )
        else:
            log.info("H mapping exists. Use `fit_H` to refit if necessary")

    def transport(
        self,
        U: NDArray[np.floating],
        Y: NDArray[np.floating],
        d: NDArray[np.floating],
        update_indices: NDArray[np.integer] | None = None,
        seed: int | None = None,
        iterative: bool = False,
    ) -> NDArray[np.floating]:
        """Transport a prior ensemble to the posterior given observations `d`.

        Each realization is mapped to the canonical (information) parametrization
        `eta = Prec_u @ u`, updated with a perturbed-observation information
        filter step using the current `H`, `Prec_u`, and `Prec_eps`, and then
        mapped back to the state space. When `update_indices` is given, only
        those components are solved for and the rest are copied from the
        prior, which is the usual speed-up for localized updates.

        Parameters
        ----------
        U : ndarray of shape (realizations, parameters)
            Prior ensemble.
        Y : ndarray of shape (realizations, responses)
            Response ensemble evaluated on `U`.
        d : ndarray of shape (responses,)
            Observed data vector.
        update_indices : ndarray of int, optional
            Indices of state components to update. Defaults to all `parameter`
            components.
        seed : int, optional
        iterative : bool, default=False

        Returns
        -------
        U_post : ndarray of shape (realizations, parameters)
            Posterior ensemble. Components not listed in `update_indices` are
            equal to those in `U`.
        """
        n, _ = U.shape
        n_y, m = Y.shape
        assert n == n_y, "Number of ensembles must be the same"
        assert d.shape == (m,), "Observations must match responses"

        # Map parameters to canonical parametrization
        canonical = self.pushforward_to_canonical(U)

        # Work out residuals and associate unexplained variance
        residuals = self.response_residual(U, Y)

        # Due to observation error
        eps = self.generate_observation_noise(
            n,
            seed=seed,
        )
        residual_noisy = residuals + eps

        # Update in canonical parametrization
        canonical_updated = self.update_canonical(
            canonical=canonical,
            residual_noisy=residual_noisy,
            d=d,
        )

        # Bring realizations back
        return self.pullback_from_canonical(
            updated_canonical=canonical_updated,
            update_indices=update_indices,
            U_prior=U,
            iterative=iterative,
        )

    # Low-level API methods
    def fit_precision(
        self,
        U: NDArray[np.floating],
        ordering_method: str = "metis",
    ) -> None:
        """
        Estimate self.Prec_u from data U w.r.t. graph self.Graph_u
        """
        assert self.Graph_u is not None, "Graph_u must be set to fit precision"
        self.Prec_u = fit_precision_cholesky(
            U=U,
            Graph_u=self.Graph_u,
            ordering_method=ordering_method,
        )
        self._validate_sparse_2d("Prec_u", self.Prec_u, square=True)

    def fit_H(
        self,
        U: NDArray[np.floating],
        Y: NDArray[np.floating],
        learning_algorithm: Literal["LASSO", "influence-boost"] = "LASSO",
    ) -> None:
        """
        Estimate H from data U using (sparse) linear regression
        """
        if learning_algorithm not in ("LASSO", "influence-boost"):
            raise ValueError(
                f"Argument `learning_algorithm` must be 'LASSO' or 'influence-boost'. "
                f"Got: {learning_algorithm}"
            )

        if learning_algorithm == "LASSO":
            self.H = lr.linear_l1_regression(
                U,
                Y,
            )
        else:
            self.H = lr.linear_boost_ic_regression(
                U,
                Y,
            )
        self._validate_sparse_2d("H", self.H)

        # Sets the `unexplained_variance` attribute on self
        self.response_residual(U=U, Y=Y)

    def pushforward_to_canonical(self, U: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Map each realization u in U to canonical space: eta = u @ Prec
        """
        log.info("Mapping realizations to canonical space")

        assert self.Prec_u is not None, "Precision must exist to pushforward"
        Eta: NDArray[np.floating] = (
            U @ self.Prec_u
        )  # Shapes: (r, param) = (r, param) @ (param, param)
        assert Eta.shape == U.shape, "Eta preserves the shape of U"
        return Eta

    def Prec_residual_noisy(self) -> sparray:
        if self.unexplained_variance is None:
            raise ValueError("`unexplained_variance` is not set.")

        # The equation below is only valid if Prec_eps is diagonal
        row_idx, col_idx, _ = sp.sparse.find(self.Prec_eps)
        if np.any(row_idx != col_idx):
            raise ValueError("Precision matrix 'Prec_eps' must be diagonal")

        eps_variances = 1.0 / self.Prec_eps.diagonal()
        residual_noisy_var = self.unexplained_variance + eps_variances
        Prec_r = diags_array(1.0 / residual_noisy_var, offsets=0, format="csc")
        assert Prec_r.shape == self.Prec_eps.shape, (
            "Residuals and noise precision should have same shape"
        )

        log.info("Total residual variance: %.4f", np.sum(residual_noisy_var))
        log.info("Unexplained variance: %.4f", np.sum(self.unexplained_variance))
        log.info("Measurement variance: %.4f", np.sum(eps_variances))
        return Prec_r

    def response_residual(
        self,
        U: NDArray[np.floating],
        Y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Residual from regression self.H for Y on U: Y - U @ H.T"""
        if self.H is None:
            raise ValueError("H is not set.")

        assert Y.shape[0] == U.shape[0], (
            "Number of realizations (ensemble members) must be equal"
        )
        assert U.shape[1] == self.H.shape[1], "Shape mismatch"
        assert Y.shape[1] == self.H.shape[0], "Shape mismatch"

        log.info("Calculating response residuals")
        # Has shape (realizations, responses)
        response_residuals: NDArray[np.floating] = Y - U @ self.H.T

        # Unexplained variance for each response
        log.info("Calculating unexplained variance")
        self.unexplained_variance = np.var(response_residuals, axis=0, ddof=0)

        return response_residuals

    def generate_observation_noise(
        self,
        n: int,
        seed: int | None = None,
    ) -> NDArray[np.floating]:
        """Sample n realizations of observation noise."""

        return generate_gaussian_noise(
            n,
            self.Prec_eps,
            seed=seed,
        )

    def update_canonical(
        self,
        canonical: NDArray[np.floating],
        residual_noisy: NDArray[np.floating],
        d: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Use information-filter equations to update (eta, Prec) using perturbed
        d
        """
        assert self.H is not None, "H must be provided of fitted"
        assert self.Prec_u is not None, "Precision must be provided of fitted"

        n, _p = canonical.shape
        n_r, m = residual_noisy.shape
        assert n == n_r, "canonical and residual_noisy must have equal samples"
        assert d.shape == (m,), "d and residual_noisy must have matching dimension"

        from sksparse.cholmod import cholesky  # noqa: PLC0415

        # Only print this if logging is on. Cholesky can be heavy
        if log.isEnabledFor(logging.INFO):
            chol_LLT = cholesky(self.Prec_u, ordering_method="metis")
            logdet_value = 2.0 * np.sum(np.log(chol_LLT.L().diagonal()))
            log.info("Prior precision log-determinant: %.3f", logdet_value)

        Prec_r = self.Prec_residual_noisy()  # This is a diagonal matrix

        # posterior canonical params
        # this is equation (46), but transposed to update each row (realizations)
        # Equivalent to:
        # upd_eta[i, :] = eta[i, :] + self.H.T @ Prec_r @ (d - residual_noisy[i, :])
        updated_canonical: NDArray[np.floating] = (
            canonical + (d - residual_noisy) @ Prec_r.T @ self.H
        )

        # posterior precision, equation (47)
        self.Prec_u = self.Prec_u + self.H.T @ Prec_r @ self.H  # Eqn (47)

        if log.isEnabledFor(logging.INFO):
            chol_LLT = cholesky(self.Prec_u, ordering_method="metis")
            logdet_value = 2.0 * np.sum(np.log(chol_LLT.L().diagonal()))
            log.info("Posterior precision log-determinant: %.3f", logdet_value)

        return updated_canonical

    def pullback_from_canonical(
        self,
        updated_canonical: NDArray[np.floating],
        update_indices: NDArray[np.integer] | None = None,
        U_prior: NDArray[np.floating] | None = None,
        iterative: bool = False,
    ) -> NDArray[np.floating]:
        """
        Solve the equation Eta = U @ Prec_u for unknown U.

        The suppose we wish to solve the matrix equation N = P @ U,
        but only some of the rows in U are to be solved for. Call these "s".
        Partitioning the matrix, we obtain
          | P_1s  P_1 | @ | u_s |    =  | n_s |
          | P_2s  P_2 |   | u   |       | n   |
        Focusing on the values u_s, we obtain the system
          P_1s @ u_s + P_1 @ u = n_s
        Which we solve for u_s:
            P_1s @ u_s = n_s - P_1 @ u

        In other words, this method uses selective updates for specified indices,
        taking into account previously calculated values of U_prior.

        Parameters
        ----------
        updated_canonical : NDArray[np.floating]
            The eta-matrix of shape (realizations, parameters).
        update_indices : NDArray[np.integer] | None, optional
            Indices to update (columns/params in U). The default is None (update all).
        U_prior : NDArray[np.floating] | None, optional
            Values in U used for indices that are not updated. The default is None.
        iterative : bool, optional
            Whether to use iterative solver or not. The default is False.

        Returns
        -------
        U : NDArray[np.floating]
            Array of shape (realizations, parameters), updated in columns
            `update_indices`. The remaining columns are copied from `U_prior`.
        """
        assert self.Prec_u is not None, "Prec_u must exist"
        assert update_indices is None or np.issubdtype(update_indices.dtype, np.integer)
        assert (update_indices is None) >= (U_prior is None), (
            "Must pass U_prior if update_indices"
        )

        from sksparse.cholmod import cholesky  # noqa: PLC0415

        log.info(
            "Mapping canonical-scaled realizations (Eta) to moment realization (U)"
        )

        # Indices to solve for 's' and complementary set 'not_s'
        all_indices = np.arange(updated_canonical.shape[1], dtype=int)
        s = all_indices if (update_indices is None) else update_indices
        assert s is not None
        not_s = np.setdiff1d(all_indices, s)

        U = np.zeros(updated_canonical.shape) if (U_prior is None) else U_prior.copy()
        if s.size == 0:
            return U

        P_ss = self.Prec_u[np.ix_(s, s)]
        P_s_not_s = self.Prec_u[np.ix_(s, not_s)]

        # === Iterative solution ===
        if iterative:
            desc = "Mapping data to moment parametrisation realization-by-realization"

            for i in tqdm(range(U.shape[0]), desc=desc):
                if not_s.size > 0:
                    rhs = updated_canonical[i, s] - P_s_not_s @ U[i, not_s]
                else:
                    rhs = updated_canonical[i, s]

                x_updated, _ = bicgstab(P_ss, rhs)
                U[i, s] = x_updated

            return U

        # === Cholesky solution ===
        chol_LLT = cholesky(P_ss, ordering_method="metis")
        if not_s.size > 0:
            rhs = updated_canonical[:, s].T - P_s_not_s @ U[:, not_s].T
        else:
            rhs = updated_canonical[:, s].T

        U[:, s] = chol_LLT.solve_A(rhs).T
        return U

    def get_update_indices(
        self,
        neighbor_propagation_order: int = 10,
    ) -> NDArray[np.integer]:
        """
        Determine indices to update based on the order of neighbor propagation.

        Parameters:
        - neighbor_propagation_order: Levels of neighbors to include.

        Returns:
        - update_indices: Array of indices that includes the initial
            predictors and their neighbors up to the specified order.
        """
        assert self.H is not None, "H must exist"
        assert self.Prec_u is not None, "Prec_u must exist"

        _, cols = self.H.nonzero()
        predictors = set(cols)
        adjacency = self.Prec_u.copy()
        all_nodes = set(predictors)  # Start with predictors

        # Initialize sets to manage nodes
        current_nodes = predictors.copy()
        new_nodes = set()

        # Iteratively find neighbors up to the specified order
        for _ in range(neighbor_propagation_order):
            temp_nodes = set()
            for col in current_nodes:
                neighbors = adjacency[:, col].nonzero()[0]
                temp_nodes.update(neighbors)

            # Update new_nodes with newly discovered nodes
            new_nodes = temp_nodes.difference(all_nodes)
            all_nodes.update(new_nodes)
            current_nodes = new_nodes.copy()

        param_num, tot_num = len(all_nodes), adjacency.shape[0]
        log.info("Retrieving %d parameters out of %d", param_num, tot_num)

        return np.array(list(all_nodes), dtype=int)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v"])
