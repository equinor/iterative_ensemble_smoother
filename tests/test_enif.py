import networkx as nx
import numpy as np
import pytest
import scipy as sp

from iterative_ensemble_smoother.enif import EnIF
from iterative_ensemble_smoother.enif_linear_regression import (
    linear_boost_ic_regression,
)
from iterative_ensemble_smoother.enif_precision_estimation import (
    fit_precision_cholesky_approximate,
)


class TestEnIF:
    def test_EnIF_snapshot(
        self,
    ):
        """The purpose of this test is to alert the developer if any changes
        change the behavior of EnIF. If this is intended, changing the
        expected value is perfectly fine."""

    @pytest.mark.parametrize("seed", range(9))
    def test_against_ESMDA(self, seed):
        pass

    @pytest.mark.suitesparse
    @pytest.mark.parametrize("seed", range(10))
    def test_affine_invariance(self, seed):
        rng = np.random.default_rng(seed)
        n_params, n_responses, n_ensemble = 50, 25, 10

        # Create data
        Graph_u = nx.binomial_graph(n_params, p=0.2, seed=42)
        covariance = np.logspace(-2, 2, num=n_responses)
        H_true = sp.sparse.csc_array(rng.normal(size=(n_responses, n_params)))
        X = rng.normal(size=(n_params, n_ensemble)) * 3 + 7
        Y = H_true @ X
        observations = np.mean(Y, axis=1)

        def run(X_in):
            # Estimate H
            H = linear_boost_ic_regression(U=X_in.T, Y=Y.T)
            assert H.shape == H_true.shape
            # Estimate precision matrix
            Prec_u = fit_precision_cholesky_approximate(U=X_in.T, Graph_u=Graph_u)
            enif = EnIF(
                covariance=covariance,
                observations=observations,
                parameter_precision=Prec_u,
                seed=seed,
                alpha=1,
            )

            enif.prepare_assimilation(Y=Y)
            cov_eps = np.var(Y - H @ X_in, axis=1, ddof=1)
            return enif.assimilate(
                X=X_in, linearized_model=H, residual_covariance=cov_eps
            )

        # Posterior parameters with no transformation
        X_raw = run(X)

        # Standardize parameters, run EnIF, then transform back
        mu = X.mean(axis=1, keepdims=True)
        sigma = X.std(axis=1, keepdims=True)
        X_std = run((X - mu) / sigma) * sigma + mu

        # Verify that we get the same answer
        np.testing.assert_allclose(X_raw, X_std)

    @pytest.mark.parametrize("solver", ["dense", "cg"])
    def test_solver_equivalence(self, solver):
        pass

    @pytest.mark.parametrize("seed", range(9))
    def test_against_gauss_linear(self, seed):
        """Test against section XX in bishop. With many ensemble members,
        the posterior mean and posterior cov should match analytical."""


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "-v",
            "-x",
        ]
    )
