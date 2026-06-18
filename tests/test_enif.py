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
from iterative_ensemble_smoother.esmda import ESMDA


class TestEnIF:
    def test_EnIF_snapshot(
        self,
    ):
        """The purpose of this test is to alert the developer if any changes
        change the behavior of EnIF. If this is intended, changing the
        expected value is perfectly fine."""

    @pytest.mark.parametrize("seed", range(9))
    def test_against_ESMDA(self, seed):
        rng = np.random.default_rng(seed)

        # (parameters, responses, realizations)
        m, n, r = 12, rng.choice([8, 12, 26]), 50

        H = rng.normal(size=(n, m))
        X = rng.normal(size=(m, r))
        Y = H @ X
        covariance = np.exp(rng.normal(size=n))
        observations = rng.normal(size=n, loc=1.0)

        # Empirical prior precision
        Lambda_x = np.linalg.inv(np.cov(X, ddof=1))

        # --- ESMDA ---
        esmda = ESMDA(covariance, observations, alpha=1, seed=seed)
        esmda.prepare_assimilation(Y=Y, truncation=1.0)
        X_esmda = esmda.assimilate_batch(X=X)

        # --- EnIF ---
        enif = EnIF(
            covariance=covariance,
            observations=observations,
            parameter_precision=Lambda_x,
            alpha=1,
            seed=seed,
            solver="dense",
        )
        enif.prepare_assimilation(Y=Y)
        X_enif = enif.assimilate(
            X=X,
            linearized_model=H,
            residual_covariance=np.zeros(n),
        )

        np.testing.assert_allclose(X_enif, X_esmda)
        assert not np.allclose(X_enif, X), "Update was trivial"

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

    @pytest.mark.parametrize("seed", range(99))
    def test_against_gauss_linear(self, seed):
        """Bishop (PRML, 2.3.3) Gauss-linear posterior.

        p(x)   = N(x | mu, C_M)
        p(d|x) = N(d | A x, C_D)
        =>  COV  = inv(inv(C_M) + A^T inv(C_D) A)
            MEAN = COV (A^T inv(C_D) d + inv(C_M) mu)
                 = mu + C_M A^T (A C_M A^T + C_D)^{-1} A (x_true - mu)

        EnIF uses the *exact* prior precision inv(C_M) and exact H, so its gain is
        the analytic Kalman gain; the only error is finite-ensemble sampling.
        Mean converges tightly; covariance is checked loosely.
        """
        rng = np.random.default_rng(seed)
        num_ensemble = 10_000
        num_inputs, num_outputs = rng.choice([2, 4, 8]), 4

        mu = rng.normal(size=num_inputs)
        C_M_factor = rng.normal(size=(num_inputs, num_inputs))
        C_M = C_M_factor.T @ C_M_factor + np.eye(num_inputs)

        A = rng.normal(size=(num_outputs, num_inputs))

        C_D_factor = rng.normal(size=(num_outputs, num_outputs))
        C_D = C_D_factor.T @ C_D_factor + np.eye(num_outputs)

        inv = np.linalg.inv
        X_true = mu + 10.0
        d = A @ X_true

        COV = inv(inv(C_M) + A.T @ inv(C_D) @ A)
        MEAN = COV @ (A.T @ inv(C_D) @ d + inv(C_M) @ mu)
        MEAN2 = mu + C_M @ A.T @ inv(A @ C_M @ A.T + C_D) @ A @ (X_true - mu)
        # Sanity-check the reference itself: two derivations must agree.
        np.testing.assert_allclose(MEAN, MEAN2)

        X_prior = rng.multivariate_normal(mean=mu, cov=C_M, size=num_ensemble).T
        Lambda_x = inv(C_M)

        enif = EnIF(
            covariance=C_D,
            observations=d,
            parameter_precision=Lambda_x,
            alpha=1,
            seed=rng,
            solver="dense",
        )
        enif.prepare_assimilation(Y=A @ X_prior)
        X_post = enif.assimilate(
            X=np.copy(X_prior),
            linearized_model=A,
            residual_covariance=np.zeros(num_outputs),
        )

        rel_mean = np.linalg.norm(X_post.mean(axis=1) - MEAN) / np.linalg.norm(MEAN)
        assert rel_mean < 0.02, f"mean relative error {rel_mean}"

        cov_post = np.cov(X_post, ddof=1)
        rel_cov = np.linalg.norm(cov_post - COV) / np.linalg.norm(COV)
        assert rel_cov < 0.75, f"cov relative error {rel_cov}"


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "-v",
            "-x",
        ]
    )
