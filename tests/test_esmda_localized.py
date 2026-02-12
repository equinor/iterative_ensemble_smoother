import pytest
import numpy as np
from iterative_ensemble_smoother.esmda_localized import (
    LocalizedESMDA,
    invert_naive,
    invert_exact,
    invert_subspace,
    invert_subspace_scaled,
)

from iterative_ensemble_smoother.esmda import ESMDA


class TestLocalizedESMDA:
    @pytest.mark.parametrize("seed", range(99))
    def test_batch_subset_invariance(self, seed):
        def indices_generator(n, rng):
            """Generate two random subsets of indices."""
            indices = np.arange(n)
            rng.shuffle(indices)
            yield indices[: n // 2]
            yield indices[n // 2 :]

        rng = np.random.default_rng(seed)
        # This property should hold for any size, inversion method, etc.
        num_obs = rng.choice([5, 10, 15])
        num_params = rng.choice([5, 10, 15])
        num_realizations = rng.choice([5, 10, 15])
        inversion = str(rng.choice(list(LocalizedESMDA._inversion_methods.keys())))
        alpha = rng.choice([1, 2, 3])  # Number of iterations

        # The linear forward map
        A = rng.normal(size=(num_obs, num_params), scale=0.1)

        def forward_model(x):
            """Forward model for a single realization."""
            return A @ x

        def F(X):
            "Vectorized forward model, applied to all realizations."
            return np.array([forward_model(x) for x in X.T]).T

        # Set up the localized ESMDA instance and the prior realizations X:
        covariance = np.logspace(-1, 1, num=num_obs)  # Covar of observations
        observations = np.zeros(num_obs)  # The observed data
        smoother = LocalizedESMDA(
            covariance=covariance,
            observations=observations,
            alpha=alpha,
            seed=seed,
            inversion=inversion,
        )

        # Set prior at N(1, 1), a little bit away from observations at 0
        X = 1 + rng.normal(size=(num_params, num_realizations))

        for iteration in range(smoother.num_assimilations()):
            # Apply forward model and prepare for assimilation
            Y = F(X)
            smoother.prepare_assimilation(Y=Y)

            # Approach 1: one batch with all parameters
            X_1 = smoother.assimilate_batch(X=X, localization_callback=None)

            # Approach 2: several batches
            X_2 = np.zeros_like(X_1)  # Empty storage
            for batch_idx in indices_generator(num_params, rng):
                X_2[batch_idx, :] = smoother.assimilate_batch(
                    X=X[batch_idx, :], localization_callback=None
                )

            # The result should be the same
            assert np.allclose(X_1, X_2)

            X = X_1

    @pytest.mark.parametrize("seed", range(99))
    def test_1D_vs_2D_covariance_invariance(self, seed):
        """Whether we use a 1D covariance or a diagonal 2D covariance
        should not change the result - but 1D is faster and uses less memory."""
        rng = np.random.default_rng(seed)
        num_obs = 1
        # This property should hold for any size, inversion method, etc.
        num_params = rng.choice([5, 10, 15])
        num_realizations = rng.choice([5, 10, 15])
        inversion = str(rng.choice(list(LocalizedESMDA._inversion_methods.keys())))
        alpha = rng.choice([1, 2, 3])  # Number of iterations

        def forward_model(x):
            """Forward model for a single realization: g(x) = sum_i x_i"""
            return np.sum(x, keepdims=True)

        def F(X):
            "Vectorized forward model, applied to all realizations."
            return np.array([forward_model(x) for x in X.T]).T

        # Set up the localized ESMDA instance and the prior realizations X:
        covariance = np.logspace(-1, 1, num=num_obs)  # Covar of observations
        observations = np.zeros(num_obs)  # The observed data
        smoother_1D_covar = LocalizedESMDA(
            covariance=covariance,
            observations=observations,
            alpha=alpha,
            seed=seed,  # Same seed
            inversion=inversion,
        )

        smoother_2D_covar = LocalizedESMDA(
            covariance=np.diag(covariance),
            observations=observations,
            alpha=alpha,
            seed=seed,  # Same seed
            inversion=inversion,
        )

        X = 1 + rng.normal(size=(num_params, num_realizations))
        for iteration in range(smoother_1D_covar.num_assimilations()):
            # Apply forward model and prepare for assimilation
            Y = F(X)
            smoother_1D_covar.prepare_assimilation(Y=Y)
            smoother_2D_covar.prepare_assimilation(Y=Y)

            # Approach 1: one batch with all parameters
            X_1 = smoother_1D_covar.assimilate_batch(X=X, localization_callback=None)
            X_2 = smoother_2D_covar.assimilate_batch(X=X, localization_callback=None)
            assert np.allclose(X_1, X_2)

            X = X_1

    @pytest.mark.parametrize("seed", range(99))
    @pytest.mark.parametrize("inversion", ["exact", "subspace"])
    def test_equivalence_with_ESMDA(self, seed, inversion):
        """With no localization, ESMDA and LocalizedESMDA should produce
        exactly the same results."""

        rng = np.random.default_rng(seed)
        # This property should hold for any size, inversion method, etc.
        num_obs = rng.choice([5, 10, 15])
        num_params = rng.choice([5, 10, 15])
        num_realizations = rng.choice([5, 10, 15])
        alpha = rng.choice([1, 2, 3])  # Number of iterations

        # IMPORTANT: if truncation is set to 1.0, then we do not get
        # numerical equivalence between ESMDA and LocalizedESMDA, because
        # small changes in implementation (e.g. order of operations), produce
        # small numerical discrepancies that are multiplied. We need to
        # 'reguarlized' a bit with truncation=0.99, which is what Emerick
        # recommends, to get equivalent results
        truncation = 0.99

        # The linear forward map
        A = rng.normal(size=(num_obs, num_params), scale=0.1)

        def forward_model(x):
            """Forward model for a single realization."""
            return A @ x

        def F(X):
            "Vectorized forward model, applied to all realizations."
            return np.array([forward_model(x) for x in X.T]).T

        # Set up the localized ESMDA instance and the prior realizations X:
        covariance = np.logspace(-1, 1, num=num_obs)  # Covar of observations
        covariance = np.ones(num_obs)  # Covar of observations
        observations = np.zeros(num_obs)  # The observed data
        esmda = ESMDA(
            covariance=covariance,
            observations=observations,
            alpha=alpha,
            seed=seed,  # Same seed
            inversion=inversion,
        )

        lesmda = LocalizedESMDA(
            covariance=np.diag(covariance),
            observations=observations,
            alpha=alpha,
            seed=seed,  # Same seed
            inversion=inversion,
        )

        X = 1 + rng.normal(size=(num_params, num_realizations))
        for iteration in range(esmda.num_assimilations()):
            # Apply forward model
            Y = F(X)

            # Assimilate with ESMDA
            X_1 = esmda.assimilate(X=X, Y=Y, truncation=truncation)

            # Assimilate with localized ESMDA.
            # If the localization callback is the identity, and the truncation
            # is 1, then it should return exactly the same result as ESMDA
            lesmda.prepare_assimilation(Y=Y, truncation=truncation)

            # Approach 1: one batch with all parameters
            X_2 = lesmda.assimilate_batch(X=X, localization_callback=None)
            assert np.allclose(X_1, X_2, rtol=1e-3)

            X = X_1


class TestLocalizedESMDAInversionMethods:
    @pytest.mark.parametrize("seed", range(99))
    @pytest.mark.parametrize("diagonal", [True, False])
    def test_that_inversion_methods_are_identical_on_uniform_covariance(
        self, seed, diagonal
    ):
        """When the covariance is constant diagonal, every single inversion
        method produces exactly the same results."""

        # Create a problem of random size
        rng = np.random.default_rng(seed)
        num_obs = rng.choice([5, 10, 15])
        num_realizations = rng.choice([5, 10, 15])
        alpha = rng.choice([0.1, 1, 10])

        # IMPORTANT: for scaled and non-scaled subspace inversion to be
        # identical, the covariances should be constant
        C_D = np.ones(num_obs) * rng.random() + 1e-3
        C_D = np.diag(C_D) if diagonal else C_D
        # delta_D is a centered forward model output Y
        Y = rng.normal(size=(num_obs, num_realizations))
        delta_D = Y - np.mean(Y, axis=1, keepdims=True)

        # With no truncation, all methods are identical
        delta_D_inv_cov1 = invert_naive(
            delta_D=delta_D, C_D=C_D, alpha=alpha, truncation=1.0
        )
        delta_D_inv_cov2 = invert_exact(
            delta_D=delta_D, C_D=C_D, alpha=alpha, truncation=1.0
        )
        delta_D_inv_cov3 = invert_subspace(
            delta_D=delta_D, C_D=C_D, alpha=alpha, truncation=1.0
        )
        delta_D_inv_cov4 = invert_subspace_scaled(
            delta_D=delta_D, C_D=C_D, alpha=alpha, truncation=1.0
        )

        # All three should be identical
        assert np.allclose(delta_D_inv_cov1, delta_D_inv_cov2)
        assert np.allclose(delta_D_inv_cov2, delta_D_inv_cov3)
        assert np.allclose(delta_D_inv_cov3, delta_D_inv_cov4)

    @pytest.mark.parametrize("seed", range(99))
    @pytest.mark.parametrize("diagonal", [True, False])
    def test_that_inversion_methods_are_identical_on_nonuniform_covariance(
        self, seed, diagonal
    ):
        """On non-uniform (different orders of magnitude) diagonal covariances,
        exact inversion and scaled subspace inversion produces the same result."""
        # Create a problem of random size
        rng = np.random.default_rng(seed)
        num_obs = rng.choice([5, 10, 15])
        num_realizations = rng.choice([5, 10, 15])
        alpha = rng.choice([0.1, 1, 10])

        # IMPORTANT: If the covariance has different orders of magnitude, then
        # naive and subspace scaled inversion is the same (for diagonal cov.)
        C_D = np.logspace(-2, 2, num=num_obs)
        C_D = np.diag(C_D) if diagonal else C_D
        # delta_D is a centered forward model output Y
        Y = rng.normal(size=(num_obs, num_realizations))
        delta_D = Y - np.mean(Y, axis=1, keepdims=True)

        # With no truncation, all methods are identical
        delta_D_inv_cov1 = invert_naive(
            delta_D=delta_D, C_D=C_D, alpha=alpha, truncation=1.0
        )
        delta_D_inv_cov2 = invert_exact(
            delta_D=delta_D, C_D=C_D, alpha=alpha, truncation=1.0
        )
        delta_D_inv_cov3 = invert_subspace_scaled(
            delta_D=delta_D, C_D=C_D, alpha=alpha, truncation=1.0
        )

        # All three should be identical
        assert np.allclose(delta_D_inv_cov1, delta_D_inv_cov2)
        assert np.allclose(delta_D_inv_cov2, delta_D_inv_cov3)


if __name__ == "__main__":
    pytest.main(
        args=[
            __file__,
            "-v",
        ]
    )
