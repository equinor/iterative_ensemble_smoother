import numpy as np
import pytest
import scipy as sp

from iterative_ensemble_smoother.esmda import ESMDA
from iterative_ensemble_smoother.esmda_inversion import normalize_alpha
from iterative_ensemble_smoother.esmda_localized import (
    LocalizedESMDA,
    invert_naive,
    invert_subspace,
)


class TestLocalizedESMDA:
    @pytest.mark.parametrize("seed", range(99))
    def test_observation_scale_invariance(self, seed):
        """ESMDA should be scale invariant, even with huge differences
        in scales. A trivial example: if we change outputs from gallons
        to liters, the result should be the same."""

        rng = np.random.default_rng(seed)
        # This property should hold for any size
        num_obs = rng.choice([5, 10, 15])
        num_params = rng.choice([5, 10, 15])
        num_realizations = rng.choice([5, 10, 15])
        alpha = rng.choice([1, 2, 3])  # Number of iterations

        # The linear map that defines the forward model
        A = rng.normal(size=(num_obs, num_params), scale=0.1)
        scales = np.logspace(-10, 10, num=num_obs)  # Extreme scaling
        A_scaled = A * scales[:, np.newaxis]

        def forward_model(x, scaled):
            """Forward model for a single realization."""
            if scaled:
                return A_scaled @ x
            return A @ x

        def F(X, scaled):
            """Forward model applied to every realization."""
            return np.array([forward_model(x, scaled) for x in X.T]).T

        # Covariances for the base (unscaled) case
        covariance = np.logspace(-1, 1, num=num_obs)  # Covar of observations

        # True parameters that create the observed values
        X_true = rng.normal(scale=0.1, size=(num_params, num_realizations))

        # Set prior at N(1, 1), a little bit away from observations at 0
        X1 = 1 + rng.normal(size=(num_params, num_realizations))
        X2 = X1.copy()

        # ================ Unscaled ================
        smoother = LocalizedESMDA(
            covariance=covariance,
            observations=F(X_true, scaled=False).mean(axis=1),
            alpha=alpha,
            seed=seed,
        )
        for _ in range(smoother.num_assimilations()):
            Y = F(X1, scaled=False)
            smoother.prepare_assimilation(Y=Y)
            X1 = smoother.assimilate_batch(X=X1)

        # ================ Scaled ================
        smoother = LocalizedESMDA(
            # Scale the covariance (notice the power of two)
            covariance=covariance * scales**2,
            observations=F(X_true, scaled=True).mean(axis=1),
            alpha=alpha,
            seed=seed,
        )
        for _ in range(smoother.num_assimilations()):
            Y = F(X2, scaled=True)
            smoother.prepare_assimilation(Y=Y)
            X2 = smoother.assimilate_batch(X=X2)

        assert np.allclose(X1, X2)

    @pytest.mark.parametrize("seed", range(99))
    def test_batch_subset_invariance(self, seed):
        def indices_generator(n, rng):
            """Generate two random subsets of indices."""
            indices = np.arange(n)
            rng.shuffle(indices)
            yield indices[: n // 2]
            yield indices[n // 2 :]

        rng = np.random.default_rng(seed)
        # This property should hold for any size
        num_obs = rng.choice([5, 10, 15])
        num_params = rng.choice([5, 10, 15])
        num_realizations = rng.choice([5, 10, 15])
        alpha = rng.choice([1, 2, 3])  # Number of iterations

        # The linear map that defines the forward model
        A = rng.normal(size=(num_obs, num_params), scale=0.1)

        def F(X):
            """Forward model applied to every realization."""
            return A @ X

        # Set up the localized ESMDA instance and the prior realizations X:
        covariance = np.logspace(-1, 1, num=num_obs)  # Covar of observations
        observations = rng.normal(size=num_obs, scale=0.01)  # The observed data
        smoother = LocalizedESMDA(
            covariance=covariance,
            observations=observations,
            alpha=alpha,
            seed=seed,
        )

        # Set prior at N(1, 1), a little bit away from observations at 0
        X = 1 + rng.normal(size=(num_params, num_realizations))

        # Loop over all iterations, defined by alpha in the initializer
        for _ in range(smoother.num_assimilations()):
            # Apply forward model and prepare for assimilation
            Y = F(X)
            smoother.prepare_assimilation(Y=Y)

            #  =======Approach 1: one batch with all parameters =======
            def cb(K):
                # Dummy callback to check that K is passed through
                assert K.shape == (num_params, num_obs)
                return 0.5 * K  # A dummy callback

            X_1 = smoother.assimilate_batch(X=X, localization_callback=cb)

            # ======= Approach 2: several batches =======
            X_2 = np.zeros_like(X_1)  # Empty storage
            for batch_idx in indices_generator(num_params, rng):

                def cb_batch(K):
                    assert K.shape == (len(batch_idx), num_obs)
                    return 0.5 * K

                X_2[batch_idx, :] = smoother.assimilate_batch(
                    X=X[batch_idx, :], localization_callback=cb_batch
                )

            # The result should be the same
            assert np.allclose(X_1, X_2)
            X = X_1  # For the next iteration

    @pytest.mark.parametrize("seed", range(99))
    def test_1D_vs_2D_covariance_invariance(self, seed):
        """Whether we use a 1D covariance or a diagonal 2D covariance
        should not change the result - but 1D is faster and uses less memory."""
        rng = np.random.default_rng(seed)
        num_obs = 1
        # This property should hold for any size
        num_params = rng.choice([5, 10, 15])
        num_realizations = rng.choice([5, 10, 15])
        alpha = rng.choice([1, 2, 3])  # Number of iterations

        def forward_model(x):
            """Forward model for a single realization: g(x) = sum_i x_i"""
            return np.sum(x, keepdims=True)

        def F(X):
            """Vectorized forward model, applied to all realizations."""
            return np.array([forward_model(x) for x in X.T]).T

        # Set up the localized ESMDA instances
        covariance = np.logspace(-10, 10, num=num_obs)
        observations = rng.normal(size=num_obs, scale=0.01)

        smoother_1D_covar = LocalizedESMDA(
            covariance=covariance,
            observations=observations,
            alpha=alpha,
            seed=seed,  # Same seed
        )

        smoother_2D_covar = LocalizedESMDA(
            covariance=np.diag(covariance),
            observations=observations,
            alpha=alpha,
            seed=seed,  # Same seed
        )

        # Prior on the observations (initial ensemble)
        X = 1 + rng.normal(size=(num_params, num_realizations))

        for _ in range(smoother_1D_covar.num_assimilations()):
            # Apply forward model and prepare smoothers for assimilation
            Y = F(X)
            smoother_1D_covar.prepare_assimilation(Y=Y)
            smoother_2D_covar.prepare_assimilation(Y=Y)

            # Apply assimilation step
            X_1 = smoother_1D_covar.assimilate_batch(X=X, localization_callback=None)
            X_2 = smoother_2D_covar.assimilate_batch(X=X, localization_callback=None)
            assert np.allclose(X_1, X_2), "Results should match exactly"
            X = X_1

    @pytest.mark.parametrize("seed", range(99))
    @pytest.mark.parametrize("dense_covariance", [False])
    def test_equivalence_with_ESMDA(self, seed, dense_covariance):
        """With no localization, ESMDA and LocalizedESMDA should produce
        exactly the same results."""

        rng = np.random.default_rng(seed)
        # This property should hold for any size
        num_obs = rng.choice([5, 10, 15])
        num_params = rng.choice([5, 10, 15])
        num_realizations = rng.choice([5, 10, 15])
        alpha = rng.choice([1, 2, 3])  # Number of iterations

        truncation = rng.choice([0.9, 0.99, 1.0])

        # Create some missing values to test that they propagate through
        # both ESMDA and LocalizedESMDA and produce the same results
        missing = rng.random(size=(num_params, num_realizations)) > 0.9
        missing[:, :2] = False  # At least two realizations have all params

        # The linear forward map
        A = rng.normal(size=(num_obs, num_params), scale=0.1)

        def F(X):
            """Forward model applied to every realization."""
            return A @ X

        # Set up the localized ESMDA instance and the prior realizations X:
        covariance = np.logspace(-10, 10, num=num_obs)  # Covar of observations
        if dense_covariance:
            factor = rng.normal(size=(num_obs, num_obs)) / num_obs
            covariance = np.diag(covariance) + factor.T @ factor

        observations = rng.normal(size=num_obs, scale=0.01)  # The observed data
        esmda = ESMDA(
            covariance=covariance,
            observations=observations,
            alpha=alpha,
            seed=seed,  # Same seed
        )

        lesmda = LocalizedESMDA(
            covariance=covariance,
            observations=observations,
            alpha=alpha,
            seed=seed,
        )

        X = 1 + rng.normal(size=(num_params, num_realizations))
        for _ in range(esmda.num_assimilations()):
            # Apply forward model
            Y = F(X)

            # Assimilate with ESMDA
            X_1 = esmda.assimilate(X=X, Y=Y, truncation=truncation, missing=missing)

            # Assimilate with localized ESMDA.
            # If the localization callback is the identity, and the truncation
            # is 1, then it should return exactly the same result as ESMDA
            lesmda.prepare_assimilation(Y=Y, truncation=truncation)

            # Assimilate with LocalizedESMDA, using identity function as callback
            X_2 = lesmda.assimilate_batch(
                X=X, localization_callback=None, missing=missing
            )
            assert np.allclose(X_1, X_2), "LocalizedESMDA should match with ESMDA"
            X = X_1

    @pytest.mark.parametrize("seed", range(9))
    def test_setting_alpha_vs_scaling_covariance(self, seed):
        """Two approaches should produce identical results:
        (1) Create instance once, then assimilate over iterations
        (2) Create instance in each iteration, with manual alpha
        """
        rng = np.random.default_rng(seed)
        num_obs = rng.choice([5, 10, 15])
        num_params = rng.choice([5, 10, 15])
        num_realizations = rng.choice([5, 10, 15])
        truncation = 0.99

        # The linear forward map
        A = rng.normal(size=(num_obs, num_params), scale=0.1)

        def F(X):
            """Vectorized forward model, applied to all realizations."""
            return A @ X

        # Pre-compute alpha values so they are identical for both approaches
        alpha = normalize_alpha(np.array([1, 2, 3, 4]))
        covariance = np.logspace(-5, 5, num=num_obs)
        observations = rng.normal(size=num_obs, scale=0.01)
        X_prior = 1 + rng.normal(size=(num_params, num_realizations))

        # === APPROACH 1 : Create instance once, then assimilate over iterations ===
        rng1 = np.random.default_rng(seed)
        esmda = LocalizedESMDA(
            covariance=covariance,
            observations=observations,
            alpha=alpha,  # Alpha will incremented by the instance
            seed=rng1,
        )

        X_1 = X_prior.copy()
        for _ in range(esmda.num_assimilations()):
            Y = F(X_1)
            # The call to `prepare_assimilation()` increments alpha
            esmda.prepare_assimilation(Y=Y, truncation=truncation)
            X_1 = esmda.assimilate_batch(X=X_1, localization_callback=None)

        # === APPROACH 2 : Create instance in each iteration, with manual alpha ===
        X_2 = X_prior.copy()
        # Creating the RNG outside of the loop, then advancing it within
        # each instance of the LocalizedESMDA class created within the loop
        rng2 = np.random.default_rng(seed)
        for alpha_i in alpha:
            esmda = LocalizedESMDA(
                covariance=covariance * alpha_i,
                observations=observations,
                alpha=1,  # Disable the effect of alpha here
                seed=rng2,
            )
            Y = F(X_2)
            esmda.prepare_assimilation(Y=Y, truncation=truncation)
            X_2 = esmda.assimilate_batch(X=X_2, localization_callback=None)

        assert np.allclose(X_1, X_2), "Posteriors should match"

    @pytest.mark.parametrize("seed", range(99))
    @pytest.mark.parametrize("diagonal", [True, False])
    def test_that_inversion_methods_are_identical(self, seed, diagonal):
        """On any covariance, exact inversion and scaled subspace
        inversion produces the same result."""
        # Create a problem of random size
        rng = np.random.default_rng(seed)
        num_obs = rng.choice([5, 10, 15])
        num_realizations = rng.choice([5, 10, 15])
        alpha = rng.choice([0.1, 1, 10])

        # Create an extreme covariance matrix
        scales = np.logspace(-3, 3, num=num_obs)
        rng.shuffle(scales)
        F = rng.normal(size=(num_obs, num_obs)) * scales
        C_D = F.T @ F

        C_D_L = sp.linalg.cholesky(C_D, lower=False)
        if diagonal:
            C_D_L = np.diag(C_D_L)

        # delta_D is a centered forward model output Y
        rng.shuffle(scales)
        Y = rng.normal(size=(num_obs, num_realizations)) * scales[:, np.newaxis]
        delta_D = Y - np.mean(Y, axis=1, keepdims=True)

        # With no truncation, all methods are identical
        delta_D_inv_cov1 = invert_naive(
            delta_D=delta_D, C_D_L=C_D_L, alpha=alpha, truncation=1.0
        )
        delta_D_inv_cov2 = invert_subspace(
            delta_D=delta_D, C_D_L=C_D_L, alpha=alpha, truncation=1.0
        )

        # These should be identical.
        # They have values that are very small, around 1e-8,
        # so comparing floats with relative precision at this level is hard
        assert np.allclose(delta_D_inv_cov1, delta_D_inv_cov2)

        assert np.min(delta_D_inv_cov1) < 10
        assert np.min(delta_D_inv_cov2) < 10


if __name__ == "__main__":
    pytest.main(
        args=[
            __file__,
            "-v",
        ]
    )
