from time import perf_counter

import numpy as np
import pytest
import scipy as sp

from iterative_ensemble_smoother.esmda_inversion import (
    invert_naive,
    invert_subspace,
    normalize_alpha,
)


class TestEsmdaInversion:
    @pytest.mark.parametrize("length", list(range(1, 101, 5)))
    def test_that_the_sum_of_normalize_alpha_is_one(self, length):
        rng = np.random.default_rng(length)
        alpha = np.exp(rng.normal(size=length))
        # Test the defining property of the function
        assert np.isclose(np.sum(1 / normalize_alpha(alpha)), 1)

    @pytest.mark.parametrize("seed", range(99))
    @pytest.mark.parametrize("diagonal", [True, False])
    def test_that_inversion_methods_are_identical(self, seed, diagonal):
        """On any covariance, exact and subspace inversion should be equal."""
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
        delta_D_inv_cov2 = np.linalg.multi_dot(
            invert_subspace(delta_D=delta_D, C_D_L=C_D_L, alpha=alpha, truncation=1.0)
        )

        # These should be identical.
        # They have values that are very small, around 1e-8,
        # so comparing floats with relative precision at this level is hard
        assert np.allclose(delta_D_inv_cov1, delta_D_inv_cov2)
        assert np.min(delta_D_inv_cov1) < 10
        assert np.min(delta_D_inv_cov2) < 10

    def test_that_inversion_does_not_mutate_input_args(self):
        """Inversion functions should not mutate input arguments."""
        rng = np.random.default_rng(42)
        num_outputs, ensemble_members = 50, 10

        # Diagonal covariance matrix
        C_D = np.diag(np.exp(rng.normal(size=num_outputs)))
        C_D_L = sp.linalg.cholesky(C_D, lower=False)

        # Set alpha to something other than 1 to test that it works
        alpha = np.exp(rng.normal())
        truncation = rng.uniform()

        # Create observations
        Y = rng.normal(size=(num_outputs, ensemble_members))
        delta_D = Y - np.mean(Y, axis=1, keepdims=True)

        args = [delta_D, C_D_L, alpha, truncation]
        args_copy = [np.copy(arg) for arg in args]

        invert_subspace(
            delta_D=delta_D, C_D_L=C_D_L, alpha=alpha, truncation=truncation
        )

        for arg, arg_copy in zip(args, args_copy):
            assert np.allclose(arg, arg_copy)


class TestSvdConvergenceFallback:
    """When the default SVD driver (gesdd) raises ``LinAlgError('SVD did not
    converge')``, ``invert_subspace`` should retry with the more robust gesvd
    driver, emit a warning, and still return the correct result.

    A genuine gesdd non-convergence is impractical to reproduce (it depends on
    the LAPACK build and hardware), so we simulate it by monkeypatching
    ``scipy.linalg.svd`` to fail on gesdd and succeed on gesvd.
    """

    @staticmethod
    def _make_problem(seed=0):
        rng = np.random.default_rng(seed)
        num_obs, num_realizations = 10, 8
        # Diagonal covariance -> C_D_L is the elementwise square root.
        C_D_L = np.sqrt(np.exp(rng.normal(size=num_obs)))
        Y = rng.normal(size=(num_obs, num_realizations))
        delta_D = Y - np.mean(Y, axis=1, keepdims=True)
        return delta_D, C_D_L

    def test_that_svd_falls_back_to_gesvd_and_warns(self, monkeypatch):
        delta_D, C_D_L = self._make_problem()
        kwargs = {
            "delta_D": delta_D,
            "C_D_L": C_D_L,
            "alpha": 1.0,
            "truncation": 1.0,
        }

        # Reference result from the normal path (gesdd succeeds).
        expected = np.linalg.multi_dot(invert_subspace(**kwargs))

        real_svd = sp.linalg.svd
        drivers_used = []

        def flaky_svd(a, *args, lapack_driver="gesdd", overwrite_a=False, **kw):
            drivers_used.append(lapack_driver)
            if lapack_driver == "gesdd":
                # LAPACK destroys A when allowed to, also when it fails, so the
                # retry only works if the first call did not overwrite G.
                if overwrite_a:
                    a.fill(np.nan)
                raise np.linalg.LinAlgError("SVD did not converge")
            return real_svd(a, *args, lapack_driver=lapack_driver, **kw)

        monkeypatch.setattr(sp.linalg, "svd", flaky_svd)

        with pytest.warns(UserWarning, match="gesdd failed"):
            result = np.linalg.multi_dot(invert_subspace(**kwargs))

        # The fallback driver was actually exercised, in order.
        assert drivers_used == ["gesdd", "gesvd"]
        # The fallback produces the same (correct) result as the normal path.
        assert np.allclose(result, expected)


def test_timing():
    """This function can be used for timing. It also doubles as a test."""

    rng = np.random.default_rng(42)

    num_inputs = 5000
    num_outputs = 500
    ensemble_members = 10

    # Covariance
    C_D_L_diag = np.exp(rng.normal(size=num_outputs))
    C_D_L = np.diag(C_D_L_diag)
    assert C_D_L_diag.ndim == 1
    assert C_D_L.ndim == 2

    alpha = rng.uniform()

    # Create observations
    D = rng.normal(size=(num_outputs, ensemble_members))
    Y = rng.normal(size=(num_outputs, ensemble_members))
    X = rng.normal(size=(num_inputs, ensemble_members))
    delta_D = Y - np.mean(Y, axis=1, keepdims=True)

    print("-" * 32)

    start_time = perf_counter()
    result_matrix = invert_naive(
        alpha=alpha, C_D_L=C_D_L, truncation=1, delta_D=delta_D
    )
    result_dense1 = np.linalg.multi_dot([X, result_matrix, (D - Y)])
    elapsed = perf_counter() - start_time
    print(f"Function: {invert_naive.__name__} on dense covariance: {elapsed:.4f} s")

    start_time = perf_counter()
    result_matrix = invert_naive(
        alpha=alpha, C_D_L=C_D_L_diag, truncation=1, delta_D=delta_D
    )
    result_diag2 = np.linalg.multi_dot([X, result_matrix, (D - Y)])
    elapsed = perf_counter() - start_time
    print(f"Function: {invert_naive.__name__} on diagonal covariance: {elapsed:.4f} s")
    assert np.allclose(result_dense1, result_diag2)

    print("-" * 32)

    print("-" * 32)

    start_time = perf_counter()
    result_matrices = invert_subspace(
        alpha=alpha, C_D_L=C_D_L, truncation=1, delta_D=delta_D
    )
    result_dense2 = np.linalg.multi_dot([X] + list(result_matrices) + [(D - Y)])
    elapsed = perf_counter() - start_time
    print(f"Function: {invert_subspace.__name__} on dense covariance: {elapsed:.4f} s")

    start_time = perf_counter()
    result_matrices = invert_subspace(
        alpha=alpha, C_D_L=C_D_L_diag, truncation=1, delta_D=delta_D
    )
    result_diag2 = np.linalg.multi_dot([X] + list(result_matrices) + [(D - Y)])
    elapsed = perf_counter() - start_time
    print(
        f"Function: {invert_subspace.__name__} on diagonal covariance: {elapsed:.4f} s"
    )
    assert np.allclose(result_dense2, result_diag2)

    print("-" * 32)

    assert np.allclose(result_dense1, result_dense2)


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "-v",
        ]
    )
