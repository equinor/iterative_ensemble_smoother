from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable

import numpy as np
import scipy as sp

if TYPE_CHECKING:
    import numpy.typing as npt


# =============================================================================
# MAIN CLASS
# =============================================================================


class SIES:
    def __init__(
        self,
        param_ensemble: npt.NDArray[np.double],
        observation_errors: npt.NDArray[np.double],
        observation_values: npt.NDArray[np.double],
        *,
        steplength_schedule: Optional[Callable[[int], float]] = None,
        seed: Optional[int] = None,
        verbosity: int = 0,
    ):
        self.iteration = 1
        self.steplength_schedule = steplength_schedule
        self.rng = np.random.default_rng(seed)
        self.X = param_ensemble
        self.d = observation_values
        self.C_dd = observation_errors
        self.A = (self.X - self.X.mean(axis=1, keepdims=True)) / np.sqrt(
            self.X.shape[1] - 1
        )

        if self.C_dd.ndim == 2:
            self.C_dd_cholesky = sp.linalg.cholesky(
                self.C_dd,
                lower=True,
                overwrite_a=False,
                check_finite=True,
            )
        else:
            self.C_dd_cholesky = np.sqrt(self.C_dd)

        # Equation (14)
        self.D = self.d.reshape(-1, 1) + sample_mvnormal(
            C_dd_cholesky=self.C_dd_cholesky, rng=self.rng, size=self.X.shape[1]
        )

        self.W = np.zeros(shape=(self.X.shape[1], self.X.shape[1]))
        self.X_i = self.X

    def newton(self, Y, step_length=0.5):
        """Implementation of Algorithm 1."""
        g_X = Y.copy()

        # Get shapes. Same notation as used in the paper.
        N = Y.shape[1]  # Ensemble members
        n = self.X.shape[0]  # Parameters (inputs)
        m = self.C_dd.shape[0]  # Responses (outputs)

        # Line 4 in Algorithm 1
        Y = (g_X - g_X.mean(axis=1, keepdims=True)) / np.sqrt(N - 1)

        # Line 5
        Omega = self.W.copy()
        Omega -= Omega.mean(axis=1, keepdims=True)
        Omega /= np.sqrt(N - 1)
        Omega.flat[:: Omega.shape[0] + 1] += 1  # Add identity in place

        # Line 6
        if n < N - 1:
            # There are fewer parameters than realizations. This means that the
            # system of equations is overdetermined, and we must solve a least
            # squares problem.

            # An alternative approach to producing A_i would be keeping the
            # returned value from the previous Newton iteration (call it X_i),
            # then computing:
            # A_i = (self.X_i - self.X_i.mean(axis=1, keepdims=True)) / np.sqrt(N - 1)
            A_i = self.A @ Omega
            S = sp.linalg.solve(
                Omega.T, np.linalg.multi_dot([Y, sp.linalg.pinv(A_i), A_i]).T
            ).T
        else:
            # The system is underdetermined
            S = sp.linalg.solve(Omega.T, Y.T).T

        # Line 7
        H = S @ self.W + self.D - g_X

        # Line 8
        assert self.W.shape == (N, N)
        assert S.shape == (m, N)
        assert self.C_dd.shape in [(m, m), (m,)]
        assert H.shape == (m, N)
        self.W = inversion_exact(
            W=self.W,
            step_length=step_length,
            S=S,
            C_dd=self.C_dd,
            H=H,
            C_dd_cholesky=self.C_dd_cholesky,
        )

        # Line 9
        return self.X + self.X @ self.W / (np.sqrt(N - 1))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def center(X):
    # Center each row, in place, so sum(row) = 0 for every row
    X -= X.mean(axis=1, keepdims=True)


def scale(X):
    # Scale each entry so a_ij = a_ij / sqrt(columns - 1)
    X /= np.sqrt(X.shape[1] - 1)


def sample_mvnormal(*, C_dd_cholesky, rng, size):
    """Draw samples from N(0, C_dd).

    Examples
    --------
    >>> C_dd_cholesky = np.diag([5, 10, 15])
    >>> rng = np.random.default_rng(42)
    >>> sample_mvnormal(C_dd_cholesky=C_dd_cholesky, rng=rng, size=2)
    array([[  1.5235854 ,  -5.19992053],
           [  7.50451196,   9.40564716],
           [-29.26552783, -19.5326926 ]])
    >>> sample_mvnormal(C_dd_cholesky=np.diag(C_dd_cholesky), rng=rng, size=2)
    array([[ 0.63920202, -1.58121296],
           [-0.16801158, -8.53043928],
           [13.19096962, 11.66687903]])
    """

    # A 2D covariance matrix was passed
    if C_dd_cholesky.ndim == 2:
        return C_dd_cholesky @ rng.standard_normal(size=(C_dd_cholesky.shape[0], size))
    else:
        return C_dd_cholesky.reshape(-1, 1) * rng.standard_normal(
            size=(C_dd_cholesky.shape[0], size)
        )


def _verify_inversion_args(*, W, step_length, S, C_dd, H, C_dd_cholesky):
    if C_dd_cholesky is not None:
        assert C_dd.shape == C_dd_cholesky.shape

    # Verify N
    assert W.shape[0] == W.shape[1]
    assert W.shape[0] == S.shape[1]
    assert W.shape[0] == H.shape[1]

    # Verify m
    assert S.shape[0] == C_dd.shape[0]
    assert S.shape[0] == H.shape[0]


# =============================================================================
# INVERSION FUNCTIONS
# =============================================================================


def inversion_naive(*, W, step_length, S, C_dd, H, C_dd_cholesky=None):
    """Naive implementation of Equation (42)."""
    _verify_inversion_args(
        W=W, step_length=step_length, S=S, C_dd=C_dd, H=H, C_dd_cholesky=C_dd_cholesky
    )

    # Since it's naive, we just create zeros here
    if C_dd.ndim == 1:
        C_dd = np.diag(C_dd)

    to_invert = S @ S.T + C_dd
    return W - step_length * (
        W - np.linalg.multi_dot([S.T, sp.linalg.inv(to_invert), H])
    )


def inversion_direct(*, W, step_length, S, C_dd, H, C_dd_cholesky=None):
    """Implementation of equation (42)."""
    _verify_inversion_args(
        W=W, step_length=step_length, S=S, C_dd=C_dd, H=H, C_dd_cholesky=C_dd_cholesky
    )

    # We define K as the solution to
    # (S @ S.T + C_dd) @ K = H, so that
    # K = (S @ S.T + C_dd)^-1 H
    # K = solve(S @ S.T + C_dd, H)
    if C_dd.ndim == 1:
        # Since S has shape (m, N), and m >> N, using BLAS is typically faster:
        # >>> S = np.random.randn(10_000, 100)
        # >>> %timeit S @ S.T
        # 512 ms ± 14.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # >>> %timeit sp.linalg.blas.dsyrk(alpha=1.0, a=S)
        # 225 ms ± 37 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # Here BLAS only computes the upper triangular part of S @ S.T, but
        # that's all we need when we call the solver with "lower=False".
        lhs = sp.linalg.blas.dsyrk(alpha=1.0, a=S)  # Equivalent to S @ S.T
        lhs.flat[:: lhs.shape[0] + 1] += C_dd
    else:
        lhs = sp.linalg.blas.dsyrk(alpha=1.0, a=S, c=C_dd, beta=1.0)  # S @ S.T + C_dd

    K = sp.linalg.solve(
        lhs,
        H,
        lower=False,
        overwrite_a=True,
        overwrite_b=False,
        check_finite=True,
        assume_a="pos",
    )
    return W - step_length * (W - np.linalg.multi_dot([S.T, K]))


def inversion_direct_corrscale(*, W, step_length, S, C_dd, H, C_dd_cholesky=None):
    """Implementation of equation (42), with correlation matrix scaling."""
    _verify_inversion_args(
        W=W, step_length=step_length, S=S, C_dd=C_dd, H=H, C_dd_cholesky=C_dd_cholesky
    )

    if C_dd.ndim == 2:
        scale_factor = 1 / np.sqrt(np.diag(C_dd))
        # Scale rows and columns by diagonal, creating correlation matrix R
        R = (C_dd * scale_factor.reshape(1, -1)) * scale_factor.reshape(-1, 1)
    else:
        scale_factor = 1 / np.sqrt(C_dd)
        # The correlation matrix R is simply the identity matrix in this case

    # Scale rows
    S = S * scale_factor.reshape(-1, 1)
    H = H * scale_factor.reshape(-1, 1)

    # We define K as the solution to
    # (S @ S.T + C_dd) @ K = H, so that
    # K = (S @ S.T + C_dd)^-1 H
    # K = solve(S @ S.T + C_dd, H)
    if C_dd.ndim == 1:
        lhs = sp.linalg.blas.dsyrk(alpha=1.0, a=S)  # S.T @ S
        lhs.flat[:: lhs.shape[0] + 1] += 1
    else:
        lhs = sp.linalg.blas.dsyrk(alpha=1.0, a=S, beta=1.0, c=R)  # S.T @ S + R

    K = sp.linalg.solve(
        lhs,
        H,
        sym_pos=False,
        lower=False,
        overwrite_a=True,
        overwrite_b=False,
        check_finite=True,
        assume_a="pos",
    )
    return W - step_length * (W - np.linalg.multi_dot([S.T, K]))


def inversion_exact(*, W, step_length, S, C_dd, H, C_dd_cholesky):
    """Implementation of equation (50)."""
    _verify_inversion_args(
        W=W, step_length=step_length, S=S, C_dd=C_dd, H=H, C_dd_cholesky=C_dd_cholesky
    )

    # Special case for diagonal covariance matrix.
    # See below for a more explanation of these computations.
    if C_dd.ndim == 1:
        K = S / C_dd_cholesky.reshape(-1, 1)
        lhs = sp.linalg.blas.dsyrk(alpha=1.0, a=K, trans=1)  # K.T @ K
        lhs.flat[:: lhs.shape[0] + 1] += 1
        C_dd_inv_H = H / C_dd.reshape(-1, 1)
        K = sp.linalg.solve(
            lhs,
            S.T @ C_dd_inv_H,
            lower=False,
            overwrite_a=True,
            overwrite_b=True,
            check_finite=True,
            assume_a="pos",
        )
        return W - step_length * (W - K)

    # Solve the equation: C_dd_cholesky @ K = S for K,
    # which is equivalent to forming K := C_dd_cholesky^-1 @ S,
    # exploiting the fact that C_dd_cholesky is lower triangular
    K = sp.linalg.blas.dtrsm(alpha=1.0, a=C_dd_cholesky, b=S, lower=1)

    # Form lhs := (S.T @ C_dd^-1 @ S + I)
    lhs = sp.linalg.blas.dsyrk(alpha=1.0, a=K, trans=1)  # K.T @ K
    lhs.flat[:: lhs.shape[0] + 1] += 1

    # Compute C_dd^-1 @ H, exploiting the fact that we have the cholesky factor
    C_dd_inv_H = sp.linalg.cho_solve((C_dd_cholesky, 1), H)

    # Solve the following for K
    # lhs @ K = S.T @ C_dd_inv_H
    K = sp.linalg.solve(
        lhs,
        S.T @ C_dd_inv_H,
        lower=False,
        overwrite_a=True,
        overwrite_b=True,
        check_finite=True,
        assume_a="pos",
    )

    return W - step_length * (W - K)


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-v", "--maxfail=1"])
