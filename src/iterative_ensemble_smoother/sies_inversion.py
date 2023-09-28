from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

import numbers

import scipy as sp  # type: ignore


def _verify_inversion_args(
    *,
    W: npt.NDArray[np.double],
    step_length: float,
    S: npt.NDArray[np.double],
    C_dd: npt.NDArray[np.double],
    H: npt.NDArray[np.double],
    C_dd_cholesky: Optional[npt.NDArray[np.double]],
) -> None:
    if C_dd_cholesky is not None:
        assert C_dd.shape == C_dd_cholesky.shape

    assert isinstance(step_length, numbers.Real)
    assert 0 < step_length <= 1.0

    # Verify N
    # assert W.shape[0] == W.shape[1]
    assert W.shape[0] == S.shape[1]
    assert W.shape[1] == H.shape[1]

    # Verify m
    assert S.shape[0] == C_dd.shape[0]
    assert S.shape[0] == H.shape[0]


def inversion_naive(
    *,
    W: npt.NDArray[np.double],
    step_length: float,
    S: npt.NDArray[np.double],
    C_dd: npt.NDArray[np.double],
    H: npt.NDArray[np.double],
    C_dd_cholesky: Optional[npt.NDArray[np.double]],
) -> npt.NDArray[np.double]:
    """A naive implementation, used for benchmarking and testing only.

    Naive implementation of Equation (42)."""
    _verify_inversion_args(
        W=W, step_length=step_length, S=S, C_dd=C_dd, H=H, C_dd_cholesky=C_dd_cholesky
    )

    # Since it's naive, we just create zeros here
    if C_dd.ndim == 1:
        C_dd = np.diag(C_dd)

    to_invert = S @ S.T + C_dd
    ans: npt.NDArray[np.double] = W - step_length * (
        W - np.linalg.multi_dot([S.T, sp.linalg.inv(to_invert), H])
    )
    return ans


def inversion_direct(
    *,
    W: npt.NDArray[np.double],
    step_length: float,
    S: npt.NDArray[np.double],
    C_dd: npt.NDArray[np.double],
    H: npt.NDArray[np.double],
    C_dd_cholesky: Optional[npt.NDArray[np.double]],
) -> npt.NDArray[np.double]:
    """A more optimized implementation of Equation (42)."""
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
    ans: npt.NDArray[np.double] = W - step_length * (W - np.linalg.multi_dot([S.T, K]))
    return ans


def inversion_direct_corrscale(
    *,
    W: npt.NDArray[np.double],
    step_length: float,
    S: npt.NDArray[np.double],
    C_dd: npt.NDArray[np.double],
    H: npt.NDArray[np.double],
    C_dd_cholesky: Optional[npt.NDArray[np.double]],
) -> npt.NDArray[np.double]:
    """Implementation of Equation (42), with correlation matrix scaling."""
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
        lower=False,
        overwrite_a=True,
        overwrite_b=False,
        check_finite=True,
        assume_a="pos",
    )
    ans: npt.NDArray[np.double] = W - step_length * (W - np.linalg.multi_dot([S.T, K]))
    return ans


def inversion_exact(
    *,
    W: npt.NDArray[np.double],
    step_length: float,
    S: npt.NDArray[np.double],
    C_dd: npt.NDArray[np.double],
    H: npt.NDArray[np.double],
    C_dd_cholesky: npt.NDArray[np.double],
) -> npt.NDArray[np.double]:
    """Implementation of Equation (50), which does inversion in the ensemble
    space (size N) instead doing it in the output space (size m >> N)."""
    _verify_inversion_args(
        W=W, step_length=step_length, S=S, C_dd=C_dd, H=H, C_dd_cholesky=C_dd_cholesky
    )

    # Special case for diagonal covariance matrix.
    # See below for a more explanation of these computations.
    if C_dd.ndim == 1:
        K = S / C_dd_cholesky.reshape(-1, 1)
        lhs = K.T @ K  # sp.linalg.blas.dsyrk(alpha=1.0, a=K, trans=1)
        lhs.flat[:: lhs.shape[0] + 1] += 1  # type: ignore
        C_dd_inv_H = H / C_dd.reshape(-1, 1)

    else:
        # Solve the equation: C_dd_cholesky @ K = S for K,
        # which is equivalent to forming K := C_dd_cholesky^-1 @ S,
        # exploiting the fact that C_dd_cholesky is lower triangular
        # K = sp.linalg.blas.dtrsm(alpha=1.0, a=C_dd_cholesky, b=S, lower=1)
        K = sp.linalg.solve(C_dd_cholesky, S)

        # Form lhs := (S.T @ C_dd^-1 @ S + I)
        lhs = K.T @ K  # sp.linalg.blas.dsyrk(alpha=1.0, a=K, trans=1)
        lhs.flat[:: lhs.shape[0] + 1] += 1  # type: ignore

        # Compute C_dd^-1 @ H, exploiting the fact that we have the cholesky factor
        C_dd_inv_H = sp.linalg.cho_solve((C_dd_cholesky, 1), H)

    # Solve the following for F
    # lhs @ F = S.T @ C_dd_inv_H
    F: npt.NDArray[np.double] = sp.linalg.solve(
        lhs,
        S.T @ C_dd_inv_H,
        lower=False,
        overwrite_a=True,
        overwrite_b=True,
        check_finite=True,
        assume_a="pos",
    )

    return W - step_length * (W - F)
