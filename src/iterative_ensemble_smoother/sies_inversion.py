"""

SIES inversions
---------------

This file contains functions for solving Equation (42) in 2019 Evensen paper.
Another useful reference is the appendix of the 2012 Emerick paper.

We wish to compute

    W - step_length * (W - S.T @ inv(S @ S.T + C_dd) @ H)             eqn (42)

and the paper outlines four ways to do so. The main implementations are:

- Section 3.1 - Direct inversion
- Section 3.2 - Exact inversion (in the ensemble subspace)
- Section 3.3 - Projected inversion (in the ensemble subspace)


Scaling
-------

In line 8 in Algorithm 1, we have to compute (S S^T + E E^T)^-1 H, where
H := (SW + D - g(X)). This is equivalent to solving the following
equation for an unknown matrix M:
    (S S^T + E E^T) M = (SW + D - g(X))
Afterward we compute S.T @ M. If we scale the rows (observed variables)
of these equations using the standard deviations, we can obtain better
conditioning numbers on the equation. This corresponds to left-multiplying
with a diagonal matrix L := sqrt(diag(C_dd)). In an experiment with
random covariance matrices, this improved the condition number ~90%
of the time (results may depend on how random covariance matrices are
generated --- I generated covariances C as E ~ stdnorm(), then C = E.T @ E).

To see the equality, note that if we scale S, E and H := (SW + D - g(X))
we obtain:
    (L S) (L S)^T + (L E) (L E)^T M_2 = L H
              L (S S^T + E E^T) L M_2 = L H
            L (S S^T + E E^T) L L^-1 M = L H
so the new solution is M_2 := L^-1 M, expressed in terms of the original M.
But when we left-multiply M_2 with S^T, we do so with a transformed S,
so we obtain S_2^T M_2 = (L S)^T (L^-1 M) = S^T M, so the solution
to the transformed system is equal to the solution of the original system.
In the implementation of scaling the right hand side (SW + D - g(X))
we first scale D - g(X), then we scale S implicitly by solving
S Sigma = L Y, instead of S Sigma = Y for S.

Another scaling technique, using the cholesky factorization, is explained in
the appendix of the 2012 Emerick paper. When C_dd is diagonal, both scaling
techniques (scaling using correlation or cholesky factors) coincide.

"""

import numbers
from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore


def calc_num_significant(
    singular_values: npt.NDArray[np.double], truncation: float
) -> int:
    """Determine the number of singular values by enforcing that less than a
    fraction truncation of the total variance be accounted for.

    Note: In e.g., scipy.linalg.pinv the following criteria is used:
        atol + rtol * max(singular_values) <= truncation
    Here we use cumsum(normalize(singular_values**2)) <= truncation

    Parameters
    ----------
    singular_values : np.ndarray
        Array with singular values, sorted in decreasing order.
    truncation : float
        Fraction of energy squared singular values to keep.

    Returns
    -------
    int
        Last index to be included in singular values array.

    Examples
    --------
    >>> singular_values = np.array([2, 2, 1, 1])
    >>> calc_num_significant(singular_values, 1.0)
    4
    >>> calc_num_significant(singular_values, 0.8)
    2
    >>> calc_num_significant(singular_values, 0.01)
    1

    >>> singular_values = np.array([3, 2, 1, 0, 0, 0])
    >>> i = calc_num_significant(singular_values, 1.0)
    >>> singular_values[:i]
    array([3, 2, 1])

    """
    assert np.all(np.diff(singular_values) <= 0), "Must be sorted decreasing"
    assert 0 < truncation <= 1

    sigma = np.cumsum(singular_values**2)
    total_sigma = sigma[-1]  # Sum of squared singular values

    relative_energy = sigma / total_sigma
    return int(np.searchsorted(relative_energy, truncation, side="left") + 1)


def _verify_inversion_args(
    *,
    W: npt.NDArray[np.double],
    step_length: float,
    S: npt.NDArray[np.double],
    C_dd: npt.NDArray[np.double],
    H: npt.NDArray[np.double],
    C_dd_cholesky: Optional[npt.NDArray[np.double]],
    truncation: float,
) -> None:
    """Verify shapes of arguments."""
    if C_dd_cholesky is not None:
        assert C_dd.shape == C_dd_cholesky.shape

    assert isinstance(step_length, numbers.Real)
    assert 0 < step_length <= 1.0

    assert isinstance(truncation, numbers.Real)
    assert 0 < truncation <= 1.0

    # Verify N
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
    truncation: float = 1.0,
) -> npt.NDArray[np.double]:
    """A naive implementation, used for benchmarking and testing only.

    Naive implementation of Equation (42)."""
    _verify_inversion_args(
        W=W,
        step_length=step_length,
        S=S,
        C_dd=C_dd,
        H=H,
        C_dd_cholesky=C_dd_cholesky,
        truncation=truncation,
    )

    # Since it's a naive approach, we just create a diagonal matrix
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
    truncation: float = 1.0,
) -> npt.NDArray[np.double]:
    """Implementation of Equation (42), with correlation matrix scaling."""
    _verify_inversion_args(
        W=W,
        step_length=step_length,
        S=S,
        C_dd=C_dd,
        H=H,
        C_dd_cholesky=C_dd_cholesky,
        truncation=truncation,
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
        # We note that we only need the upper-triangular part of S @ S.T + I in
        # the sp.linalg.solve routine, and that the BLAS routine dsyrk can compute
        # this, via sp.linalg.blas.dsyrk(alpha=1.0, a=S).
        # However, we chose NOT to use BLAS 'directly' since it makes the code
        # less readable, and the gains in speed are marginal.

        lhs = S @ S.T
        np.fill_diagonal(lhs, lhs.diagonal() + 1)
    else:
        lhs = S @ S.T + R

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


def inversion_subspace_exact(
    *,
    W: npt.NDArray[np.double],
    step_length: float,
    S: npt.NDArray[np.double],
    C_dd: npt.NDArray[np.double],
    H: npt.NDArray[np.double],
    C_dd_cholesky: npt.NDArray[np.double],
    truncation: float = 1.0,
) -> npt.NDArray[np.double]:
    """Implementation of Equation (50), which performs inversion in the ensemble
    space (size N) instead doing it in the output space (size m >> N)."""
    _verify_inversion_args(
        W=W,
        step_length=step_length,
        S=S,
        C_dd=C_dd,
        H=H,
        C_dd_cholesky=C_dd_cholesky,
        truncation=truncation,
    )

    # Correlation scaling
    if C_dd.ndim == 2:
        scale_factor = 1 / np.sqrt(np.diag(C_dd))
        # Scale rows and columns by diagonal, creating correlation matrix R
        # R = (C_dd * scale_factor.reshape(1, -1)) * scale_factor.reshape(-1, 1)
        R_cholesky = C_dd_cholesky * scale_factor.reshape(-1, 1)
    else:
        scale_factor = 1 / np.sqrt(C_dd)
        # The correlation matrix R is simply the identity matrix in this case

    # Scale rows
    S = S * scale_factor.reshape(-1, 1)
    H = H * scale_factor.reshape(-1, 1)

    # Special case for diagonal covariance matrix.
    # See below for a more explanation of these computations.
    if C_dd.ndim == 1:
        lhs = S.T @ S
        np.fill_diagonal(lhs, lhs.diagonal() + 1)
        C_dd_inv_H = H

    else:
        # Solve the equation: C_dd_cholesky @ K = S for K,
        # which is equivalent to forming K := C_dd_cholesky^-1 @ S,
        # exploiting the fact that C_dd_cholesky is lower triangular
        K = sp.linalg.solve_triangular(a=R_cholesky, b=S, lower=True)

        # Form lhs := (S.T @ C_dd^-1 @ S + I)
        lhs = K.T @ K
        np.fill_diagonal(lhs, lhs.diagonal() + 1)

        # Compute C_dd^-1 @ H, exploiting the fact that we have the cholesky factor
        C_dd_inv_H = sp.linalg.cho_solve((R_cholesky, 1), H)

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


def inversion_subspace_projected(
    *,
    W: npt.NDArray[np.double],
    step_length: float,
    S: npt.NDArray[np.double],
    C_dd: npt.NDArray[np.double],
    H: npt.NDArray[np.double],
    C_dd_cholesky: npt.NDArray[np.double],
    truncation: float = 1.0,
) -> npt.NDArray[np.double]:
    """Implementation of section 3.3 : 'Ensemble Subspace Inversion Using Full C_dd'
    with correlation scaling. This scales the covariance matrix to a correlation
    matrix. The idea is that the relative importance of the variables, not the magnitude
    of the numbers in the variables, will determine the truncation.

    Note that this is an APPROXIMATE inversion in general.
    When N >= m and truncation is 1.0, then there is no approximation.

    """
    _verify_inversion_args(
        W=W,
        step_length=step_length,
        S=S,
        C_dd=C_dd,
        H=H,
        C_dd_cholesky=C_dd_cholesky,
        truncation=truncation,
    )
    assert 0 < truncation <= 1.0

    # Correlation scaling
    if C_dd.ndim == 2:
        scale_factor = 1 / np.sqrt(np.diag(C_dd))
        # Scale rows and columns by diagonal, creating correlation matrix R
        # R = (C_dd * scale_factor.reshape(1, -1)) * scale_factor.reshape(-1, 1)
        R_chol = C_dd_cholesky * scale_factor.reshape(-1, 1)
    else:
        scale_factor = 1 / np.sqrt(C_dd)
        # The correlation matrix R is simply the identity matrix in this case

    # Scale rows
    S = S * scale_factor.reshape(-1, 1)
    H = H * scale_factor.reshape(-1, 1)

    # Equation (58)
    U, Sigma, VT = sp.linalg.svd(
        S,
        full_matrices=False,
        compute_uv=True,
        overwrite_a=False,
        check_finite=True,
        lapack_driver="gesdd",
    )

    # Compute Sigma^+
    num_significant = calc_num_significant(Sigma, truncation)
    Sigma_inv = 1 / Sigma[:num_significant]
    U = U[:, :num_significant]
    VT = VT[:num_significant, :]

    # Equation (59), using the cholesky factor: C_dd_cholesky @ C_dd_cholesky.T = C_dd
    if C_dd_cholesky.ndim == 2:
        K = np.linalg.multi_dot([U.T, R_chol]) * Sigma_inv.reshape(-1, 1)
    else:
        K = U.T * Sigma_inv.reshape(-1, 1)
    # Compute SVD, which is equivalent to taking eigendecomposition
    # since C_tilde is PSD. Using eigh() is faster than svd().
    # Note that svd() returns eigenvalues in decreasing order, while eigh()
    # returns eigenvalues in increasing order.
    # driver="evr" => fastest option
    Lambda, Z = sp.linalg.eigh(K @ K.T, driver="evr", overwrite_a=True)

    # Equation (60)
    K = np.linalg.multi_dot([U * Sigma_inv, Z])
    K *= 1 / (1 + Lambda) ** 0.5

    # Equation (42), but with (S @ S.T + C_dd)^{-1} expressed as K @ K.T
    F: npt.NDArray[np.double] = np.linalg.multi_dot([S.T, K, K.T, H])
    return W - step_length * (W - F)
