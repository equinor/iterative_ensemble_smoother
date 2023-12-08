from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore


def empirical_covariance_upper(X: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Compute the upper triangular part of the empirical covariance matrix X
    with shape (parameters, ensemble_size).

    Examples
    --------
    >>> X = np.array([[-2.4, -0.3,  0.7,  0.2,  1.1],
    ...               [-1.5,  0.4, -0.4, -0.9,  1. ],
    ...               [-0.1, -0.4, -0. , -0.5,  1.1]])
    >>> empirical_covariance_upper(X)
    array([[1.873, 0.981, 0.371],
           [0.   , 0.997, 0.392],
           [0.   , 0.   , 0.407]])

    Naive computation:

    >>> empirical_cross_covariance(X, X)
    array([[1.873, 0.981, 0.371],
           [0.981, 0.997, 0.392],
           [0.371, 0.392, 0.407]])
    """
    _, num_observations = X.shape
    X = (X - np.mean(X, axis=1, keepdims=True)) / np.sqrt(num_observations - 1)
    # https://www.math.utah.edu/software/lapack/lapack-blas/dsyrk.html
    XXT: npt.NDArray[np.double] = sp.linalg.blas.dsyrk(alpha=1.0, a=X)
    return XXT


def empirical_cross_covariance(
    X: npt.NDArray[np.double], Y: npt.NDArray[np.double]
) -> npt.NDArray[np.double]:
    """Both X and Y have shape (parameters, ensemble_size).

    We use this function instead of np.cov to handle cross-correlation,
    where X and Y have a different number of parameters.

    Examples
    --------
    >>> X = np.array([[-2.4, -0.3,  0.7],
    ...               [ 0.2,  1.1, -1.5]])
    >>> Y = np.array([[ 0.4, -0.4, -0.9],
    ...               [ 1. , -0.1, -0.4],
    ...               [-0. , -0.5,  1.1],
    ...               [-1.8, -1.1,  0.3]])
    >>> empirical_cross_covariance(X, Y)
    array([[-1.035     , -1.15833333,  0.66      ,  1.56333333],
           [ 0.465     ,  0.36166667, -1.08      , -1.09666667]])

    Verify against numpy.cov

    >>> np.cov(X, rowvar=True, ddof=1)
    array([[ 2.50333333, -0.99666667],
           [-0.99666667,  1.74333333]])
    >>> empirical_cross_covariance(X, X)
    array([[ 2.50333333, -0.99666667],
           [-0.99666667,  1.74333333]])

    """
    assert X.shape[1] == Y.shape[1], "Ensemble size must be equal"

    # https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices
    # Subtract mean. Even though the equation says E[(X - E[X])(Y - E[Y])^T],
    # we actually only need to subtract the mean value from one matrix, since
    # (X - E[X])(Y - E[Y])^T = E[(X - E[X])Y] - E[(X - E[X])E[Y]^T]
    # = E[(X - E[X])Y] - E[(0)E[Y]^T] = E[(X - E[X])Y]
    # We choose to subtract from the matrix with the smaller number of rows
    if X.shape[0] > Y.shape[0]:
        Y = Y - np.mean(Y, axis=1, keepdims=True)
    else:
        X = X - np.mean(X, axis=1, keepdims=True)

    # Compute outer product and divide
    cov = X @ Y.T / (X.shape[1] - 1)
    assert cov.shape == (X.shape[0], Y.shape[0])
    return cov


def normalize_alpha(alpha: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Assure that sum_i (1/alpha_i) = 1.

    This is Eqn (22) in :cite:t:`EMERICK2013`.

    Examples
    --------
    >>> alpha = np.arange(10) + 1
    >>> np.sum(1/normalize_alpha(alpha))
    1.0
    """
    factor = np.sum(1 / alpha)
    rescaled: npt.NDArray[np.double] = alpha * factor
    return rescaled


def singular_values_to_keep(
    singular_values: npt.NDArray[np.double], truncation: float = 1.0
) -> int:
    """Find the index of the singular values to keep when truncating.

    Examples
    --------
    >>> singular_values = np.array([3, 2, 1, 0, 0, 0])
    >>> i = singular_values_to_keep(singular_values, truncation=1.0)
    >>> singular_values[:i]
    array([3, 2, 1])

    >>> singular_values = np.array([4, 3, 2, 1])
    >>> i = singular_values_to_keep(singular_values, truncation=1.0)
    >>> singular_values[:i]
    array([4, 3, 2, 1])

    >>> singular_values = np.array([4, 3, 2, 1])
    >>> singular_values_to_keep(singular_values, truncation=0.95)
    4
    >>> singular_values_to_keep(singular_values, truncation=0.9)
    3
    >>> singular_values_to_keep(singular_values, truncation=0.7)
    2

    """
    assert np.all(
        np.diff(singular_values) <= 0
    ), "Singular values must be sorted decreasing"
    assert 0 < truncation <= 1, "Threshold must be in range (0, 1]"
    singular_values = np.array(singular_values, dtype=float)

    # Take cumulative sum and normalize
    cumsum = np.cumsum(singular_values)
    cumsum /= cumsum[-1]
    return int(np.searchsorted(cumsum, v=truncation, side="left") + 1)


# =============================================================================
# INVERSION FUNCTIONS
# =============================================================================
# All of these functions compute (exactly, or approximately), the product
#
#  C_MD @ inv(C_DD + alpha * C_D) @ (D - Y)
#
# where C_MD = empirical_cross_covariance(X, Y) = center(X) @ center(Y).T
#               / (X.shape[1] - 1)
#       C_DD = empirical_cross_covariance(Y, Y) = center(Y) @ center(Y).T
#               / (Y.shape[1] - 1)
#
# The methods can be classified as
#   - exact : with truncation=1.0, these methods compute the exact solution
#   - exact : with truncation<1.0, these methods may approximate the solution
#   - approximate: if ensemble_members <= num_outputs, then the solution is
#                  always approximated, regardless of the truncation
#   - approximate: if ensemble_members > num_outputs, then the solution is
#                  exact when truncation is 1.0

# Every inversion function has the form
# inversion_<exact/approximate>_<name>


def inversion_exact_naive(
    *,
    alpha: float,
    C_D: npt.NDArray[np.double],
    D: npt.NDArray[np.double],
    Y: npt.NDArray[np.double],
    X: npt.NDArray[np.double],
    truncation: float = 1.0,
) -> npt.NDArray[np.double]:
    """Naive inversion, used for testing only.

    Computes C_MD @ inv(C_DD + alpha * C_D) @ (D - Y) naively.
    """
    # Naive implementation of Equation (3) in Emerick (2013)
    C_MD = empirical_cross_covariance(X, Y)
    C_DD = empirical_cross_covariance(Y, Y)
    C_D = np.diag(C_D) if C_D.ndim == 1 else C_D
    return C_MD @ sp.linalg.inv(C_DD + alpha * C_D) @ (D - Y)  # type: ignore


def inversion_exact_cholesky(
    *,
    alpha: float,
    C_D: npt.NDArray[np.double],
    D: npt.NDArray[np.double],
    Y: npt.NDArray[np.double],
    X: Optional[npt.NDArray[np.double]],
    truncation: float = 1.0,
    return_T: bool = False,
) -> npt.NDArray[np.double]:
    """Computes an exact inversion using `sp.linalg.solve`, which uses a
    Cholesky factorization in the case of symmetric, positive definite matrices.

    The goal is to compute: C_MD @ inv(C_DD + alpha * C_D) @ (D - Y)

    First we solve (C_DD + alpha * C_D) @ T = (D - Y) for T, so that
    T = inv(C_DD + alpha * C_D) @ (D - Y), then we compute
    C_MD @ T, but we don't explicitly form C_MD, since it might be more
    efficient to perform the matrix products in another order.
    """
    C_DD = empirical_covariance_upper(Y)  # Only compute upper part

    # Arguments for sp.linalg.solve
    solver_kwargs = {
        "overwrite_a": True,
        "overwrite_b": True,
        "assume_a": "pos",  # Assume positive definite matrix (use cholesky)
        "lower": False,  # Only use the upper part while solving
    }

    # Compute T := sp.linalg.inv(C_DD + alpha * C_D) @ (D - Y)
    if C_D.ndim == 2:
        # C_D is a covariance matrix
        C_DD += alpha * C_D  # Save memory by mutating
        T = sp.linalg.solve(C_DD, D - Y, **solver_kwargs)
    elif C_D.ndim == 1:
        # C_D is an array, so add it to the diagonal without forming diag(C_D)
        C_DD.flat[:: C_DD.shape[1] + 1] += alpha * C_D
        T = sp.linalg.solve(C_DD, D - Y, **solver_kwargs)

    # Center matrix
    Y = Y - np.mean(Y, axis=1, keepdims=True)
    _, num_ensemble = Y.shape

    # Don't left-multiply the X
    if return_T:
        return (Y.T @ T) / (num_ensemble - 1)  # type: ignore

    return np.linalg.multi_dot([X, Y.T / (num_ensemble - 1), T])  # type: ignore


def inversion_exact_lstsq(
    *,
    alpha: float,
    C_D: npt.NDArray[np.double],
    D: npt.NDArray[np.double],
    Y: npt.NDArray[np.double],
    X: npt.NDArray[np.double],
    truncation: float = 1.0,
) -> npt.NDArray[np.double]:
    """Computes inversion using least squares. While this method can deal with
    rank-deficient C_D, it should not be used since it's very slow.
    """
    C_DD = empirical_cross_covariance(Y, Y)

    # A covariance matrix was given
    if C_D.ndim == 2:
        lhs = C_DD + alpha * C_D
    # A diagonal covariance matrix was given as a vector
    else:
        lhs = C_DD
        lhs.flat[:: lhs.shape[0] + 1] += alpha * C_D

    # T = lhs^-1 @ (D - Y)
    # lhs @ T = (D - Y)
    ans, *_ = sp.linalg.lstsq(
        lhs, D - Y, overwrite_a=True, overwrite_b=True, lapack_driver="gelsy"
    )

    # Compute C_MD := X @ center(Y).T / (Y.shape[1] - 1)
    Y_shift = (Y - np.mean(Y, axis=1, keepdims=True)) / (Y.shape[1] - 1)
    return np.linalg.multi_dot([X, Y_shift.T, ans])  # type: ignore


def inversion_exact_rescaled(
    *,
    alpha: float,
    C_D: npt.NDArray[np.double],
    D: npt.NDArray[np.double],
    Y: npt.NDArray[np.double],
    X: npt.NDArray[np.double],
    truncation: float = 1.0,
) -> npt.NDArray[np.double]:
    """Compute a rescaled inversion.

    See Appendix A.1 in :cite:t:`emerickHistoryMatchingTimelapse2012`
    for details regarding this approach.
    """
    C_DD = empirical_cross_covariance(Y, Y)

    if C_D.ndim == 2:
        # Eqn (57). Cholesky factorize the covariance matrix C_D
        # TODO: here we compute the cholesky factor in every call, but C_D
        # never changes. it would be better to compute it once
        C_D_L = sp.linalg.cholesky(C_D, lower=True)  # Lower triangular cholesky
        C_D_L_inv, _ = sp.linalg.lapack.dtrtri(
            C_D_L, lower=1, overwrite_c=1
        )  # Invert lower triangular using BLAS routine
        C_D_L_inv /= np.sqrt(alpha)

        # Eqn (59). Form C_tilde
        # TODO: Use BLAS routine for triangular times dense matrix
        # sp.linalg.blas.strmm(alpha=1, a=C_D_L_inv, b=C_DD, lower=1)
        C_tilde = C_D_L_inv @ C_DD @ C_D_L_inv.T
        C_tilde.flat[:: C_tilde.shape[0] + 1] += 1.0  # Add to diagonal

    # When C_D is a diagonal covariance matrix, there is no need to perform
    # the cholesky factorization
    elif C_D.ndim == 1:
        C_D_L_inv = 1 / np.sqrt(C_D * alpha)
        C_tilde = (C_D_L_inv * (C_DD * C_D_L_inv).T).T
        C_tilde.flat[:: C_tilde.shape[0] + 1] += 1.0  # Add to diagonal

    # Eqn (60). Compute SVD, which is equivalent to taking eigendecomposition
    # since C_tilde is PSD. Using eigh() is faster than svd().
    # Note that svd() returns eigenvalues in decreasing order, while eigh()
    # returns eigenvalues in increasing order.
    # driver="evr" => fastest option
    s, U = sp.linalg.eigh(C_tilde, driver="evr", overwrite_a=True)
    # Truncate the SVD ( U_r @ np.diag(s_r) @ U_r.T == C_tilde )
    # N_n is the number of observations
    # N_e is the number of members in the ensemble
    N_n, N_e = Y.shape

    idx = singular_values_to_keep(s[::-1], truncation=truncation)
    N_r = min(N_n, N_e - 1, idx)  # Number of values in SVD to keep
    U_r, s_r = U[:, -N_r:], s[-N_r:]
    # U_r @ np.diag(s_r) @ U_r.T == C_tilde

    # Eqn (61). Compute symmetric term once first, then multiply together and
    # finally multiply with (D - Y)
    term = C_D_L_inv.T @ U_r if C_D.ndim == 2 else (C_D_L_inv * U_r.T).T

    # Compute the first factors, which make up C_MD
    Y_shift = (Y - np.mean(Y, axis=1, keepdims=True)) / (N_e - 1)

    return np.linalg.multi_dot(  # type: ignore
        [X, Y_shift.T, term / s_r, term.T, (D - Y)]
    )


def inversion_exact_subspace_woodbury(
    *,
    alpha: float,
    C_D: npt.NDArray[np.double],
    D: npt.NDArray[np.double],
    Y: npt.NDArray[np.double],
    X: npt.NDArray[np.double],
    truncation: float = 1.0,
) -> npt.NDArray[np.double]:
    """Use the Woodbury lemma to compute the inversion.

    This approach uses the Woodbury lemma to compute:
        C_MD @ inv(C_DD + alpha * C_D) @ (D - Y)

    Since C_DD = U @ U.T, where U := center(Y) / sqrt(N_e - 1), we can use:

    (A + U @ U.T)^-1 = A^-1 - A^-1 @ U @ (1 + U.T @ A^-1 @ U )^-1 @ U.T @ A^-1

    to compute inv(C_DD + alpha * C_D).
    """

    # Woodbury:
    # (A + U @ U.T)^-1 = A^-1 - A^-1 @ U @ (1 + U.T @ A^-1 @ U )^-1 @ U.T @ A^-1

    # Compute D_delta. N_n = number of outputs, N_e = number of ensemble members
    N_n, N_e = Y.shape
    D_delta = Y - np.mean(Y, axis=1, keepdims=True)  # Subtract average
    D_delta /= np.sqrt(N_e - 1)

    # Compute the first factors, which make up C_MD
    # X_shift = (X - np.mean(X, axis=1, keepdims=True)) / np.sqrt(N_e - 1)

    # A full covariance matrix was given
    if C_D.ndim == 2:
        # Invert C_D
        # TODO: This inverse could be cached
        C_D_inv = np.linalg.inv(C_D) / alpha

        # Compute the center part of the rhs in woodburry
        center = np.linalg.multi_dot([D_delta.T, C_D_inv, D_delta])
        center.flat[:: center.shape[0] + 1] += 1.0  # Add to diagonal

        # Compute the symmetric term of the rhs in woodbury
        term = C_D_inv @ D_delta

        # Compute the woodbury inversion, then return
        inverted = C_D_inv - np.linalg.multi_dot([term, sp.linalg.inv(center), term.T])
        return np.linalg.multi_dot(  # type: ignore
            [X, D_delta.T / np.sqrt(N_e - 1), inverted, (D - Y)]
        )

    # A diagonal covariance matrix was given as a 1D array.
    # Same computation as above, but exploit the diagonal structure
    else:
        C_D_inv = 1 / (C_D * alpha)  # Invert diagonal
        center = np.linalg.multi_dot([D_delta.T * C_D_inv, D_delta])
        center.flat[:: center.shape[0] + 1] += 1.0
        UT_D = D_delta.T * C_D_inv
        inverted = np.diag(C_D_inv) - np.linalg.multi_dot(
            [UT_D.T, sp.linalg.inv(center), UT_D]
        )
        return np.linalg.multi_dot(  # type: ignore
            [X, D_delta.T / np.sqrt(N_e - 1), inverted, (D - Y)]
        )


def inversion_subspace(
    *,
    alpha: float,
    C_D: npt.NDArray[np.double],
    D: npt.NDArray[np.double],
    Y: npt.NDArray[np.double],
    X: Optional[npt.NDArray[np.double]],
    truncation: float = 1.0,
    return_T: bool = False,
) -> npt.NDArray[np.double]:
    """See Appendix A.2 in :cite:t:`emerickHistoryMatchingTimelapse2012`.

    This is an approximate solution. The approximation is that when
    U, w, V.T = svd(D_delta)
    then we assume that U @ U.T = I.
    This is not true in general, for instance:

    >>> Y = np.array([[2, 0],
    ...               [0, 0],
    ...               [0, 0]])
    >>> D_delta = Y - np.mean(Y, axis=1, keepdims=True)  # Subtract average
    >>> D_delta
    array([[ 1., -1.],
           [ 0.,  0.],
           [ 0.,  0.]])
    >>> U, w, VT = sp.linalg.svd(D_delta)
    >>> U, w
    (array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]]), array([1.41421356, 0.        ]))
    >>> U[:, :1] @ np.diag(w[:1]) @ VT[:1, :] # Reconstruct D_Delta
    array([[ 1., -1.],
           [ 0.,  0.],
           [ 0.,  0.]])
    >>> U[:, :1] @ U[:, :1].T # But U_r @ U_r.T != I
    array([[1., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])

    """

    # N_n is the number of observations
    # N_e is the number of members in the ensemble
    N_n, N_e = Y.shape

    # Subtract the mean of every observation, see Eqn (67)
    D_delta = Y - np.mean(Y, axis=1, keepdims=True)  # Subtract average

    # Eqn (68)
    # TODO: Approximately 50% of the time in the function is spent here
    # consider using randomized svd for further speed gains
    U, w, _ = sp.linalg.svd(D_delta, overwrite_a=True, full_matrices=False)

    # Clip the singular value decomposition
    idx = singular_values_to_keep(w, truncation=truncation)
    N_r = min(N_n, N_e - 1, idx)  # Number of values in SVD to keep
    U_r, w_r = U[:, :N_r], w[:N_r]

    # Eqn (70). First compute the symmetric term, then form X
    U_r_w_inv = U_r / w_r
    if C_D.ndim == 1:
        X1 = (N_e - 1) * np.linalg.multi_dot([U_r_w_inv.T * C_D * alpha, U_r_w_inv])
    else:
        X1 = (N_e - 1) * np.linalg.multi_dot([U_r_w_inv.T, alpha * C_D, U_r_w_inv])

    # Eqn (72)
    # Z, T, _ = sp.linalg.svd(X, overwrite_a=True, full_matrices=False)
    # Compute SVD, which is equivalent to taking eigendecomposition
    # since X is PSD. Using eigh() is faster than svd().
    # Note that svd() returns eigenvalues in decreasing order, while eigh()
    # returns eigenvalues in increasing order.
    # driver="evr" => fastest option
    T, Z = sp.linalg.eigh(X1, driver="evr", overwrite_a=True)

    # Eqn (74).
    # C^+ = (N_e - 1) hat{C}^+
    #     = (N_e - 1) (U / w @ Z) * (1 / (1 + T)) (U / w @ Z)^T
    #     = (N_e - 1) (term) * (1 / (1 + T)) (term)^T
    # and finally we multiiply by (D - Y)
    term = U_r_w_inv @ Z

    if return_T:
        return np.linalg.multi_dot(  # type: ignore
            [D_delta.T, (term / (1 + T)), term.T, (D - Y)]
        )

    # Compute C_MD = center(X) @ center(Y).T / (num_ensemble - 1)
    return np.linalg.multi_dot(  # type: ignore
        [X, D_delta.T, (term / (1 + T)), term.T, (D - Y)]
    )


def inversion_rescaled_subspace(
    *,
    alpha: float,
    C_D: npt.NDArray[np.double],
    D: npt.NDArray[np.double],
    Y: npt.NDArray[np.double],
    X: npt.NDArray[np.double],
    truncation: float = 1.0,
) -> npt.NDArray[np.double]:
    """
    See Appendix A.2 in :cite:t:`emerickHistoryMatchingTimelapse2012`.

    Subspace inversion with rescaling.
    """
    # TODO: I don't understand why this approach is not approximate, when
    # `inversion_subspace` is approximate

    N_n, N_e = Y.shape
    D_delta = Y - np.mean(Y, axis=1, keepdims=True)  # Subtract average

    if C_D.ndim == 2:
        # Eqn (76). Cholesky factorize the covariance matrix C_D
        # TODO: here we compute the cholesky factor in every call, but C_D
        # never changes. it would be better to compute it once
        C_D_L = sp.linalg.cholesky(C_D * alpha, lower=True)  # Lower triangular cholesky
        # Here C_D_L is C^{1/2} in equation (57)
        # assert np.allclose(C_D_L @ C_D_L.T, C_D * alpha)
        C_D_L_inv, _ = sp.linalg.lapack.dtrtri(
            C_D_L, lower=1
        )  # Invert lower triangular

        # Use BLAS to compute product of lower triangular matrix C_D_L_inv and D_Delta
        # This line is equal to C_D_L_inv @ D_delta
        C_D_L_times_D_delta = sp.linalg.blas.dtrmm(
            alpha=1.0, a=C_D_L_inv, b=D_delta, lower=1
        )

    else:
        # Same as above, but C_D is a vector
        C_D_L_inv = 1 / np.sqrt(alpha * C_D)  # Invert the Cholesky factor a diagonal
        C_D_L_times_D_delta = (D_delta.T * C_D_L_inv).T

    U, w, _ = sp.linalg.svd(C_D_L_times_D_delta, overwrite_a=True, full_matrices=False)
    idx = singular_values_to_keep(w, truncation=truncation)

    # assert np.allclose(VT @ VT.T, np.eye(VT.shape[0]))
    N_r = min(N_n, N_e - 1, idx)  # Number of values in SVD to keep
    U_r, w_r = U[:, :N_r], w[:N_r]

    # Eqn (78) - taking into account that C_D_L_inv could be an array
    term = C_D_L_inv.T @ (U_r / w_r) if C_D.ndim == 2 else ((U_r / w_r).T * C_D_L_inv).T
    T_r = (N_e - 1) / w_r**2  # Equation (79)
    diag = 1 / (1 + T_r)

    # Compute C_MD
    return np.linalg.multi_dot(  # type: ignore
        [X, D_delta.T, (term * diag), term.T, (D - Y)]
    )


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
            "-v",
            # "-k simple",
        ]
    )
