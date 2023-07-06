#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp


def empirical_covariance_upper(X):
    """Compute the upper part of the empirical covariance.

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
    num_variables, num_observationservations = X.shape
    X = X - np.mean(X, axis=1, keepdims=True)
    # https://www.math.utah.edu/software/lapack/lapack-blas/dsyrk.html
    XXT = sp.linalg.blas.dsyrk(alpha=1, a=X)
    XXT /= num_observationservations - 1
    return XXT


def empirical_cross_covariance(X, Y):
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
    """

    assert X.shape[1] == Y.shape[1], "Ensemble size must be equal"
    X = X - np.mean(X, axis=1, keepdims=True)
    Y = Y - np.mean(Y, axis=1, keepdims=True)
    cov = X @ Y.T / (X.shape[1] - 1)
    assert cov.shape == (X.shape[0], Y.shape[0])
    return cov


def normalize_alpha(alpha):
    """Assure that sum_i (1/alpha_i) = 1

    Examples
    --------
    >>> alpha = np.arange(10) + 1
    >>> np.sum(1/normalize_alpha(alpha))
    1.0
    """
    factor = np.sum(1 / alpha)
    return alpha * factor


# =============================================================================
# INVERSION FUNCTIONS
# =============================================================================
def inversion_naive(*, alpha, C_D, D, Y):
    """Naive inversion, used for testing only.

    Computes inv(C_DD + alpha * C_D) @ (D - Y) naively.
    """
    C_DD = empirical_cross_covariance(Y, Y)

    # Naive implementation of Equation (3) in Ensemble smoother with multiple...
    C_D = np.diag(C_D) if C_D.ndim == 1 else C_D

    return sp.linalg.inv(C_DD + alpha * C_D) @ (D - Y)


def inversion_exact(*, alpha, C_D, D, Y):
    """Computes an exact inversion using `sp.linalg.solve`, which uses a
    Cholesky factorization.
    """
    C_DD = empirical_covariance_upper(Y)  # Only compute upper part
    solver_kwargs = {
        "overwrite_a": True,
        "overwrite_b": True,
        "assume_a": "pos",  # Assume positive definite matrix
        "lower": False,  # Only use the upper part while solving
    }

    # Compute K := sp.linalg.inv(C_DD + alpha * C_D) @ (D - Y)
    if C_D.ndim == 2:
        # C_D is a covariance matrix
        K = sp.linalg.solve(C_DD + alpha * C_D, D - Y, **solver_kwargs)
    elif C_D.ndim == 1:
        # C_D is an array, so add it to the diagonal without forming diag(C_D)
        C_DD.flat[:: C_DD.shape[1] + 1] += alpha * C_D
        K = sp.linalg.solve(C_DD, D - Y, **solver_kwargs)
    return K


def inversion_lstsq(*, alpha, C_D, D, Y):
    """Computes inversion uses least squares."""
    C_DD = empirical_cross_covariance(Y, Y)

    # A covariance matrix was given
    if C_D.ndim == 2:
        lhs = C_DD + alpha * C_D
    # A diagonal covariance matrix was given as a vector
    else:
        lhs = C_DD
        lhs.flat[:: lhs.shape[0] + 1] += alpha * C_D

    # K = lhs^-1 @ (D - Y)
    # lhs @ K = (D - Y)
    ans, *_ = sp.linalg.lstsq(lhs, D - Y, overwrite_a=True, overwrite_b=True)
    return ans


def inversion_rescaled(*, alpha, C_D, D, Y):
    """See Appendix A.1 in Emerick et al (2012)"""
    C_DD = empirical_cross_covariance(Y, Y)

    if C_D.ndim == 2:
        # Eqn (57). Cholesky factorize the covariance matrix C_D
        # TODO: here we compute the cholesky factor in every call, but C_D
        # never changes. it would be better to compute it once
        C_D_L = sp.linalg.cholesky(C_D * alpha, lower=True)  # Lower triangular cholesky
        C_D_L_inv, _ = sp.linalg.lapack.dtrtri(
            C_D_L, lower=1
        )  # Invert lower triangular

        # Eqn (59). Form C_tilde
        # sp.linalg.blas.strmm(alpha=1, a=C_D_L_inv, b=C_DD, lower=1)
        C_tilde = C_D_L_inv @ C_DD @ C_D_L_inv.T
        C_tilde.flat[:: C_tilde.shape[0] + 1] += 1  # Add to diagonal

    # When C_D is a diagonal covariance matrix, there is no need to perform
    # the cholesky factorization
    elif C_D.ndim == 1:
        C_D_L_inv = 1 / np.sqrt(C_D * alpha)
        C_tilde = (C_D_L_inv * (C_DD * C_D_L_inv).T).T
        C_tilde.flat[:: C_tilde.shape[0] + 1] += 1  # Add to diagonal

    # Eqn (60). Compute SVD (we could do eigenvalues, since C_tilde is
    # symmetric, but it turns out that computing the SVD is faster)
    # On a 1000 x 1000 example, 80% of the time is spent here
    U, s, _ = sp.linalg.svd(C_tilde, overwrite_a=True, full_matrices=False)
    # TODO: truncate SVD

    # Eqn (61). Compute symmetric term once first, then multiply together and
    # finally multiply with (D - Y)
    term = C_D_L_inv.T @ U if C_D.ndim == 2 else (C_D_L_inv * U.T).T
    return np.linalg.multi_dot([term / s, term.T, (D - Y)])


def inversion_subspace_woodbury(*, alpha, C_D, D, Y):
    """See Appendix A.2 in Emerick et al (2012)"""

    # C_D = np.diag(C_D) if C_D.ndim == 1 else C_D

    # Woodbury: (A + U @ U.T)^-1 = A^-1 - A^-1 @ U @ (1 + U.T @ A^-1 @ U )^-1 @ U.T @ A^-1
    N_n, N_e = Y.shape
    D_delta = Y - np.mean(Y, axis=1, keepdims=True)  # Subtract average
    D_delta /= np.sqrt(N_e - 1)

    if C_D.ndim == 2:
        C_D_inv = np.linalg.inv(C_D) / alpha

        center = np.linalg.multi_dot([D_delta.T, C_D_inv, D_delta])
        center.flat[:: center.shape[0] + 1] += 1.0

        term = C_D_inv @ D_delta
        inverted = C_D_inv - np.linalg.multi_dot([term, sp.linalg.inv(center), term.T])

        return inverted @ (D - Y)

    else:
        C_D_inv = 1 / (C_D * alpha)  # Diagonal matrix
        center = np.linalg.multi_dot([D_delta.T * C_D_inv, D_delta])
        center.flat[:: center.shape[0] + 1] += 1.0
        UT_D = D_delta.T * C_D_inv
        inverted = np.diag(C_D_inv) - np.linalg.multi_dot(
            [UT_D.T, sp.linalg.inv(center), UT_D]
        )

        return inverted @ (D - Y)


def inversion_subspace(*, alpha, C_D, D, Y):
    """See Appendix A.2 in Emerick et al (2012)

    This is an approximate solution. The approximation is that when
    U, w, V.T = svd(D_delta)
    then we assume that U_r @ U_r.T = I.
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

    # TODO: Incorporate this
    C_D = np.diag(C_D) if C_D.ndim == 1 else C_D

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
    # w_cumsum = np.cumsum(w)
    # w_cumsum /= w_cumsum[-1]
    # idx_to_keep = np.searchsorted(w_cumsum, 0.999, side="left") + 1

    N_r = min(N_n, N_e - 1)  # Number of values in SVD to keep
    U_r, w_r = U[:, :N_r], w[:N_r]

    # Eqn (70). First compute the symmetric term, then form X
    U_r_w_inv = U_r / w_r
    X = (N_e - 1) * np.linalg.multi_dot([U_r_w_inv.T, alpha * C_D, U_r_w_inv])

    # Eqn (72)
    Z, T, _ = sp.linalg.svd(X, overwrite_a=True, full_matrices=False)

    # Eqn (74).
    # C^+ = (N_e - 1) hat{C}^+
    #     = (N_e - 1) (U / w @ Z) * (1 / (1 + T)) (U / w @ Z)^T
    #     = (N_e - 1) (term) * (1 / (1 + T)) (term)^T
    # and finally we multiiply by (D - Y)
    term = U_r_w_inv @ Z
    return (N_e - 1) * np.linalg.multi_dot([(term / (1 + T)), term.T, (D - Y)])


def inversion_rescaled_subspace(*, alpha, C_D, D, Y):
    """See Appendix A.2 in Emerick et al (2012)"""
    pass


# =============================================================================
# TESTS
# =============================================================================

import pytest


class TestEsmdaInversion:
    @pytest.mark.parametrize(
        "function",
        [
            inversion_naive,
            inversion_exact,
            inversion_rescaled,
            inversion_lstsq,
            inversion_subspace_woodbury,
        ],
    )
    def test_inversion_on_a_simple_example(self, function):
        """Test on one of the simplest cases imaginable.

        If C_D = diag([1, 1, 1]) and Y = array([[2, 0],
                                                [0, 0],
                                                [0, 0]])

        then C_DD = diag([2, 0, 0])

        and inv(C_DD + C_D) = diag([1/3, 1, 1])
        """

        # Create positive symmetric definite covariance C_D
        C_D = np.identity(3)

        # Create observations
        D = np.ones((3, 2))
        Y = np.array([[2, 0], [0, 0], [0, 0]])

        # inv(diag([3, 1, 1])) @ (D - Y)
        K0 = function(alpha=1, C_D=C_D, D=D, Y=Y)
        assert np.allclose(K0, np.array([[-1 / 3, 1 / 3], [1.0, 1.0], [1.0, 1.0]]))

        # Same thing, but with a diagonal covariance represented as an array
        K0 = function(alpha=1, C_D=np.diag(C_D), D=D, Y=Y)
        assert np.allclose(K0, np.array([[-1 / 3, 1 / 3], [1.0, 1.0], [1.0, 1.0]]))

    def test_that_inverions_are_equal_with_few_ensemble_members(self, k=10):
        emsemble_members = k - 1

        # Create positive symmetric definite covariance C_D
        E = np.random.randn(k, k)
        C_D = E.T @ E

        # Set alpha to something other than 1 to test that it works
        alpha = 3

        # Create observations
        D = np.random.randn(k, emsemble_members)
        Y = np.random.randn(k, emsemble_members)

        # All non-subspace methods
        K1 = inversion_naive(alpha=alpha, C_D=C_D, D=D, Y=Y)
        K2 = inversion_exact(alpha=alpha, C_D=C_D, D=D, Y=Y)
        K3 = inversion_rescaled(alpha=alpha, C_D=C_D, D=D, Y=Y)
        K4 = inversion_lstsq(alpha=alpha, C_D=C_D, D=D, Y=Y)

        assert np.allclose(K1, K2)
        assert np.allclose(K1, K3)
        assert np.allclose(K1, K4)

    @pytest.mark.parametrize(
        "function",
        [
            # Exact inversions
            inversion_naive,
            inversion_exact,
            inversion_rescaled,
            inversion_lstsq,
            inversion_subspace_woodbury,
            # Approximate inversions (same result as long as ensemble_members > num_outputs)
            inversion_subspace,
        ],
    )
    @pytest.mark.parametrize(
        "num_outputs,num_emsemble",
        [(10, 11), (10, 20), (10, 100), (100, 101), (100, 200), (100, 500)],
    )
    def test_that_all_inversions_all_equal_with_many_ensemble_members(
        self, function, num_outputs, num_emsemble
    ):

        assert num_emsemble > num_outputs
        np.random.seed(num_outputs + num_emsemble)

        # Create positive symmetric definite covariance C_D
        E = np.random.randn(num_outputs, num_outputs)
        C_D = E.T @ E

        # Set alpha to something other than 1 to test that it works
        alpha = np.exp(np.random.randn())

        # Create observations
        D = np.random.randn(num_outputs, num_emsemble)
        Y = np.random.randn(num_outputs, num_emsemble)

        K1 = inversion_naive(alpha=alpha, C_D=C_D, D=D, Y=Y)
        K2 = function(alpha=alpha, C_D=C_D, D=D, Y=Y)

        assert np.allclose(K1, K2)

    @pytest.mark.parametrize("ratio_ensemble_members_over_outputs", [0.5, 1, 2])
    @pytest.mark.parametrize("num_outputs", [10, 50, 100])
    @pytest.mark.parametrize(
        "function",
        [
            inversion_naive,
            inversion_exact,
            inversion_rescaled,
            inversion_lstsq,
            inversion_subspace_woodbury,
            inversion_subspace,
        ],
    )
    def test_that_inversion_methods_work_with_covariance_matrix_and_variance_vector(
        self, ratio_ensemble_members_over_outputs, num_outputs, function
    ):

        emsemble_members = int(num_outputs / ratio_ensemble_members_over_outputs)
        np.random.seed(num_outputs + emsemble_members)

        # Diagonal covariance matrix
        C_D = np.diag(np.exp(np.random.randn(num_outputs)))
        C_D_full = np.diag(C_D)

        # Set alpha to something other than 1 to test that it works
        alpha = np.exp(np.random.randn())

        # Create observations
        D = np.random.randn(num_outputs, emsemble_members)
        Y = np.random.randn(num_outputs, emsemble_members)

        result_diagonal = function(alpha=alpha, C_D=C_D, D=D, Y=Y)
        result_dense = function(alpha=alpha, C_D=C_D_full, D=D, Y=Y)

        assert np.allclose(result_diagonal, result_dense)


def test_timing(num_outputs=100, num_ensemble=25):

    k = num_outputs
    emsemble_members = num_ensemble

    E = np.random.randn(k, k)
    C_D = E.T @ E
    C_D = np.diag(np.exp(np.random.randn(k)))  # Diagonal covariance matrix
    C_D_full = np.diag(C_D)

    # Set alpha to something other than 1 to test that it works
    alpha = 3

    # Create observations
    D = np.random.randn(k, emsemble_members)
    Y = np.random.randn(k, emsemble_members)

    exact_inversion_funcs = [
        inversion_naive,
        inversion_exact,
        inversion_rescaled,
        inversion_lstsq,
    ]

    from time import perf_counter

    print("-" * 32)

    for func in exact_inversion_funcs:
        start_time = perf_counter()
        result_matrix = func(alpha=alpha, C_D=C_D, D=D, Y=Y)
        elapsed_time = round(perf_counter() - start_time, 4)
        print(f"Function: {func.__name__} on dense covariance: {elapsed_time} s")

        start_time = perf_counter()
        result_vector = func(alpha=alpha, C_D=C_D_full, D=D, Y=Y)
        elapsed_time = round(perf_counter() - start_time, 4)
        print(f"Function: {func.__name__} on diagonal covariance: {elapsed_time} s")
        assert np.allclose(result_matrix, result_vector)

        print("-" * 32)

    subspace_inversion_funcs = [inversion_subspace, inversion_subspace_woodbury]

    from time import perf_counter

    for func in subspace_inversion_funcs:
        start_time = perf_counter()
        result_matrix = func(alpha=alpha, C_D=C_D, D=D, Y=Y)
        elapsed_time = round(perf_counter() - start_time, 4)
        print(f"Function: {func.__name__} on dense covariance: {elapsed_time} s")

        start_time = perf_counter()
        result_vector = func(alpha=alpha, C_D=C_D_full, D=D, Y=Y)
        elapsed_time = round(perf_counter() - start_time, 4)
        print(f"Function: {func.__name__} on diagonal covariance: {elapsed_time} s")
        assert np.allclose(result_matrix, result_vector)

        print("-" * 32)


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
            "-v",
            # "-k test_inversions_when_there_are_few_ensemble_members",
        ]
    )
