"""
Contains (publicly available, but not officially supported) experimental
features of iterative_ensemble_smoother
"""
import numpy as np

rng = np.random.default_rng()

# mypy: ignore-errors

import numpy as np
import scipy as sp


from typing import Optional, Tuple

import numpy as np


import numpy.typing as npt


def covariance_to_correlation(
    C: npt.NDArray[np.double],
) -> Tuple[Optional[npt.NDArray[np.double]], npt.NDArray[np.double]]:
    """The input C is either (1) a 2D covariance matrix or (2) a 1D array of
    standard deviations. This function normalizes the covariance matrix to a
    correlation matrix (unit diagonal).

    Examples
    --------
    >>> C = np.array([[4., 1., 0.],
    ...               [1., 4., 1.],
    ...               [0., 1., 4.]])
    >>> corr_mat, stds = covariance_to_correlation(C)
    >>> corr_mat
    array([[1.  , 0.25, 0.  ],
           [0.25, 1.  , 0.25],
           [0.  , 0.25, 1.  ]])
    >>> stds
    array([2., 2., 2.])
    """
    assert isinstance(C, np.ndarray) and C.ndim in (1, 2)

    # A covariance matrix was passed
    if C.ndim == 2:
        standard_deviations = np.sqrt(C.diagonal())

        # Create a correlation matrix from a covariance matrix
        # https://en.wikipedia.org/wiki/Covariance_matrix#Relation_to_the_correlation_matrix

        # Divide every column
        correlation_matrix = C / standard_deviations.reshape(1, -1)

        # Divide every row
        correlation_matrix /= standard_deviations.reshape(-1, 1)

        return correlation_matrix, standard_deviations

    # A vector of standard deviations was passed
    correlation_matrix = None
    standard_deviations = C
    return None, standard_deviations


def calc_num_significant(singular_values, truncation):
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
    total_sigma = sigma[-1]  # Sum of all squared singular values

    relative_energy = sigma / total_sigma
    return np.searchsorted(relative_energy, truncation, side="left") + 1


def truncated_svd_inv_sigma(S, truncation, full_matrices=False):
    """
    Compute truncated SVD of a matrix S, keeping a fraction `truncation` of the
    total energy (singular values squared).

    Examples
    --------
    >>> S = np.diag([100, 1, 0]) #  Create a singular matrix
    >>> truncated_svd_inv_sigma(S, 1.0)
    (array([[1., 0.],
           [0., 1.],
           [0., 0.]]), array([0.01, 1.  ]), array([[1., 0., 0.],
           [0., 1., 0.]]))

    Keep only 90 % of energy, removing one non-zero singular value:

    >>> S = np.diag([100, 1, 0])
    >>> truncated_svd_inv_sigma(S, 0.9)
    (array([[1.],
           [0.],
           [0.]]), array([0.01]), array([[1., 0., 0.]]))

    """

    U, singular_values, VT = sp.linalg.svd(
        S,
        full_matrices=full_matrices,
        compute_uv=True,
        overwrite_a=False,
        check_finite=True,
        lapack_driver="gesdd",
    )

    if isinstance(truncation, float):
        num_significant = calc_num_significant(singular_values, truncation)
    else:
        num_significant = truncation

    assert np.all(
        singular_values[:num_significant] > 0
    ), "Must have positive singular values"

    inverted_singular_values = np.zeros_like(singular_values)
    inverted_singular_values[:num_significant] = 1 / singular_values[:num_significant]

    return (
        U[:, :num_significant],
        inverted_singular_values[:num_significant],
        VT[:num_significant, :],
    )


def exact_inversion(W, S, H, steplength):
    """Compute exact inversion, assuming identity error covariance matrix.

    This is equation (51) in the paper.

    W has shape (num_ensemble, num_ensemble)
    S has shape (num_outputs, num_ensemble)
    H has shape (num_outputs, num_ensemble)

    Returns
    -------
    W : np.ndarray
        Updated matrix W_i+1.

    Examples
    --------
    >>> W = np.array([[1, 2], [3, 4]])
    >>> S = np.array([[1, 2], [3, 4], [5, 6]])
    >>> H = np.array([[0, 1], [1, 0], [1, 1]])

    Naive computation:

    >>> W - 1 * (W - np.linalg.inv(S.T @ S + np.eye(2)) @ S.T @ H)
    array([[ 0.13793103, -0.0862069 ],
           [ 0.06896552,  0.20689655]])

    This function:

    >>> exact_inversion(W, S, H, 1.0)
    array([[ 0.13793103, -0.0862069 ],
           [ 0.06896552,  0.20689655]])

    """
    # Section 3.2 - Exact inversion assuming diagonal error covariance matrix

    # Equation (51) states that
    # W_{i+1} = W_i - step * (W_i - (S.T @ S + I)^-1 @ S.T @ H)
    # Here we form
    # C = S.T @ S + I
    # This is a square symmetric matrix, instead of taking the SVD we
    # call sp.linalg.solve. This is equally fast.

    # Form the matrix C
    _, ensemble_size = S.shape
    C = S.T @ S  # TODO: Only form upper part of this matrix
    # Add the identity matrix in place
    # See: https://github.com/numpy/numpy/blob/db4f43983cb938f12c311e1f5b7165e270c393b4/numpy/lib/index_tricks.py#L786
    C.flat[:: ensemble_size + 1] += 1

    # Compute term = (S.T @ S + I)^-1 @ S.T @ H
    try:
        # Here we set the right hand side as the product, instead of
        # setting the right hand side as S.T, and then right-multiplying by H.
        # option 1: solve(C, S.T @ H)
        # option 2: solve(C, S.T) @ H
        # the reason is that with typical dimensions (num_ensemble << num_outputs)
        # the first option is faster.
        term = np.linalg.solve(C, S.T @ H)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Fit produces NaNs. Check your response matrix for outliers or use an inversion type with truncation."
        )

    return W - steplength * (W - term)


def create_coefficient_matrix(Y, R, E, D, inversion, truncation, W, steplength):
    """Creates the coefficient matrix W_i defined in line 8 in Algorithm 1.

    Examples
    --------
    >>> Y = np.array([[1, 2], [3, 4]])
    >>> E = np.array([[2, 0], [1, 3]])
    >>> R = E @ E.T
    >>> W0 = np.array([[2, 1], [2, 2]])
    >>> D = np.array([[0, 1], [1, 1]])
    >>> create_coefficient_matrix(Y, R, E, D, "naive", 1.0, W0, 1.0)
    array([[0.76348548, 0.63347165],
           [1.94605809, 1.79944675]])
    >>> create_coefficient_matrix(Y, R, E, D, "exact_r", 1.0, W0, 1.0)
    array([[0.76348548, 0.63347165],
           [1.94605809, 1.79944675]])
    >>> create_coefficient_matrix(Y, R, E, D, "subspace_re", 1.0, W0, 1.0)
    array([[0.76348548, 0.63347165],
           [1.94605809, 1.79944675]])

    Diagonal covariance matrix:

    >>> R = np.eye(2)
    >>> E = np.eye(2)
    >>> create_coefficient_matrix(Y, R, E, D, "naive", 1.0, W0, 1.0)
    array([[1.07964602, 0.75516224],
           [2.43362832, 2.25958702]])
    >>> create_coefficient_matrix(Y, R, E, D, "exact", 1.0, W0, 1.0)
    array([[1.07964602, 0.75516224],
           [2.43362832, 2.25958702]])
    """

    assert inversion in ("naive", "exact", "exact_r", "subspace_re")

    _, ensemble_size = W.shape

    # Line 5 in Algorithm 1
    Omega = W - W.mean(axis=1, keepdims=True)
    Omega /= np.sqrt(ensemble_size - 1)

    # Add the identity matrix in place
    # See: https://github.com/numpy/numpy/blob/db4f43983cb938f12c311e1f5b7165e270c393b4/numpy/lib/index_tricks.py#L786
    Omega.flat[: ensemble_size * ensemble_size : ensemble_size + 1] += 1.0

    # Line 6 of Algorithm 1, also Section 5
    # Solving for the average sensitivity matrix.
    S = sp.linalg.solve(Omega.T, Y.T, lower=False, assume_a="gen").T

    # Line 7 of Algorithm 1, also Section 2.6
    # Similar to the innovation term.
    # Differs in that `D` here is defined as dobs + E - Y instead of just dobs +
    # E as in the paper
    H = D + S @ W

    # With R=I the subspace inversion (ies_inversion=1) with
    # singular value trucation=1.000 gives exactly the same solution as the exact
    # inversion (`ies_inversion`=Inversion::exact).
    #
    # With very large data sets it is likely that the inversion becomes poorly
    # conditioned and a trucation=1.0 is not a good choice. In this case
    # `ies_inversion` other than Inversion::exact and truncation set to less
    # than 1.0 could stabilize the algorithm.

    # Naive computation, used for testing purposes only
    if inversion == "naive":
        # Compute K = S.T @ sp.linalg.inv(S @ S.T + E @ E.T, overwrite_a=True) @ H
        to_invert = S @ S.T + E @ E.T
        # Add a small number to the diagonal to improve the condition number
        # when the matrix is close to singular (positive semidefinite)
        to_invert.flat[:: to_invert.shape[0] + 1] += 1e-12
        K = S.T @ sp.linalg.solve(to_invert, H, assume_a="sym", overwrite_a=True)
        return W - steplength * (W - K)
    elif inversion == "exact":  # Section 3.2 in paper
        return exact_inversion(W, S, H, steplength)

    # Implements parts of Eq. 14.31 in the book Data Assimilation,
    # The Ensemble Kalman Filter, 2nd Edition by Geir Evensen.

    elif inversion == "exact_r":  # Section 3.3 in paper
        eig, X1 = lowrankCinv(S, R, truncation)
        # Compute W2 @ diag(lambda_inv) @ W2.T @ H
        X3 = np.linalg.multi_dot([S.T, X1 * eig, X1.T, H])
        # Line 9 in algorithm 1
        W = W - steplength * (W - X3)
        return W
    elif inversion == "subspace_re":  # Section 3.4 in paper
        _, ensemble_size = S.shape
        nsc = 1.0 / np.sqrt(ensemble_size - 1.0)
        eig, X1 = lowrankE(S, E * nsc, truncation)
        # Compute W2 @ diag(lambda_inv) @ W2.T @ H
        X3 = np.linalg.multi_dot([S.T, X1 * eig, X1.T, H])
        # Line 9 in algorithm 1
        W = W - steplength * (W - X3)
        return W
    else:
        raise Exception(f"Invalid inversion {inversion}")


def lowrankE(S, E, truncation):
    """Compute inverse of S @ S.T + E @ E.T by projecting E @ E.T onto S.

    This is not an exact inversion, but an approximation.

    See the following section in the 2009 book by Evensen:
        14.3.1 Derivation of the pseudo inverse

    Also see section 3.4 in the paper

    Returns
    -------
    eig, X1 : tuple
        A vector and a matrix so that X1 @ np.diag(eig) @ X1.T = inv(S @ S.T + E @ E.T).

    Examples
    --------
    >>> S = np.array([[1, 2, 3], [3, 4, 5], [4, 3, 1]])
    >>> E = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]])

    Naive computation:

    >>> np.linalg.inv(S @ S.T + E @ E.T)
    array([[ 0.38012959, -0.22030238,  0.03239741],
           [-0.22030238,  0.18070554, -0.07559395],
           [ 0.03239741, -0.07559395,  0.09935205]])

    Using this function:

    >>> eig, X1 = lowrankE(S, E, truncation=1.0)
    >>> X1 @ np.diag(eig) @ X1.T
    array([[ 0.38012959, -0.22030238,  0.03239741],
           [-0.22030238,  0.18070554, -0.07559395],
           [ 0.03239741, -0.07559395,  0.09935205]])
    """

    U0, inv_sigma0, _ = truncated_svd_inv_sigma(S, truncation)

    # Equation 14.51
    X0 = (U0 * inv_sigma0).T @ E  # Same as diag(inv_sigma) @ U0.T @ E

    # Equation 14.52
    U1, sigma1, _ = sp.linalg.svd(X0, full_matrices=False)

    # Equation 14.55
    X1 = (U0 * inv_sigma0) @ U1

    eig = 1 / (1 + sigma1**2)

    # Equation 14.54 is X1 @ diag(eig) @ X1.T
    return eig, X1


def lowrankCinv(S, R, truncation):
    """Invert S @ S.T + R by projecting R onto S.

    This is not an exact inversion, but an approximation.

    See the following section in the 2009 book by Evensen:
        14.2.1 Derivation of the subspace pseudo inverse

    Also see section 3.3 in the paper

    Returns
    -------
    eig, X1 : tuple
        A vector and a matrix so that X1 @ np.diag(eig) @ X1.T = inv(S @ S.T + E @ E.T).

    Examples
    --------
    >>> S = np.array([[1, 2, 9], [3, 4, 1], [3, 0, 1]])
    >>> R = np.array([[10, 2, 0], [2, 10, 2], [0, 2, 10]])

    Naive computation:

    >>> np.linalg.inv(S @ S.T + R)
    array([[ 0.01231611, -0.00632911, -0.0035922 ],
           [-0.00632911,  0.03797468, -0.01898734],
           [-0.0035922 , -0.01898734,  0.06354772]])

    Using this function:

    >>> eig, X1 = lowrankCinv(S, R, truncation=1.0)
    >>> X1 @ np.diag(eig) @ X1.T
    array([[ 0.01231611, -0.00632911, -0.0035922 ],
           [-0.00632911,  0.03797468, -0.01898734],
           [-0.0035922 , -0.01898734,  0.06354772]])
    """

    # Equations (14.19) / (14.20)
    U0, inv_sig0, _ = truncated_svd_inv_sigma(S, truncation)

    # Equation (14.26)
    U0_inv_sigma = U0 * inv_sig0

    # If a 1D array was passed for covariance, then R (correlation matrix) is None
    if R is None:
        X0 = U0_inv_sigma.T @ U0_inv_sigma
    else:
        X0 = np.linalg.multi_dot([U0_inv_sigma.T, R, U0_inv_sigma])

    U1, sig, _ = sp.linalg.svd(X0, full_matrices=False)

    # Equation (14.30)
    X1 = (U0 * inv_sig0) @ U1

    # Equation (14.29)
    eig = 1 / (1 + sig)
    return eig, X1


def ensemble_smoother_update_step_row_scaling(
    response_ensemble,
    A_with_row_scaling,
    observation_errors,
    observation_values,
    noise=None,
    truncation=0.98,
    inversion="exact",
):
    """This is an experimental feature."""
    ensemble_size = response_ensemble.shape[1]
    if noise is None:
        num_obs = len(observation_values)
        noise = rng.standard_normal(size=(num_obs, ensemble_size))

    # Columns of E should be sampled from N(0,Cdd) and centered, Evensen 2019
    if len(observation_errors.shape) == 2:
        E = np.linalg.cholesky(observation_errors) @ noise
    else:
        E = np.linalg.cholesky(np.diag(observation_errors**2)) @ noise
    E = E @ (
        np.identity(ensemble_size)
        - np.ones((ensemble_size, ensemble_size)) / ensemble_size
    )

    R, observation_errors = covariance_to_correlation(observation_errors)

    D = (E + observation_values.reshape(num_obs, 1)) - response_ensemble
    D = (D.T / observation_errors).T
    E = (E.T / observation_errors).T
    response_ensemble = (response_ensemble.T / observation_errors).T
    for A, row_scale in A_with_row_scaling:
        W = create_coefficient_matrix(
            (response_ensemble - response_ensemble.mean(axis=1, keepdims=True))
            / np.sqrt(ensemble_size - 1),
            R,
            E,
            D,
            inversion,
            truncation,
            np.zeros((ensemble_size, ensemble_size)),
            1.0,
        )
        I = np.identity(ensemble_size)
        transition_matrix = I + W / np.sqrt(ensemble_size - 1)
        row_scale.multiply(A, transition_matrix)
    return A_with_row_scaling
