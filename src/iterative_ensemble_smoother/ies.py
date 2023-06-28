#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp


# =============================================================================
# int calc_num_significant(const VectorXd &singular_values, double truncation) {
#   int num_significant = 0;
#   double total_sigma2 = singular_values.squaredNorm();
#
#   /*
#    * Determine the number of singular values by enforcing that
#    * less than a fraction @truncation of the total variance be
#    * accounted for.
#    */
#   double running_sigma2 = 0;
#   for (auto sig : singular_values) {
#     if (running_sigma2 / total_sigma2 <
#         truncation) { /* Include one more singular value ? */
#       num_significant++;
#       running_sigma2 += sig * sig;
#     } else
#       break;
#   }
#
#   return num_significant;
# }
# =============================================================================


def calc_num_significant(singular_values, truncation):
    """Determine the number of singular values by enforcing that less than a
    fraction truncation of the total variance be accounted for.

    Parameters
    ----------
    singular_values : np.ndarray
        Array with singular values, sorted in decreasing order.
    truncation : float
        Fraction of energy squared singular values to keep.

    Returns
    -------
    None.

    Examples
    --------
    >>> singular_values = np.array([2, 2, 1, 1])
    >>> calc_num_significant(singular_values, 1.0)
    4
    >>> calc_num_significant(singular_values, 0.8)
    2
    >>> calc_num_significant(singular_values, 0.01)
    0

    """
    assert np.all(np.diff(singular_values) <= 0), "Must be sorted decreasing"
    assert 0 < truncation <= 1

    sigma = np.cumsum(singular_values**2)
    total_sigma = sigma[-1]  # Sum of all squared singular values

    relative_energy = sigma / total_sigma
    return np.searchsorted(relative_energy, truncation, side="right")


# =============================================================================
# MatrixXd genX3(const MatrixXd &W, const MatrixXd &D, const VectorXd &eig) {
#   const int nrmin = std::min(D.rows(), D.cols());
#   // Corresponds to (I + \Lambda_1)^{-1} since `eig` has already been
#   // transformed.
#   MatrixXd Lambda_inv = eig(Eigen::seq(0, nrmin - 1)).asDiagonal();
#   MatrixXd X1 = Lambda_inv * W.transpose();
#
#   MatrixXd X2 = X1 * D;
#   MatrixXd X3 = W * X2;
#
#   return X3;
# }
# =============================================================================


def genX3(W, D, eig):
    r"""
    * Implements parts of Eq. 14.31 in the book Data Assimilation,
    * The Ensemble Kalman Filter, 2nd Edition by Geir Evensen.
    * Specifically, this implements
    * X_1 (I + \Lambda_1)^{-1} X_1^T (D - M[A^f])

    Examples
    --------
    >>> W = np.array([[1, 2], [3,4]])
    >>> D = np.array([[3, 1], [5, 9]])
    >>> eig = np.array([10, 1])
    >>> genX3(W, D, eig)
    array([[232, 356],
           [644, 992]])
    >>> W @ np.diag(eig) @ W.T @ D
    array([[232, 356],
           [644, 992]])

    """
    # X3 = X1 * diag(eig) * X1' * H (Similar to Eq. 14.31, Evensen (2007))

    nrmin = min(D.shape)

    lambda_inv = eig[:nrmin]  # Keep first `nrmin` values

    # Compute W @ diag(lambda_inv) @ W.T @ D
    return np.linalg.multi_dot([W * lambda_inv, W.T, D])


# =============================================================================
# int svdS(const MatrixXd &S, const std::variant<double, int> &truncation,
#          VectorXd &inv_sig0, MatrixXd &U0) {
#
#   int num_significant = 0;
#
#   auto svd = S.bdcSvd(ComputeThinU);
#   U0 = svd.matrixU();
#   VectorXd singular_values = svd.singularValues();
#
#   if (std::holds_alternative<int>(truncation)) {
#     num_significant = std::get<int>(truncation);
#   } else {
#     num_significant =
#         calc_num_significant(singular_values, std::get<double>(truncation));
#   }
#
#   inv_sig0 = singular_values.cwiseInverse();
#
#   inv_sig0(Eigen::seq(num_significant, Eigen::last)).setZero();
#
#   return num_significant;
# }
# =============================================================================


def svdS(S, truncation):
    """


    Parameters
    ----------
    S : TYPE
        DESCRIPTION.
    truncation : TYPE
        DESCRIPTION.

    Returns
    -------
    U0 : TYPE
        DESCRIPTION.
    singular_values : TYPE
        DESCRIPTION.
    num_significant : TYPE
        DESCRIPTION.

    Examples
    --------
    >>> S = np.array([[1, 2], [3, 4]])
    >>> svdS(S, 1)
    (array([[-0.40455358, -0.9145143 ],
           [-0.9145143 ,  0.40455358]]), array([0.1829831, 0.       ]), 1)

    On a singular matrix

    >>> S = np.array([[1, 2], [2, 4]])
    >>> svdS(S, 0.99)
    (array([[-0.4472136 , -0.89442719],
           [-0.89442719,  0.4472136 ]]), array([0., 0.]), 0)
    """

    U0, singular_values, Vh = sp.linalg.svd(
        S,
        full_matrices=False,
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

    inv_sig0 = np.zeros_like(singular_values)
    inv_sig0[:num_significant] = 1 / singular_values[:num_significant]

    return U0, inv_sig0, num_significant


# =============================================================================
# /**
#  * Section 3.2 - Exact inversion assuming diagonal error covariance matrix
#  */
# void exact_inversion(MatrixXd &W, const MatrixXd &S, const MatrixXd &H,
#                      double ies_steplength) {
#   int ens_size = S.cols();
#
#   MatrixXd C = S.transpose() * S;
#   C.diagonal().array() += 1;
#
#   auto svd = C.bdcSvd(Eigen::ComputeFullV);
#
#   W = W - ies_steplength *
#               (W - svd.matrixV() *
#                        svd.singularValues().cwiseInverse().asDiagonal() *
#                        svd.matrixV().transpose() * S.transpose() * H);
# }
# =============================================================================


def exact_inversion(W, S, H, steplength):
    """


    Parameters
    ----------
    W : TYPE
        DESCRIPTION.
    S : TYPE
        DESCRIPTION.
    H : TYPE
        DESCRIPTION.
    steplength : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

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
    # This is a square symmetric matrix, so taking the eigenvalue decomposition
    # is the same as the SVD
    # V @ np.diag(u) @ V.T = C
    # And the inverse of C becomes
    # C^-1 = (S.T @ S + I)^-1 = V.T @ diag(1/u) @ V

    # TODO: Why not follow equation (52) in the paper here?

    # Create the matrix C
    _, ensemble_size = S.shape
    C = S.T @ S
    # Add the identity matrix in place
    # See: https://github.com/numpy/numpy/blob/db4f43983cb938f12c311e1f5b7165e270c393b4/numpy/lib/index_tricks.py#L786
    C.flat[:: ensemble_size + 1] += 1

    # Compute the correction term that multiplies the step length
    u, V = np.linalg.eig(C)

    # assert np.allclose(np.linalg.inv(C), V @ np.diag(1/u) @ V.T)
    # assert np.allclose(np.linalg.inv(C), (V/u) @ V.T)

    correction = W - np.linalg.multi_dot([(V / u), V.T, S.T, H])
    return W - steplength * correction


def add_identity(a):
    """Mutates the argument in place, adding the identity matrix.


    Parameters
    ----------
    A : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    Examples
    --------
    >>> A = np.array([[1, 1], [1, 1]])
    >>> add_identity(A)
    >>> A
    array([[2, 1],
           [1, 2]])

    >>> A = np.array([[1, 1], [1, 1], [1, 1]])
    >>> add_identity(A)
    >>> A
    array([[2, 1],
           [1, 2],
           [1, 1]])

    >>> A = np.array([[1, 1, 1], [1, 1, 1]])
    >>> add_identity(A)
    >>> A
    array([[2, 1, 1],
           [1, 2, 1]])
    """
    if a.ndim != 2:
        raise ValueError("array must be 2-d")
    end = None

    # Explicit, fast formula for the common case.  For 2-d arrays, we
    # accept rectangular ones.
    step = a.shape[1] + 1
    # This is needed to don't have tall matrix have the diagonal wrap.
    end = a.shape[1] * a.shape[1]

    # Write the value out into the diagonal.
    a.flat[:end:step] += 1


# =============================================================================
# MatrixXd create_coefficient_matrix(py::EigenDRef<MatrixXd> Y,
#                                    std::optional<py::EigenDRef<MatrixXd>> R,
#                                    py::EigenDRef<MatrixXd> E,
#                                    py::EigenDRef<MatrixXd> D,
#                                    const Inversion ies_inversion,
#                                    const std::variant<double, int> &truncation,
#                                    MatrixXd &W, double ies_steplength)
#
# {
#   const int ens_size = Y.cols();
#
#   /* Line 5 of Algorithm 1 */
#   MatrixXd Omega =
#       (1.0 / sqrt(ens_size - 1.0)) * (W.colwise() - W.rowwise().mean());
#   Omega.diagonal().array() += 1.0;
#
#   /* Solving for the average sensitivity matrix.
#      Line 6 of Algorithm 1, also Section 5
#   */
#   Omega.transposeInPlace();
#   MatrixXd S = Omega.fullPivLu().solve(Y.transpose()).transpose();
#
#   /* Similar to the innovation term.
#      Differs in that `D` here is defined as dobs + E - Y instead of just dobs +
#      E as in the paper. Line 7 of Algorithm 1, also Section 2.6
#   */
#   MatrixXd H = D + S * W;
#
#   /*
#    * With R=I the subspace inversion (ies_inversion=1) with
#    * singular value trucation=1.000 gives exactly the same solution as the exact
#    * inversion (`ies_inversion`=Inversion::exact).
#    *
#    * With very large data sets it is likely that the inversion becomes poorly
#    * conditioned and a trucation=1.0 is not a good choice. In this case
#    * `ies_inversion` other than Inversion::exact and truncation set to less
#    * than 1.0 could stabilize the algorithm.
#    */
#
#   if (ies_inversion == Inversion::exact) {
#     exact_inversion(W, S, H, ies_steplength);
#   } else {
#     subspace_inversion(W, ies_inversion, E, R, S, H, truncation,
#                        ies_steplength);
#   }
#
#   return W;
# }
# =============================================================================


def create_coefficient_matrix(Y, R, E, D, inversion, truncation, W, steplength):

    assert inversion in ("exact", "subspace_exact_r", "subspace_re")

    _, ensemble_size = Y.shape

    # This corresponds to line 4 in Algorithm 1
    Omega = Y - Y.mean(axis=1, keepdims=True)
    Omega /= np.sqrt(ensemble_size - 1)

    # Add the identity matrix in place
    # See: https://github.com/numpy/numpy/blob/db4f43983cb938f12c311e1f5b7165e270c393b4/numpy/lib/index_tricks.py#L786
    Omega.flat[: ensemble_size * ensemble_size : ensemble_size + 1] += 1

    # Solving for the average sensitivity matrix.
    # Line 6 of Algorithm 1, also Section 5
    S = sp.linalg.solve(
        Omega.T,
        Y.T,
        lower=False,
        overwrite_a=False,
        overwrite_b=False,
        check_finite=True,
        assume_a="gen",
        transposed=False,
    )

    # Similar to the innovation term.
    # Differs in that `D` here is defined as dobs + E - Y instead of just dobs +
    # E as in the paper. Line 7 of Algorithm 1, also Section 2.6
    H = D + S @ W

    if inversion == "exact":
        return exact_inversion(W, S, H, steplength)
    else:
        pass

    return None


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-v"])
