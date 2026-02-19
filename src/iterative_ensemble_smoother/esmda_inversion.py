"""
ESMDA inversion
---------------

This module contains inversion methods, i.e. methods for computing the equation:

    C_MD @ inv(C_DD + alpha * C_D) @ (D - Y)

This might seem like a simple task, but several issues make the computation
of this matrix product interesting. It all comes down to exploiting structure
and dimensionality: the order of matrix multiplication affects the big-Oh,
we want to avoid explicitly computing the inverse, and we can use the Woodbury
identity along with a Singular Value Decomposition (SVD) to speed things up further.

There are essentially two main methods:

    (1) Exact inversion: use Cholesky to invert C_DD + alpha * C_D
    (2) Subspace inversion: use the Woodbury identity and an SVD

When reading about these methods in the papers, e.g.:

    - "History Matching Time-lapse Seismic Data Using the Ensemble Kalman
       Filter with Multiple Data Assimilations", Emerick and Reynolds
    - "Analysis of the Performance of Ensemble-based Assimilation of
       Production and Seismic Data", Emerick

The exposition is quite hard to understand. In the first paper above, the authors
cite an example where water-cut data could not be matched, and therefore claim
that one should rescale the equation in subspace inversion before taking SVD.
In the second paper, the authors suggest that scaling should be done based
only on diagonal elements of C_D in subspace inversion.

I found this hard to understand, because it was not clear what parts of the
linear algebra is exact, what are approximations, and what is due to numerics.
Therefore I have written about the scaling methods below.

Exact inversion and scaling
---------------------------

In exact inversion, all scaling is done to help with numerical issues.
Without scaling, a LinAlgError might raise due to a poor condition number.

The solver sp.linalg.solve uses the Cholesky factorization when the appropriate
argument is passed. We wish to compute inv(C_DD + alpha * C_D) @ (D - Y), but the
matrix C_DD + alpha * C_D may be ill-conditioned. There are three approaches to
deal with this: (1) do not scale, (2) scale with diag(C_D) and (3) scale with the
Cholesky factor of C_D.

See also:

    - https://en.wikipedia.org/wiki/Preconditioner
    - https://en.wikipedia.org/wiki/Condition_number

Below is code that shows these approaches. We avoid numerical tricks, like
not forming diagonal matrices, specialized diagonal solvers, etc., in order
to show the mathematics. For simplicity, suppose we want to solve Ax = b, where
A is positive definite.

Prepare some data:

>>> import numpy as np
>>> import scipy as sp
>>> rng = np.random.default_rng(42)
>>> F = rng.normal(size=(3, 3))
>>> G = rng.normal(size=(3, 3))
>>> C_D = G.T @ G + np.diag([1e-5, 1, 1e5])
>>> A = (F.T @ F + C_D).round(1)
>>> A
array([[ 2.50000e+00, -3.20000e+00, -8.00000e-01],
       [-3.20000e+00,  8.20000e+00,  2.60000e+00],
       [-8.00000e-01,  2.60000e+00,  1.00004e+05]])
>>> b = rng.normal(size=3).round(1)

Approach 1: No scaling
----------------------

We do not perform any scaling and simply compute the result.
Notice the large condition number (here it is not large enough to lead
to numerical issues, but if it is even larger then it can).

>>> float(np.linalg.cond(A))
93913.686...
>>> sp.linalg.solve(A, b, assume_a='positive definite').round(2)
array([ 0.72,  0.28, -0.  ])

Approach 2: Scale with diagonal
-------------------------------

We scale with the diagonal entries, then solve the system:
   (S @ A @ S) (S^-1 @ x) = S @ b
Notice below that using S = 1 / sqrt(diag(C_D)) as a two-sided diagonal
preconditioner leads to a much smaller condition number.

>>> S = np.diag(1 / np.sqrt(np.diag(C_D)))
>>> S.round(3)
array([[0.825, 0.   , 0.   ],
       [0.   , 0.561, 0.   ],
       [0.   , 0.   , 0.003]])

>>> float(np.linalg.cond(S @ A @ S))
6.182...
>>> y = sp.linalg.solve(S @ A @ S, S @ b, assume_a='positive definite')
>>> x = S @ y
>>> x.round(2)
array([ 0.72,  0.28, -0.  ])

Approach 3: Scale with Cholesky factor
--------------------------------------

We scale with the Cholesky factor. Notice how this brings the condition
number even further down.

>>> S = np.linalg.cholesky(C_D)
>>> A_c = np.linalg.inv(S) @ A @ np.linalg.inv(S).T
>>> A_c.round(2)
array([[ 1.7 , -0.79, -0.  ],
       [-0.79,  2.  ,  0.  ],
       [-0.  ,  0.  ,  1.  ]])

>>> float(np.linalg.cond(A_c))
2.651...
>>> y = sp.linalg.solve(A_c, np.linalg.inv(S) @ b, assume_a='positive definite')
>>> x = np.linalg.inv(S.T) @ y
>>> x.round(2)
array([ 0.72,  0.28, -0.  ])

Subspace inversion
------------------

Suppose we want to solve Ax = b, where A is positive definite. In fact A
has a structure like A := (F F.T + C_D), where C_D is positive definite
and F is tall and thin (many more rows than columns). Here is an example:

>>> F = rng.normal(size=(4, 2))
>>> G = rng.normal(size=(4, 4))
>>> C_D = G.T @ G
>>> A = (F @ F.T + C_D)

Subspace inversion means using the Woodbury identity and an SVD when inverting A.
The version of Woodbury that we'll use is the following:

    (F F.T + I)^-1 = I - F (F.T F + I)^-1 F.T

But in order to use this identity, we need to get rid of C_D in the expression
for A and somehow transform it into the identity matrix. We'll see that the trick
is to use the Cholesky factorization of C_D. In the papers, the following is not
obvious:

    If we do not use the Cholesky factor, we compute the wrong answer.

Assume that we can find an invertible scaling matrix S such that S^-1 C_D S^-T = I.
The matrix S is the Cholesky factor. Using S, we can write:

    A = (F F.T + C_D) = S (S^-1 F F.T S^-T + S^-1 C_D S^-T) S^T = S (G G.T + I) S^T,

where we defined G = S^-1 F. If we choose S = I, or S = diag(1 / sqrt(diag(C_D))),
then the equation above is an approximation and does not hold.

Inverting A then boils down to computing A^-1 = S^-T (G G.T + I)^-1 S^-1,
and this is where the Woodbury identity comes in. Let us now look at some code.
We take the SVD of G, to get U, W, V.T = svd(G) = svd(S^-1 F).

>>> S = np.linalg.cholesky(C_D)
>>> G = np.linalg.inv(S) @ F

By the Woodbury identity, we have:

    (G G.T + I)^-1 = I - G (G.T G + I)^-1 G.T

Let us verify that this is exact:

>>> np.linalg.inv(np.eye(4) + G @ G.T).round(1)
array([[ 0.9,  0. , -0.3, -0.1],
       [ 0. ,  1. ,  0.1,  0.1],
       [-0.3,  0.1,  0.4, -0.2],
       [-0.1,  0.1, -0.2,  0.4]])

>>> (np.eye(4) -  G @ np.linalg.inv(np.eye(2) + G.T @ G) @ G.T).round(1)
array([[ 0.9,  0. , -0.3, -0.1],
       [ 0. ,  1. ,  0.1,  0.1],
       [-0.3,  0.1,  0.4, -0.2],
       [-0.1,  0.1, -0.2,  0.4]])

At this point, one approach would be to use Woodbury identity directly, which
avoids inverting a large matrix and instead inverts a much smaller matrix.
If G has shape (m, n), with m >> n, then this has cost O(n^3) for the inversion,
plus a cost of O(nm^2) to form the full product I - G (G.T G + I)^-1 G.T.

The more common alternative, which is not less work computationally, is to
take the SVD of the matrix G. Assume U, W, V.T = svd(G), then:

    I - G (I + G.T  G)^-1 G.T =
    I -  G (I + (U W V.T).T (U W V.T))^-1 G.T =
    I -  G (I + (V W U.T) (U W V.T))^-1 G.T

What remains in this equation is to compute the middle factor. If we use
G = U W V.T at this point to simplify, we need to assume that V V.T = I,
but this only holds when rows >= columns. However, the above holds for
*any* G, regardless of shape. To show this, we again appeal to the Woodbury
identity, this time the version: (I + UV)^-1 = I - U (I + VU)^-1 V.
Note that while V V.T != I in general, V.T V = I and U.T U = I always holds:

    (I + (V W U.T)(U W V.T))^-1 = (I + (V W**2) V.T)^-1               (U.T U = I)
                                = I - V W**2 (I + V.T V W**2)^-1 V.T  (woodbury)
                                = I - V W**2 (I + W**2)^-1 V.T        (V.T V = I)

Now we substitute this middle term back. Starting from the first equation to the last:

    (G G.T + I)^-1
    =  I - G (I + G.T  G)^-1 G.T =
    = I -  G [ (I + (V W U.T) (U W V.T))^-1 ] G.T            (substitute)
    = I -  G [ I - V W**2 (I + W**2)^-1 V.T ] G.T            (substitute)
    = I - U W V.T [ I - V W**2 (I + W**2)^-1 V.T ] V W U.T   (G = U W V.T)
    = I - U W [ I - W**2 (I + W**2)^-1] W U.T                (V.T V = I, always)
    = I - U diag( w**2 / (1 + w**2)) U.T                     (algebra on diags)

We can now verify this result numerically:

>>> U, W, Vt = sp.linalg.svd(G, full_matrices=False)
>>> V = Vt.T

>>> np.linalg.inv(np.eye(4) + G @ G.T).round(1)
array([[ 0.9,  0. , -0.3, -0.1],
       [ 0. ,  1. ,  0.1,  0.1],
       [-0.3,  0.1,  0.4, -0.2],
       [-0.1,  0.1, -0.2,  0.4]])

>>> (np.eye(4) - U @ np.diag(W**2 / (1 + W**2)) @ U.T).round(1)
array([[ 0.9,  0. , -0.3, -0.1],
       [ 0. ,  1. ,  0.1,  0.1],
       [-0.3,  0.1,  0.4, -0.2],
       [-0.1,  0.1, -0.2,  0.4]])

To summarize subspace inversion:

- Unlike in exact inversion, scaling F by S = cholesky(C_D) is not optional or
  due to numerics. To use the Woodbury identity we need to transform (whiten) C_D
  into the identity matrix I. If we do not perform this step, we compute the wrong
  answer. Of course, if C_D happens to be diagonal, then
  cholesky(C_D) = diag(1/sqrt(diag(C_D))) and it works out.
- If cholesky(C_D) ~= diag(constant), then every covariance is roughly the same
  and we could get away with taking the SVD of F instead of G = S^-1 @ F.
  However, this reverses the logic: in reality we must take the SVD of G,
  we just so happen to get away with the wrong computation in some cases.
- The equations work regardless of the size of F. But working in the subspace
  is more efficient when F is tall and thin, which is almost always the case
  in ensemble smoothers, since observations >> realizations.
- The truncation of the SVD may help a bit numerically if columns or rows of F
  are nearly identical. This can happen if columns are highly correlated.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore


def empirical_covariance_upper(Y: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Compute the upper triangular part of the empirical covariance matrix Y
    with shape (parameters, ensemble_size).

    Examples
    --------
    >>> Y = np.array([[-2.4, -0.3,  0.7,  0.2,  1.1],
    ...               [-1.5,  0.4, -0.4, -0.9,  1. ],
    ...               [-0.1, -0.4, -0. , -0.5,  1.1]])
    >>> empirical_covariance_upper(Y)
    array([[1.873, 0.981, 0.371],
           [0.   , 0.997, 0.392],
           [0.   , 0.   , 0.407]])

    Naive computation:

    >>> empirical_cross_covariance(Y, Y)
    array([[1.873, 0.981, 0.371],
           [0.981, 0.997, 0.392],
           [0.371, 0.392, 0.407]])
    """
    _, num_observations = Y.shape
    if num_observations <= 1:
        raise ValueError("Need at least two observations to compute covariance")
    Y = (Y - np.mean(Y, axis=1, keepdims=True)) / np.sqrt(num_observations - 1)
    # https://www.math.utah.edu/software/lapack/lapack-blas/dsyrk.html
    YYT: npt.NDArray[np.double] = sp.linalg.blas.dsyrk(alpha=1.0, a=Y)
    return YYT


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
    if X.shape[1] <= 1:
        raise ValueError("Need at least two observations to compute covariance")

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
    # If X is a large matrix, it might be stored as a float32 array to save memory.
    # However, if Y is of type float64,
    # the resulting cross-covariance matrix will be float64,
    # potentially doubling the memory usage even if X is float32.
    # To prevent unnecessary memory consumption,
    # we cast Y to the same data type as X before computing the dot product.
    # This ensures that the output cross-covariance matrix uses memory efficiently
    # while retaining the precision dictated by X's data type.
    cov = X @ Y.astype(X.dtype).T / (X.shape[1] - 1)
    assert cov.shape == (X.shape[0], Y.shape[0])
    return cov


def normalize_alpha(alpha: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Assure that sum_i (1/alpha_i) = 1.

    This is Eqn (22) in :cite:t:`EMERICK2013`.

    Examples
    --------
    >>> alpha = np.arange(10) + 1
    >>> float(np.sum(1/normalize_alpha(alpha)))
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
    assert np.all(np.diff(singular_values) <= 0), (
        "Singular values must be sorted decreasing"
    )
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
    U, w, _ = sp.linalg.svd(D_delta, full_matrices=False)

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
