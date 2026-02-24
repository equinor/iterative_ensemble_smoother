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

The exposition of these methods can be quite hard to follow in papers like:

    - "History Matching Time-lapse Seismic Data Using the Ensemble Kalman
       Filter with Multiple Data Assimilations", Emerick and Reynolds
    - "Analysis of the Performance of Ensemble-based Assimilation of
       Production and Seismic Data", Emerick

In the first paper above, the authors
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

Let S be the Cholesky factor of C_D, such that S^-1 C_D S^-T = I.
We can then write:

    A = (F F.T + C_D) = S (S^-1 F F.T S^-T + S^-1 C_D S^-T) S^T = S (G G.T + I) S^T,

where we defined G = S^-1 F. If we choose S = I, or S = diag(1 / sqrt(diag(C_D))),
then the equation above is an approximation and does not hold.

Inverting A then boils down to computing A^-1 = S^-T (G G.T + I)^-1 S^-1,
and this is where the Woodbury identity comes in. Let us now look at some code.

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

One approach is to use Woodbury identity directly, which
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
*any* G, regardless of shape. To demonstrate this, we use the specific Woodbury
identity variant: (I + UV)^-1 = I - U (I + VU)^-1 V.
Note that while V V.T != I in general, V.T V = I and U.T U = I always holds:

    (I + (V W U.T)(U W V.T))^-1 = (I + (V W**2) V.T)^-1               (U.T U = I)
                                = I - V W**2 (I + V.T V W**2)^-1 V.T  (woodbury)
                                = I - V W**2 (I + W**2)^-1 V.T        (V.T V = I)

Now we substitute this middle term back. Starting from the first equation to the last:

    (G G.T + I)^-1
    = I - G (I + G.T  G)^-1 G.T
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

from typing import Union

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore

from iterative_ensemble_smoother.utils import adjust_for_missing


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


def inversion_exact_naive(
    *,
    alpha: float,
    C_D_L: npt.NDArray[np.double],
    D: npt.NDArray[np.double],
    Y: npt.NDArray[np.double],
    X: npt.NDArray[np.double],
    truncation: float = 1.0,
    missing: Union[npt.NDArray[np.bool_], None] = None,
) -> npt.NDArray[np.double]:
    """Naive inversion, used for testing only.

    Computes C_MD @ inv(C_DD + alpha * C_D) @ (D - Y) naively.

    C_D_L is the upper cholesky factor. C_D_L.T @ C_D_L = C_D
    """
    if missing is not None:
        X = adjust_for_missing(X, missing=missing)

    # Naive implementation of Equation (3) in Emerick (2013)
    C_MD = empirical_cross_covariance(X, Y)
    C_DD = empirical_cross_covariance(Y, Y)
    C_D = np.diag(C_D_L**2) if C_D_L.ndim == 1 else C_D_L.T @ C_D_L
    return C_MD @ sp.linalg.inv(C_DD + alpha * C_D) @ (D - Y)  # type: ignore


def inversion_subspace(
    *,
    alpha: float,
    C_D_L: npt.NDArray[np.double],
    D: npt.NDArray[np.double],
    Y: npt.NDArray[np.double],
    X: npt.NDArray[np.double],
    truncation: float = 1.0,
    missing: Union[npt.NDArray[np.bool_], None] = None,
    return_T: bool = False,
) -> npt.NDArray[np.double]:
    """Computes

        C_MD @ inv(C_DD + alpha * C_D) @ (D - Y)

    by taking a single SVD. See Appendix A.2 and comments in code.
    """
    # Quick verification of shapes
    assert alpha >= 0, "Alpha must be non-negative"
    assert C_D_L.shape[0] == Y.shape[0], "Number of observations must match"
    assert D.shape == Y.shape, "Y must match D in shape"

    # This method is based on Appendix A.2 in "History Matching Time-lapse
    # Seismic Data Using ..." by Emerick. We start with the equation:
    #   X @ D_delta.T @ inv(D_delta @ D_delta.T + (N_e - 1) * alpha * C_D) @ (D - Y)
    # Then we whiten the middle factor with S = cholesky(alpha * C_D) to obtain:
    #   S^-1 @ [G @ G.T + (N_e - 1) * I]^-1 S^-T,  where G = S^-T @ D_delta
    # Now take SVD of G, then follow the steps outlined at the top in this module.

    # Shapes
    N_n, N_e = Y.shape
    D_delta = Y - np.mean(Y, axis=1, keepdims=True)  # Subtract average

    # If the matrix C_D is 2D, then C_D_L is the (upper) Cholesky factor
    if C_D_L.ndim == 2:
        # Computes G := inv(sqrt(alpha) * C_D_L.T) @ D_delta
        G = sp.linalg.solve_triangular(
            np.sqrt(alpha) * C_D_L, D_delta, lower=False, trans=1
        )

    # If the matrix C_D is 1D, then C_D_L is the square-root of C_D
    else:
        G = D_delta / (np.sqrt(alpha) * C_D_L[:, np.newaxis])

    # Take the SVD and truncate it. N_r is the number of values to keep
    U, w, _ = sp.linalg.svd(G, overwrite_a=True, full_matrices=False)
    idx = singular_values_to_keep(w, truncation=truncation)
    N_r = min(N_n, N_e - 1, idx)  # Number of values in SVD to keep
    U_r, w_r = U[:, :N_r], w[:N_r]

    # Compute the symmetric terms
    if C_D_L.ndim == 2:
        # Computes term := np.linalg.inv(np.sqrt(alpha) * C_D_L) @ U_r @ np.diag(1/w_r)
        term = sp.linalg.solve_triangular(
            np.sqrt(alpha) * C_D_L, (U_r / w_r[np.newaxis, :]), lower=False
        )
    else:
        term = (U_r / w_r[np.newaxis, :]) / (np.sqrt(alpha) * C_D_L)[:, np.newaxis]

    # Diagonal matrix represented as vector
    diag = w_r**2 / (w_r**2 + N_e - 1)

    if return_T:  # Return transition matrix without X at the start
        return np.linalg.multi_dot(  # type: ignore
            [D_delta.T, term * diag, term.T, (D - Y)]
        )

    if missing is not None:
        X = adjust_for_missing(X, missing=missing)

    assert X.shape[1] == Y.shape[1], "Number of ensemble members must match"
    return np.linalg.multi_dot(  # type: ignore
        [X, D_delta.T, term * diag, term.T, (D - Y)]
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
