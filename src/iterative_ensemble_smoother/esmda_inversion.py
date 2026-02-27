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

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore


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


def invert_naive(
    *,
    delta_D: npt.NDArray[np.double],
    C_D_L: npt.NDArray[np.double],
    alpha: float,
    truncation: float,
) -> npt.NDArray[np.double]:
    r"""Naive implementation of the equation:

    (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)

    This function should only be used for testing and verification.
    """
    _, N_e = delta_D.shape  # Number of ensemble members

    covariance = np.diag(C_D_L**2) if C_D_L.ndim == 1 else C_D_L.T @ C_D_L
    return delta_D.T @ np.linalg.inv(
        delta_D @ delta_D.T + alpha * (N_e - 1) * covariance
    )


def invert_subspace(
    *,
    delta_D: npt.NDArray[np.double],
    C_D_L: npt.NDArray[np.double],
    alpha: float,
    truncation: float,
) -> npt.NDArray[np.double]:
    r"""Subspace inversion implementation of the equation:

    (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)

    See the appendix in the 2016 Emerick paper for details:
    https://doi.org/10.1016/j.petrol.2016.01.029

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> rng = np.random.default_rng(42)

    Create matrix delta_D (centered responses)

    >>> Y = rng.normal(size=(5, 3))
    >>> delta_D = Y - np.mean(Y, axis=1, keepdims=True)

    Create matrix C_D_L (Cholesky of covariance C_D)

    >>> F = rng.normal(size=(5, 5))
    >>> C_D = F.T @ F
    >>> C_D_L = sp.linalg.cholesky(C_D, lower=False)

    Now compute delta_D.T @ inv(delta_D @ delta_D.T + alpha (N_1 - 1) C_D)
    naively:

    >>> N_e = Y.shape[1]
    >>> alpha = 1/3
    >>> delta_D.T @ np.linalg.inv(delta_D @ delta_D.T + alpha * (N_e - 1) * C_D)
    array([[-0.08790298,  0.19940823, -0.00894643, -0.26919301, -0.07465476],
           [-0.31262364, -0.21350368, -0.04888574, -0.26139685,  0.35487784],
           [ 0.40052661,  0.01409545,  0.05783217,  0.53058986, -0.28022308]])

    Using this function. Notice that it returns matrices that the user
    must multiply together:

    >>> delta_DT, factor1, factor2 = invert_subspace(delta_D=delta_D, C_D_L=C_D_L,
    ...                                              alpha=alpha, truncation=1.0)
    >>> np.linalg.multi_dot([delta_DT, factor1, factor2])
    array([[-0.08790298,  0.19940823, -0.00894643, -0.26919301, -0.07465476],
           [-0.31262364, -0.21350368, -0.04888574, -0.26139685,  0.35487784],
           [ 0.40052661,  0.01409545,  0.05783217,  0.53058986, -0.28022308]])


    """
    # Quick verification of shapes
    assert alpha >= 0, "Alpha must be non-negative"

    # Shapes
    N_n, N_e = delta_D.shape

    # The LAPACK routine trtrs is the one called by sp.linalg.solve_triangular.
    # However, solve_triangular will always upcast to float64, even if inputs
    # are float32. Here we get the correct LAPACK routine based on inputs.
    # (To avoid float64 cast, we use alpha**0.5 instead of np.sqrt(alpha))
    dtrtrs = sp.linalg.get_lapack_funcs("trtrs", arrays=(C_D_L, delta_D))

    # If the matrix C_D is 2D, then C_D_L is the (upper) Cholesky factor
    if C_D_L.ndim == 2:
        # Computes G := inv(sqrt(alpha) * C_D_L.T) @ D_delta
        # G = sp.linalg.solve_triangular(
        #     alpha**0.5 * C_D_L, delta_D, lower=False, trans=1
        # )
        G, info = dtrtrs(alpha**0.5 * C_D_L, delta_D, lower=0, trans=1)
        assert info == 0

    # If the matrix C_D is 1D, then C_D_L is the square-root of C_D
    else:
        G = delta_D / (alpha**0.5 * C_D_L[:, np.newaxis])

    # Take the SVD and truncate it. N_r is the number of values to keep
    U, w, _ = sp.linalg.svd(G, overwrite_a=True, full_matrices=False)
    idx = singular_values_to_keep(w, truncation=truncation)
    N_r = min(N_n, N_e - 1, idx)  # Number of values in SVD to keep
    U_r, w_r = U[:, :N_r], w[:N_r]

    # Compute the symmetric terms
    if C_D_L.ndim == 2:
        # Computes term := np.linalg.inv(np.sqrt(alpha) * C_D_L) @ U_r @ np.diag(1/w_r)
        # term = sp.linalg.solve_triangular(
        #     np.sqrt(alpha) * C_D_L, (U_r / w_r[np.newaxis, :]), lower=False
        # )
        term, info = dtrtrs(
            alpha**0.5 * C_D_L, (U_r / w_r[np.newaxis, :]), lower=0, trans=0
        )
        assert info == 0
    else:
        term = (U_r / w_r[np.newaxis, :]) / (alpha**0.5 * C_D_L)[:, np.newaxis]

    # Diagonal matrix represented as vector
    diag = w_r**2 / (w_r**2 + N_e - 1)
    return (delta_D.T, term * diag, term.T)


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
