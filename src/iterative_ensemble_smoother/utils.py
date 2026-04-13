from __future__ import annotations

import collections
import logging
import numbers
import warnings
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


def clip_correlation_matrix(
    corr_XY: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Clip correlation array to range [-1, 1]."""

    # Perform checks and clip values to [-1, 1]
    eps = 1e-8
    min_value, max_value = corr_XY.min(), corr_XY.max()
    if not ((max_value <= 1 + eps) and (min_value >= -1 - eps)):
        msg = "Cross-correlation matrix has entries not in [-1, 1]."
        msg += f"The min and max values are: {min_value} and {max_value}"
        msg += "Entries will be clipped to the range [-1, 1]."
        warnings.warn(msg)

    return np.clip(corr_XY, a_min=-1, a_max=1, out=corr_XY)


def groupby_rows(
    A: npt.NDArray[np.bool_],
) -> Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.bool_]]]:
    """Yields pairs (row_indices, columns).

    The usage is that A is a boolean matrix with shape (params, responses)
    indicating for each parameter which responses to keep. This function
    tells us, for each group of parameters, which responses to update.

    Examples
    --------
    >>> A = np.array([[ True, False,  True],
    ...               [False,  True, False],
    ...               [False,  True, False],
    ...               [ True, False,  True],
    ...               [ True,  True, False]])
    >>> for param_idx, response_idx in groupby_rows(A):
    ...     print(param_idx, response_idx)
    [0 3] [ True False  True]
    [1 2] [False  True False]
    [4] [ True  True False]
    """
    if not np.issubdtype(A.dtype, np.bool_):
        raise ValueError(f"A must be a boolean array, got dtype: {A.dtype}")

    # Find unique rows efficiently by packing booleans into bytes.
    #
    # The naive approach -- np.unique(A, axis=0) -- is slow because NumPy
    # converts each row into a structured dtype with one field per column
    # (e.g. [('f0', bool), ('f1', bool), ...]), then sorts by comparing
    # fields one at a time.  With 5000 columns that means up to 5000
    # per-field dispatches for every comparison during the sort.
    #
    # Instead we:
    #  1. Pack every 8 boolean columns into one uint8 byte with packbits,
    #     shrinking a 5000-column row to ~625 bytes.
    #  2. View each packed row as a single np.void blob (an unstructured
    #     opaque byte sequence).  Unlike a structured dtype, np.void has
    #     no named fields, so NumPy compares two elements with a single
    #     memcmp call over the contiguous byte block -- not 5000 separate
    #     field comparisons.
    #  3. Pass these keys to np.unique, which now sorts and deduplicates
    #     with the fast memcmp comparisons.
    #
    # This is a lossless encoding (packbits is a bijection on fixed-width
    # boolean rows), so every distinct boolean row maps to a distinct key.
    packed = np.packbits(A, axis=1)
    key_dtype = np.dtype((np.void, packed.shape[1]))
    keys = np.ascontiguousarray(packed).view(key_dtype).ravel()
    _, inverse = np.unique(keys, return_inverse=True)

    groups = collections.defaultdict(list)
    for row_idx, group in enumerate(inverse):
        groups[group].append(row_idx)

    for indices in groups.values():
        first_idx = indices[0]
        yield np.array(indices, dtype=np.int_), A[first_idx, :]


def groupby_rows_float(
    A: npt.NDArray[np.bool_], resolution: int = 8
) -> Iterator[npt.NDArray[np.int_]]:
    """Yields integer arrays with row_indices, such that rows are similar.

    The `resolution` is the number of buckets to group the floating point
    numbers into. For instance, if the entries of A are in the range [-1, 1],
    then a resolution of 8 will bucket the numbers into
    [(-1, -0.75), (-0.75, -0.5), ..., (0.75, 1.0)].
    If the resolution is 8, then 3 bits will be used, since 2**3 = 8.
    The resolution must be a power of two.

    The usage is that A is a float matrix with shape (params, responses)
    indicating for each parameter which responses to keep. This function
    tells us, for each group of parameters, which responses to update.

    Examples
    --------
    >>> A = np.array([[-1.0,  -0.9],
    ...               [-1.0,  -0.9],
    ...               [-0.96,  0.24],
    ...               [ 0.22,  0.23],
    ...               [ 0.23,  0.23],
    ...               [ 0.26,  0.99],
    ...               [ 0.39,  0.99],
    ...               [ 0.49,  0.99],
    ...               [-0.58, -0.74],
    ...               [-0.37,  1.0]])
    >>> for param_idx in groupby_rows_float(A):
    ...     print(param_idx)
    [0 1]
    [2]
    [8]
    [9]
    [3 4]
    [5 6 7]
    """
    if not np.issubdtype(A.dtype, np.floating):
        raise ValueError(f"A must be a float array, got dtype: {A.dtype}")
    if not isinstance(resolution, numbers.Integral):
        raise TypeError("`resolution` must be a integer")
    if resolution < 2 or resolution > 256:
        raise ValueError("`resolution` must be in the range [2, 256]")
    if (resolution & (resolution - 1)) != 0:
        raise ValueError("`resolution` must be a power of two, e.g. 2, 4, 8, ...")

    minimum = np.min(A)
    maximum = np.max(A)
    difference = maximum - minimum
    assert resolution <= 256
    assert (resolution & (resolution - 1)) == 0, "Must be a power of two"
    # Map to integers in 0...num_bits-1 (bucket the floats)
    A_integer = np.floor((A - minimum) / difference * resolution).astype(np.uint8)
    A_integer = np.minimum(A_integer, resolution - 1, out=A_integer)

    keys = np.ascontiguousarray(A_integer)
    key_dtype = np.dtype((np.void, A_integer.shape[1]))
    keys = keys.view(key_dtype).ravel()
    _, inverse = np.unique(keys, return_inverse=True)

    # Group row indices using argsort instead of a Python loop
    order = np.argsort(inverse, kind="stable")
    splits = np.flatnonzero(np.diff(inverse[order])) + 1
    for group in np.split(order, splits):
        yield group


def masked_std(
    X: npt.NDArray[np.floating], *, missing: npt.NDArray[np.bool_]
) -> npt.NDArray[np.floating]:
    """Computes a masked std for each row in X.

    Examples
    --------
    >>> X = np.array([[ 0.65,  0.74,  0.54, -0.67],
    ...               [ 0.23,  0.12,  0.22,  0.87],
    ...               [ 0.22,  0.68,  0.07,  0.29]])

    Let us encode some missing data:

    >>> missing = np.array([[0, 0, 0, 0],
    ...                     [0, 0, 0, 1],
    ...                     [0, 0, 1, 1]], dtype=np.bool_)

    For each row, the standard deviations are:


    >>> for i in range(X.shape[0]):
    ...     print(float(np.std(X[i, ~missing[i, :]], ddof=1)))
    0.6617401302626281
    0.060827625302982205
    0.3252691193458119

    This function computes the same standard deviations, but vectorized:

    >>> masked_std(X, missing=missing)
    array([0.66174013, 0.06082763, 0.32526912])
    """

    N_e = X.shape[1]  # Ensemble members / realizations
    n_available = N_e - np.sum(missing, axis=1, keepdims=True)  # Non-missing params

    # Need at least two valid ensemble members per parameter
    if np.any(n_available < 2):
        msg = (
            "One or several parameters have too few valid ensemble members (need >=2)."
        )
        raise ValueError(msg)

    valid = np.logical_not(missing)
    X_masked = valid * X  # Set missing values to zero

    # Compute mean values, taking missing into account
    X_means = np.sum(X_masked, axis=1, keepdims=True) / n_available

    # Center the matrix
    X_centered = (X_masked - X_means) * valid

    result: npt.NDArray[np.floating] = np.sqrt(
        np.sum(X_centered**2, axis=1, keepdims=True) / (n_available - 1)
    ).ravel()
    return result


def adjust_for_missing(
    X: npt.NDArray[np.floating], *, missing: npt.NDArray[np.bool_]
) -> npt.NDArray[np.floating]:
    """Removes missing values from X, such that the cross-covariance product

        center(X) @ center(Y).T / (N_e - 1)

    remains correct even in the presence of missing parameters in some
    ensemble members (realizations). Mutates the "missing" argument.

    Examples
    --------
    >>> X = np.array([[ 0.65,  0.74,  0.54, -0.67],
    ...               [ 0.23,  0.12,  0.22,  0.87],
    ...               [ 0.22,  0.68,  0.07,  0.29]])

    Let us encode some missing data:

    >>> missing = np.array([[0, 0, 0, 0],
    ...                     [0, 0, 0, 1],
    ...                     [0, 0, 1, 1]], dtype=np.bool_)

    The second parameter (row) is missing from the last realization (column).
    The third parameter (row) is missing from last two realizations (columns).

    If we compute the cross-covariance directly, we get the wrong answer.
    (Recall that we don't have to center both matrices, centering one is enough.)

    >>> Y = np.array([[ 0.59,  0.71,  0.79, -0.35],
    ...               [-0.46,  0.86, -0.19, -1.28]])
    >>> (X - np.mean(X, axis=1, keepdims=True)) @ Y.T / (X.shape[1] - 1)
    array([[ 0.34063333,  0.47648333],
           [-0.17873333, -0.2576    ],
           [ 0.0061    ,  0.14538333]])

    Only the top row is actually correct, since only the first parameter has no
    missing values. The remaining entries are wrong, because missing values
    are not taken into account. Let us compute a few entries by hand.

    The entry in the second row, first column should be:

    >>> x = np.array([0.23,  0.12,  0.22])
    >>> y = np.array([0.59,  0.71,  0.79])
    >>> float((x - np.mean(x)) @ y / (3 - 1))
    -0.0011999...

    The bottom-right entry should be:

    >>> x = np.array([0.22, 0.68])
    >>> y = np.array([-0.46,  0.86])
    >>> float((x - np.mean(x)) @ y / (2 - 1))
    0.30360000...

    Now let us use our function. Notice that the entries match up to numerical
    accuracy:

    >>> adjust_for_missing(X, missing=missing) @ Y.T / (X.shape[1] - 1)
    array([[ 0.34063333,  0.47648333],
           [-0.0012    , -0.04215   ],
           [ 0.0276    ,  0.3036    ]])

    To summarize, this function prepares a matrix X for the cross-covariance
    computation, such that the computation is correct even is the presence
    of missing values.
    """
    N_e = X.shape[1]  # Ensemble members / realizations
    n_available = N_e - np.sum(missing, axis=1, keepdims=True)  # Non-missing params

    # Need at least two valid ensemble members per parameter
    if np.any(n_available < 2):
        msg = (
            "One or several parameters have too few valid ensemble members (need >=2)."
        )
        raise ValueError(msg)

    valid = np.logical_not(missing)
    X_masked = valid * X  # Set missing values to zero
    # Compute mean values, taking missing into account
    X_means = np.sum(X_masked, axis=1, keepdims=True) / n_available

    # Center the matrix
    X_centered = X_masked - X_means

    # Scale by number of ensemble members. I think of this as a scaling of
    # the final covariance matrix, but it works to apply it to X directly
    # due to linearity: a[:, None] * (X @ Y.T) == (a[:, None] * X) @ Y.T
    X_centered *= (N_e - 1) / (n_available - 1)

    # Mask to zero in anticipation of C = X @ Y.T, so that in the product
    # zero values are accounted for in the sum-of-products in C_ij
    result: npt.NDArray[np.floating] = X_centered * valid
    return result


def sample_mvnormal(
    *,
    C_dd_cholesky: npt.NDArray[np.floating],
    rng: np.random.Generator,
    size: int,
) -> npt.NDArray[np.floating]:
    """Draw samples from the multivariate normal N(0, C_dd).

    We write this function from scratch to avoid factoring the covariance
    matrix every time we sample, and we want to exploit diagonal covariance
    matrices in terms of computation and memory. More specifically:

        - numpy.random.multivariate_normal factors the covariance in every call
        - scipy.stats.Covariance.from_diagonal stores off diagonal zeros

    So the best choice was to write sampling from scratch.


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
    # Standard normal samples
    z = rng.standard_normal(size=(C_dd_cholesky.shape[0], size))

    # A 2D covariance matrix was passed
    if C_dd_cholesky.ndim == 2:
        return C_dd_cholesky @ z

    # A 1D diagonal of a covariance matrix was passed
    return C_dd_cholesky.reshape(-1, 1) * z


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-v"])
