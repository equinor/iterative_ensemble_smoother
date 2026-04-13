from __future__ import annotations

import collections
import logging
import numbers
import warnings
from typing import TYPE_CHECKING, Iterator

import numpy as np
import psutil

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


def calc_max_number_of_layers_per_batch_for_distance_localization(
    nx: int,
    ny: int,
    nz: int,
    num_obs: int,
    nreal: int,
    bytes_per_float: int = 8,  # float64 as default here
) -> int:
    """Calculate number of layers from a 3D field parameter that can be updated
    within available memory. Distance-based localization requires two large matrices
    the Kalman gain matrix K and the localization scaling matrix RHO, both have size
    equal to number of field parameter values times number of observations.
    Therefore, a batching algorithm is used where only a subset of parameters
    is used when calculating the Schur product of RHO and K matrix in the update
    algorithm. This function calculates number of batches and
    number of grid layers of field parameter values that can fit
    into the available memory for one batch accounting for a safety margin.

    The available memory is checked using the `psutil` library, which provides
    information about system memory usage.
    From `psutil` documentation:
    - available:
        the memory that can be given instantly to processes without the
        system going into swap.
        This is calculated by summing different memory values depending
        on the platform and it is supposed to be used to monitor actual
        memory usage in a cross platform fashion.

    Parameters
    ----------
    nx : int
        Grid size in I-direction (local x-axis direction).
    ny : int
        Grid size in J-direction (local y-axis direction).
    nz : int
        Grid size in K-direction (number of layers).
    num_obs : int
        Number of observations.
    nreal : int
        Number of realizations.
    bytes_per_float : int, optional
        Number of bytes per float (4 or 8). Default is 8.

    Returns
    -------
    int
        Max number of layers that can be updated in one batch to
        avoid memory problems.
    """

    memory_safety_factor = 0.8
    num_params = nx * ny * nz
    num_param_per_layer = nx * ny

    # Rough estimate of necessary number of float variables
    sum_floats = 0
    sum_floats += num_params * num_obs  # K matrix before Schur product
    sum_floats += num_params * num_obs  # RHO matrix
    sum_floats += num_params * num_obs  # K matrix after Schur product
    sum_floats += int(num_params * nreal * 2.5)  # X_prior, X_prior_batch, M_delta
    sum_floats += int(num_params * nreal * 1.5)  # X_post and X_post_batch
    sum_floats += num_obs * nreal * 2  # D matrix and internal matrices
    sum_floats += num_obs * nreal * 2  # Y matrix and internal matrices

    # Check available memory
    available_memory_in_bytes = psutil.virtual_memory().available * memory_safety_factor

    # Required memory
    total_required_memory_per_field_param = sum_floats * bytes_per_float

    # Minimum number of batches
    min_number_of_batches = int(
        np.ceil(total_required_memory_per_field_param / available_memory_in_bytes)
    )

    max_nlayer_per_batch = int(nz / min_number_of_batches)

    if max_nlayer_per_batch == 0:
        # Batch size cannot be less than 1 layer
        memory_one_batch = num_param_per_layer * bytes_per_float
        raise MemoryError(
            "The required memory to update one grid layer or one 2D surface is "
            "larger than available memory.\n"
            "Cannot split the update into batch size less than one complete "
            "grid layer for 3D field or one surface for 2D fields."
            f"Required memory for one batch is about: {memory_one_batch / 10**9} GB\n"
            f"Available memory is about: {available_memory_in_bytes / 10**9} GB"
        )

    logger.debug(
        "Calculate batch size for updating of field parameter:\n"
        f" Number of parameters in field param: {num_params}\n"
        f" Required number of floats to update one field parameter: {sum_floats}\n"
        " Available memory per field param update: "
        f"{available_memory_in_bytes / 10**9} GB\n"
        " Required memory total to update a field parameter: "
        f"{total_required_memory_per_field_param / 10**9} GB\n"
        f" Number of layers in one batch: {max_nlayer_per_batch}"
    )

    return max_nlayer_per_batch


def gaspari_cohn(
    distances: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Gaspari--Cohn distance-based localization scaling function.

    For each normalised distance d, returns a scaling factor in [0, 1]
    used as elements in the localization matrix (rho).
    For d >= 2 the value is 0.

    This is an implementation of Eq. (4.10) in Section 4.3
    ("Compactly supported 5th-order piecewise rational functions") of
    :cite:t:`gaspari1999construction`.

    References
    ----------
    Gaspari, G. and Cohn, S.E. (1999), Construction of correlation functions
    in two and three dimensions. Q.J.R. Meteorol. Soc., 125: 723-757.
    https://doi.org/10.1002/qj.49712555417

    Parameters
    ----------
    distances : np.ndarray
        Vector of values for normalized distances.

    Returns
    -------
    np.ndarray
        Values of scaling factors for each value of input distance.

    Examples
    --------
    The function equals 1 at d=0, 5/24 at d=1, and 0 for d>=2:

    >>> import numpy as np
    >>> gaspari_cohn(np.array([0.0, 1.0, 2.0, 3.0]))
    array([1.        , 0.20833333, 0.        , 0.        ])

    The input array is not modified:

    >>> d = np.array([0.5, 1.5])
    >>> _ = gaspari_cohn(d)
    >>> d
    array([0.5, 1.5])
    """
    if not np.all(distances >= 0):
        raise ValueError(f"Distances must be positive. Min: {np.min(distances)}")
    scaling_factor = np.zeros_like(distances)

    d2 = distances**2
    d3 = d2 * distances
    d4 = d3 * distances
    d5 = d4 * distances

    near = distances <= 1
    scaling_factor[near] = (
        -1 / 4 * d5[near] + 1 / 2 * d4[near] + 5 / 8 * d3[near] - 5 / 3 * d2[near] + 1
    )

    mid = (distances > 1) & (distances <= 2)
    scaling_factor[mid] = (
        1 / 12 * d5[mid]
        - 1 / 2 * d4[mid]
        + 5 / 8 * d3[mid]
        + 5 / 3 * d2[mid]
        - 5 * distances[mid]
        + 4
        - 2 / 3 / distances[mid]
    )

    # Clip to [0, 1] to suppress tiny negative artefacts from
    # floating-point arithmetic at the d=2 boundary.
    np.clip(scaling_factor, 0.0, 1.0, out=scaling_factor)
    return scaling_factor


def calc_rho_for_2d_grid_layer(
    *,
    nx: int,
    ny: int,
    xinc: float,
    yinc: float,
    obs_xpos: npt.NDArray[np.floating],
    obs_ypos: npt.NDArray[np.floating],
    obs_main_range: npt.NDArray[np.floating],
    obs_perp_range: npt.NDArray[np.floating],
    obs_anisotropy_angle: npt.NDArray[np.floating],
    right_handed_grid_indexing: bool = True,
) -> npt.NDArray[np.floating]:
    """Calculate elements of the localization matrix (rho) for a 2D grid layer.

    For each observation, the distance to every grid cell centre is computed
    and passed through the Gaspari--Cohn scaling function to obtain rho.
    Only lateral distances (horizontal distances in the (x, y) plane,
    ignoring depth) are considered, so every depth layer of a 3D grid
    shares the same cell centres and produces identical rho values;
    a single 2D calculation therefore covers all depth layers.
    All observation positions are given in the local grid coordinate
    system.

    Each observation n is described by its position
    (obs_xpos[n], obs_ypos[n]) and its localization ellipse
    (obs_main_range[n], obs_perp_range[n], obs_anisotropy_angle[n]).

    Grid cells are addressed by a flat index m that encodes the 2D cell
    index (i, j):
        m = j + i * ny                (left-handed grid indexing)
        m = (ny - j - 1) + i * ny    (right-handed grid indexing)

    The 2D distance from observation n to grid cell m = (i, j) is:
        d[m, n] = dist(
            (obs_xpos[n], obs_ypos[n]),
            ((i + 0.5) * xinc, (j + 0.5) * yinc),
        )

    where (i + 0.5) * xinc and (j + 0.5) * yinc are the x- and y-coordinates
    of the centre of grid cell (i, j) in the local coordinate system.

    The localization matrix element for cell m and observation n is:
        rho[m, n] = gaspari_cohn(d[m, n])

    Parameters
    ----------
    nx : int
        Number of grid cells in x-direction of local coordinate system.
    ny : int
        Number of grid cells in y-direction of local coordinate system.
    xinc : float
        Grid cell size in x-direction.
    yinc : float
        Grid cell size in y-direction.
    obs_xpos : np.ndarray
        Observations x coordinates in local coordinates.
    obs_ypos : np.ndarray
        Observations y coordinates in local coordinates.
    obs_main_range : np.ndarray
        Semi-axis length of the localization ellipse along the principal axis
        (the axis oriented at ``obs_anisotropy_angle`` relative to the
        local x-axis).
    obs_perp_range : np.ndarray
        Semi-axis length of the localization ellipse perpendicular to the
        principal axis. Equal to ``obs_main_range`` gives a circle; smaller
        gives an elongated ellipse.
    obs_anisotropy_angle : np.ndarray
        Orientation of the principal axis of the localization ellipse in
        degrees relative to the local x-axis. An angle of 0 aligns the
        principal axis with the x-axis of the local coordinate system.
    right_handed_grid_indexing : bool, optional
        Whether to use right-handed grid indexing. Default is True.

    Returns
    -------
    np.ndarray
        Localization matrix (rho) of shape ``(nx, ny, nobs)`` for one
        layer of a 3D grid or for a 2D surface grid.
    """
    if nx <= 0:
        raise ValueError("`nx` must be positive")
    if ny <= 0:
        raise ValueError("`ny` must be positive")

    if xinc <= 0.0:
        raise ValueError("`xinc` must be positive")
    if yinc <= 0.0:
        raise ValueError("`yinc` must be positive")

    # Center points of each grid cell in field parameter grid
    x_local = (np.arange(nx) + 0.5) * xinc
    if right_handed_grid_indexing:
        # y coordinate decreases from max to min
        y_local = (np.arange(ny - 1, -1, -1) + 0.5) * yinc
    else:
        # y coordinate increases from min to max
        y_local = (np.arange(ny) + 0.5) * yinc

    # Validate that all observation arrays are 1-D
    for name, arr in [
        ("obs_xpos", obs_xpos),
        ("obs_ypos", obs_ypos),
        ("obs_main_range", obs_main_range),
        ("obs_perp_range", obs_perp_range),
        ("obs_anisotropy_angle", obs_anisotropy_angle),
    ]:
        if arr.ndim != 1:
            raise ValueError(f"`{name}` must be 1-D, got {arr.ndim}-D")

    # Number of observations
    nobs = obs_xpos.shape[0]
    if obs_ypos.shape[0] != nobs:
        raise ValueError("Number of coordinates must match number of observations")
    if obs_anisotropy_angle.shape[0] != nobs:
        raise ValueError(
            "Number of ellipse orientation angles must match number of observations"
        )
    if obs_main_range.shape[0] != nobs:
        raise ValueError(
            "Number of ellipse main range values must match number of observations"
        )
    if obs_perp_range.shape[0] != nobs:
        raise ValueError(
            "Number of ellipse perpendicular range values must match number"
            " of observations"
        )
    if np.any(obs_main_range <= 0.0):
        raise ValueError("All main-range values for all observations must be positive")
    if np.any(obs_perp_range <= 0.0):
        raise ValueError(
            "All perpendicular-range values for all observations must be positive"
        )

    # Build flattened grid coordinates directly, avoiding intermediate
    # (nx, ny) arrays.
    # With "ij" indexing, meshgrid followed by flatten is equivalent to:
    #   x repeated ny times per x-value: [x0,x0,...,x1,x1,...,xn,xn,...]
    #   y tiled nx times:                [y0,y1,...,y0,y1,...,y0,y1,...]
    mesh_x_coord_flat = np.repeat(x_local, ny).reshape(-1, 1)  # (nx * ny, 1)
    mesh_y_coord_flat = np.tile(y_local, nx).reshape(-1, 1)  # (nx * ny, 1)

    # Observation coordinates and parameters
    obs_xpos = obs_xpos[np.newaxis, :]  # (1, nobs)
    obs_ypos = obs_ypos[np.newaxis, :]  # (1, nobs)
    obs_main_range = obs_main_range[np.newaxis, :]  # (1, nobs)
    obs_perp_range = obs_perp_range[np.newaxis, :]  # (1, nobs)
    obs_anisotropy_angle = obs_anisotropy_angle[np.newaxis, :]  # (1, nobs)

    # Compute displacement between grid points and observations
    dX = mesh_x_coord_flat - obs_xpos  # (nx * ny, nobs)
    dY = mesh_y_coord_flat - obs_ypos  # (nx * ny, nobs)

    # Compute rotation parameters
    rotation = np.deg2rad(obs_anisotropy_angle)
    cos_angle = np.cos(rotation)  # (1, nobs)
    sin_angle = np.sin(rotation)  # (1, nobs)

    # Rotate and scale displacements to local coordinate system defined by
    # the two half axes of the influence ellipse. First coordinate (local x)
    # is in direction defined by anisotropy angle and local y is
    # perpendicular to that.
    # Scale the distance by the ranges to get a normalized distance
    # (with value 1 at the edge of the ellipse)
    dX_ellipse = (dX * cos_angle + dY * sin_angle) / obs_main_range  # (nx * ny, nobs)
    dY_ellipse = (-dX * sin_angle + dY * cos_angle) / obs_perp_range  # (nx * ny, nobs)

    # Compute distances in the elliptical coordinate system
    distances = np.hypot(dX_ellipse, dY_ellipse)  # (nx * ny, nobs)
    # Apply the scaling function
    return gaspari_cohn(distances).reshape((nx, ny, nobs))


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-v"])
