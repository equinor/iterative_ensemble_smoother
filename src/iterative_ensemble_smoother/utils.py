from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import psutil

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


def masked_std(X, *, missing):
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

    # Need at least two observations per parameter
    if np.any(n_available < 2):
        msg = "One or several parameters have too few observations (need >=2)."
        raise ValueError(msg)

    X_masked = np.logical_not(missing) * X  # Set missing values to zero

    # Compute mean values, taking missing into account
    X_means = np.sum(X_masked, axis=1, keepdims=True) / n_available

    # Center the matrix
    X_centered = (X_masked - X_means) * np.logical_not(missing)

    return np.sqrt(
        np.sum(X_centered**2, axis=1, keepdims=True) / (n_available - 1)
    ).ravel()


def adjust_for_missing(
    X: npt.NDArray[np.double], *, missing: npt.NDArray[np.bool_]
) -> npt.NDArray[np.double]:
    """Removes missing values from X, such that the cross-covariance product

        center(X) @ center(Y).T / (N_e - 1)

    remains correct even in the presence of missing parameters in some
    ensemble members (realizations).

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

    # Need at least two observations per parameter
    if np.any(n_available < 2):
        msg = "One or several parameters have too few observations (need >=2)."
        raise ValueError(msg)

    X_masked = np.logical_not(missing) * X  # Set missing values to zero
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
    return X_centered * np.logical_not(missing)


def steplength_exponential(
    iteration: int,
    min_steplength: float = 0.3,
    max_steplength: float = 0.6,
    halflife: float = 1.5,
) -> float:
    r"""
    Compute a suitable step length for the update step.

    This is an implementation of Eq. (49), which calculates a suitable step length for
    the update step, from the book: \"Formulating the history matching problem with
    consistent error statistics", written by :cite:t:`evensen2021formulating`.

    Examples
    --------
    >>> [steplength_exponential(i) for i in [1, 2, 3, 4]]
    [0.6, 0.48898815748423097, 0.41905507889761495, 0.375]
    >>> [steplength_exponential(i, 0.0, 1.0, 1.0) for i in [1, 2, 3, 4]]
    [1.0, 0.5, 0.25, 0.125]
    >>> [steplength_exponential(i, 0.0, 1.0, 0.5) for i in [1, 2, 3, 4]]
    [1.0, 0.25, 0.0625, 0.015625]
    >>> [steplength_exponential(i, 0.5, 1.0, 1.0) for i in [1, 2, 3]]
    [1.0, 0.75, 0.625]

    """
    assert max_steplength > min_steplength
    assert iteration >= 1
    assert halflife > 0

    delta = max_steplength - min_steplength
    exponent = -(iteration - 1) / halflife
    return min_steplength + delta * 2**exponent


def _validate_inputs(
    parameters: npt.NDArray[np.double],
    covariance: npt.NDArray[np.double],
    observations: npt.NDArray[np.double],
) -> None:
    # Check types
    inputs = [parameters, covariance, observations]
    names = ["parameters", "covariances", "observations"]
    for input_, name in zip(inputs, names):
        if not isinstance(input_, np.ndarray):
            raise TypeError(f"Argument '{name}' must be numpy nd.array")

    assert covariance.ndim in (1, 2)
    assert parameters.ndim == 2

    assert covariance.shape[0] == observations.shape[0]
    assert covariance.shape[0] == observations.shape[0]


def sample_mvnormal(
    *,
    C_dd_cholesky: npt.NDArray[np.double],
    rng: np.random._generator.Generator,
    size: int,
) -> npt.NDArray[np.double]:
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


def localization_scaling_function(
    distances: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate scaling factor to be used as values in RHO matrix.

    Calculate scaling factor to be used as values in RHO matrix in distance-based
    localization. The scaling function implements the commonly used function
    published by Gaspari and Cohn. For input normalized distance >= 2,
    the value will be 0.

    Parameters
    ----------
    distances : np.ndarray
        Vector of values for normalized distances.

    Returns
    -------
    np.ndarray
        Values of scaling factors for each value of input distance.
    """
    # "gaspari-cohn"
    # Commonly used in distance-based localization
    # Is exact 0 for normalized distance > 2.
    scaling_factor = distances
    d2 = distances**2
    d3 = d2 * distances
    d4 = d3 * distances
    d5 = d4 * distances
    s = -1 / 4 * d5 + 1 / 2 * d4 + 5 / 8 * d3 - 5 / 3 * d2 + 1
    scaling_factor[distances <= 1] = s[distances <= 1]
    s = (
        1 / 12 * d5
        - 1 / 2 * d4
        + 5 / 8 * d3
        + 5 / 3 * d2
        - 5 * distances
        + 4
        - 2 / 3 * 1 / distances
    )
    scaling_factor[(distances > 1) & (distances <= 2)] = s[
        (distances > 1) & (distances <= 2)
    ]
    scaling_factor[distances > 2] = 0.0

    return scaling_factor


def calc_rho_for_2d_grid_layer(
    nx: int,
    ny: int,
    xinc: float,
    yinc: float,
    obs_xpos: npt.NDArray[np.float64],
    obs_ypos: npt.NDArray[np.float64],
    obs_main_range: npt.NDArray[np.float64],
    obs_perp_range: npt.NDArray[np.float64],
    obs_anisotropy_angle: npt.NDArray[np.float64],
    right_handed_grid_indexing: bool = True,
) -> npt.NDArray[np.float64]:
    """Calculate scaling values (RHO matrix elements) for a set of observations
    with associated localization ellipse. The method will first
    calculate the distances from each observation position to each grid cell
    center point of all grid cells for a 2D grid.
    The localization method will only consider lateral distances, and it is
    therefore sufficient to calculate the distances in 2D.
    All input observation positions are in the local grid coordinate system
    to simplify the calculation of the distances.

    The position: xpos[n], ypos[n] and
    localization ellipse defined by obs_main_range[n],obs_perp_range[n],
    obs_anisotropy_angle[n]) refers to observation[n].

    The distance between an observation with index n and a grid cell (i,j) is
    d[m,n] = dist((xpos_obs[n],ypos_obs[n]),(xpos_field[i,j],ypos_field[i,j]))

    RHO[[m,n] = scaling(d)
    where m = j + i * ny for left-handed grid index origo and
          m = (ny - j - 1) + i * ny for right-handed grid index origo
    Note that since d[m,n] does only depend on observation index n and
    grid cell index (i,j). The values for RHO is
    calculated for the combination ((i,j), n) and this covers
    one grid layer in ertbox grid or a 2D surface grid.

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
        Localization ellipse first range.
    obs_perp_range : np.ndarray
        Localization ellipse second range.
    obs_anisotropy_angle : np.ndarray
        Localization ellipse orientation relative to local coordinate system in degrees.
    right_handed_grid_indexing : bool, optional
        Whether to use right-handed grid indexing. Default is True.

    Returns
    -------
    np.ndarray
        Rho matrix values for one layer of the 3D ertbox grid or for a 2D surface grid.
    """
    # Center points of each grid cell in field parameter grid
    x_local = (np.arange(nx, dtype=np.float64) + 0.5) * xinc
    if right_handed_grid_indexing:
        # y coordinate decreases from max to min
        y_local = (np.arange(ny - 1, -1, -1, dtype=np.float64) + 0.5) * yinc
    else:
        # y coordinate increases from min to max
        y_local = (np.arange(ny, dtype=np.float64) + 0.5) * yinc
    mesh_x_coord, mesh_y_coord = np.meshgrid(x_local, y_local, indexing="ij")

    # Number of observations
    nobs = len(obs_xpos)
    assert nobs == len(obs_ypos), (
        "Number of coordinates must match number of observations"
    )
    assert nobs == len(obs_anisotropy_angle), (
        "Number of ellipse orientation angles must match number of observations"
    )
    assert nobs == len(obs_main_range), (
        "Number of ellipse main range values must match number of observations"
    )
    assert nobs == len(obs_perp_range), (
        "Number of ellipse second range values must match number of observations"
    )
    assert np.all(obs_main_range > 0.0), (
        "All range values for all observations must be positive"
    )
    assert np.all(obs_perp_range > 0.0), (
        "All range values for all observations must be positive"
    )

    # Expand grid coordinates to match observations
    mesh_x_coord_flat = mesh_x_coord.flatten()[:, np.newaxis]  # (nx * ny, 1)
    mesh_y_coord_flat = mesh_y_coord.flatten()[:, np.newaxis]  # (nx * ny, 1)

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

    # Rotate and scale displacements to local coordinate system defined
    # by the two half axes of the influence ellipse. First coordinate (local x) is in
    # direction defined by anisotropy angle and local y is perpendicular to that.
    # Scale the distance by the ranges to get a normalized distance
    # (with value 1 at the edge of the ellipse)
    dX_ellipse = (dX * cos_angle + dY * sin_angle) / obs_main_range  # (nx * ny, nobs)
    dY_ellipse = (-dX * sin_angle + dY * cos_angle) / obs_perp_range  # (nx * ny, nobs)

    # Compute distances in the elliptical coordinate system
    distances = np.hypot(dX_ellipse, dY_ellipse)  # (nx * ny, nobs)
    # Apply the scaling function
    return localization_scaling_function(distances).reshape((nx, ny, nobs))


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-v"])
