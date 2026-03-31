"""
Script to test 1D gaussian field updates with ESMDA with various localizations
and implementations of the algorithm.
"""

import time

import gaussianfft as grf  # noqa
import numpy as np
from numpy import typing as npt

from iterative_ensemble_smoother import ESMDA
from iterative_ensemble_smoother.esmda_localized import LocalizedESMDA
from iterative_ensemble_smoother.experimental import (
    DistanceESMDA,
)
from iterative_ensemble_smoother.utils import (
    #    calc_max_number_of_layers_per_batch_for_distance_localization,
    calc_rho_for_2d_grid_layer,
)


def calculate_rho_1d(
    N_m: int, obs_index: int, localization_radius: float, xinc: float = 1.0
) -> npt.NDArray[np.float64]:
    model_grid = np.arange(N_m) * xinc
    distances = np.abs(model_grid - obs_index * xinc)
    # Ordinary definition of range,
    # rho is approximately 0.05 at localization_radius
    return np.exp(-3.0 * (distances / localization_radius) ** 2).reshape(-1, 1)


def calculate_rho_2d(
    Nx: int, Ny: int, x_obs: int, y_obs: int, localization_radius: float
) -> npt.NDArray[np.float64]:
    xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny))
    distances_2d = np.sqrt((xx - x_obs) ** 2 + (yy - y_obs) ** 2)
    distances = distances_2d.flatten()
    # Ordinary definition of range,
    # rho is approximately 0.05 at localization_radius
    return np.exp(-3.0 * (distances / localization_radius) ** 2).reshape(-1, 1)


def calculate_rho_3d(
    Nx: int,
    Ny: int,
    Nz: int,
    x_obs: int,
    y_obs: int,
    z_obs: int,
    localization_radius: float,
) -> npt.NDArray[np.float64]:
    # Create 3D coordinate grids
    zz, yy, xx = np.meshgrid(np.arange(Nz), np.arange(Ny), np.arange(Nx), indexing="ij")
    # Calculate 3D Euclidean distance from every point to the observation
    distances_3d = np.sqrt((xx - x_obs) ** 2 + (yy - y_obs) ** 2 + (zz - z_obs) ** 2)
    distances = distances_3d.flatten()
    # Ordinary definition of range,
    # rho is approximately 0.05 at localization_radius
    return np.exp(-3.0 * (distances / localization_radius) ** 2).reshape(-1, 1)


def draw_3D_field(
    mean: float,
    stdev: float,
    xinc: float,
    yinc: float,
    zinc: float,
    nx: int,
    ny: int,
    nz: int,
    nreal: int,
    main_corr_range: float,
    perp_corr_range: float,
    vert_corr_range: float,
    start_seed: int = 42,
    corr_func_name: str = "matern32",
    power: float = 1.9,
    azimuth: float = 0.0,
    dip: float = 0.0,
    write_progress: bool = False,
    use_4_byte_float: bool = False,
) -> npt.NDArray[np.float64]:
    # Draw prior ensemble of 3D field. Ensemble size is nreal
    # Returns prior ensemble drawn and covariance matrix

    # Initialize start seed

    grf.seed(start_seed)

    # Define spatial correlation function for gaussian fields
    # to be simulated.
    if corr_func_name == "general_exponential":
        variogram = grf.variogram(
            corr_func_name,
            main_corr_range,
            perp_corr_range,
            vert_corr_range,
            azimuth,
            dip,
            power,
        )
    else:
        variogram = grf.variogram(
            corr_func_name,
            main_corr_range,
            perp_corr_range,
            vert_corr_range,
            azimuth,
            dip,
        )

    nparam = nx * ny * nz
    if use_4_byte_float:
        X_prior = np.zeros((nparam, nreal), dtype=np.float32)
    else:
        X_prior = np.zeros((nparam, nreal), dtype=np.float64)
    # Flatten 3D array in F order
    for real_number in range(nreal):
        if write_progress and real_number % 10 == 0:
            print(f"  Sim real nr: {real_number}")
        field_values = grf.simulate(variogram, nx, xinc, ny, yinc, nz, zinc)
        X_prior[:, real_number] = (
            field_values.reshape((nx, ny, nz), order="F").flatten(order="C") * stdev
            + mean
        )

    return X_prior


def draw_random_obs(rng, nobs, nx, ny, nz, obs_err_std):
    nparam = nx * ny * nz
    assert nobs < nparam
    # Define a grid resolution with specified size
    xinc = 50.0
    yinc = 50.0
    zinc = 1.0
    right_handed_grid_indexing = True
    # Draw some observation values (Use same seed every time)
    observations = rng.normal(loc=0.5, scale=0.05, size=nobs)

    # Choose some observation values, errors and positions
    # Draw some position of the observations, ensure no observations at same position
    # since the response values are equal to field values (only one response variable).
    # Multiple response values can have same position.

    # Draw i index, j_index, k_index for grid cell to be used as observed.
    if nobs == 1:
        obs_xpos = np.array([(nx / 2) * xinc])
        obs_ypos = np.array([(ny / 2) * yinc])
        obs_zpos = np.array([(nz / 2) * zinc])
        i_indices = np.array([int(nx / 2)])
        j_indices = np.array([int(ny / 2)])
        k_indices = np.array([int(nz / 2)])
        unique_obs_indices = k_indices + j_indices * nz + i_indices * nz * ny
    else:
        if nobs > nparam:
            raise ValueError(
                "Cannot draw more observations than number of parameters in the field"
            )
        unique_obs_indices = rng.choice(range(nparam), size=nobs, replace=False)
        unique_obs_indices = np.sort(unique_obs_indices)
        i_indices = (unique_obs_indices // (nz * ny)).astype(int)
        j_indices = ((unique_obs_indices % (nz * ny)) // nz).astype(int)
        k_indices = (unique_obs_indices % nz).astype(int)
        if right_handed_grid_indexing:
            # Right-handed grid indexing
            obs_ypos = ((ny - j_indices - 1) + 0.5) * yinc
        else:
            obs_ypos = (j_indices + 0.5) * yinc

        obs_xpos = (i_indices + 0.5) * xinc
        obs_zpos = (k_indices + 0.5) * zinc

    # Set observation error
    obs_var_vector = np.zeros(nobs, dtype=np.float64)
    obs_var_vector[:] = obs_err_std**2
    return (
        observations,
        obs_var_vector,
        obs_xpos,
        obs_ypos,
        obs_zpos,
        i_indices,
        j_indices,
        k_indices,
        unique_obs_indices,
    )


def test_update_params_3D(
    snapshot,
    nx: int,
    ny: int,
    nz: int,
    nobs: int,
    nreal: int,
    field_mean: float,
    field_std: float,
    rel_corr_length: float,
    obs_err_std: float,
    rel_localization_range: float,
    case_with_some_zero_variance_field_params: bool,
    seed: int,
):
    # The field parameter is assumed to belong to a box grid
    # with specified nx,ny,nz and grid increments xinc,yinc,zinc
    # The observation position is assumed to be within the same coordinate
    # system as the grid. The grid cell center point coordinates are
    # x[i,j,k] = xinc * (i + 0.5)  i=0,.. nx-1
    # y[i,j,k] = yinc * (j + 0.5)  j=0,.. ny-1
    # z[i,j,k] = zinc * (k + 0.5)  k=0,.. nz-1
    # Z coordinate is not used when calculating RHO matrix, but is used here
    # to define observation values.
    xinc = 50.0
    yinc = 50.0
    zinc = 1.0
    xlength = xinc * nx
    ylength = yinc * ny
    zlength = zinc * nz
    corr_range = max(xlength, ylength) * rel_corr_length
    vert_range = zlength * rel_corr_length
    fraction_of_field_values_with_zero_variance = 0.1

    # Draw prior gaussian fields with spatial correlations
    X_prior = draw_3D_field(
        field_mean,
        field_std,
        xinc,
        yinc,
        zinc,
        nx,
        ny,
        nz,
        nreal,
        corr_range,
        corr_range,
        vert_range,
        seed,
        corr_func_name="gaussian",
    )

    X_prior_3D = X_prior.reshape((nx, ny, nz, nreal))
    rng = np.random.default_rng(seed)
    nparam = nx * ny * nz
    (
        observations,
        obs_var_vector,
        obs_xpos,
        obs_ypos,
        obs_zpos,
        i_indices,
        j_indices,
        k_indices,
        unique_obs_indices,
    ) = draw_random_obs(rng, nobs, nx, ny, nz, obs_err_std)

    # Choose localization range around each obs
    typical_field_size = min(xlength, ylength)
    obs_main_range = np.zeros(nobs, dtype=np.float64)
    obs_main_range[:] = typical_field_size * rel_localization_range
    obs_perp_range = np.zeros(nobs, dtype=np.float64)
    obs_perp_range[:] = typical_field_size * rel_localization_range
    obs_anisotropy_angle = np.zeros(nobs, dtype=np.float64)

    # Calculate rho_for one layer
    rho_2D = calc_rho_for_2d_grid_layer(
        nx,
        ny,
        xinc,
        yinc,
        obs_xpos,
        obs_ypos,
        obs_main_range,
        obs_perp_range,
        obs_anisotropy_angle,
        right_handed_grid_indexing=True,
    )
    # Set responses for each observation equal to the X_prior for simplicity
    # (Forward model is identity Y = X in observation points + small random noise)
    # Note cannot have observations with response with 0 variance
    add_response_variability = rng.normal(loc=0, scale=0.01, size=(nreal, nobs))
    Y = X_prior_3D[i_indices, j_indices, k_indices, :] + add_response_variability.T

    if case_with_some_zero_variance_field_params:
        # Set same field value in all realizations for selected grid cells
        # They should not be updated since the ensemble variance is 0 for those
        # values. The selected grid cells must not be selected as observed.
        nconst_values = int(
            fraction_of_field_values_with_zero_variance * nparam
        )  # Choose a portion of field values to be constant
        unique_indices_const = rng.choice(
            range(nparam), size=nconst_values, replace=False
        )
        print(f"Number of selected grid indices: {len(unique_indices_const)}")
        # Reject indices corresponding to observed grid cells
        unique_indices_const = np.setdiff1d(unique_indices_const, unique_obs_indices)
        # Usually inactive cell values can be set to 0 for all realizations,
        # but choose something else here for visualization purpose
        X_prior[unique_indices_const, :] = 10.0
        print(f"Number of field values with 0 variance: {len(unique_indices_const)}")

    # Initialize Distance based localization object
    alpha = np.array([1.0])
    dl_smoother = DistanceESMDA(
        covariance=obs_var_vector, observations=observations, alpha=alpha, seed=rng
    )

    # Call the function to be tested here
    # Note that no field values with 0 variance is removed
    # from the update calculation in this function,
    # but that is ok since there will be no change of
    # field parameters with 0 variance anyway.
    X_post = dl_smoother.update_params(
        X_prior,
        Y,
        rho_2D,
        nz,
    )
    X_post_3D = X_post.reshape((nx, ny, nz, nreal))
    # Mean and stdev over ensemble of field parameters
    #    X_prior_mean_3D = X_prior_3D.mean(axis=3)
    X_post_mean_3D = X_post_3D.mean(axis=3)
    #    X_prior_stdev_3D = X_prior_3D.std(axis=3)
    X_post_stdev_3D = X_post_3D.std(axis=3)
    # Difference between post and prior mean and stdev
    #    X_diff_mean_3D = X_post_mean_3D - X_prior_mean_3D
    #    X_diff_stdev_3D = X_post_stdev_3D - X_prior_stdev_3D
    print(f"X_post_mean_3D.shape = {X_post_mean_3D.shape}")
    print(f"X_post_stdev_3D.shape = {X_post_stdev_3D.shape}")


def draw_1D_field(
    mean: float,
    stdev: float,
    xinc: float,
    corr_func_name: str,
    corr_range: float,
    nparam: int,
    nsim_param: int,
    nreal: int,
    nu: float = 1.5,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # Draw prior ensemble of 1D field with nparam. Ensemble size is nreal
    # Returns prior ensemble drawn and covariance matrix

    variance = stdev**2
    assert corr_func_name in ["exponential", "general_exponential", "gaussian", "spherical","matern32","matern72"]
    # Generate distance matrix
    x_coords = (np.arange(nparam) + 0.5) * xinc
    distances = np.abs(x_coords[:, None] - x_coords[None, :]) / corr_range
    print(f"corr_range:  {corr_range=}")
    print(f"distances:  {distances=}")
    # Compute covariance matrix based on correlation function
    if corr_func_name == "exponential":
        cov_matrix = variance * np.exp(-3.0 * distances)
    elif corr_func_name == "gaussian":
        cov_matrix = variance * np.exp(-3.0 * distances**2)
    elif corr_func_name == "general_exponential":
        cov_matrix = variance * np.exp(-3.0 * np.power(distances, nu))
    else:
        raise ValueError("Unsupported correlation function")

    # Create mean array
#    mean_values = np.full((nparam,), mean, dtype=np.float64)

    # Generate random fields with multivariate normal distribution
#    fields = rng.multivariate_normal(mean_values, cov_matrix, size=nreal).T


    # Define spatial correlation function for gaussian fields
    # to be simulated.
    if corr_func_name == "general_exponential":
        variogram = grf.variogram(
            corr_func_name,
            corr_range,
            power=nu,
        )
    else:
        variogram = grf.variogram(
            corr_func_name,
            corr_range,
        )
    assert nsim_param >= nparam
    start_index = int((nsim_param -nparam)/2)
    end_index = int(start_index + nparam)
    print(f"start_index: {start_index}  end_index: {end_index}")
    X_prior = np.zeros((nparam, nreal), dtype=np.double)
    for real_number in range(nreal):
        if real_number % 2000 == 0:
            print(f"Real number: {real_number}")
        field_values = grf.simulate(variogram, nsim_param, xinc)
        X_prior[:, real_number] = field_values[start_index:end_index] * stdev + mean
    

    return X_prior, cov_matrix


def generate_alpha_vector(nalpha):
    """Generate alpha vector with length nalpha
    where sum of the inverse of alpha is 1.
    """
    # Define alpha[k] = [2**(m-1) + 2**(m-2) + ... + 2**(0)] / 2**(k)
    # such that alpha[k] = (2**m -1)/(2**k)
    # Then sum (1/alpha[k]) for k=0,1,2,..m-1  is 1.0
    # The alpha values are reduced with a factor 1/2 for each iteration
    # Special case: m = 3 gives alpha_vector = [7, 3.5, 1.75]
    return (2**nalpha - 1) / 2 ** np.arange(nalpha)


def predicted_mean_field_values(
    obs_vector: npt.NDArray[np.float64],
    obs_index_vector: npt.NDArray[np.int32],
    field_cov_matrix: npt.NDArray[np.float64],
    obs_err_covariance: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculates simple kriging estimate of expected value. This is used
    to check that DL-ESMDA and ordinary ESMDA are close to the kriging estimate
    when number of realizations becomes large.
    """
    # Extract the observed covariance matrix
    # (corresponding to the response covariance matrix)
    # Here simple kriging with observation error is used, and
    # response is equal to the simulated field values in
    # position of the observations.
    obs_cov_matrix = field_cov_matrix[np.ix_(obs_index_vector, obs_index_vector)]
    cov_mat = obs_cov_matrix + obs_err_covariance
    # Compute the inverse covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_mat)

    # Extract the covariance matrix between the observations and the full field
    field_obs_cov = field_cov_matrix[obs_index_vector, :]

    # Predicted mean values (broadcast calculation for all parameters at once)
    return field_obs_cov.T @ inv_cov_matrix @ obs_vector


# def calc_obs_error_cov(
#     obs_error_std: npt.NDArray[np.float64],
#     xpos: npt.NDArray[np.float64],
#     corr_func_name: str,
#     obs_corr_range: float,
#     nu: float = 1.5,
# ) -> npt.NDArray[np.float64]:
#     nobs = obs_error_std.shape[0]
#     cov = np.zeros((nobs, nobs), dtype=np.float64)
#     if obs_corr_range == 0:
#         # Diagonal covariance matrix
#         np.fill_diagonal(cov, obs_error_std**2)
#     else:
#         for i in range(nobs):
#             x1 = xpos[i]
#             sigma1 = obs_error_std[i]
#             for j in range(nobs):
#                 sigma2 = obs_error_std[j]
#                 x2 = xpos[j]
#                 d = np.abs(x1 - x2) / obs_corr_range
#                 if corr_func_name == "exponential":
#                     cov[i, j] = sigma1 * sigma2 * math.exp(-3.0 * d)
#                 elif corr_func_name == "gaussian":
#                     cov[i, j] = sigma1 * sigma2 * math.exp(-3.0 * d**2)
#                 else:
#                     cov[i, j] = sigma1 * sigma2 * math.exp(-3.0 * np.power(d, nu))
#     return cov


def calc_obs_error_cov(
    obs_error_std: npt.NDArray[np.float64],
    xpos: npt.NDArray[np.float64],
    corr_func_name: str,
    obs_corr_range: float,
    nu: float = 1.5,
) -> npt.NDArray[np.float64]:
    nobs = obs_error_std.shape[0]


    assert corr_func_name in ["gaussian", "exponential","general_exponential"]
    if obs_corr_range == 0:
        # Diagonal covariance matrix
        cov = np.zeros((nobs, nobs), dtype=np.float64)
        np.fill_diagonal(cov, obs_error_std**2)
    else:
        # Precompute pairwise distances divided by obs_corr_range
        pairwise_distances = np.abs(xpos[:, None] - xpos[None, :]) / obs_corr_range

        # Precompute the outer product of obs_error_std
        outer_std = np.outer(obs_error_std, obs_error_std)

        if corr_func_name == "exponential":
            corr = np.exp(-3.0 * pairwise_distances)
        elif corr_func_name == "gaussian":
            corr = np.exp(-3.0 * pairwise_distances**2)
        else:
            corr = np.exp(-3.0 * np.power(pairwise_distances, nu))

        # Element-wise multiplication to get covariance matrix
        cov = outer_std * corr
    print(f"Obs covariance: {cov=}")
    return cov


def draw_prior_ensemble(
        seed: int,
        mean:float,
        stdev: float,
        xinc:float,
        field_corr_func_name: str,
        nparam: int,
        nreal: int,
        field_corr_rel_range: float,
        nu: float,
):
    # Draw observations from same distriution as the gaussian field
    length = xinc * nparam
    field_corr_range = length * field_corr_rel_range
    grf.seed(seed)  # Seed is global to the class 
    nsim_param = int(1.5*nparam)
    print("Simulate Gaussian fields")
    X_prior, field_cov_matrix = draw_1D_field(
        mean,
        stdev,
        xinc,
        field_corr_func_name,
        field_corr_range,
        nparam,
        nsim_param,
        nreal,
        nu,
    )
    return X_prior, field_cov_matrix

def draw_obs(
        mean:float,
        stdev: float,
        xinc:float,
        field_corr_func_name: str,
        nparam: int,
        nobs: int,
        field_corr_rel_range: float,
        nu: float,
        draw_position: bool = False,
):
    # Draw observations from same distriution as the gaussian field
    length = xinc * nparam
    field_corr_range = length * field_corr_rel_range

    print(f"Draw observations")
    nsim_param = int(1.5*nparam)
    X_obs, _ = draw_1D_field(
        mean,
        stdev,
        xinc,
        field_corr_func_name,
        field_corr_range,
        nparam,
        nsim_param,
        1,
        nu,
    )
    
    # Draw position of observations
    if draw_position:
        obs_index_vector = np.random.choice(nparam, size=nobs)
    else:
        obs_index_vector = np.linspace(1, nparam - 1, nobs, dtype=np.int32)
    obs_vector = X_obs[obs_index_vector, 0]
    print(f"nobs: {obs_vector.shape[0]}")
    return  obs_vector, obs_index_vector

def calculate_obs_covariance(
        nparam: int,
        nobs: int,
        xinc:float,
        obs_corr_func_name: str,
        obs_index_vector: npt.NDArray[np.int32],
        obs_corr_rel_range:float,
        obs_err_std: float,
        nu: float,
):
    length = xinc * nparam
    obs_corr_range = length * obs_corr_rel_range
    obs_std_vector = obs_err_std * np.ones(nobs, dtype=np.float64)
    xpos = (obs_index_vector) * xinc

    print("Calculate covariance matrix for observations")
    C_D = calc_obs_error_cov(obs_std_vector, xpos, obs_corr_func_name, obs_corr_range, nu)
    return  C_D


def test_distance_based_localization_on_1D_corr_field(
    X_prior: npt.NDArray[np.double],
    field_cov_matrix: npt.NDArray[np.double],
    xinc: float,
    obs_vector: npt.NDArray[np.double],
    obs_index_vector: npt.NDArray[np.int32],
    C_D: npt.NDArray[np.double],
    relative_localization_radius: float,
    seed: int,
    use_localization: bool,
    case: str,
    use_esmda: bool = False,
    use_esmda_dist: bool = False,
    use_esmda_local: bool = True,
    calc_simple_kriging:bool = False,
) -> None:
    write_csv_files = True
    nparam =X_prior.shape[0]
    nreal = X_prior.shape[1]
    nobs = obs_vector.shape[0]
    localization_radius = nparam * xinc * relative_localization_radius


    alpha = np.array([1.0])
    # Calculate theoretical posterior mean
    # assuming prior mean equal 0.
    # Assume forward model is identity  X = g(X)
    # Use simple kriging estimate for theoretical posterior mean
    if calc_simple_kriging:
        print("Calculate Simple Kriging estimate")
        mean_predicted_field = predicted_mean_field_values(
            obs_vector, obs_index_vector, field_cov_matrix, C_D
        )

    # Predict observations `Y` using the identity model `g(x) = x`
    Y = X_prior[obs_index_vector, :]

    # Using a simple Gaussian decay for rho.

    print("Calculate RHO")
    rho_matrix = np.zeros((nparam, nobs), dtype=np.float64)
    if use_localization:
        for i in range(nobs):
            rho = calculate_rho_1d(
                nparam,
                obs_index_vector[i],
                localization_radius,
                xinc=xinc,
            )
            rho_matrix[:, i] = rho[:, 0]
    else:
        rho_matrix[:, :] = 1.0

    # Define Schur-product
    def localization_callback(
        K: npt.NDArray[np.double],
    ) -> npt.NDArray[np.double]:
        assert K.shape == rho_matrix.shape
        return K * rho_matrix

    if use_esmda:
        start_time = time.time()
        print("Use ESMDA")
        esmda = ESMDA(covariance=C_D, observations=obs_vector, alpha=alpha, seed=seed)
        X_posterior_global = esmda.assimilate(X=X_prior, Y=Y)
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time:.3f}")

        # Mean and stdev of ensemble of posterior field
        print("Calculate mean and stdev of ensembles")
        X_post_mean_global = X_posterior_global.mean(axis=1)
        X_post_std_global = X_posterior_global.std(axis=1)

    if use_esmda_local:
        start_time = time.time()
        print("Use LocalizedESMDA")
        esmda_local = LocalizedESMDA(
            covariance=C_D,
            observations=obs_vector,
            alpha=alpha,
            seed=seed,
        )

        esmda_local.prepare_assimilation(Y=Y, truncation=0.99999)
        X_posterior_local = esmda_local.assimilate_batch(
            X=X_prior,
            localization_callback=localization_callback,
        )
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time:.3f}")

        # Mean and stdev of ensemble of posterior field
        print("Calculate mean and stdev of ensembles")
        X_post_mean_local = X_posterior_local.mean(axis=1)
        X_post_std_local = X_posterior_local.std(axis=1)

    if use_esmda_dist:
        start_time = time.time()
        print("Use DistanceESMDA")
        esmda_distance = DistanceESMDA(
            covariance=C_D, observations=obs_vector, alpha=alpha, seed=seed
        )
        X_posterior = esmda_distance.assimilate(
            X=X_prior, Y=Y, rho=rho_matrix, truncation=0.999
        )
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time:.3f}")

        # Mean and stdev of ensemble of posterior field
        print("Calculate mean and stdev of ensembles")
        X_post_mean_dist = X_posterior.mean(axis=1)
        X_post_std_dist = X_posterior.std(axis=1)

    # Difference with theoretical mean based on simple kriging is calculated.
    # Note that ESMDA in this test should approach the kriging estimate
    # which is the theoretical limit of the posterior mean when nreal -> infinity
    # For DL-ESMDA only when localization range is much larger than the spatial
    # correlation length one can expect the posterior mean approaches the
    # simple kriging estimate in this test. But DL-ESMDA will be closer to
    # the simple kriging estimate for practical purposes where nreal is not large.
    # The tolerance specified in this test is specified such that the estimated
    # standard deviation of the differences between DL-ESMDA and simple kriging
    # estimate is less than the tolerances. Note that when nreal increases, the
    # tolerances can be reduced, but will never approach 0 as long as
    # localization range is finite. However, for ESMDA, the tolerance
    # could in theory approach 0 when nreal approach infinity2

    # X_diff_dist_sk = X_post_mean_dist - mean_predicted_field
    # X_diff_global_sk = X_post_mean_global - mean_predicted_field
    # X_diff_local_sk = X_post_mean_local - mean_predicted_field

    # X_diff_dist_abs = np.abs(X_diff_dist_sk)
    # X_diff_global_abs = np.abs(X_diff_global_sk)
    # X_diff_local_abs = np.abs(X_diff_local_sk)

    # est_std_diff_global = X_diff_global_abs.std()
    # est_std_diff_dist = X_diff_dist_abs.std()
    # est_std_diff_local = X_diff_local_abs.std()

    # X_dist_diff_max = np.max(X_diff_dist_abs)
    # X_local_diff_max = np.max(X_diff_local_abs)
    # X_global_diff_max = np.max(X_diff_global_abs)

    print(f"Number of real: {nreal}")
    # print(f"Max difference using DL ESMDA : {X_dist_diff_max}")
    # print(f"Max difference using LOCAL ESMDA : {X_local_diff_max}")
    # print(f"Max difference using ordinary ESMDA: {X_global_diff_max}")
    # print(
    #     "Estimated std of difference between DL and simple kriging: "
    #     f"{est_std_diff_dist}"
    # )
    # print(
    #     "Estimated std of difference between LOCAL and simple kriging: "
    #     f"{est_std_diff_local}"
    # )
    # print(
    #     "Estimated std of difference between Global and simple kriging: "
    #     f"{est_std_diff_global}"
    # )
    if write_csv_files:
        # To verify convergence to simple kriging estimate,
        # various csv files are created
        filename = "obs_" + str(nobs) + ".csv"
        print(f"Write file: {filename}")
        np.savetxt(filename, obs_vector, delimiter=",", fmt="%.6f")
        print(f"obs: {obs_vector}")
        filename = "obs_index_" + str(nobs) + ".csv"
        print(f"Write file: {filename}")
        np.savetxt(filename, obs_index_vector, delimiter=",", fmt="%d")
        print(f"obs_index: {obs_index_vector}")
        if use_esmda_dist:
            filename = "X_post_mean_dist_" + case + ".csv"
            print(f"Write file: {filename}")
            np.savetxt(filename, X_post_mean_dist, delimiter=",", fmt="%.6f")

            filename = "X_post_std_dist_" + case + ".csv"
            print(f"Write file: {filename}")
            np.savetxt(filename, X_post_std_dist, delimiter=",", fmt="%.6f")
        if use_esmda_local:
            filename = "X_post_mean_local_" + case + ".csv"
            print(f"Write file: {filename}")
            np.savetxt(filename, X_post_mean_local, delimiter=",", fmt="%.6f")

            filename = "X_post_std_local_" + case + ".csv"
            print(f"Write file: {filename}")
            np.savetxt(filename, X_post_std_local, delimiter=",", fmt="%.6f")
        if use_esmda:
            filename = "X_post_mean_global_" + case + ".csv"
            print(f"Write file: {filename}")
            np.savetxt(filename, X_post_mean_global, delimiter=",", fmt="%.6f")

            filename = "X_post_std_global_" + case + ".csv"
            print(f"Write file: {filename}")
            np.savetxt(filename, X_post_std_global, delimiter=",", fmt="%.6f")
        if calc_simple_kriging:
            filename = "X_post_SK_" + case + ".csv"
            print(f"Write file: {filename}")
            np.savetxt(filename, mean_predicted_field, delimiter=",", fmt="%.6f")

        # Mean and stdev of ensemble of prior field
        X_prior_mean = X_prior.mean(axis=1)
        filename = "X_prior_mean_" + case + ".csv"
        print(f"Write file: {filename}")
        np.savetxt(filename, X_prior_mean, delimiter=",", fmt="%.6f")
        print(f"Max abs of X_prior_mean: {np.max(np.abs(X_prior_mean))}")

        # # Diff between prior and posterior
        # X_diff_dist_prior_post_mean = X_post_mean_dist - X_prior_mean
        # filename = "X_diff_dist_prior_post_mean_" + case + ".csv"
        # print(f"Write file: {filename}")
        # np.savetxt(filename, X_diff_dist_prior_post_mean, delimiter=",", fmt="%.6f")

        # X_diff_local_prior_post_mean = X_post_mean_local - X_prior_mean
        # filename = "X_diff_local_prior_post_mean_" + case + ".csv"
        # print(f"Write file: {filename}")
        # np.savetxt(filename, X_diff_local_prior_post_mean, delimiter=",", fmt="%.6f")

        # X_diff_prior_post_mean_global = X_post_mean_global - X_prior_mean
        # filename = "X_diff_mean_global_" + case + ".csv"
        # print(f"Write file: {filename}")
        # np.savetxt(filename, X_diff_prior_post_mean_global, delimiter=",", fmt="%.6f")

        # # Diff between posterior and simple kriging estimate
        # filename = "X_diff_mean_sk_" + case + ".csv"
        # print(f"Write file: {filename}")
        # np.savetxt(filename, X_diff_local_sk, delimiter=",", fmt="%.6f")

        # filename = "X_diff_mean_global_sk_" + case + ".csv"
        # print(f"Write file: {filename}")
        # np.savetxt(filename, X_diff_global_sk, delimiter=",", fmt="%.6f")


if __name__ == "__main__":
    import math
    NPARAM = 1000
    NREAL = 100
    NOBS = 200
    FIELD_MEAN = 0.0
    FIELD_STDEV = 1.0
    XINC = 1.0
    FIELD_CORR_FUNC_NAME = "general_exponential"
#    FIELD_CORR_FUNC_NAME = "exponential"
    OBS_CORR_FUNC_NAME = "exponential"
    FIELD_CORR_EXPONENT = 1.9
    FIELD_RELATIVE_CORR_RANGE = 0.05
    FIELD_RELATIVE_LOCALIZATION_RANGE = 0.1
#    OBS_RELATIVE_CORR_RANGE = 0.001
    OBS_STD_ERR = 1.0
    OBS_STD_ERR = 2.58
    OBS_STD_ERR = math.sqrt(NOBS/30)  # 2.58
    OBS_STD_ERR = math.sqrt(NOBS/60)  # 1.82
    OBS_CORR_EXPONENT = 1.9
    SEED1 = 987654321
    SEED2 = 781609981
    USE_LOCALIZATION = False

    X_prior, field_cov_matrix = draw_prior_ensemble(
        SEED1,
        FIELD_MEAN,
        FIELD_STDEV,
        XINC,
        FIELD_CORR_FUNC_NAME,
        NPARAM,
        NREAL,
        FIELD_RELATIVE_CORR_RANGE,
        FIELD_CORR_EXPONENT)
    print(f"Field cov matrix: {field_cov_matrix=}")
    obs_vector, obs_index_vector = draw_obs(
        FIELD_MEAN,
        FIELD_STDEV,
        XINC,
        FIELD_CORR_FUNC_NAME,
        NPARAM,
        NOBS,
        FIELD_RELATIVE_CORR_RANGE,
        FIELD_CORR_EXPONENT)

    obs_relative_corr_range_vector = np.arange(0.0,0.3, 0.01)
    for i in range(len(obs_relative_corr_range_vector)):
        obs_relative_corr_range = obs_relative_corr_range_vector[i]
        if  USE_LOCALIZATION:
            CASE = f"N_{NREAL}_std_{OBS_STD_ERR:.2f}_obsrange_{obs_relative_corr_range:.2f}_nobs_{NOBS}_{OBS_CORR_FUNC_NAME}_L_{FIELD_RELATIVE_LOCALIZATION_RANGE}"
        else:
            CASE = f"N_{NREAL}_std_{OBS_STD_ERR:.2f}_obsrange_{obs_relative_corr_range:.2f}_nobs_{NOBS}_{OBS_CORR_FUNC_NAME}"

        print(f"Case: {CASE}")

        C_D = calculate_obs_covariance(
                NPARAM,
                NOBS,
                XINC,
                OBS_CORR_FUNC_NAME,
                obs_index_vector,
                obs_relative_corr_range,
                OBS_STD_ERR,
                OBS_CORR_EXPONENT)

        test_distance_based_localization_on_1D_corr_field(
            X_prior,
            field_cov_matrix,
            XINC,
            obs_vector,
            obs_index_vector,
            C_D,
            FIELD_RELATIVE_LOCALIZATION_RANGE,
            SEED2,
            USE_LOCALIZATION,
            CASE,
            use_esmda=False,
            use_esmda_dist=False,
            use_esmda_local=True)
        
