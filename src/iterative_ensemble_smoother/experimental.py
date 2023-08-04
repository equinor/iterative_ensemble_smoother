"""
Contains (publicly available, but not officially supported) experimental
features of iterative_ensemble_smoother
"""
import numpy as np

rng = np.random.default_rng()

from iterative_ensemble_smoother.utils import (
    covariance_to_correlation,
)

from iterative_ensemble_smoother.ies import create_coefficient_matrix


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
