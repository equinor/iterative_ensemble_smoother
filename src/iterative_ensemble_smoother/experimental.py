"""
Contains (publicly available, but not officially supported) experimental
features of iterative_ensemble_smoother
"""
import numpy as np

rng = np.random.default_rng()

from ._ies import InversionType, make_D, make_E, make_X


def ensemble_smoother_update_step_row_scaling(
    response_ensemble,
    A_with_row_scaling,
    observation_errors,
    observation_values,
    noise=None,
    truncation=0.98,
    inversion=InversionType.EXACT,
):
    """This is an experimental feature."""
    ensemble_size = response_ensemble.shape[1]
    if noise is None:
        num_obs = len(observation_values)
        noise = rng.standard_normal(size=(num_obs, ensemble_size))

    E = make_E(observation_errors, noise)
    R = np.identity(len(observation_errors), dtype=np.double)
    D = make_D(observation_values, E, response_ensemble)
    D = (D.T / observation_errors).T
    E = (E.T / observation_errors).T
    response_ensemble = (response_ensemble.T / observation_errors).T
    for (A, row_scale) in A_with_row_scaling:
        X = make_X(
            (response_ensemble - response_ensemble.mean(axis=1, keepdims=True))
            / np.sqrt(ensemble_size - 1),
            R,
            E,
            D,
            inversion,
            truncation,
            np.zeros((ensemble_size, ensemble_size)),
            1.0,
            1,
        )
        row_scale.multiply(A, X)
    return A_with_row_scaling
