"""
Contains (publicly available, but not officially supported) experimental
features of iterative_ensemble_smoother
"""
import numpy as np

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
    Y = response_ensemble
    if noise is None:
        noise = np.random.rand(*Y.shape)

    E = make_E(observation_errors, noise)
    R = np.identity(len(observation_errors), dtype=np.double)
    D = make_D(observation_values, E, Y)
    D = (D.T / observation_errors).T
    E = (E.T / observation_errors).T
    Y = (Y.T / observation_errors).T
    for (A, row_scale) in A_with_row_scaling:
        X = make_X(
            Y,
            R,
            E,
            D,
            inversion,
            truncation,
            np.zeros((Y.shape[1], Y.shape[1])),
            1.0,
            1,
        )
        row_scale.multiply(A, X)
    return A_with_row_scaling
