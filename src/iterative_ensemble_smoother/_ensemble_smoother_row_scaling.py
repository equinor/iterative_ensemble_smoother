import numpy as np

from ._ies import InversionType, make_D, make_E, make_X


def ensemble_smoother_update_step_row_scaling(
    response_matrix,
    parameters_with_row_scaling,
    observation_errors,
    observation_values,
    noise=None,
    truncation=0.98,
    inversion=InversionType.EXACT,
):
    """This is an experimental feature."""
    S = response_matrix
    A_with_row_scaling = parameters_with_row_scaling
    if noise is None:
        noise = np.random.rand(*S.shape)

    E = make_E(observation_errors, noise)
    R = np.identity(len(observation_errors), dtype=np.double)
    D = make_D(observation_values, E, S)
    D = (D.T / observation_errors).T
    E = (E.T / observation_errors).T
    S = (S.T / observation_errors).T
    for (A, row_scale) in A_with_row_scaling:
        X = make_X(
            A,
            S,
            R,
            E,
            D,
            inversion,
            truncation,
            np.zeros((S.shape[1], S.shape[1])),
            1.0,
            1,
        )
        row_scale.multiply(A, X)
    return A_with_row_scaling
