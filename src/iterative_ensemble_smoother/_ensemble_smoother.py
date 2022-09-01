from typing import TYPE_CHECKING, Optional

import numpy as np

from ._ies import InversionType, make_D, make_E, make_X

if TYPE_CHECKING:
    import numpy.typing as npt


def ensemble_smoother_update_step(
    sensitivity_matrix: "npt.NDArray[np.double]",
    centered_anomaly_matrix: "npt.NDArray[np.double]",
    observation_errors: "npt.NDArray[np.double]",
    observation_values: "npt.NDArray[np.double]",
    noise: Optional["npt.NDArray[np.double]"] = None,
    truncation: float = 0.98,
    inversion: InversionType = InversionType.EXACT,
):
    """Perform one step of the ensemble smoother algorithm

    :param sensitivity_matrix: Matrix of responses from the :term:`forward model`.
        Has shape (number of observations, number of realizations). (S in Evensen et. al)
    :param centered_anomaly_matrix: Matrix of sampled model parameters. Has shape
        (number of parameters, number of realizations) (A in Evensen et. al).
    :param observation_errors: List of measurement of errors for each observation.
    :param observation_values: List of observations.
    :param noise: Optional list of noise used in the algorithm, Has same shape as
        response matrix.
    :param truncation: float used to determine the number of significant singular
        values. Defaults to 0.98 (ie. 98% significant values).
    :param inversion: The type of subspace inversion used in the algorithm, defaults
        to exact.
    """
    S = sensitivity_matrix
    A = centered_anomaly_matrix
    if noise is None:
        noise = np.random.rand(*S.shape)
    E = make_E(observation_errors, noise)
    R = np.identity(len(observation_errors), dtype=np.double)
    D = make_D(observation_values, E, S)
    D = (D.T / observation_errors).T
    E = (E.T / observation_errors).T
    S = (S.T / observation_errors).T

    X = make_X(
        A, S, R, E, D, inversion, truncation, np.zeros((S.shape[1], S.shape[1])), 1.0, 1
    )
    return A @ X
