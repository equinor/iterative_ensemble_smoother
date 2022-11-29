from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt

from ._ies import InversionType, make_D, make_E, make_X
from iterative_ensemble_smoother.utils import _compute_AA_projection

if TYPE_CHECKING:
    import numpy.typing as npt


def ensemble_smoother_update_step(
    response_ensemble: "npt.NDArray[np.double]",
    parameter_ensemble: "npt.NDArray[np.double]",
    observation_errors: "npt.NDArray[np.double]",
    observation_values: "npt.NDArray[np.double]",
    noise: Optional["npt.NDArray[np.double]"] = None,
    truncation: float = 0.98,
    inversion: InversionType = InversionType.EXACT,
    projection: bool = True,
):
    """Perform one step of the ensemble smoother algorithm

    :param response_ensemble: Matrix of responses from the :term:`forward model`.
        Has shape (number of observations, number of realizations). (Y in Evensen et. al)
    :param parameter_ensemble: Matrix of sampled model parameters. Has shape
        (number of parameters, number of realizations) (A in Evensen et. al).
    :param observation_errors: List of measurement of errors for each observation.
    :param observation_values: List of observations.
    :param noise: Optional list of noise used in the algorithm, Has same shape as
        response matrix.
    :param truncation: float used to determine the number of significant singular
        values. Defaults to 0.98 (ie. 98% significant values).
    :param inversion: The type of subspace inversion used in the algorithm, defaults
        to exact.
    :param projection: Whether to project response matrix.
    """
    parameters = parameter_ensemble.shape[0]
    realizations = parameter_ensemble.shape[1]
    responses = response_ensemble.shape[0]
    if noise is None:
        noise = np.random.rand(responses, realizations)
    E = make_E(observation_errors, noise)
    R = np.identity(len(observation_errors), dtype=np.double)
    D = make_D(observation_values, E, response_ensemble)
    D = (D.T / observation_errors).T
    E = (E.T / observation_errors).T
    response_ensemble = (response_ensemble.T / observation_errors).T

    if projection and (parameters < realizations - 1):
        AA_projection = _compute_AA_projection(parameter_ensemble)
        response_ensemble = response_ensemble @ AA_projection

    X = make_X(
        (response_ensemble - response_ensemble.mean(axis=1, keepdims=True))
        / np.sqrt(realizations - 1),
        R,
        E,
        D,
        inversion,
        truncation,
        np.zeros((realizations, realizations)),
        1.0,
        1,
    )
    return parameter_ensemble @ X
