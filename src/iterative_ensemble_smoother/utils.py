from __future__ import annotations
from typing import Tuple, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

from ._ies import InversionType


def _validate_inputs(
    response_ensemble: npt.NDArray[np.double],
    noise: Optional[npt.NDArray[np.double]],
    observation_errors: npt.NDArray[np.double],
    observation_values: npt.NDArray[np.double],
    param_ensemble: Optional[npt.NDArray[np.double]] = None,
) -> None:
    if len(response_ensemble.shape) != 2:
        raise ValueError(
            "response_ensemble must be a matrix of size (number of responses by number of realizations)"
        )

    num_responses = response_ensemble.shape[0]
    ensemble_size = response_ensemble.shape[1]

    if response_ensemble.shape[1] != ensemble_size:
        raise ValueError(
            "response_ensemble and parameter_ensemble must have the same number of columns"
        )

    if noise is not None and noise.shape[1] != ensemble_size:
        raise ValueError(
            "noise and response_ensemble must have the same number of columns"
        )

    if noise is not None and noise.shape[0] != num_responses:
        raise ValueError(
            "noise and response_ensemble must have the same number of rows"
        )

    if len(observation_errors) != len(observation_values):
        raise ValueError(
            "observation_errors and observation_values must have the same number of elements"
        )

    if len(observation_values) != num_responses:
        raise ValueError(
            "observation_values must have the same number of elements as there are responses"
        )

    if param_ensemble is not None and len(param_ensemble.shape) != 2:
        raise ValueError(
            "parameter_ensemble must be a matrix of size (number of parameters by number of realizations)"
        )

    if param_ensemble is not None and param_ensemble.shape[1] != ensemble_size:
        raise ValueError(
            "param_ensemble and response_ensemble must have the same number of columns"
        )


def _create_errors(
    observation_errors: npt.NDArray[np.double],
    inversion: InversionType,
) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
    if len(observation_errors.shape) == 2:
        R = observation_errors
        observation_errors = np.sqrt(observation_errors.diagonal())
        R = (R.T / R.diagonal()).T
    elif len(observation_errors.shape) == 1 and inversion == InversionType.EXACT_R:
        R = np.identity(len(observation_errors))
    else:
        R = None
    return R, observation_errors
