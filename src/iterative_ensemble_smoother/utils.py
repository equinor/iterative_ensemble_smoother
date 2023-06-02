from __future__ import annotations
from typing import Tuple, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

from ._ies import InversionType


def steplength_exponential(
    iteration: int,
    min_steplength: float = 0.3,
    max_steplength: float = 0.6,
    halflife: float = 1.5,
) -> float:
    """
    This is an implementation of Eq. (49), which calculates a suitable step length for
    the update step, from the book:

    Geir Evensen, Formulating the history matching problem with consistent error statistics,
    Computational Geosciences (2021) 25:945 –970

    Examples
    --------
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


def response_projection(
    param_ensemble: npt.NDArray[np.double],
) -> npt.NDArray[np.double]:
    """A^+A projection is necessary when the parameter matrix has fewer rows than
    columns, and when the forward model is non-linear. Section 2.4.3
    """
    ensemble_size = param_ensemble.shape[1]
    A = (param_ensemble - param_ensemble.mean(axis=1, keepdims=True)) / np.sqrt(
        ensemble_size - 1
    )
    projection: npt.NDArray[np.double] = np.linalg.pinv(A) @ A
    return projection


def _validate_inputs(
    response_ensemble: npt.NDArray[np.double],
    observation_errors: npt.NDArray[np.double],
    observation_values: npt.NDArray[np.double],
    param_ensemble: Optional[npt.NDArray[np.double]] = None,
) -> None:
    if response_ensemble.ndim != 2:
        raise ValueError(
            "response_ensemble must be a matrix of size (number of responses by number of realizations)"
        )

    num_responses = response_ensemble.shape[0]
    ensemble_size = response_ensemble.shape[1]

    if response_ensemble.shape[1] != ensemble_size:
        raise ValueError(
            "response_ensemble and parameter_ensemble must have the same number of columns"
        )

    if observation_errors.ndim == 2:
        if observation_errors.shape[0] != observation_errors.shape[1]:
            raise ValueError(
                "observation_errors as covariance matrix must be a square matrix"
            )
        if observation_errors.shape[0] != len(observation_values):
            raise ValueError(
                "observation_errors covariance matrix must match size of observation_values"
            )
        if not np.all(np.abs(observation_errors - observation_errors.T) < 1e-8):
            raise ValueError(
                "observation_errors as covariance matrix must be symmetric"
            )
    elif len(observation_errors) != len(observation_values):
        raise ValueError(
            "observation_errors and observation_values must have the same number of elements"
        )

    if len(observation_values) != num_responses:
        raise ValueError(
            "observation_values must have the same number of elements as there are responses"
        )

    if param_ensemble is not None and param_ensemble.ndim != 2:
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
    if observation_errors.ndim == 2:
        R = observation_errors
        observation_errors = np.sqrt(observation_errors.diagonal())
        # The line below is equivalent to:
        # R = np.diag(1 / observation_errors) @ R @ np.diag(1 / observation_errors)
        R = R * np.outer(1 / observation_errors, 1 / observation_errors)

    elif observation_errors.ndim == 1 and inversion == InversionType.EXACT_R:
        R = np.identity(len(observation_errors))

    else:
        R = None

    return R, observation_errors
