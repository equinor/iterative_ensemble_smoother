from typing import TYPE_CHECKING, Optional

import numpy as np

rng = np.random.default_rng()

from ._ies import InversionType, make_D, make_E, create_transition_matrix
from iterative_ensemble_smoother.utils import _compute_AA_projection, _validate_inputs

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
    _validate_inputs(
        response_ensemble,
        parameter_ensemble,
        noise,
        observation_errors,
        observation_values,
    )

    num_params = parameter_ensemble.shape[0]
    ensemble_size = parameter_ensemble.shape[1]
    if noise is None:
        num_obs = len(observation_values)
        noise = rng.standard_normal(size=(num_obs, ensemble_size))

    E = make_E(observation_errors, noise)
    R = np.identity(len(observation_errors), dtype=np.double)
    D = make_D(observation_values, E, response_ensemble)
    D = (D.T / observation_errors).T
    E = (E.T / observation_errors).T
    response_ensemble = (response_ensemble.T / observation_errors).T

    if projection and (num_params < ensemble_size - 1):
        AA_projection = _compute_AA_projection(parameter_ensemble)
        response_ensemble = response_ensemble @ AA_projection

    X = create_transition_matrix(
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
    return parameter_ensemble @ X
