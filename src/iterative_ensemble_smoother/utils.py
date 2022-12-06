import numpy as np
import numpy.typing as npt


def _compute_AA_projection(A: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """A^+A projection is necessary when the parameter matrix has fewer rows than
    columns, and when the forward model is non-linear. Section 2.4.3
    """
    _, _, vh = np.linalg.svd(A - A.mean(axis=1, keepdims=True), full_matrices=False)
    return vh.T @ vh


def _validate_inputs(
    response_ensemble: npt.NDArray[np.double],
    parameter_ensemble: npt.NDArray[np.double],
    noise: npt.NDArray[np.double],
    observation_errors: npt.NDArray[np.double],
    observation_values: npt.NDArray[np.double],
) -> None:

    if len(response_ensemble.shape) != 2:
        raise ValueError(
            "response_ensemble must be a matrix of size (number of responses by number of realizations)"
        )

    if len(parameter_ensemble.shape) != 2:
        raise ValueError(
            "parameter_ensemble must be a matrix of size (number of parameters by number of realizations)"
        )

    num_responses = response_ensemble.shape[0]
    ensemble_size = parameter_ensemble.shape[1]

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
