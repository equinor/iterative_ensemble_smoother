from __future__ import annotations
from typing import Tuple, Optional, TYPE_CHECKING

import numpy as np
import scipy as sp  # type: ignore

if TYPE_CHECKING:
    import numpy.typing as npt


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

    Examples
    --------
    >>> A = np.arange(9, dtype=float).reshape(3,3)
    >>> response_projection(A)
    array([[ 0.5,  0. , -0.5],
           [ 0. ,  0. ,  0. ],
           [-0.5,  0. ,  0.5]])

    Equivalent to:

    >>> C = (A - A.mean(axis=1, keepdims=True)) / np.sqrt(3 - 1)
    >>> np.linalg.pinv(C) @ C
    array([[ 0.5,  0. , -0.5],
           [ 0. ,  0. ,  0. ],
           [-0.5,  0. ,  0.5]])
    """
    ensemble_size = param_ensemble.shape[1]
    A = param_ensemble - param_ensemble.mean(axis=1, keepdims=True)
    A /= np.sqrt(ensemble_size - 1)

    # The code below is equivalent to np.linalg.pinv(A) @ A, since:
    # pinv(A) @ A = [V (1/s) U.T] @ [U s V.T]
    #             = V (1/s) @ s V.T

    # Compute SVD
    U, s, VT = sp.linalg.svd(A, full_matrices=False)

    # This code is adapted from
    # https://github.com/numpy/numpy/blob/db4f43983cb938f12c311e1f5b7165e270c393b4/numpy/linalg/linalg.py#L2015
    rcond = np.asarray(1e-15)
    cutoff = rcond[..., np.newaxis] * np.amax(s, axis=-1, keepdims=True)

    # Only keep columns in V corresponding to "large" singular values
    VT_filtered = VT[(s > cutoff), :]

    projection: npt.NDArray[np.double] = VT_filtered.T @ VT_filtered
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


def covariance_to_correlation(
    C: npt.NDArray[np.double],
) -> Tuple[Optional[npt.NDArray[np.double]], npt.NDArray[np.double]]:
    """The input C is either (1) a 2D covariance matrix or (2) a 1D array of
    standard deviations. This function normalizes the covariance matrix to a
    correlation matrix (unit diagonal).

    Examples
    --------
    >>> C = np.array([[4., 1., 0.],
    ...               [1., 4., 1.],
    ...               [0., 1., 4.]])
    >>> corr_mat, stds = covariance_to_correlation(C)
    >>> corr_mat
    array([[1.  , 0.25, 0.  ],
           [0.25, 1.  , 0.25],
           [0.  , 0.25, 1.  ]])
    >>> stds
    array([2., 2., 2.])
    """
    assert isinstance(C, np.ndarray) and C.ndim in (1, 2)

    # A covariance matrix was passed
    if C.ndim == 2:
        standard_deviations = np.sqrt(C.diagonal())

        # Create a correlation matrix from a covariance matrix
        # https://en.wikipedia.org/wiki/Covariance_matrix#Relation_to_the_correlation_matrix

        # Divide every column
        correlation_matrix = C / standard_deviations.reshape(1, -1)

        # Divide every row
        correlation_matrix /= standard_deviations.reshape(-1, 1)

        return correlation_matrix, standard_deviations

    # A vector of standard deviations was passed
    correlation_matrix = None
    standard_deviations = C
    return None, standard_deviations


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-v"])
