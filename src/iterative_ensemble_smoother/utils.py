from __future__ import annotations
from typing import Tuple, Optional, TYPE_CHECKING

import numpy as np

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
    >>> [steplength_exponential(i) for i in [1, 2, 3, 4]]
    [0.6, 0.48898815748423097, 0.41905507889761495, 0.375]
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

    # TODO: Since pinv(A) takes the SVD, it seems like using the SVD directly
    # to compute pinv(A) @ A directly without forming pinv(A) explicitly should
    # be faster, but a quick timing showed no such result. Might be worth
    # looking into.
    ans: npt.NDArray[np.double] = np.linalg.pinv(A) @ A
    return ans


def _validate_inputs(
    parameters: npt.NDArray[np.double],
    covariance: npt.NDArray[np.double],
    observations: npt.NDArray[np.double],
) -> None:
    # Check types
    inputs = [parameters, covariance, observations]
    names = ["parameters", "covariances", "observations"]
    for input_, name in zip(inputs, names):
        if not isinstance(input_, np.ndarray):
            raise TypeError(f"Argument '{name}' must be numpy nd.array")

    assert covariance.ndim in (1, 2)
    assert parameters.ndim == 2

    assert covariance.shape[0] == observations.shape[0]
    assert covariance.shape[0] == observations.shape[0]


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


def sample_mvnormal(*, C_dd_cholesky, rng, size):
    """Draw samples from the multivariate normal N(0, C_dd).

    We write this function from scratch here we can to avoid factoring the
    covariance matrix every time we sample, and we want to exploit diagonal
    covariance matrices in terms of computation and memory. More specifically:

        - numpy.random.multivariate_normal factors the covariance in every call
        - scipy.stats.Covariance.from_diagonal stores off diagonal zeros

    So the best choice was to write sampling from scratch.


    Examples
    --------
    >>> C_dd_cholesky = np.diag([5, 10, 15])
    >>> rng = np.random.default_rng(42)
    >>> sample_mvnormal(C_dd_cholesky=C_dd_cholesky, rng=rng, size=2)
    array([[  1.5235854 ,  -5.19992053],
           [  7.50451196,   9.40564716],
           [-29.26552783, -19.5326926 ]])
    >>> sample_mvnormal(C_dd_cholesky=np.diag(C_dd_cholesky), rng=rng, size=2)
    array([[ 0.63920202, -1.58121296],
           [-0.16801158, -8.53043928],
           [13.19096962, 11.66687903]])


    """

    # Standard normal samples
    z = rng.standard_normal(size=(C_dd_cholesky.shape[0], size))

    # A 2D covariance matrix was passed
    if C_dd_cholesky.ndim == 2:
        return C_dd_cholesky @ z

    # A 1D diagonal of a covariance matrix was passed
    else:
        return C_dd_cholesky.reshape(-1, 1) * z


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-v"])
