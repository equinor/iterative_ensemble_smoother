from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def steplength_exponential(
    iteration: int,
    min_steplength: float = 0.3,
    max_steplength: float = 0.6,
    halflife: float = 1.5,
) -> float:
    r"""
    Compute a suitable step length for the update step.

    This is an implementation of Eq. (49), which calculates a suitable step length for
    the update step, from the book: \"Formulating the history matching problem with
    consistent error statistics", written by :cite:t:`evensen2021formulating`.

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


def sample_mvnormal(
    *,
    C_dd_cholesky: npt.NDArray[np.double],
    rng: np.random._generator.Generator,
    size: int,
) -> npt.NDArray[np.double]:
    """Draw samples from the multivariate normal N(0, C_dd).

    We write this function from scratch to avoid factoring the covariance
    matrix every time we sample, and we want to exploit diagonal covariance
    matrices in terms of computation and memory. More specifically:

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
