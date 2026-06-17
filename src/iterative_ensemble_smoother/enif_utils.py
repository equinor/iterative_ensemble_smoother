import logging

import numpy as np
import scipy as sp
from numpy.typing import NDArray
from scipy.sparse import sparray
from sksparse.cholmod import cholesky

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def generate_gaussian_noise(
    n: int, Prec: sparray, seed: int | None = None
) -> NDArray[np.floating]:
    """
    Generates 'n' samples of Gaussian noise with precision 'Prec'.

    Parameters
    ----------
    n : int
        The number of samples to generate.
    Prec : scipy.sparse.sparray
        The precision matrix for the Gaussian noise, assumed to be sparse.

    Returns
    -------
    np.ndarray
        The Gaussian noise array of shape (n, m), where Prec has shape (m, m).

    Examples
    --------
    >>> F = sp.sparse.random_array(shape=(6, 6), density=0.3, random_state=42)
    >>> Prec = sp.sparse.csc_array(F.T @ F + np.diag([1, 2, 4, 8, 16, 32]))
    >>> generate_gaussian_noise(n=3, Prec=Prec, seed=1).round(2)
    array([[ 0.35,  0.19, -0.33, -0.19, -0.32,  0.1 ],
           [ 0.82,  0.02, -0.07,  0.2 ,  0.23,  0.01],
           [ 0.33,  0.36, -0.19,  0.11,  0.11, -0.05]])

    >>> Prec = sp.sparse.diags_array(Prec.diagonal())
    >>> generate_gaussian_noise(n=3, Prec=Prec, seed=2).round(2)
    array([[ 0.19, -0.34, -0.17, -0.84,  0.45,  0.2 ],
           [-0.33,  0.51,  0.12, -0.19,  0.24, -0.05],
           [-0.33, -0.52,  0.19, -0.03,  0.14, -0.1 ]])
    """
    if n < 1:
        raise ValueError(f"`n` should be g.e. 1, got {n}")

    m = Prec.shape[0]
    rng = np.random.default_rng(seed)

    # If precision matrix is diagonal
    row_idx, col_idx, _ = sp.sparse.find(Prec)
    if np.all(row_idx == col_idx):
        # Scale is the standard deviations. diag(Prec) = 1 / variances
        scale = np.sqrt(1 / Prec.diagonal())
        return rng.normal(
            loc=0,
            scale=scale,
            size=(n, m),
        )

    # General case: precision matrix is not diagonal
    z = rng.normal(size=(m, n))

    # The simplest, but naive, way to sample is to compute:
    # cov = np.linalg.inv(Prec.todense())
    # C = np.linalg.cholesky(cov)
    # assert np.allclose(C @ C.T, cov)
    # return (np.linalg.cholesky(cov) @ z).T

    # Suppose C @ C.T = Cov, then our samples are eps = C @ z.
    # If we take cholesky of Prec, we obtain L @ L.T = P @ Prec @ P.T.
    # Rearrange to (P.T @ L) @ (P.T @ L).T = Prec, then take inverse
    # to see that C = inv((P.T @ L).T). Thus the equation eps = C @ z
    # becomes the system (P.T @ L).T @ eps = z, which we solve for eps below
    factor = cholesky(Prec)
    v = factor.solve_Lt(z, use_LDLt_decomposition=False)
    return factor.apply_Pt(v).T


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v"])
