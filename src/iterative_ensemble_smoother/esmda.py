"""
Ensemble Smoother with Multiple Data Assimilation (ES-MDA)
----------------------------------------------------------

Implementation of the 2013 paper "Ensemble smoother with multiple data assimilation"

References
----------

Emerick, A.A., Reynolds, A.C. History matching time-lapse seismic data using
the ensemble Kalman filter with multiple data assimilations.
Comput Geosci 16, 639–659 (2012). https://doi.org/10.1007/s10596-012-9275-5

Alexandre A. Emerick, Albert C. Reynolds.
Ensemble smoother with multiple data assimilation.
Computers & Geosciences, Volume 55, 2013, Pages 3-15, ISSN 0098-3004,
https://doi.org/10.1016/j.cageo.2012.03.011.

https://gitlab.com/antoinecollet5/pyesmda

https://helper.ipam.ucla.edu/publications/oilws3/oilws3_14147.pdf

"""

import numbers
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore

from iterative_ensemble_smoother.esmda_inversion import (
    inversion_exact_cholesky,
    inversion_subspace,
    normalize_alpha,
)


class ESMDA:
    """Initialize Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

    The implementation follows the 2013 paper by Emerick et al.

    Parameters
    ----------
    C_D : np.ndarray
        Covariance matrix of outputs of shape (num_outputs, num_outputs).
        If a 1D array is passed, it represents a diagonal covariance matrix.
    observations : np.ndarray
        1D array of shape (num_inputs,) representing real-world observations.
    alpha : int or 1D np.ndarray, optional
        If an integer `alpha` is given, an array with length `alpha` and
        elements `alpha` is constructed. If an 1D array is given, it is
        normalized so sum_i 1/alpha_i = 1 and used. The default is 5, which
        corresponds to np.array([5, 5, 5, 5, 5]).
    seed : integer or numpy.random._generator.Generator, optional
        A seed or numpy.random._generator.Generator used for random number
        generation. The argument is passed to numpy.random.default_rng().
        The default is None.
    inversion : str, optional
        Which inversion method to use. The default is "exact".

    Examples
    --------
    >>> C_D = np.diag([1, 1, 1])
    >>> observations = np.array([1, 2, 3, 4])
    >>> esmda = ESMDA(C_D, observations)

    """

    # Available inversion methods. The inversion methods all compute
    # C_MD @ (C_DD + alpha * C_D)^(-1)  @ (D - Y)
    _inversion_methods = {
        "exact": inversion_exact_cholesky,
        "subspace": inversion_subspace,
    }

    def __init__(
        self,
        C_D: npt.NDArray[np.double],
        observations: npt.NDArray[np.double],
        alpha: Union[int, npt.NDArray[np.double]] = 5,
        seed: Union[np.random._generator.Generator, int, None] = None,
        inversion: str = "exact",
    ) -> None:
        # Validate inputs
        if not (isinstance(C_D, np.ndarray) and C_D.ndim in (1, 2)):
            raise TypeError("Argument `C_D` must be a NumPy array of dimension 1 or 2.")

        if C_D.ndim == 2 and C_D.shape[0] != C_D.shape[1]:
            raise ValueError("Argument `C_D` must be square if it's 2D.")

        if not (isinstance(observations, np.ndarray) and observations.ndim == 1):
            raise TypeError("Argument `observations` must be a 1D NumPy array.")

        if not observations.shape[0] == C_D.shape[0]:
            raise ValueError("Shapes of `observations` and `C_D` must match.")

        if not (
            (isinstance(alpha, np.ndarray) and alpha.ndim == 1)
            or isinstance(alpha, numbers.Integral)
        ):
            raise TypeError("Argument `alpha` must be an integer or a 1D NumPy array.")

        if not (
            isinstance(seed, (int, np.random._generator.Generator)) or seed is None
        ):
            raise TypeError(
                "Argument `seed` must be an integer or numpy.random._generator.Generator."
            )

        if not isinstance(inversion, str):
            raise TypeError(
                f"Argument `inversion` must be a string in {tuple(self._inversion_methods.keys())}"
            )
        if inversion not in self._inversion_methods.keys():
            raise ValueError(
                f"Argument `inversion` must be a string in {tuple(self._inversion_methods.keys())}"
            )

        # Store data
        self.observations = observations
        self.iteration = 0
        self.rng = np.random.default_rng(seed)
        self.inversion = inversion

        # Alpha can either be an integer (num iterations) or a list of weights
        if isinstance(alpha, np.ndarray) and alpha.ndim == 1:
            self.alpha = normalize_alpha(alpha)
        elif isinstance(alpha, numbers.Integral):
            self.alpha = normalize_alpha(np.ones(alpha))
            assert np.allclose(self.alpha, normalize_alpha(self.alpha))
        else:
            raise TypeError("Alpha must be integer or 1D array.")

        # Only compute the covariance factorization once
        # If it's a full matrix, we gain speedup by only computing cholesky once
        # If it's a diagonal, we gain speedup by never having to compute cholesky
        num_outputs = C_D.shape[0]

        if isinstance(C_D, np.ndarray) and C_D.ndim == 2:
            self.C_D_L = sp.linalg.cholesky(C_D, lower=False)
        elif isinstance(C_D, np.ndarray) and C_D.ndim == 1:
            self.C_D_L = np.sqrt(C_D)
        else:
            raise TypeError("Argument `C_D` must be 1D or 2D array")

        self.C_D = C_D
        assert isinstance(self.C_D, np.ndarray) and self.C_D.ndim in (1, 2)

    def num_assimilations(self) -> int:
        return len(self.alpha)

    def assimilate(
        self,
        X: npt.NDArray[np.double],
        Y: npt.NDArray[np.double],
        ensemble_mask: Optional[npt.NDArray[np.bool_]] = None,
        overwrite: bool = False,
        truncation: float = 1.0,
    ) -> npt.NDArray[np.double]:
        """Assimilate data and return an updated ensemble X_posterior.

        Parameters
        ----------
        X : np.ndarray
            2D array of shape (num_inputs, num_ensemble_members).
        Y : np.ndarray
            2D array of shape (num_ouputs, num_ensemble_members).
        ensemble_mask : np.ndarray
            1D boolean array of length `num_ensemble_members`, describing which
            ensemble members are active. Inactive realizations are ignored.
            Defaults to all active.
        overwrite : bool
            If True, then arguments X and Y may be overwritten.
            If False, then the method will not permute inputs in any way.
        truncation : float
            How large a fraction of the singular values to keep in the inversion
            routine. Must be a float in the range (0, 1]. A lower number means
            a more approximate answer and a slightly faster computation.

        Returns
        -------
        X_posterior : np.ndarray
            2D array of shape (num_inputs, num_ensemble_members).

        """
        if self.iteration >= self.num_assimilations():
            raise Exception("No more assimilation steps to run.")

        # Verify shapes
        _, num_ensemble = X.shape
        num_outputs, num_emsemble2 = Y.shape
        assert (
            num_ensemble == num_emsemble2
        ), "Number of ensemble members in X and Y must match"
        assert (ensemble_mask is None) or (
            ensemble_mask.ndim == 1 and len(ensemble_mask) == num_ensemble
        )
        if not np.issubdtype(X.dtype, np.floating):
            raise TypeError("Argument `X` must be contain floats")
        if not np.issubdtype(Y.dtype, np.floating):
            raise TypeError("Argument `Y` must be contain floats")

        assert 0 < truncation <= 1.0

        # Do not overwrite input arguments
        if not overwrite:
            X, Y = np.copy(X), np.copy(Y)

        # If no ensemble mask was given, we use the entire ensemble
        if ensemble_mask is None:
            ensemble_mask = np.ones(num_ensemble, dtype=bool)

        # No ensemble members means no update
        if ensemble_mask.sum() == 0:
            return X

        # Line 2 (b) in the description of ES-MDA in the 2013 Emerick paper

        # Draw samples from zero-centered multivariate normal with cov=alpha * C_D,
        # and add them to the observations. Notice that
        # if C_D = L L.T by the cholesky factorization, then drawing y from
        # a zero cented normal means that y := L @ z, where z ~ norm(0, 1)
        # Therefore, scaling C_D by alpha is equivalent to scaling L with sqrt(alpha)
        size = (num_outputs, ensemble_mask.sum())
        if self.C_D.ndim == 2:
            D = self.observations.reshape(-1, 1) + np.sqrt(
                self.alpha[self.iteration]
            ) * self.C_D_L @ self.rng.normal(size=size)
        else:
            D = self.observations.reshape(-1, 1) + np.sqrt(
                self.alpha[self.iteration]
            ) * self.rng.normal(size=size) * self.C_D_L.reshape(-1, 1)
        assert D.shape == (num_outputs, ensemble_mask.sum())

        # Line 2 (c) in the description of ES-MDA in the 2013 Emerick paper
        # Choose inversion method, e.g. 'exact'. The inversion method computes
        # C_MD @ sp.linalg.inv(C_DD + C_D_alpha) @ (D - Y)
        inversion_func = self._inversion_methods[self.inversion]

        # Update and return
        X[:, ensemble_mask] += inversion_func(
            alpha=self.alpha[self.iteration],
            C_D=self.C_D,
            D=D,
            Y=Y[:, ensemble_mask],
            X=X[:, ensemble_mask],
            truncation=truncation,
        )

        self.iteration += 1
        return X


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
        ]
    )
