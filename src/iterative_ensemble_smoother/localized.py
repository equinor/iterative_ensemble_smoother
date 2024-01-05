# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 09:27:34 2024

@author: TODL
"""

from iterative_ensemble_smoother.esmda import BaseESMDA
import numbers
from abc import ABC
from typing import Union

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore

from iterative_ensemble_smoother.esmda_inversion import (
    inversion_exact_cholesky,
    inversion_subspace,
    normalize_alpha,
)
from iterative_ensemble_smoother.utils import sample_mvnormal


class LassoESMDA(BaseESMDA):
    """
    Implement an Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

    The implementation follows :cite:t:`EMERICK2013`.

    Parameters
    ----------
    covariance : np.ndarray
        Either a 1D array of diagonal covariances, or a 2D covariance matrix.
        The shape is either (num_observations,) or (num_observations, num_observations).
        This is C_D in Emerick (2013), and represents observation or measurement
        errors. We observe d from the real world, y from the model g(x), and
        assume that d = y + e, where the error e is multivariate normal with
        covariance given by `covariance`.
    observations : np.ndarray
        1D array of shape (num_observations,) representing real-world observations.
        This is d_obs in Emerick (2013).
    alpha : int or 1D np.ndarray, optional
        Multiplicative factor for the covariance.
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
        See the dictionary ESMDA._inversion_methods for more information.

    Examples
    --------
    >>> covariance = np.diag([1, 1, 1])
    >>> observations = np.array([1, 2, 3])
    >>> esmda = ESMDA(covariance, observations)

    """

    # Available inversion methods. The inversion methods all compute
    # C_MD @ (C_DD + alpha * C_D)^(-1)  @ (D - Y)
    _inversion_methods = {
        "exact": inversion_exact_cholesky,
        "subspace": inversion_subspace,
    }

    def __init__(
        self,
        covariance: npt.NDArray[np.double],
        observations: npt.NDArray[np.double],
        alpha: Union[int, npt.NDArray[np.double]] = 5,
        seed: Union[np.random._generator.Generator, int, None] = None,
        inversion: str = "exact",
    ) -> None:
        """Initialize the instance."""

        super().__init__(covariance=covariance, observations=observations, seed=seed)

        if not (
            (isinstance(alpha, np.ndarray) and alpha.ndim == 1)
            or isinstance(alpha, numbers.Integral)
        ):
            raise TypeError("Argument `alpha` must be an integer or a 1D NumPy array.")

        if not isinstance(inversion, str):
            raise TypeError(
                "Argument `inversion` must be a string in "
                f"{tuple(self._inversion_methods.keys())}, but got {inversion}"
            )
        if inversion not in self._inversion_methods.keys():
            raise ValueError(
                "Argument `inversion` must be a string in "
                f"{tuple(self._inversion_methods.keys())}, but got {inversion}"
            )

        # Store data
        self.inversion = inversion

        # Alpha can either be an integer (num iterations) or a list of weights
        if isinstance(alpha, np.ndarray) and alpha.ndim == 1:
            self.alpha = normalize_alpha(alpha)
        elif isinstance(alpha, numbers.Integral):
            self.alpha = normalize_alpha(np.ones(alpha))
            assert np.allclose(self.alpha, normalize_alpha(self.alpha))
        else:
            raise TypeError("Alpha must be integer or 1D array.")

    def num_assimilations(self) -> int:
        return len(self.alpha)

    def assimilate(
        self,
        X: npt.NDArray[np.double],
        Y: npt.NDArray[np.double],
        *,
        overwrite: bool = False,
        truncation: float = 1.0,
    ) -> npt.NDArray[np.double]:
        """Assimilate data and return an updated ensemble X_posterior.

            X_posterior = smoother.assimilate(X, Y)

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (num_parameters, ensemble_size). Each row
            corresponds to a parameter in the model, and each column corresponds
            to an ensemble member (realization).
        Y : np.ndarray
            2D array of shape (num_observations, ensemble_size), containing
            responses when evaluating the model at X. In other words, Y = g(X),
            where g is the forward model.
        overwrite : bool
            If True, then arguments X and Y may be overwritten and mutated.
            If False, then the method will not mutate inputs in any way.
            Setting this to True might save memory.
        truncation : float
            How large a fraction of the singular values to keep in the inversion
            routine. Must be a float in the range (0, 1]. A lower number means
            a more approximate answer and a slightly faster computation.

        Returns
        -------
        X_posterior : np.ndarray
            2D array of shape (num_parameters, ensemble_size).

        """
        if self.iteration >= self.num_assimilations():
            raise Exception("No more assimilation steps to run.")

        # Verify shapes
        _, num_ensemble = X.shape
        num_outputs, num_emsemble2 = Y.shape
        assert (
            num_ensemble == num_emsemble2
        ), "Number of ensemble members in X and Y must match"
        if not np.issubdtype(X.dtype, np.floating):
            raise TypeError("Argument `X` must contain floats")
        if not np.issubdtype(Y.dtype, np.floating):
            raise TypeError("Argument `Y` must contain floats")

        assert 0 < truncation <= 1.0

        # Do not overwrite input arguments
        if not overwrite:
            X, Y = np.copy(X), np.copy(Y)

        # Line 2 (b) in the description of ES-MDA in the 2013 Emerick paper

        # Draw samples from zero-centered multivariate normal with cov=alpha * C_D,
        # and add them to the observations. Notice that
        # if C_D = L L.T by the cholesky factorization, then drawing y from
        # a zero cented normal means that y := L @ z, where z ~ norm(0, 1)
        # Therefore, scaling C_D by alpha is equivalent to scaling L with sqrt(alpha)
        D = self.perturb_observations(
            ensemble_size=num_ensemble, alpha=self.alpha[self.iteration]
        )
        assert D.shape == (num_outputs, num_ensemble)

        # Line 2 (c) in the description of ES-MDA in the 2013 Emerick paper
        # Choose inversion method, e.g. 'exact'. The inversion method computes
        # C_MD @ sp.linalg.inv(C_DD + C_D_alpha) @ (D - Y)
        inversion_func = self._inversion_methods[self.inversion]

        # Update and return
        X += inversion_func(
            alpha=self.alpha[self.iteration],
            C_D=self.C_D,
            D=D,
            Y=Y,
            X=X,
            truncation=truncation,
        )

        self.iteration += 1
        return X

    def compute_transition_matrix(
        self,
        Y: npt.NDArray[np.double],
        *,
        alpha: float,
        truncation: float = 1.0,
    ) -> npt.NDArray[np.double]:
        """Return a matrix T such that X_posterior = X_prior + X_prior @ T.

        The purpose of this method is to facilitate row-by-row, or batch-by-batch,
        updates of X. This is useful if X is too large to fit in memory.

        Parameters
        ----------
        Y : np.ndarray
            2D array of shape (num_observations, ensemble_size), containing
            responses when evaluating the model at X. In other words, Y = g(X),
            where g is the forward model.
        alpha : float
            The covariance inflation factor. The sequence of alphas should
            obey the equation sum_i (1/alpha_i) = 1. However, this is NOT enforced
            in this method call. The user/caller is responsible for this.
        truncation : float
            How large a fraction of the singular values to keep in the inversion
            routine. Must be a float in the range (0, 1]. A lower number means
            a more approximate answer and a slightly faster computation.

        Returns
        -------
        T : np.ndarray
            A matrix T such that X_posterior = X_prior + X_prior @ T.
            It has shape (num_ensemble_members, num_ensemble_members).
        """

        # Recall the update equation:
        # X += C_MD @ (C_DD + alpha * C_D)^(-1)  @ (D - Y)
        # X += X @ center(Y).T / (N-1) @ (C_DD + alpha * C_D)^(-1) @ (D - Y)
        # We form T := center(Y).T / (N-1) @ (C_DD + alpha * C_D)^(-1) @ (D - Y),
        # so that
        # X_new = X_old + X_old @ T
        # or
        # X += X @ T

        D = self.perturb_observations(ensemble_size=Y.shape[1], alpha=alpha)
        inversion_func = self._inversion_methods[self.inversion]
        return inversion_func(
            alpha=alpha,
            C_D=self.C_D,
            D=D,
            Y=Y,
            X=None,  # We don't need X to compute the factor T
            truncation=truncation,
            return_T=True,  # Ensures that we don't need X
        )


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
        ]
    )
