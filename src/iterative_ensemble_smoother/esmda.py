"""
Ensemble Smoother with Multiple Data Assimilation (ES-MDA)
----------------------------------------------------------

Implementation of the 2013 paper "Ensemble smoother with multiple data assimilation"

This implementation follows the paper, but with some additions:

- Allows batching the parameters. This is useful if you have many parameters
  that you need to update, e.g. 10 million. In that case you might want to
  read 1 million from disk, update those, then write them back to disk, assimilate
  the next 1 million, etc.
- Deals with missing combinations of parameters in ensembles.

We take a layered approach in the implementation:

1. Computations that need to be done once are performed in class initialization.
   Example: computing the Cholesky factor of the observation covariance
2. Computations that are done once per smoothing iteration are performed in the
   method `.prepare_assimilation()`.
   Example: computing the inversion of (Y @ Y.T + C_D)
3. Computations that are done done once per parameter group are performed in the
   method `.assimilate_batch()`. If you wish to assimilate all parameters in
   one batch, then just pass all parameters to this method.
   Example: computing the full product C_MD @ inv(C_DD + alpha * C_D) @ (D - Y)

References
----------

- Emerick, A.A., Reynolds, A.C. History matching time-lapse seismic data using
  the ensemble Kalman filter with multiple data assimilations.
  Comput Geosci 16, 639–659 (2012). https://doi.org/10.1007/s10596-012-9275-5
- Alexandre A. Emerick, Albert C. Reynolds.
  Ensemble smoother with multiple data assimilation.
  Computers & Geosciences, Volume 55, 2013, Pages 3-15, ISSN 0098-3004,
  https://doi.org/10.1016/j.cageo.2012.03.011
- https://gitlab.com/antoinecollet5/pyesmda
- https://helper.ipam.ucla.edu/publications/oilws3/oilws3_14147.pdf

"""

import numbers
from abc import ABC
from typing import Union

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore

from iterative_ensemble_smoother.esmda_inversion import (
    invert_subspace,
    normalize_alpha,
)
from iterative_ensemble_smoother.utils import adjust_for_missing, sample_mvnormal


class BaseESMDA(ABC):
    """Base class for all ESMDA classes.

    This class defines every method, apart from `assimilate_batch()`,
    which each subclass implements in their own way.
    """

    ALLOWED_DTYPES = (np.float16, np.float32, np.float64)

    def __init__(
        self,
        covariance: npt.NDArray[np.double],
        observations: npt.NDArray[np.double],
        alpha: Union[int, npt.NDArray[np.double]] = 5,
        seed: Union[np.random._generator.Generator, int, None] = None,
    ) -> None:
        """
        Parameters
        ----------
        covariance : np.ndarray
            Either a 1D array of diagonal covariances, or a 2D covariance matrix.
            The shape is (num_observations,) or (num_observations, num_observations).
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
        """

        # Validate inputs
        if not (isinstance(covariance, np.ndarray) and covariance.ndim in (1, 2)):
            raise TypeError(
                "Argument `covariance` must be a NumPy array of dimension 1 or 2."
            )

        if covariance.ndim == 2 and covariance.shape[0] != covariance.shape[1]:
            raise ValueError("Argument `covariance` must be square if it's 2D.")

        if not (isinstance(observations, np.ndarray) and observations.ndim == 1):
            raise TypeError("Argument `observations` must be a 1D NumPy array.")

        if not observations.shape[0] == covariance.shape[0]:
            raise ValueError("Shapes of `observations` and `covariance` must match.")

        if not (
            isinstance(seed, (int, np.random._generator.Generator)) or seed is None
        ):
            raise TypeError(
                "Argument `seed` must be an integer "
                "or numpy.random._generator.Generator."
            )

        if observations.dtype not in self.ALLOWED_DTYPES:
            raise ValueError(
                f"'observations' has unsupported dtype {observations.dtype}"
            )
        if covariance.dtype not in self.ALLOWED_DTYPES:
            raise ValueError(f"'covariance' has unsupported dtype {covariance.dtype}")
        if observations.dtype != covariance.dtype:
            raise ValueError(
                f"dtype mismatch: 'observations' is {observations.dtype}"
                "'covariance' is {covariance.dtype}"
            )

        self._dtype = observations.dtype

        # Store data
        self.observations = observations
        self.iteration = -1
        self.rng = np.random.default_rng(seed)

        # Only compute the covariance factorization once
        # If it's a full matrix, we gain speedup by only computing cholesky once
        # If it's a diagonal, we gain speedup by never having to compute cholesky
        self.C_D = covariance.copy()
        if covariance.ndim == 2:
            self.C_D_L = sp.linalg.cholesky(self.C_D, lower=False)
        elif covariance.ndim == 1:
            self.C_D_L = np.sqrt(self.C_D)
        else:
            raise TypeError("Argument `covariance` must be 1D or 2D array")

        if not (
            (isinstance(alpha, np.ndarray) and alpha.ndim == 1)
            or isinstance(alpha, numbers.Integral)
        ):
            raise TypeError("Argument `alpha` must be an integer or a 1D NumPy array.")

        # Alpha can either be an integer (num iterations) or a list of weights
        if isinstance(alpha, np.ndarray) and alpha.ndim == 1:
            self.alpha = normalize_alpha(alpha)
        elif isinstance(alpha, numbers.Integral):
            self.alpha = normalize_alpha(np.ones(alpha))
            assert np.allclose(self.alpha, normalize_alpha(self.alpha))
        else:
            raise TypeError("Alpha must be integer or 1D array.")

        self.alpha = self.alpha.astype(self._dtype)  # Convert to same dtype

    def num_assimilations(self) -> int:
        return len(self.alpha)

    def perturb_observations(
        self, *, ensemble_size: int, alpha: float
    ) -> npt.NDArray[np.double]:
        """Create a matrix D with perturbed observations.

        In the Emerick (2013) paper, the matrix D is defined in section 6.
        See section 2(b) of the ES-MDA algorithm in the paper.

        Parameters
        ----------
        ensemble_size : int
            The ensemble size, i.e., the number of columns in the returned array,
            which is of shape (num_observations, ensemble_size).
        alpha : float
            The covariance inflation factor. The sequence of alphas should
            obey the equation sum_i (1/alpha_i) = 1. However, this is NOT enforced
            in this method call. The user/caller is responsible for this.

        Returns
        -------
        D : np.ndarray
            Each column consists of perturbed observations,
            observation std is scaled by sqrt(alpha).
        """
        # Draw samples from zero-centered multivariate normal with cov=alpha * C_D,
        # and add them to the observations. Notice that
        # if C_D = L @ L.T by the cholesky factorization, then drawing y from
        # a zero cented normal means that y := L @ z, where z ~ norm(0, 1).
        # Therefore, scaling C_D by alpha is equivalent to scaling L with sqrt(alpha).
        samples = sample_mvnormal(
            C_dd_cholesky=self.C_D_L, rng=self.rng, size=ensemble_size
        ).astype(self.C_D_L.dtype)

        D: npt.NDArray[np.double] = (
            self.observations[:, np.newaxis] + (alpha**0.5) * samples
        )
        assert D.shape == (len(self.observations), ensemble_size)
        return D

    def prepare_assimilation(
        self,
        *,
        Y: npt.NDArray[np.double],
        truncation: float = 0.99,
        overwrite: bool = False,
    ) -> None:
        r"""Prepare assimilation of one or several batches of parameters.

        This method call pre-computes everything that is needed to assimilate
        a set of batches once. This saves time since we do not have to repeat
        the same computations for every single batch.

        Parameters
        ----------
        Y : np.ndarray
            2D array of shape (num_observations, ensemble_size), containing
            responses when evaluating the model at X. In other words, Y = g(X),
            where g is the forward model.
        truncation : float
            How large a fraction of the singular values to keep in the inversion
            routine (if the inversion routine supports it). Must be a float in
            the range (0, 1]. A lower number means a more approximate answer and a
            slightly faster computation. The default is 0.99, which is recommended
            by Emerick in the reference paper.
        overwrite: bool
            If False (the default), the input array will not be overwritten (mutated).
            If True, the method may overwrite the input array.

        Returns
        -------
        self
            The instance with mutated state.

        Notes
        -----
        In the equation:

            M + (\delta M)
                (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1)
                (D_obs - D)

        This method call corresponds to computing:

            - The next-to-last factors: (\delta D)^T [(\delta D) (\delta D)^T +
                                                      \alpha (N_e - 1) C_D]^(-1)
            - The last factor: (D_obs - D)

        The total internal storage is: 2 * ensemble_size * num_observations,
        since shapes are:

            - The next-to-last factors: (ensemble_size, num_observations)
            - The last factor: (num_observations, ensemble_size)
        """
        assert Y.ndim == 2
        assert 0 < truncation <= 1.0
        if not np.issubdtype(Y.dtype, np.floating):
            raise TypeError("Argument `Y` must contain floats")

        if Y.dtype != self._dtype:
            raise ValueError(
                f"'Y' has dtype {Y.dtype}, but class was "
                "initialized with dtype {self._dtype}"
            )

        if self.iteration >= self.num_assimilations():
            raise Exception("No more assimilation steps to run.")

        if not overwrite:
            Y = Y.copy()

        self.truncation = truncation
        self.iteration += 1

        D = Y  # Switch from API notation to paper notation
        N_d, N_e = D.shape  # (num_observations, ensemble_size)
        assert N_e >= 2, "Must have at least two ensemble members"
        assert N_d == self.observations.shape[0], "Shape mismatch"

        # Compute the last factor
        alpha = self.alpha[self.iteration]
        D_obs = self.perturb_observations(ensemble_size=N_e, alpha=alpha)
        self.D_obs_minus_D = D_obs - D

        # Compute parts of the Kalman gain
        D -= np.mean(D, axis=1, keepdims=True)  # Center observations
        self.delta_DT, self.term_diag, self.termT = invert_subspace(
            delta_D=D, C_D_L=self.C_D_L, alpha=alpha, truncation=truncation
        )

    def _compute_delta_M(
        self,
        *,
        X: npt.NDArray[np.floating],
        missing: Union[npt.NDArray[np.bool_], None] = None,
    ) -> npt.NDArray[np.floating]:
        """Prepare delta_M := X - center(X), dealing with missing values
        as needed."""
        if not np.issubdtype(X.dtype, np.floating):
            raise TypeError("Argument `X` must contain floats")
        if not (missing is None or np.issubdtype(missing.dtype, np.bool_)):
            raise TypeError("Argument `missing_mask` must contain booleans")
        if missing is not None and (not X.shape == missing.shape):
            raise ValueError(f"Shapes must match: {X.shape=} != {missing.shape}")
        if not X.ndim == 2:
            raise ValueError("X must have shape (num_parameters_batch, ensemble_size)")

        assert X.ndim == 2
        N_m, N_e = X.shape  # (num_parameters, ensemble_size)

        if X.dtype != self._dtype:
            raise ValueError(
                f"'X' has dtype {X.dtype}, but class was "
                "initialized with dtype {self._dtype}"
            )

        # Center the parameters, possibly accounting for missing data
        if missing is not None:
            return adjust_for_missing(X, missing=missing)
        return X - np.mean(X, axis=1, keepdims=True)


class ESMDA(BaseESMDA):
    """
    Implement an Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

    The implementation follows :cite:t:`EMERICK2013`.

    Examples
    --------

    A full example where the forward model maps 10 parameters to 3 outputs.
    We will use 100 realizations. First we define the forward model:

    >>> rng = np.random.default_rng(42)
    >>> A = rng.normal(size=(3, 10))
    >>> def forward_model(x):
    ...     return A @ x

    Then we set up the ESMDA instance and the prior realizations X:

    >>> covariance = np.ones(3, dtype=float)  # Covariance of the observations / outputs
    >>> observations = np.array([1, 2, 3], dtype=float)  # The observed data
    >>> esmda = ESMDA(covariance, observations, alpha=3, seed=42)
    >>> X = rng.normal(size=(10, 100))

    To assimilate data, we iterate over the assimilation steps:

    >>> for iteration in range(esmda.num_assimilations()):
    ...     # Apply the forward model in each realization in the ensemble
    ...     Y = np.array([forward_model(x) for x in X.T]).T
    ...     esmda.prepare_assimilation(Y=Y)
    ...     X = esmda.assimilate_batch(X=X)  # Update X
    """

    def assimilate_batch(
        self,
        *,
        X: npt.NDArray[np.floating],
        missing: Union[npt.NDArray[np.bool_], None] = None,
        overwrite: bool = False,
    ) -> npt.NDArray[np.floating]:
        """Assimilate a batch of parameters against all observations.

        The internal storage used by the class is 2 * ensemble_size * num_observations,
        so a good batch size that is of the same order of magnitude as the internal
        storage is 2 * num_observations. This is only a rough guideline.

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (num_parameters_batch, ensemble_size). Each row
            corresponds to a parameter in the model, and each column corresponds
            to an ensemble member (realization).
        missing : np.ndarray or None
            A boolean 2D array of shape (num_parameters_batch, ensemble_size).
            If an entry is True, then that value is assumed missing. This can
            happen if the ensemble members use different grids, where each
            ensemble member has a slightly different grid layout. If None,
            then all entries are assumed to be valid.
        overwrite: bool
            If False (the default), the input array will not be overwritten (mutated).
            If True, the method may overwrite the input array.

        Returns
        -------
        X_posterior : np.ndarray
            2D array of shape (num_parameters_batch, ensemble_size).

        """
        if not overwrite:
            X = X.copy()
            missing = missing if missing is None else missing.copy()
        if not hasattr(self, "delta_DT"):
            raise Exception("The method `prepare_assmilation` must be called.")
        N_m, N_e = X.shape  # (num_parameters, ensemble_size)
        assert N_e == self.delta_DT.shape[0], "Dimension mismatch"

        # In standard ESMDA, we simplify compute the product in a good order
        delta_M = self._compute_delta_M(X=X, missing=missing)
        X += np.linalg.multi_dot(
            [delta_M, self.delta_DT, self.term_diag, self.termT, self.D_obs_minus_D]
        )
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
