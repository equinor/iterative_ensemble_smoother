from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

from iterative_ensemble_smoother.utils import (
    _validate_inputs,
    covariance_to_correlation,
    steplength_exponential,
    response_projection,
)

from iterative_ensemble_smoother.ies import create_coefficient_matrix


class SIES:
    """SIES performs the update step of the Subspace Iterative Ensemble Smoother
    algorithm.

    :param ensemble_size: The number of realizations in the ensemble model.
    :param steplength_schedule: A function that takes the iteration number (starting at 1) and returns steplength.
    :param seed: Integer used to seed the random number generator.
    """

    _inversion_methods = ("naive", "exact", "exact_r", "subspace_re")

    def __init__(
        self,
        *,
        steplength_schedule: Optional[Callable[[int], float]] = None,
        seed: Optional[int] = None,
    ):
        self.iteration = 1
        self.steplength_schedule = steplength_schedule
        self.rng = np.random.default_rng(seed)

    def _get_E(
        self,
        *,
        observation_errors: npt.NDArray[np.double],
        observation_values: npt.NDArray[np.double],
        ensemble_size: int,
    ) -> npt.NDArray[np.double]:
        """Draw samples from N(0, Cdd). Use cached values if already drawn."""

        # Return cached values if they exist
        if hasattr(self, "E_"):
            return self.E_

        # A covariance matrix was passed
        # Columns of E should be sampled from N(0, Cdd) and centered, Evensen 2019
        if observation_errors.ndim == 2:
            E = self.rng.multivariate_normal(
                mean=np.zeros_like(observation_values),
                cov=observation_errors,
                size=ensemble_size,
                method="cholesky",  # An order of magnitude faster than 'svd'
            ).T
        # A vector of standard deviations was passed
        else:
            E = self.rng.normal(
                loc=0,
                scale=observation_errors,
                size=(ensemble_size, len(observation_errors)),
            ).T

        # Center values, removing one degree of freedom
        E -= E.mean(axis=1, keepdims=True)

        self.E_: npt.NDArray[np.double] = E
        return self.E_

    def fit(
        self,
        response_ensemble: npt.NDArray[np.double],
        observation_errors: npt.NDArray[np.double],
        observation_values: npt.NDArray[np.double],
        *,
        truncation: float = 0.98,
        step_length: Optional[float] = None,
        ensemble_mask: Optional[npt.NDArray[np.bool_]] = None,
        inversion: str = "exact",
        param_ensemble: Optional[npt.NDArray[np.double]] = None,
    ) -> None:
        """Perform one step of the iterative ensemble smoother algorithm

        :param response_ensemble: Matrix of responses from the :term:`forward model`.
            Has shape (number of observations, number of realizations).
            (Y in Evensen et. al)
        :param observation_errors: 1D array of measurement errors (standard deviations)
                                   for each observation, or covariance matrix
                                   if errors are correlated.
        :param observation_values: 1D array of observations.
        :param truncation: float used to determine the number of significant singular
            values. Defaults to 0.98 (ie. 98% significant values).
        :param step_length: The step length to be used in the algorithm,
            defaults to using the method described in Eq. 49 Geir Evensen,
            Formulating the history matching problem with consistent error
            statistics, Computational Geosciences (2021) 25:945 –970
        :param ensemble_mask: 1D array describing which realizations are active. Defaults
            to all active. Inactive realizations are ignored.
        :param inversion: The type of subspace inversion used in the algorithm, defaults
            to exact.
        :param param_ensemble: All parameters input to dynamical model used to
            generate responses.
            Must be passed if total number of parameters is
            less than ensemble_size - 1 and the dynamical model is non-linear.
        """

        # ---------------------------------------------------------------------
        # ----------------- Input validation and setup ------------------------
        # ---------------------------------------------------------------------

        _validate_inputs(
            response_ensemble,
            observation_errors,
            observation_values,
            param_ensemble=param_ensemble,
        )

        num_obs = len(observation_values)
        ensemble_size = response_ensemble.shape[1]

        # Determine the step length
        if step_length is None:
            if self.steplength_schedule is None:
                step_length = steplength_exponential(self.iteration)
            else:
                step_length = self.steplength_schedule(self.iteration)

        assert 0 < step_length <= 1, "Step length must be in (0, 1]"

        if ensemble_mask is None:
            ensemble_mask = np.ones(ensemble_size, dtype=bool)
        self.ensemble_mask = ensemble_mask

        # If it's the first time the method is called, create coeff matrix
        if not hasattr(self, "coefficient_matrix"):
            self.coefficient_matrix = np.zeros(shape=(ensemble_size, ensemble_size))

        # ---------------------------------------------------------------------
        # ----------- Computations corresponding to algorithm 1 ---------------
        # ---------------------------------------------------------------------

        # Draw samples from N(0, C_dd)
        E = self._get_E(
            observation_errors=observation_errors,
            observation_values=observation_values,
            ensemble_size=ensemble_size,
        )
        assert E.shape == (num_obs, ensemble_size)

        # Transform covariance matrix to correlation matrix
        R, observation_errors_std = covariance_to_correlation(observation_errors)

        # Store D as defined by Equation (14) in Evensen (2019)
        self.D_ = E + observation_values.reshape(num_obs, 1)
        D = self.D_ - response_ensemble

        # Note on scaling
        # -------------------
        # In line 8 in Algorithm 1, we have to compute (S S^T + E E^T)^-1 H, where
        # H := (SW + D - g(X)). This is equivalent to solving the following
        # equation for an unknown matrix M:
        #     (S S^T + E E^T) M = (SW + D - g(X))
        # Afterwards we compute S.T M. If we scale the rows (observed variables)
        # of these equations using the standard deviations, we can obtain better
        # conditioning numbers on the equation. This corresponds to left-multiplying
        # with a diagonal matrix L := sqrt(diag(C_dd)). In an experiment with
        # random covariance matrices, this improved the condition number ~90%
        # of the time (results may depend on how random covariance matrices are
        # generated --- I generated covariances C as E ~ stdnorm(), then C = E.T @ E).
        # To see the equality, note that if we scale S, E and H := (SW + D - g(X))
        # we obtain:
        #     (L S) (L S)^T + (L E) (L E)^T M_2 = L H
        #               L (S S^T + E E^T) L M_2 = L H
        #            L (S S^T + E E^T) L L^-1 M = L H
        # so the new solution is M_2 := L^-1 M, expressed in terms of the original M.
        # But when we left-multiply M_2 with S^T, we do so with a transformed S,
        # so we obtain S_2^T M_2 = (L S)^T (L^-1 M) = S^T M, so the solution
        # to the transformed system is equal to the solution of the original system.
        # In the implementation of scaling the right hand side (SW + D - g(X))
        # we first scale D - g(X), then we scale S implicitly by solving
        # S Sigma = L Y, instead of S Sigma = Y for S.

        # Scale D and E with observation error standard deviations.
        D /= observation_errors_std.reshape(num_obs, 1)

        # Here we have to make a new copy of E, since if not we would
        # divide the same E by the standard deviations in every iteration
        E = E / observation_errors_std.reshape(num_obs, 1)

        # See section 2.4 in the paper
        if param_ensemble is not None:
            response_ensemble = response_ensemble @ response_projection(param_ensemble)

        _response_ensemble = response_ensemble / observation_errors_std.reshape(
            num_obs, 1
        )
        _response_ensemble -= _response_ensemble.mean(axis=1, keepdims=True)
        _response_ensemble /= np.sqrt(ensemble_size - 1)

        W: npt.NDArray[np.double] = create_coefficient_matrix(  # type: ignore
            _response_ensemble,
            R,  # Correlation matrix or None (if 1D array was passed)
            E,  # Samples from multivariate normal
            D,  # D - G(x) in line 7 in algorithm 1
            inversion,
            truncation,
            self.coefficient_matrix[np.ix_(ensemble_mask, ensemble_mask)],
            step_length,
        )

        if np.isnan(W).sum() != 0:
            raise ValueError(
                "Fit produces NaNs. Check your response matrix for outliers or use an inversion type with truncation."
            )

        self.iteration += 1

        # Put the values back into the coefficient matrix
        self.coefficient_matrix[np.ix_(ensemble_mask, ensemble_mask)] = W

    def update(self, param_ensemble: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        # Line 9 of Algorithm 1
        ensemble_size = self.ensemble_mask.sum()

        # First get W, then divide by square root, then add identity matrix
        # Equivalent to (I + W / np.sqrt(ensemble_size - 1))
        transition_matrix: npt.NDArray[np.double] = self.coefficient_matrix[
            np.ix_(self.ensemble_mask, self.ensemble_mask)
        ]
        transition_matrix /= np.sqrt(ensemble_size - 1)
        transition_matrix.flat[:: ensemble_size + 1] += 1.0

        return param_ensemble @ transition_matrix

    def __repr__(self) -> str:
        return "SIES()"


class ES:
    def __init__(
        self,
        seed: Optional[int] = None,
    ) -> None:
        self.smoother: Optional[SIES] = None
        self.seed = seed

    def fit(
        self,
        response_ensemble: npt.NDArray[np.double],
        observation_errors: npt.NDArray[np.double],
        observation_values: npt.NDArray[np.double],
        *,
        truncation: float = 0.98,
        inversion: str = "exact",
        param_ensemble: Optional[npt.NDArray[np.double]] = None,
    ) -> None:
        self.smoother = SIES(seed=self.seed)
        self.smoother.fit(
            response_ensemble,
            observation_errors,
            observation_values,
            truncation=truncation,
            step_length=1.0,
            ensemble_mask=None,
            inversion=inversion,
            param_ensemble=param_ensemble,
        )

    def update(self, param_ensemble: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        assert self.smoother is not None
        return self.smoother.update(param_ensemble)

    def __repr__(self) -> str:
        return "ES()"
