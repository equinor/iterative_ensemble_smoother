from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

from iterative_ensemble_smoother._ies import InversionType, create_coefficient_matrix
from iterative_ensemble_smoother.utils import (
    _validate_inputs,
    _create_errors,
    steplength_exponential,
    response_projection,
)


class SIES:
    """SIES performs the update step of the Subspace Iterative Ensemble Smoother
    algorithm. See `Evensen[1]`_.

    :param ensemble_size: The number of realizations in the ensemble model.
    :param steplength_schedule: A function that takes the iteration number (starting at 1) and returns steplength.
    :param seed: Integer used to seed the random number generator.
    """

    def __init__(
        self,
        ensemble_size: int,
        *,
        steplength_schedule: Optional[Callable[[int], float]] = None,
        seed: Optional[int] = None,
    ):
        self._initial_ensemble_size = ensemble_size
        self.iteration_nr = 1
        self.steplength_schedule = steplength_schedule
        self.coefficient_matrix = np.zeros(shape=(ensemble_size, ensemble_size))
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def fit(
        self,
        response_ensemble: npt.NDArray[np.double],
        observation_errors: npt.NDArray[np.double],
        observation_values: npt.NDArray[np.double],
        *,
        truncation: float = 0.98,
        step_length: Optional[float] = None,
        ensemble_mask: Optional[npt.NDArray[np.bool_]] = None,
        inversion: InversionType = InversionType.EXACT,
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
                step_length = steplength_exponential(self.iteration_nr)
            else:
                step_length = self.steplength_schedule(self.iteration_nr)

        assert 0 < step_length <= 1, "Step length must be in (0, 1]"

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
                loc=0, scale=observation_errors, size=(ensemble_size, num_obs)
            ).T

        assert E.shape == (num_obs, ensemble_size)

        E -= E.mean(axis=1, keepdims=True)

        R, observation_errors_std = _create_errors(observation_errors, inversion)

        # Store D as defined by Equation (14) in Evensen (2019)
        self.D_ = E + observation_values.reshape(num_obs, 1)
        D = self.D_ - response_ensemble

        # Scale D and E with observation error standard deviations.
        D /= observation_errors_std.reshape(num_obs, 1)
        E /= observation_errors_std.reshape(num_obs, 1)

        if param_ensemble is not None:
            projected_response = response_ensemble @ response_projection(param_ensemble)
            _response_ensemble = projected_response / observation_errors_std.reshape(
                num_obs, 1
            )
        else:
            _response_ensemble = response_ensemble / observation_errors_std.reshape(
                num_obs, 1
            )
        _response_ensemble -= _response_ensemble.mean(axis=1, keepdims=True)
        _response_ensemble /= np.sqrt(ensemble_size - 1)

        if ensemble_mask is None:
            ensemble_mask = np.ones(ensemble_size, dtype=bool)

        self.ensemble_mask = ensemble_mask

        W: npt.NDArray[np.double] = create_coefficient_matrix(
            _response_ensemble,
            R,
            E,
            D,
            inversion,
            truncation,
            self.coefficient_matrix[np.ix_(ensemble_mask, ensemble_mask)],
            step_length,
        )

        if np.isnan(W).sum() != 0:
            raise ValueError(
                "Fit produces NaNs. Check your response matrix for outliers or use an inversion type with truncation."
            )

        self.iteration_nr += 1

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
        return f"SIES(ensemble_size={self._initial_ensemble_size})"


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
        inversion: InversionType = InversionType.EXACT,
        param_ensemble: Optional[npt.NDArray[np.double]] = None,
    ) -> None:
        self.smoother = SIES(ensemble_size=response_ensemble.shape[1], seed=self.seed)
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
