from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

rng = np.random.default_rng()

from ._ies import InversionType, make_D, make_E, update_A
from iterative_ensemble_smoother.utils import (
    _compute_AA_projection,
    _validate_inputs,
    _create_errors,
)


class IterativeEnsembleSmoother:
    """IterativeEnsembleSmoother performs the update step of the iterative ensemble ensemble
    algorithm. See `Evensen[1]`_.

    :param ensemble_size: The number of realizations in the ensemble model.
    :param max_steplength: parameter used to tweaking the step length.
    :param min_steplength: parameter used to tweaking the step length.
    :param dec_steplength: parameter used to tweaking the step length.
    """

    def __init__(
        self,
        ensemble_size: int,
        max_steplength: float = 0.6,
        min_steplength: float = 0.3,
        dec_steplength: float = 2.5,
    ):
        self.iteration_nr = 1
        self._initial_ensemble_size = ensemble_size
        self.max_steplength = max_steplength
        self.min_steplength = min_steplength
        self.dec_steplength = dec_steplength
        self.coefficient_matrix = np.zeros(shape=(ensemble_size, ensemble_size))

    def _get_steplength(self, iteration_nr: int) -> float:
        """
        This is an implementation of Eq. (49), which calculates a suitable step length for
        the update step, from the book:

        Geir Evensen, Formulating the history matching problem with consistent error statistics,
        Computational Geosciences (2021) 25:945 –970
        """
        steplength = self.min_steplength + (
            self.max_steplength - self.min_steplength
        ) * pow(2, -(iteration_nr - 1) / (self.dec_steplength - 1))
        return steplength

    def update_step(
        self,
        response_ensemble: npt.NDArray[np.double],
        parameter_ensemble: npt.NDArray[np.double],
        observation_errors: npt.NDArray[np.double],
        observation_values: npt.NDArray[np.double],
        noise: Optional[npt.NDArray[np.double]] = None,
        truncation: float = 0.98,
        projection: bool = True,
        step_length: Optional[float] = None,
        ensemble_mask: Optional[npt.NDArray[np.bool_]] = None,
        observation_mask: Optional[npt.NDArray[np.bool_]] = None,
        inversion: InversionType = InversionType.EXACT,
    ) -> npt.NDArray[np.double]:
        """Perform one step of the iterative ensemble smoother algorithm

        :param response_ensemble: Matrix of responses from the :term:`forward model`.
            Has shape (number of active observations, number of realizations).
            (Y in Evensen et. al)
            Assumes that `observation_mask` and `ensemble_mask` have been applied, for example
            `response_ensemble = response_ensemble[observation_mask, :]`
        :param parameter_ensemble: Matrix of sampled model parameters. Has shape
            (number of active parameters, number of realizations) (A in Evensen et. al).
            Assumes that `ensemble_mask` has been applied, for example
            `parameter_ensemble = parameter_ensemble[:, ensemble_mask]`
        :param observation_errors: List of active measurement of errors for each observation.
            Assumes that `observation_mask` has been applied, for example
            `observation_errors = observation_errors[observation_mask]`
        :param observation_values: List of active observations.
            Assumes that `observation_mask` has been applied, for example
            `observation_value = observation_values[observation_mask]`
        :param noise: Optional list of noise used in the algorithm, Has same shape as
            response matrix.
        :param truncation: float used to determine the number of significant singular
            values. Defaults to 0.98 (ie. 98% significant values).
        :param projection: Whether to project response matrix.
        :param step_length: The step length to be used in the algorithm,
            defaults to using the method described in Eq. 49 Geir Evensen,
            Formulating the history matching problem with consistent error
            statistics, Computational Geosciences (2021) 25:945 –970
        :param ensemble_mask: An array describing which realizations are active. Defaults
            to all active. Inactive realizations are ignored.
        :param observation_mask: An array describing which observations are active. Defaults
            to all active. Inactive observations are ignored.
            To be deprecated.
        :param inversion: The type of subspace inversion used in the algorithm, defaults
            to exact.
        """

        R, observation_errors = _create_errors(observation_errors, inversion)

        _validate_inputs(
            response_ensemble,
            parameter_ensemble,
            noise,
            observation_errors,
            observation_values,
        )

        num_obs = len(observation_values)
        # Note that this may differ from self._initial_ensemble_size,
        # as realizations may get deactivated between iterations.
        num_params = parameter_ensemble.shape[0]
        ensemble_size = parameter_ensemble.shape[1]
        if step_length is None:
            step_length = self._get_steplength(self.iteration_nr)
        if noise is None:
            noise = rng.standard_normal(size=(num_obs, ensemble_size))

        E = make_E(observation_errors, noise)
        D = make_D(observation_values, E, response_ensemble)
        D = (D.T / observation_errors).T

        E = (E.T / observation_errors).T
        response_ensemble = (response_ensemble.T / observation_errors).T

        if projection and (num_params < self._initial_ensemble_size - 1):
            AA_projection = _compute_AA_projection(parameter_ensemble)
            response_ensemble = response_ensemble @ AA_projection

        if ensemble_mask is None:
            ensemble_mask = np.array([True] * ensemble_size)

        coefficient_matrix = update_A(
            parameter_ensemble,
            (response_ensemble - response_ensemble.mean(axis=1, keepdims=True))
            / np.sqrt(self._initial_ensemble_size - 1),
            R,
            E,
            D,
            self.coefficient_matrix[ensemble_mask, :][:, ensemble_mask],
            inversion,
            truncation,
            step_length,
        )

        self.coefficient_matrix[
            np.outer(ensemble_mask, ensemble_mask)
        ] = coefficient_matrix.ravel()
        self.iteration_nr += 1
        return parameter_ensemble
