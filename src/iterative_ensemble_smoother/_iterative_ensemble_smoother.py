from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

rng = np.random.default_rng()

from ._ies import InversionType, create_coefficient_matrix
from iterative_ensemble_smoother.utils import (
    _validate_inputs,
    _create_errors,
)


def _response_projection(
    param_ensemble: npt.NDArray[np.double],
) -> npt.NDArray[np.double]:
    """A^+A projection is necessary when the parameter matrix has fewer rows than
    columns, and when the forward model is non-linear. Section 2.4.3
    """
    ensemble_size = param_ensemble.shape[1]
    A = (param_ensemble - param_ensemble.mean(axis=1, keepdims=True)) / np.sqrt(
        ensemble_size - 1
    )
    projection: npt.NDArray[np.double] = np.linalg.pinv(A) @ A
    return projection


class SIES:
    """SIES performs the update step of the Subspace Iterative Ensemble Smoother
    algorithm. See `Evensen[1]`_.

    :param ensemble_size: The number of realizations in the ensemble model.
    :param max_steplength: parameter used to tweaking the step length.
    :param min_steplength: parameter used to tweaking the step length.
    :param dec_steplength: parameter used to tweaking the step length.
    """

    def __init__(
        self,
        ensemble_size: int,
        *,
        max_steplength: float = 0.6,
        min_steplength: float = 0.3,
        dec_steplength: float = 2.5,
    ):
        self._initial_ensemble_size = ensemble_size
        self.iteration_nr = 1
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
        if step_length is None:
            step_length = self._get_steplength(self.iteration_nr)

        # A covariance matrix was passed
        # Columns of E should be sampled from N(0,Cdd) and centered, Evensen 2019
        if len(observation_errors.shape) == 2:
            E = rng.multivariate_normal(
                mean=np.zeros_like(num_obs), cov=observation_errors, size=ensemble_size
            ).T
        # A vector of standard deviations was passed
        else:
            E = rng.normal(
                loc=0, scale=observation_errors, size=(ensemble_size, num_obs)
            ).T

        assert E.shape == (num_obs, ensemble_size)

        E -= E.mean(axis=1, keepdims=True)

        R, observation_errors = _create_errors(observation_errors, inversion)

        D = (E + observation_values.reshape(num_obs, 1)) - response_ensemble

        # Scale D and E with observation error standard deviations.
        D /= observation_errors.reshape(num_obs, 1)
        E /= observation_errors.reshape(num_obs, 1)

        if param_ensemble is not None:
            projected_response = response_ensemble @ _response_projection(
                param_ensemble
            )
            _response_ensemble = projected_response / observation_errors.reshape(
                num_obs, 1
            )
        else:
            _response_ensemble = response_ensemble / observation_errors.reshape(
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
        return (
            f"SIES(ensemble_size={self._initial_ensemble_size}, "
            f"max_steplength={self.max_steplength}, min_steplength={self.min_steplength}, "
            f"dec_steplength={self.dec_steplength})"
        )


class ES:
    def __init__(self) -> None:
        self.smoother: Optional[SIES] = None

    def fit(
        self,
        response_ensemble: npt.NDArray[np.double],
        observation_errors: npt.NDArray[np.double],
        observation_values: npt.NDArray[np.double],
        *,
        noise: Optional[npt.NDArray[np.double]] = None,
        truncation: float = 0.98,
        inversion: InversionType = InversionType.EXACT,
        param_ensemble: Optional[npt.NDArray[np.double]] = None,
    ) -> None:
        self.smoother = SIES(ensemble_size=response_ensemble.shape[1])
        self.smoother.fit(
            response_ensemble,
            observation_errors,
            observation_values,
            noise=noise,
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
