from typing import TYPE_CHECKING, Optional

import numpy as np

from ._ies import InversionType, ModuleData, init_update, make_D, make_E, update_A

if TYPE_CHECKING:
    import numpy.typing as npt


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
        self._module_data = ModuleData(ensemble_size)
        self._ensemble_size = ensemble_size
        self.max_steplength = max_steplength
        self.min_steplength = min_steplength
        self.dec_steplength = dec_steplength

    @property
    def iteration_nr(self):
        """The number of update steps that have been run"""
        return self._module_data.iteration_nr

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
        sensitivity_matrix: "npt.NDArray[np.double]",
        centered_anomaly_matrix: "npt.NDArray[np.double]",
        observation_errors: "npt.NDArray[np.double]",
        observation_values: "npt.NDArray[np.double]",
        noise: Optional["npt.NDArray[np.double]"] = None,
        truncation: float = 0.98,
        step_length: Optional[float] = None,
        ensemble_mask: Optional["npt.ArrayLike"] = None,
        observation_mask: Optional["npt.ArrayLike"] = None,
        inversion: InversionType = InversionType.EXACT,
    ):
        """Perform one step of the iterative ensemble smoother algorithm

        :param sensitivity_matrix: Matrix of responses from the :term:`forward model`.
            Has shape (number of observations, number of realizations). (S in Evensen et al.)
        :param centered_anomaly_matrix: Matrix of sampled model parameters. Has shape
            (number of parameters, number of realizations). (A in Evensen et al.)
        :param observation_errors: List of measurement of errors for each observation.
        :param observation_values: List of observations.
        :param noise: Optional list of noise used in the algorithm, Has same shape as
            response matrix.
        :param truncation: float used to determine the number of significant singular
            values. Defaults to 0.98 (ie. 98% significant values).
        :param step_length: The step length to be used in the algorithm,
            defaults to using the method described in Eq. 49 Geir Evensen,
            Formulating the history matching problem with consistent error
            statistics, Computational Geosciences (2021) 25:945 –970
        :param ensemble_mask: An array describing which realizations are active. Defaults
            to all active. Inactive realizations are ignored.
        :param observation_mask: An array describing which observations are active. Defaults
            to all active. Inactive observations are ignored.
        :param inversion: The type of subspace inversion used in the algorithm, defaults
            to exact.
        """
        S = sensitivity_matrix
        A = centered_anomaly_matrix
        if step_length is None:
            step_length = self._get_steplength(self._module_data.iteration_nr)
        if noise is None:
            noise = np.random.rand(*S.shape)
        if ensemble_mask is None:
            ensemble_mask = np.array([True] * self._ensemble_size)
        if observation_mask is None:
            observation_mask = np.array([True] * len(observation_values))
        if not A.flags.fortran:
            A = np.asfortranarray(A)

        E = make_E(observation_errors, noise)
        R = np.identity(len(observation_errors), dtype=np.double)
        D = make_D(observation_values, E, S)
        D = (D.T / observation_errors).T
        E = (E.T / observation_errors).T
        S = (S.T / observation_errors).T

        init_update(self._module_data, ensemble_mask, observation_mask)

        update_A(
            self._module_data,
            A,
            S,
            R,
            E,
            D,
            inversion,
            truncation,
            step_length,
        )
        self._module_data.iteration_nr += 1
        return A
