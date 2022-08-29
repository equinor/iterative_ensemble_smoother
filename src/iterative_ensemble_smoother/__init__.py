""" Module which implements the iterative ensemble smoother history matching algorithm.

See  Evensen, G. "Analysis of iterative ensemble smoothers for solving inverse
problems." for details about the algorithm.

"""
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from ._ies import Config, ModuleData, init_update, inversion_type, make_D, make_E

if TYPE_CHECKING:
    import numpy.typing as npt


def make_X(
    Y: "npt.NDArray[np.double]",
    R: "npt.NDArray[np.double]",
    E: "npt.NDArray[np.double]",
    D: "npt.NDArray[np.double]",
    A: "npt.NDArray[np.double]" = np.empty(shape=(0, 0)),
    ies_inversion: inversion_type = inversion_type.EXACT,
    truncation: Union[float, int] = 0.98,
    W0: Optional["npt.NDArray[np.double]"] = None,
    step_length: float = 1.0,
    iteration: int = 1,
) -> Any:
    if W0 is None:
        W0 = np.zeros((Y.shape[1], Y.shape[1]))
    return _ies.make_X(
        A,
        Y,
        R,
        E,
        D,
        ies_inversion,
        truncation,
        W0,
        step_length,
        iteration,
    )


def update_A(
    data: ModuleData,
    A: "npt.NDArray[np.double]",
    Y: "npt.NDArray[np.double]",
    R: "npt.NDArray[np.double]",
    E: "npt.NDArray[np.double]",
    D: "npt.NDArray[np.double]",
    ies_inversion: inversion_type = inversion_type.EXACT,
    truncation: Union[float, int] = 0.98,
    step_length: float = 1.0,
) -> None:

    if not A.flags.fortran:
        A = np.asfortranarray(A)
    _ies.update_A(data, A, Y, R, E, D, ies_inversion, truncation, step_length)
    return A


def ensemble_smoother_update_step(
    S,
    A,
    observation_errors,
    observation_values,
    noise=None,
    truncation=0.98,
    inversion=inversion_type.EXACT,
):
    if noise is None:
        noise = np.random.rand(*S.shape)
    E = make_E(observation_errors, noise)
    R = np.identity(len(observation_errors), dtype=np.double)
    D = make_D(observation_values, E, S)
    D = (D.T / observation_errors).T
    E = (E.T / observation_errors).T
    S = (S.T / observation_errors).T

    X = make_X(
        S,
        R,
        E,
        D,
        A,
        ies_inversion=inversion,
        truncation=truncation,
    )
    return A @ X


class IterativeEnsembleSmoother:
    """IterativeEnsembleSmoother performs the update step of the iterative ensemble ensemble
    algorithm. See `Evensen[1]`_.

    :param ensemble_size: The number of realizations in the ensemble model.
    :param max_steplength: parameter used to tweaking the step length.
    :param min_steplength: parameter used to tweaking the step length.
    :param dec_steplength: parameter used to tweaking the step length.
    """

    def __init__(
        self, ensemble_size, max_steplength=0.6, min_steplength=0.3, dec_steplength=2.5
    ):
        self._module_data = ModuleData(ensemble_size)
        self._config = Config(True)
        self._ensemble_size = ensemble_size
        self.max_steplength = max_steplength
        self.min_steplength = min_steplength
        self.dec_steplength = dec_steplength

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
        S,
        A,
        observation_errors,
        observation_values,
        noise=None,
        truncation=0.98,
        step_length=None,
        ensemble_mask=None,
        observation_mask=None,
        inversion=inversion_type.EXACT,
    ):
        if step_length is None:
            step_length = self._get_steplength(self._module_data.iteration_nr)
        if noise is None:
            noise = np.random.rand(*S.shape)
        if ensemble_mask is None:
            ensemble_mask = [True] * self._ensemble_size
        if observation_mask is None:
            observation_mask = [True] * len(observation_values)

        E = make_E(observation_errors, noise)
        R = np.identity(len(observation_errors), dtype=np.double)
        D = make_D(observation_values, E, S)
        D = (D.T / observation_errors).T
        E = (E.T / observation_errors).T
        S = (S.T / observation_errors).T

        init_update(self._module_data, ensemble_mask, observation_mask)

        A = update_A(
            self._module_data,
            A,
            S,
            R,
            E,
            D,
            ies_inversion=inversion,
            truncation=truncation,
            step_length=step_length,
        )
        self._module_data.iteration_nr += 1
        return A


def ensemble_smoother_update_step_row_scaling(
    S,
    A_with_row_scaling,
    observation_errors,
    observation_values,
    noise=None,
    truncation=0.98,
    inversion=inversion_type.EXACT,
):
    if noise is None:
        noise = np.random.rand(*S.shape)

    E = make_E(observation_errors, noise)
    R = np.identity(len(observation_errors), dtype=np.double)
    D = make_D(observation_values, E, S)
    D = (D.T / observation_errors).T
    E = (E.T / observation_errors).T
    S = (S.T / observation_errors).T
    for (A, row_scale) in A_with_row_scaling:
        X = make_X(
            S,
            R,
            E,
            D,
            A,
            ies_inversion=inversion,
            truncation=truncation,
        )
        row_scale.multiply(A, X)
    return A_with_row_scaling


__all__ = [
    "ensemble_smoother_update_step",
    "IterativeEnsembleSmoother",
    "ensemble_smoother_update_step_row_scaling",
]
