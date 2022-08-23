from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

# pylint: disable=import-error
from ._ies import Config, ModuleData, init_update, inversion_type, make_D, make_E

if TYPE_CHECKING:
    import numpy.typing as npt


def make_X(  # pylint: disable=too-many-arguments
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
    return _ies.make_X(  # pylint: disable=no-member, c-extension-no-member
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


def update_A(  # pylint: disable=too-many-arguments
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
        raise TypeError("A matrix must be F_contiguous")
    _ies.update_A(  # pylint: disable=no-member, c-extension-no-member
        data, A, Y, R, E, D, ies_inversion, truncation, step_length
    )


def ensemble_smoother_update_step(
    S,
    A,
    observation_errors,
    observation_values,
    noise,
    truncation=0.98,
    inversion=inversion_type.EXACT,
):
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
    def __init__(
        self, ensemble_size, max_steplength=0.6, min_steplength=0.3, dec_steplength=2.5
    ):
        self._module_data = ModuleData(ensemble_size)
        self._config = Config(True)
        self.max_steplength = max_steplength
        self.min_steplength = min_steplength
        self.dec_steplength = dec_steplength

    def _get_steplength(self, iteration_nr: int) -> float:
        # This is an implementation of Eq. (49) from the book:
        # Geir Evensen, Formulating the history matching problem with consistent error statistics,
        # Computational Geosciences (2021) 25:945 –970: https://doi.org/10.1007/s10596-021-10032-7
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
        truncation=0.98,
        noise=None,
        step_length=None,
        ensemble_mask=None,
        observation_mask=None,
        inversion=inversion_type.EXACT,
    ):
        if step_length is None:
            step_length = self._get_steplength(self._module_data.iteration_nr)

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
    noise,
    truncation=0.98,
    inversion=inversion_type.EXACT,
):
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
