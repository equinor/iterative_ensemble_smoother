from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

from iterative_ensemble_smoother.utils import (
    _validate_inputs,
    _create_errors,
    steplength_exponential,
    response_projection,
)

from iterative_ensemble_smoother.ies import create_coefficient_matrix


class SIES:
    """
    Initialize a Subspace Iterative Ensemble Smoother (SIES) instance.

    This is an implementation of the algorithm described in the paper:
    Efficient Implementation of an Iterative Ensemble Smoother for Data Assimilation and Reservoir History Matching
    written by Evensen et al (2019), URL: https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full

    The default step length is described in equation (49) in the paper
    Formulating the history matching problem with consistent error statistics
    written by Geir Evensen (2021), URL: https://link.springer.com/article/10.1007/s10596-021-10032-7

    Parameters
    ----------
    steplength_schedule : Optional[Callable[[int], float]], optional
        A function that takes the iteration number (starting at 1) and returns steplength.
        The default is None, which defaults to using an exponential decay.
        See the references or the function `steplength_exponential`.
    seed : Optional[int], optional
        Integer used to seed the random number generator.. The default is None.

    Examples
    --------
    >>> steplength_schedule = lambda iteration: 0.8 * 2**(-iteration - 1)
    >>> smoother = SIES(steplength_schedule=steplength_schedule, seed=42)
    """

    def __init__(
        self,
        *,
        steplength_schedule: Optional[Callable[[int], float]] = None,
        seed: Optional[int] = None,
    ):
        self.iteration_nr = 1
        self.steplength_schedule = steplength_schedule
        self.seed = seed

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
        """Perform one Gauss-Newton step and update the coefficient matrix W.

        Parameters
        ----------
        response_ensemble : npt.NDArray[np.double]
            A 2D array of reponses from the model g(X) of shape (observations, ensemble_size).
            This matrix is Y in Evensen (2019).
        observation_errors : npt.NDArray[np.double]
            Either a 1D array of standard deviations, or a 2D covariance matrix.
            This is C_dd in Evensen (2019), and represents observation or measurement
            errors. We observe d from the real world, y from the model g(x), and
            assume that d = y + e, where e is multivariate normal with covariance
            or standard devaiations given by observation_errors.
        observation_values : npt.NDArray[np.double]
            A 1D array of observations, with shape (observations,).
            This is d in Evensen (2019).
        truncation : float, optional
            A value in the range [0, 1], used to determine the number of
            significant singular values. The default is 0.98.
        step_length : Optional[float], optional
            If given, this value (in the range [0, 1]) overrides the step length
            schedule that was provided at initialization. The default is None.
        ensemble_mask : Optional[npt.NDArray[bool]], optional
            A 1D array of booleans describing which realizations in the ensemble
            that are are active. The default is None, which means all realizations
            are active. Inactive realizations are ignored (not updated).
            Must be of shape (ensemble_size,).
        inversion : InversionType, optional
            The type of subspace inversion used in the algorithm.
            The default is InversionType.EXACT.
        param_ensemble : Optional[npt.NDArray[np.double]], optional
            All parameters input to dynamical model used to generate responses.
            Must be passed if total number of parameters is less than
            ensemble_size - 1 and the dynamical model g(x) is non-linear.
            This is X in Evensen (2019), and has shape (parameters, ensemble_size).
            The default is None. See section 2.4.3 in Evensen (2019).

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

        # If it's the first time the method is called, create coeff matrix
        if not hasattr(self, "coefficient_matrix"):
            self.coefficient_matrix = np.zeros(shape=(ensemble_size, ensemble_size))

        # Seed here so we draw the same samples in every iteration (call to update())
        rng = np.random.default_rng(self.seed)

        # A covariance matrix was passed
        # Columns of E should be sampled from N(0, Cdd) and centered, Evensen 2019
        if observation_errors.ndim == 2:
            E = rng.multivariate_normal(
                mean=np.zeros_like(observation_values),
                cov=observation_errors,
                size=ensemble_size,
                method="cholesky",  # An order of magnitude faster than 'svd'
            ).T
        # A vector of standard deviations was passed
        else:
            E = rng.normal(
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

        W: npt.NDArray[np.double] = create_coefficient_matrix(  # type: ignore
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
        """Update the parameter ensemble (X in Evensen (2019)).

        Parameters
        ----------
        param_ensemble : npt.NDArray[np.double]
            The (prior) parameter ensemble. The same `param_ensemble` should be
            used in each Gauss-Newton step.

        Returns
        -------
        np.ndarray
            Updated parameter ensemble.

        """
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
