from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

from iterative_ensemble_smoother.ies import create_coefficient_matrix
from iterative_ensemble_smoother.utils import (
    SiesInversionType,
    covariance_to_correlation,
    response_projection,
    steplength_exponential,
    validate_inputs,
    validate_observations,
)


class SIES:
    """
    Subspace Iterative Ensemble Smoother (SIES).

    This is an implementation of the algorithm described in the paper:
    Efficient Implementation of an Iterative Ensemble Smoother for Data Assimilation and Reservoir History Matching
    written by :cite:t:`evensen2019efficient`.

    The default step length is described in equation (49) in the paper
    Formulating the history matching problem with consistent error statistics
    written by :cite:t:`evensen2021formulating`.

    Attributes
    ----------
    observation_errors : npt.NDArray[np.double]
        Either a 1D array of standard deviations, or a 2D covariance matrix.
        This is C_dd in :cite:t:`evensen2019efficient`, and represents observation or measurement
        errors. We observe d from the real world, y from the model g(x), and
        assume that d = y + e, where e is multivariate normal with covariance
        or standard devaiations given by observation_errors.
    observation_values : npt.NDArray[np.double]
        A 1D array of observations, with shape (observations,).
        This is d in :cite:t:`evensen2019efficient`.
    iteration: int
        Current iteration number (starts at 1).
    steplength_schedule : Optional[Callable[[int], float]], optional
        A function that takes as input the iteration number (starting at 1) and
        returns steplength (a float in the range (0, 1]).
        The default is None, which defaults to using an exponential decay.
        See the references or the function `steplength_exponential`.
    seed : Optional[int], optional
        Integer used to seed the random number generator. The default is None.
    rng: np.random.RandomState
        Pseudorandom number generator state used to generate samples.
    coefficient_matrix: npt.NDArray[np.double]
        Transition matrix used for the update. This is W in
        :cite:t:`evensen2019efficient`.
    ensemble_mask: Optional[npt.NDArray[np.bool_]]
        A 1D array of booleans describing which realizations in the ensemble
        that are are active. The default is None, which means all realizations
        are active. Inactive realizations are ignored (not updated).
        Must be of shape (ensemble_size,).
    E_: npt.NDArray[np.double]
        Samples from N(0, Cdd) used to perturb the residuals D.

    Examples
    --------
    >>> obs = np.random.random(50)
    >>> obs_std = np.random.random(50)
    >>> ensemble_size = 100
    >>> steplength_schedule = lambda iteration: 0.8 * 2**(-iteration - 1)
    >>> smoother = SIES(obs, obs_std, ensemble_size, steplength_schedule=steplength_schedule, seed=42)
    """

    __slots__ = [
        "observation_values",
        "observation_errors",
        "iteration",
        "steplength_schedule",
        "rng",
        "coefficient_matrix",
        "ensemble_mask",
        "E_",
    ]

    def __init__(
        self,
        observation_errors: npt.NDArray[np.double],
        observation_values: npt.NDArray[np.double],
        ensemble_size: int,
        *,
        steplength_schedule: Optional[Callable[[int], float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        observation_errors : npt.NDArray[np.double]
            Either a 1D array of standard deviations, or a 2D covariance matrix.
            This is C_dd in :cite:t:`evensen2019efficient`,
            and represents observation or measurement
            errors. We observe d from the real world, y from the model g(x), and
            assume that d = y + e, where e is multivariate normal with covariance
            or standard devaiations given by observation_errors.
        observation_values : npt.NDArray[np.double]
            A 1D array of observations, with shape (observations,).
            This is d in :cite:t:`evensen2019efficient`.
        ensemble_size: int
            Number of members in the ensemble (realizations).
        steplength_schedule : Optional[Callable[[int], float]], optional
            A function that takes as input the iteration number (starting at 1) and
            returns steplength (a float in the range (0, 1]).
            The default is None, which defaults to using an exponential decay.
            See the references or the function `steplength_exponential`.
        seed : Optional[int], optional
            Integer used to seed the random number generator. The default is None.

        """
        self.iteration = 1
        self.steplength_schedule = steplength_schedule
        self.rng = np.random.default_rng(seed)
        self.coefficient_matrix: npt.NDArray[np.double] = np.zeros(
            (ensemble_size, ensemble_size), dtype=np.double
        )
        self.ensemble_mask: Optional[npt.NDArray[np.bool_]] = None

        # check the input shapes
        validate_observations(
            observation_errors,
            observation_values,
        )

        self.observation_values = observation_values
        self.observation_errors = observation_errors

        # Draw samples from N(0, Cdd) that will be used for all the fit
        self.E_: npt.NDArray[np.double] = self._get_E(
            observation_values=observation_values,
            observation_errors=observation_errors,
            ensemble_size=ensemble_size,
        )

    @property
    def num_obs(self) -> int:
        """Return the number of observed values."""
        return self.E_.shape[0]

    @property
    def ensemble_size(self) -> int:
        """Return the number of members in the ensemble."""
        return self.E_.shape[1]

    @property
    def D(self) -> npt.NDArray[np.double]:
        """
        Perturbed observed values.

        This is D, as defined by Equation (14) in :cite:t:`evensen2019efficient`.
        """
        return self.E_ + self.observation_values.reshape(self.num_obs, 1)

    def _get_E(
        self,
        *,
        observation_errors: npt.NDArray[np.double],
        observation_values: npt.NDArray[np.double],
        ensemble_size: int,
    ) -> npt.NDArray[np.double]:
        """Draw samples from N(0, Cdd)."""

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
        return E - E.mean(axis=1, keepdims=True)

    def fit(
        self,
        response_ensemble: npt.NDArray[np.double],
        *,
        truncation: float = 0.98,
        step_length: Optional[float] = None,
        ensemble_mask: Optional[npt.NDArray[np.bool_]] = None,
        inversion: SiesInversionType = SiesInversionType.EXACT,
        param_ensemble: Optional[npt.NDArray[np.double]] = None,
    ) -> None:
        """
        Perform one Gauss-Newton step and update the coefficient matrix W.

        To apply the coefficient matrix W to the ensemble, call update() after fit().

        Parameters
        ----------
        response_ensemble : npt.NDArray[np.double]
            A 2D array of reponses from the model g(X) of shape
            (observations, ensemble_size).
            This matrix is Y in :cite:t:`evensen2019efficient`.
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

        # ---------------------------------------------------------------------
        # ----------------- Input validation and setup ------------------------
        # ---------------------------------------------------------------------

        validate_inputs(
            inversion,
            response_ensemble,
            self.observation_values,
            param_ensemble=param_ensemble,
        )

        # Some realization might be ignored
        effective_ensemble_size: int = response_ensemble.shape[1]

        # Determine the step length
        if step_length is None:
            if self.steplength_schedule is None:
                step_length = steplength_exponential(self.iteration)
            else:
                step_length = self.steplength_schedule(self.iteration)

        assert 0 < step_length <= 1, "Step length must be in (0, 1]"

        if ensemble_mask is None:
            ensemble_mask = np.ones(effective_ensemble_size, dtype=np.bool_)
        self.ensemble_mask = ensemble_mask

        # ---------------------------------------------------------------------
        # ----------- Computations corresponding to algorithm 1 ---------------
        # ---------------------------------------------------------------------

        # Transform covariance matrix to correlation matrix
        R, observation_errors_std = covariance_to_correlation(self.observation_errors)

        # Di - g(Xi + AWi)
        residuals = self.D[:, self.ensemble_mask] - response_ensemble

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
        residuals /= observation_errors_std.reshape(self.num_obs, 1)

        # Here we have to make a new copy of E, since if not we would
        # divide the same E by the standard deviations in every iteration
        E = self.E_ / observation_errors_std.reshape(self.num_obs, 1)

        # See section 2.4 in the paper
        if param_ensemble is not None:
            response_ensemble = response_ensemble @ response_projection(param_ensemble)

        _response_ensemble = response_ensemble / observation_errors_std.reshape(
            self.num_obs, 1
        )

        _response_ensemble -= _response_ensemble.mean(axis=1, keepdims=True)
        _response_ensemble /= np.sqrt(effective_ensemble_size - 1)

        W: npt.NDArray[np.double] = create_coefficient_matrix(  # type: ignore
            Y=_response_ensemble,
            R=R,  # Correlation matrix or None (if 1D array was passed)
            E=E,  # Samples from multivariate normal
            D=residuals,  # D - G(x) in line 7 in algorithm 1
            inversion=inversion,
            truncation=truncation,
            W=self.coefficient_matrix[np.ix_(ensemble_mask, ensemble_mask)],
            steplength=step_length,
        )

        if np.isnan(W).sum() != 0:
            raise ValueError(
                "Fit produces NaNs. Check your response matrix for outliers or use an inversion type with truncation."
            )

        self.iteration += 1

        # Put the values back into the coefficient matrix
        self.coefficient_matrix[np.ix_(ensemble_mask, ensemble_mask)] = W

    def update(self, param_ensemble: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
        """
        Update the parameter ensemble (X in :cite:t:`evensen2019efficient`).

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
        if self.ensemble_mask is not None:
            ensemble_size: int = self.ensemble_mask.sum()
            transition_matrix: npt.NDArray[np.double] = self.coefficient_matrix[
                np.ix_(self.ensemble_mask, self.ensemble_mask)
            ]
        else:
            ensemble_size = self.coefficient_matrix.shape[1]
            transition_matrix = self.coefficient_matrix

        # First get W, then divide by square root, then add identity matrix
        # Equivalent to (I + W / np.sqrt(ensemble_size - 1))
        transition_matrix /= np.sqrt(ensemble_size - 1)
        transition_matrix.flat[:: ensemble_size + 1] += 1.0

        return param_ensemble @ transition_matrix

    def __repr__(self) -> str:
        return "SIES()"


class ES:
    """
    Ensemble smoother.

    Wrapper for :class:`SIES`.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the instance."""
        self.smoother: Optional[SIES] = None
        self.seed = seed

    def fit(
        self,
        response_ensemble: npt.NDArray[np.double],
        observation_errors: npt.NDArray[np.double],
        observation_values: npt.NDArray[np.double],
        *,
        truncation: float = 0.98,
        inversion: SiesInversionType = SiesInversionType.EXACT,
        param_ensemble: Optional[npt.NDArray[np.double]] = None,
    ) -> None:
        self.smoother = SIES(
            observation_errors,
            observation_values,
            response_ensemble.shape[1],
            seed=self.seed,
        )
        self.smoother.fit(
            response_ensemble,
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
