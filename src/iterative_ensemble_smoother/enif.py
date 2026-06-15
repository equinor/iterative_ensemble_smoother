"""
Ensemble Information Filter (EnIF)
----------------------------------

The EnIF algorithm roughly consists of three main ideas:

    1. Work with a form of the Kalman gain that requires (a) H, a linearization
       of the forward map Y = h(X) and (b) Prec(X), the prior parameter precision.
    2. Estimate H using sparse linear regression: Y = HX + r, where r are residuals
    3. Estimate Prec(X) using graph adjacency information if possible

The EnIF formulation estimates (H, Prec(X))
while ESMDA estimates (Cov(X, Y), Cov(Y, Y)).

The advantage of estimating H and Prec(X) is that we can lean on high-dimensional
regression algorithms (Lasso, linear boosting, etc.) to estimate H and
lean on covariance estimation algorithms (Graphical Lasso, Triangular Transport, etc.).
In the implementation below the user is assumed to supply their own (H, Prec(X)).
This decouples the estimation of (H, Prec(X)) from the main assimilation step.

One could argue that we should call it an Information Smoother instead,
but EnIF class name was chosen because it is the appreviation already in use.

References
----------

- An Ensemble Information Filter:
  Retrieving Markov-information from the SPDE discretisation
  Berent Ånund Strømnes Lunde
  https://arxiv.org/abs/2501.09016


Examples
--------
>>> import numpy as np, scipy as sp
>>> rng = np.random.default_rng(42)

Create problem size with parameters >> responses >> realizations.

>>> num_responses = 50
>>> num_params = 100
>>> num_realizations = 15
>>> alpha = 3

Create a true forward model:

>>> H = sp.sparse.random_array(shape=(num_responses, num_params),
...                            density=0.1, rng=rng)
>>> def forward(X):
...     linear = H @ X
...     return linear + 1e-3 * linear**2

Diagonal observation (measurement) errors, prior ensemble, observations:

>>> covariance = np.logspace(-1, 1, num=num_responses)
>>> X_prior = rng.normal(size=(num_params, num_realizations))
>>> observations = forward(X_prior[:, 0] + 1).ravel()

Prior parameter precision. Here we have generated X ~ N(0, 1), so we know
that the prior parameter precision is eye(num_params) and do not need to
estimate it from the prior.

>>> parameter_precision = sp.sparse.diags_array(np.ones(num_params))

Create the smoother:

>>> enif = EnIF(covariance=covariance, observations=observations,
...             parameter_precision=parameter_precision, alpha=alpha,
...             seed=42, solver="cg")

Assimilate data:

>>> X = np.copy(X_prior)
>>> for _ in range(enif.num_assimilations()):
...
...     # Apply the forward model
...     Y = forward(X)
...     enif.prepare_assimilation(Y=Y)
...     residual_covariance = np.var(Y - H @ X, axis=1, ddof=1)
...     X = enif.assimilate(X=X, linearized_model=H,
...                         residual_covariance=residual_covariance)
"""

from typing import Union

import numpy as np
import numpy.typing as npt

from iterative_ensemble_smoother.enif_utils import SPDSolver
from iterative_ensemble_smoother.esmda import BaseESMDA


class EnIF(BaseESMDA):
    """
    Implement the Ensemble Information Filter (here really a *smoother*).

    Examples
    --------

    A full example where the forward model maps 10 parameters to 3 responses.
    We will use 100 realizations. First we define the forward model:

    >>> rng = np.random.default_rng(42)
    >>> H = rng.normal(size=(3, 10))
    >>> def forward_model(x):
    ...     return H @ x

    Then we set up the EnIF instance and the prior realizations X:

    >>> covariance = np.ones(3, dtype=float)
    >>> observations = np.array([1, 2, 3], dtype=float)
    >>> parameter_precision = np.eye(10)
    >>> smoother = EnIF(covariance=covariance, observations=observations,
    ...                 parameter_precision=np.eye(10), alpha=3, seed=42)
    >>> X = rng.normal(size=(10, 100))

    To assimilate data, we iterate over assimilation steps:

    >>> for iteration in range(smoother.num_assimilations()):
    ...     # Apply the forward model
    ...     Y = forward_model(X)
    ...     smoother.prepare_assimilation(Y=Y)
    ...     X = smoother.assimilate(X=X, linearized_model=H,
    ...                             residual_covariance=np.zeros(3))
    ...     parameter_precision = smoother.parameter_precision
    >>> X[:3, :5].round(1)
    array([[ 1.8, -0.5, -0. , -0. ,  0.8],
           [-0.8, -1.2,  0.5,  0.6, -1.1],
           [-0.3, -0.2,  1.2,  1.6, -0.4]])
    """

    def __init__(
        self,
        *,
        covariance: npt.NDArray[np.floating],
        observations: npt.NDArray[np.floating],
        parameter_precision: npt.NDArray[np.floating],
        alpha: Union[int, npt.NDArray[np.floating]] = 5,
        seed: Union[np.random.Generator, int, None] = None,
        solver: str = "cg",
        solver_options: Union[dict["str", object], None] = None,
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
        parameter_precision: sp.sparse.sparray
            Sparse parameter precision of shape (num_parameters, num_parameters).
        alpha : int or 1D np.ndarray, optional
            Multiplicative factor for the covariance.
            If an integer `alpha` is given, an array with length `alpha` and
            elements `alpha` is constructed. If an 1D array is given, it is
            normalized so sum_i 1/alpha_i = 1 and used. The default is 5, which
            corresponds to np.array([5, 5, 5, 5, 5]).
        seed : integer or numpy.random.Generator, optional
            A seed or numpy.random.Generator used for random number
            generation. The argument is passed to numpy.random.default_rng().
            The default is None.
        solver : str
            Which solver to use. The options are:
                - "cholesky" for a sparse cholesky factorization solver
                - "cg" for conjugate gradients
                - "dense" to cast all sparse matrices to dense and solve with scipy
        solver_options : dict
            Dictionary of solver options. The options are:
                - "cholesky": "ordering_method" and other arguments are passed
                  to "sksparse.cholmod.cholesky"
                - "cg": "rtol", "atol", "maxiter" and other arguments are passed
                  to "scipy.sparse.linalg.cg"
        """
        self.solver = solver
        self.solver_options = solver_options
        self.parameter_precision = parameter_precision

        # Defaults for solvers
        if self.solver_options is None and self.solver == "cholesky":
            self.solver_options = {"ordering_method": "metis"}
        if self.solver_options is None and self.solver == "cg":
            self.solver_options = {"maxiter": 15}

        # Creat solver for solving the main sym. pos. def. equation:
        # (Prec_x + H.T @ Prec @ H) X = H.T @ Prec @ (D - (Y + E))
        self.spd_solver = SPDSolver(
            Prec_x=parameter_precision,
            solver=self.solver,
            solver_options=self.solver_options,
        )
        super().__init__(
            covariance=covariance, observations=observations, alpha=alpha, seed=seed
        )

    def prepare_assimilation(  # type: ignore[override]
        self,
        *,
        Y: npt.NDArray[np.floating],
        observation_perturbations: Union[npt.NDArray[np.floating], None] = None,
    ) -> None:
        r"""Prepare assimilation of parameters.

        Parameters
        ----------
        Y : np.ndarray
            2D array of shape (num_observations, ensemble_size), containing
            responses when evaluating the model at X. In other words, Y = g(X),
            where g is the forward model.
        observation_perturbations: np.ndarray or None
            2D array of shape (num_observations, ensemble_size) containing
            additive perturbations drawn from the observation error distribution,
            i.e. ``observation_perturbations ~ N(0, C_D)``.
            The method will apply the ``sqrt(alpha)`` scaling internally,
            consistent with :meth:`perturb_observations`.
            If None, perturbed observations are generated internally.

        Returns
        -------
        self
            The instance with mutated state.
        """
        assert Y.ndim == 2
        if not np.issubdtype(Y.dtype, np.floating):
            raise TypeError("Argument `Y` must contain floats")

        if Y.dtype != self.observations.dtype:
            raise ValueError(
                f"'Y' must have dtype {self.observations.dtype}, got {Y.dtype}"
            )

        if self.iteration >= self.num_assimilations():
            raise Exception("No more assimilation steps to run.")

        self.iteration += 1

        D = Y  # Switch from API notation to paper notation
        N_d, N_e = D.shape  # (num_observations, ensemble_size)
        assert N_e >= 2, "Must have at least two ensemble members"
        assert N_d == self.observations.shape[0], "Shape mismatch"

        # Compute the last factor
        alpha = self.alpha[self.iteration]
        if observation_perturbations is not None:
            if observation_perturbations.shape != D.shape:
                raise ValueError(
                    "observation_perturbations must have shape "
                    f"{D.shape}, got {observation_perturbations.shape}"
                )
            if observation_perturbations.dtype != self.observations.dtype:
                raise ValueError(
                    f"'observation_perturbations' must have dtype "
                    f"{self.observations.dtype}, got {observation_perturbations.dtype}"
                )
            # Scale the perturbations by sqrt(alpha),
            # consistent with perturb_observations.
            D_obs = (
                self.observations[:, np.newaxis]
                + (alpha**0.5) * observation_perturbations
            )
        else:
            D_obs = self.perturb_observations(ensemble_size=N_e, alpha=alpha)
        self.D_obs_minus_D = D_obs - D

    def assimilate_batch(
        self, *args: object, **kwargs: object
    ) -> npt.NDArray[np.floating]:
        # Override the base class method so users do not call it
        msg = "The EnIF class cannot assimilate batches\n Use .assimilate()"
        raise NotImplementedError(msg)

    def assimilate(
        self,
        *,
        X: npt.NDArray[np.floating],
        linearized_model: npt.NDArray[np.floating],
        residual_covariance: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Assimilate parameters against all observations.

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (num_parameters, ensemble_size). Each row
            corresponds to a parameter in the model, and each column corresponds
            to an ensemble member (realization).
        linearized_model : np.ndarray or None
            A 2D sparse array of shape (num_observations, num_parameters) that
            represents a linearization of the forward model. Sometimes called H.
            Created by regressing Y on X, such that Y = H @ X.
        residual_covariance : np.ndarray or None
            A 2D array of shape (num_observations, num_observations) that
            represents the covariance of the residuals of the linearized model.
            If r = Y - H @ X, then the residual covariance is cov(r).
            If a 1D array is passed, the residual covariance is assumed linear,
            e.g. np.var(Y - H @ X, axis=1, ddof=1).

        Returns
        -------
        X_posterior : np.ndarray
            2D array of shape (num_parameters, ensemble_size).

        """
        # Switch from API notation to more dense mathematical notation
        H = linearized_model
        Cov_r = residual_covariance
        Cov_eps = self.alpha[self.iteration] * self.C_D
        innovation = self.D_obs_minus_D

        # Compute right-hand side of equation:
        if Cov_r.ndim == 1 and Cov_eps.ndim == 1:
            Prec_eps_r = 1 / (Cov_r + Cov_eps)
            RHS = (H.T * Prec_eps_r) @ innovation
        else:
            Cov_r = Cov_r if Cov_r.ndim == 2 else np.diag(Cov_r)
            Cov_eps = Cov_eps if Cov_eps.ndim == 2 else np.diag(Cov_eps)
            Prec_eps_r = np.linalg.inv(Cov_r + Cov_eps)
            RHS = H.T @ Prec_eps_r @ innovation

        # Add terms to the left hand side
        self.spd_solver.add(H=H, Prec_eps_r=Prec_eps_r)

        # Solve for change in X
        delta_X = self.spd_solver.solve(RHS)
        X_posterior: npt.NDArray[np.floating] = X + delta_X
        return X_posterior


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
        ]
    )
