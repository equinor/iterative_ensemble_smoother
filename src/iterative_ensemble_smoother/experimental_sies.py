#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 06:49:22 2023

@author: tommy
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable

import numpy as np
import scipy as sp

if TYPE_CHECKING:
    import numpy.typing as npt

from iterative_ensemble_smoother.utils import (
    _validate_inputs,
    covariance_to_correlation,
    steplength_exponential,
    response_projection,
)

from iterative_ensemble_smoother.ies import create_coefficient_matrix


def center(X):
    # Center each row, in place, so sum(row) = 0 for every row
    X -= X.mean(axis=1, keepdims=True)


def scale(X):
    # Scale each entry so a_ij = a_ij / sqrt(columns - 1)
    X /= np.sqrt(X.shape[1] - 1)


def sample_mvnormal(*, C_dd_cholesky, rng, size):
    """Draw samples from N(0, C_dd).

    Examples
    --------
    >>> C_dd_cholesky = np.diag([5, 10, 15])
    >>> rng = np.random.default_rng(42)
    >>> sample_mvnormal(C_dd_cholesky=C_dd_cholesky, rng=rng, size=2)
    array([[  1.5235854 ,  -5.19992053],
           [  7.50451196,   9.40564716],
           [-29.26552783, -19.5326926 ]])
    >>> sample_mvnormal(C_dd_cholesky=np.diag(C_dd_cholesky), rng=rng, size=2)
    array([[ 0.63920202, -1.58121296],
           [-0.16801158, -8.53043928],
           [13.19096962, 11.66687903]])
    """

    # A 2D covariance matrix was passed
    if C_dd_cholesky.ndim == 2:
        return C_dd_cholesky @ rng.standard_normal(size=(C_dd_cholesky.shape[0], size))
    else:
        return C_dd_cholesky.reshape(-1, 1) * rng.standard_normal(
            size=(C_dd_cholesky.shape[0], size)
        )


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
        param_ensemble: npt.NDArray[np.double],
        observation_errors: npt.NDArray[np.double],
        observation_values: npt.NDArray[np.double],
        *,
        steplength_schedule: Optional[Callable[[int], float]] = None,
        seed: Optional[int] = None,
        verbosity: int = 0,
    ):
        self.iteration = 1
        self.steplength_schedule = steplength_schedule
        self.rng = np.random.default_rng(seed)
        self.X = param_ensemble
        self.d = observation_values
        self.C_dd = (
            np.diag(observation_errors)
            if observation_errors.ndim == 1
            else observation_errors
        )
        self.A = (self.X - self.X.mean(axis=1, keepdims=True)) / np.sqrt(
            self.X.shape[1] - 1
        )

        if self.C_dd.ndim == 2:
            self.C_dd_cholesky = sp.linalg.cholesky(
                self.C_dd, lower=False, overwrite_a=False, check_finite=True
            )
        else:
            self.C_dd_cholesky = np.sqrt(self.C_dd)

        # Equation (14)
        self.D = self.d.reshape(-1, 1) + sample_mvnormal(
            C_dd_cholesky=self.C_dd_cholesky, rng=self.rng, size=self.X.shape[1]
        )

        self.W = np.zeros(shape=(self.X.shape[1], self.X.shape[1]))
        self.X_i = self.X

    def newton(self, Y, step_length=0.5):

        g_X = Y.copy()

        # Get shapes
        N = Y.shape[1]  # Ensemble members
        n = self.X.shape[0]  # Parameters (inputs)
        m = self.C_dd.shape[0]  # Responses (outputs)

        # Line 4 in Algorithm 1
        Y = (g_X - g_X.mean(axis=1, keepdims=True)) / np.sqrt(N - 1)

        # Line 5
        Omega = self.W.copy()
        Omega.flat[:: Omega.shape[0] + 1] += 1

        # Line 6
        if n < N - 1:
            print("Case 2.4.3")
            A_i0 = (self.X_i - self.X_i.mean(axis=1, keepdims=True)) / np.sqrt(N - 1)
            A_i = self.A @ Omega
            # assert np.allclose(A_i0, A_i)
            ST = sp.linalg.solve(
                Omega.T, np.linalg.multi_dot([Y, sp.linalg.pinv(A_i), A_i]).T
            )

        # =============================================================================
        #             ST, *_ = sp.linalg.lstsq(
        #                 A_i.T, Y.T
        #             )
        #
        #             S = ST.T @ A_i
        # =============================================================================

        else:

            print(f"Solving sytem of size lhs = {Omega.T.shape}, rhs= {Y.T.shape}")
            ST = sp.linalg.solve(Omega.T, Y.T)
        S = ST.T

        # Line 7
        H = S @ self.W + self.D - g_X

        center = False
        if center:
            factor = np.diag(1 / np.sqrt(np.diag(self.C_dd)))
            C_dd = factor @ self.C_dd @ factor
            S = factor @ S
            H = factor @ H
        else:
            C_dd = self.C_dd

        # Line 8
        to_invert = S @ S.T + C_dd

        import matplotlib.pyplot as plt

        # plt.imshow(to_invert)
        # plt.show()

        print(np.mean(S), np.mean(C_dd), np.mean(to_invert))
        K, *_ = sp.linalg.lstsq(to_invert, H)

        print(np.linalg.det(to_invert))
        print(np.linalg.det(sp.linalg.inv(to_invert)))
        #        print(np.linalg.det(ST @ sp.linalg.inv(to_invert) @ H))
        #       print(np.linalg.det((self.W - ST @ sp.linalg.inv(to_invert) @ H)))
        # to_invert.flat[:: to_invert.shape[1] + 1] += 1e-12
        self.W = self.W - step_length * (self.W - S.T @ np.linalg.inv(to_invert) @ H)

        self.X_i = self.X + self.X @ self.W / (np.sqrt(N - 1))
        return self.X_i


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-v"])

    import numpy as np
    import matplotlib.pyplot as plt

    np.set_printoptions(suppress=True)
    rng = np.random.default_rng(12345)

    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rcParams.update({"font.size": 10})

    import iterative_ensemble_smoother as ies

    # %%
    def plot_ensemble(observations, X, title=None):
        """Utility function for plotting the ensemble."""
        fig, axes = plt.subplots(1, 4, figsize=(9, 2.5))

        if title:
            fig.suptitle(title, fontsize=14)

        x = np.arange(len(observations))
        axes[0].plot(x, observations, label="Observations", zorder=10, lw=3)
        for simulation in G(X).T:
            axes[0].plot(x, simulation, color="black", alpha=0.33, zorder=0)
        axes[0].legend()

        axes[1].set_title("Interest rate")
        axes[1].hist(X[0, :], bins="fd")
        axes[1].axvline(x=INTEREST_RATE, label="Truth", color="black", ls="--")
        axes[1].legend()

        axes[2].set_title("Deposit")
        axes[2].hist(X[1, :], bins="fd")
        axes[2].axvline(x=DEPOSIT, label="Truth", color="black", ls="--")
        axes[2].legend()

        axes[3].scatter(*X)
        axes[3].scatter([INTEREST_RATE], [DEPOSIT], label="Truth", color="black", s=50)
        axes[3].legend()

        fig.tight_layout()

        return fig, axes

    # %% [markdown]
    # ## Define synthetic truth and use it to create noisy observations

    # %%
    num_observations = 25  # Number of years we simulate the mutual fund account
    num_ensemble = 1000  # Number of ensemble members

    def g(interest_rate, deposit, obs=None):
        """Simulate a mutual fund account, starting with year 0.

        g is linear in deposit, but non-linear in interest_rate.
        """
        obs = obs or num_observations

        saved = 0
        for year in range(obs):
            yield saved
            saved = saved * interest_rate + deposit

    # Test the function
    assert list(g(1.1, 100, obs=4)) == [0, 100.0, 210.0, 331.0]

    def G(X):
        """Run model g(x) on every column in X."""
        return np.array([np.array(list(g(*i))) for i in X.T]).T

    # True inputs, unknown to us
    INTEREST_RATE = 1.05
    DEPOSIT = 1000
    X_true = np.array([INTEREST_RATE, DEPOSIT])

    # Real world observations
    observations = np.array(list(g(*X_true))) * (
        1 + rng.standard_normal(size=num_observations) / 10
    )

    # Priors for interest rate and deposit - quite wide (see plot below)
    X_prior_interest_rate = 2 ** rng.normal(loc=0, scale=0.1, size=num_ensemble)
    X_prior_deposit = np.exp(rng.normal(loc=6, scale=0.5, size=num_ensemble))

    X_prior = np.vstack([X_prior_interest_rate, X_prior_deposit])
    assert X_prior.shape == (2, num_ensemble)

    # %%
    plot_ensemble(observations, X_prior, title="Prior distribution")
    plt.show()

    # %% [markdown]
    # ## Create and run ESMDA - with one iteration

    # %%
    smoother = SIES(
        param_ensemble=X_prior,
        observation_errors=1 + (observations * 0.1) ** 2,
        observation_values=observations,
        seed=42,
    )

    X_i = np.copy(X_prior)
    for iteration in range(25):

        X_i = smoother.newton(Y=G(X_i), step_length=0.3)

        if np.any(X_i > 100_000):
            print("Breaking due to large values")
            break

        plot_ensemble(observations, X_i, title=f"Iteration {iteration+1}")
        plt.show()

    1 / 0

    # --------------------------------------

    def subspace_ies(
        X: npt.NDArray[np.double],
        D: npt.NDArray[np.double],
        g: Callable[[npt.NDArray[np.double]], float],
        linear: bool = False,
        step_size: float = 0.3,
        iterations: int = 2,
    ) -> npt.NDArray[np.double]:
        """Updates ensemble of parameters according to the Subspace
        Iterative Ensemble Smoother (Evensen 2019).

        :param X: sample from prior parameter distribution, i.e. ensemble.
        :param D: observations perturbed with noise having observation uncertainty.
        :param g: the forward model g:Re**parameter_size -> R^m
        :param linear: if g is a linear forward model.
        :param step_size: the step size of an ensemble-weight update at each iteration.
        :param iterations: number of iterations in the udpate algorithm.
        """
        parameters, realizations = X.shape
        projection = not linear and parameters < realizations
        if projection:
            print("Using projection matrix")
        m = D.shape[0]
        W = np.zeros((realizations, realizations))
        Xi = X.copy()
        I = np.identity(realizations)
        centering_matrix = (
            I - np.ones((realizations, realizations)) / realizations
        ) / np.sqrt(realizations - 1)
        E = D @ centering_matrix
        for i in range(iterations):
            gXi = G(Xi)
            Y = gXi @ centering_matrix
            if projection:
                Ai = Xi @ centering_matrix  ## NB: Using Xi not X
                projection_matrix = np.linalg.pinv(Ai) @ Ai
                Y = Y @ projection_matrix
            Ohmega = I + W @ centering_matrix
            S = np.linalg.solve(Ohmega.T, Y.T).T
            H = S @ W + D - gXi
            W = W - step_size * (W - S.T @ np.linalg.inv(S @ S.T + E @ E.T) @ H)
            Xi = X @ (I + W / np.sqrt(realizations - 1))
            # print(np.mean(Xi, axis=1))
            yield Xi

    X_i = np.copy(X_prior)
    for X_i in subspace_ies(
        X_i, smoother.D, g, linear=False, iterations=10, step_size=0.9
    ):

        plot_ensemble(observations, X_i, title=f"Iteration {iteration+1}")
        plt.show()
