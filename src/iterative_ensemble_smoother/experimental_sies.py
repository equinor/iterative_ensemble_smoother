from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Union

import numpy as np
import scipy as sp  # type: ignore

if TYPE_CHECKING:
    import numpy.typing as npt


from iterative_ensemble_smoother.sies_inversion import inversion_exact, inversion_naive
from iterative_ensemble_smoother.utils import _validate_inputs, sample_mvnormal


class SIES:
    """
    Initialize a Subspace Iterative Ensemble Smoother (SIES) instance.

    This is an implementation of the algorithm described in the paper:
    Efficient Implementation of an Iterative Ensemble Smoother for Data Assimilation and Reservoir History Matching
    written by Evensen et al (2019),
    URL: https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full

    Parameters
    ----------
    parameters : npt.NDArray[np.double]
        A 2D array of shape (num_parameters, ensemble_size). Each row corresponds
        to a parameter in the model, and each column corresponds to an ensemble
        member (realization). This is X in Evensen (2019).
    covariance : npt.NDArray[np.double]
        Either a 1D array of diagonal covariances, or a 2D covariance matrix.
        The shape is either (num_parameters,) or (num_parameters, num_parameters).
        This is C_dd in Evensen (2019), and represents observation or measurement
        errors. We observe d from the real world, y from the model g(x), and
        assume that d = y + e, where e is multivariate normal with covariance
        given by `covariance`.
    observations : npt.NDArray[np.double]
        A 1D array of observations, with shape (num_observations,).
        This is d in Evensen (2019).
    inversion : str
        The type of subspace inversion used in the algorithm.
        See the dictionary `SIES.inversion_funcs` for more information.
    seed : Optional[int], optional
        Integer used to seed the random number generator. The default is None.
    """

    inversion_funcs = {"exact": inversion_exact, "naive": inversion_naive}

    def __init__(
        self,
        parameters: npt.NDArray[np.double],
        covariance: npt.NDArray[np.double],
        observations: npt.NDArray[np.double],
        *,
        inversion: str = "exact",
        seed: Union[None, int, np.random._generator.Generator] = None,
    ):
        _validate_inputs(
            parameters=parameters, covariance=covariance, observations=observations
        )

        self.rng = np.random.default_rng(seed)
        self.inversion = self.inversion_funcs[inversion]
        self.X = parameters
        self.d = observations
        self.C_dd = covariance
        self.A = (self.X - self.X.mean(axis=1, keepdims=True)) / np.sqrt(
            self.X.shape[1] - 1
        )

        # Create and store the cholesky factorization of C_dd
        if self.C_dd.ndim == 2:
            # With lower=True, the cholesky factorization obeys:
            # C_dd_cholesky @ C_dd_cholesky.T = C_dd
            self.C_dd_cholesky = sp.linalg.cholesky(
                self.C_dd,
                lower=True,
                overwrite_a=False,
                check_finite=True,
            )
        else:
            self.C_dd_cholesky = np.sqrt(self.C_dd)

        # Equation (14)
        self.D = (
            self.d.reshape(-1, 1)
            + sample_mvnormal(
                C_dd_cholesky=self.C_dd_cholesky, rng=self.rng, size=self.X.shape[1]
            )
        ).astype(parameters.dtype, copy=False)

        self.W = np.zeros(
            shape=(self.X.shape[1], self.X.shape[1]), dtype=parameters.dtype
        )

    def evaluate_objective(
        self, W: npt.NDArray[np.double], Y: npt.NDArray[np.double]
    ) -> npt.NDArray[np.double]:
        """Evaluate the objective function in Equation (18), taking the mean
        over each ensemble member.

        Parameters
        ----------
        W : npt.NDArray[np.double]
            Current weight matrix W, which represents the current X_i as a linear
            combination of the prior. See equation (17) and line 9 in Algorithm 1.
        Y : npt.NDArray[np.double]
            Reponses when evaluating the model at X_i. In other words, Y = g(X_i).

        Returns
        -------
        float
            The objective function value. Lower objective is better.

        """
        (prior, likelihood) = self.evaluate_objective_elementwise(W, Y)
        answer: npt.NDArray[np.double] = (prior + likelihood).mean()
        return answer

    def evaluate_objective_elementwise(
        self, W: npt.NDArray[np.double], Y: npt.NDArray[np.double]
    ) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        """Equation (18) elementwise and termwise, returning a tuple of vectors
        (prior, likelihood).

        Parameters
        ----------
        W : npt.NDArray[np.double]
            Current weight matrix W, which represents the current X_i as a linear
            combination of the prior. See equation (17) and line 9 in Algorithm 1.
        Y : npt.NDArray[np.double]
            Reponses when evaluating the model at X_i. In other words, Y = g(X_i).

        Returns
        -------
        prior : npt.NDArray[np.double]
            A 1D array representing distance from prior for each ensemble member.
        likelihood : npt.NDArray[np.double]
            A 1D array representing distance from observations for each ensemble member.

        """
        # Evaluate the elementwise prior term
        # Evaluate np.diag(W.T @ W) = (W**2).sum(axis=0)
        prior = (W**2).sum(axis=0)

        if self.C_dd.ndim == 2:
            # Evaluate the elementwise likelihood term
            # Evaluate np.diag((g(X_i) - D).T @ C_dd^{-1} @ (g(X_i) - D))
            #        = np.diag((Y - D).T @ C_dd^{-1} @ (Y - D))
            # First let A := Y - D, then use the cholesky factors C_dd = L @ L.T
            # to write (Y - D).T @ C_dd^{-1} @ (Y - D)
            #             A.T @ (L @ L.T)^-1 @ A
            #             (L^-1 @ A).T @ (L^-1 @ A)
            # To compute the expression above, we solve K := L^-1 @ A,
            # then we compute np.diag(K.T @ K) as (K**2).sum(axis=0)
            A = Y - self.D
            K = sp.linalg.blas.dtrsm(alpha=1.0, a=self.C_dd_cholesky, b=A, lower=1)
            likelihood = (K**2).sum(axis=0)
            # assert np.allclose(likelihood, np.diag(A.T @ np.linalg.inv(self.C_dd) @ A))

        else:
            # If C_dd is diagonal, then (L^-1 @ A) = A / L.reshape(-1, 1)
            A = Y - self.D
            K = A / self.C_dd_cholesky.reshape(-1, 1)
            likelihood = (K**2).sum(axis=0)
            # assert np.allclose(likelihood, np.diag(A.T @ np.diag(1/self.C_dd) @ A))

        return (prior, likelihood)

    def sies_iteration(
        self,
        responses: npt.NDArray[np.double],
        step_length: float = 0.5,
        ensemble_mask: Optional[npt.NDArray[np.bool_]] = None,
    ) -> npt.NDArray[np.double]:
        """Perform a single SIES iteration (Gauss-Newton step).

        This method implements lines 4-9 in Algorithm 1.
        It returns a updated X and updates internal state W.


        Parameters
        ----------
        responses : npt.NDArray[np.double]
            The model evaluated at X_i. In other words, responses = g(X_i).
            This is Y in the paper.
        step_length : float, optional
            Step length for Gauss-Newton. The default is 0.5.

        Returns
        -------
        npt.NDArray[np.double]
            Updated parameter ensemble.

        """
        Y = responses

        if ensemble_mask is not None:
            # Lines 4 through 8
            proposed_W = self.propose_W_masked(
                Y, ensemble_mask=ensemble_mask, step_length=step_length
            )
            self.W[:, ensemble_mask] = proposed_W

        else:
            # Lines 4 through 8
            proposed_W = self.propose_W(Y, step_length=step_length)
            self.W = proposed_W

        # Line 9
        N = self.X.shape[1]  # Ensemble members
        ans: npt.NDArray[np.double] = self.X + self.X @ self.W / np.sqrt(N - 1)
        return ans

    def propose_W(
        self, responses: npt.NDArray[np.double], step_length: float = 0.5
    ) -> npt.NDArray[np.double]:
        """Returns a proposal for W_i, without updating the internal W.

        This is an implementation of lines 4-8 in Algorithm 1.

        Parameters
        ----------
        responses : npt.NDArray[np.double]
            The model evaluated at X_i. In other words, responses = g(X_i).
            This is Y in the paper.
        step_length : float, optional
            Step length for Gauss-Newton. The default is 0.5.

        Returns
        -------
        W_i : npt.NDArray[np.double]
            A proposal for a new W in the algorithm.

        """
        Y = responses
        assert Y.ndim == 2
        assert Y.shape[0] == self.C_dd.shape[0]
        g_X = Y.copy()

        # Get shapes. Same notation as used in the paper.
        N = self.X.shape[1]  # Ensemble members
        n = self.X.shape[0]  # Parameters (inputs)
        m = self.C_dd.shape[0]  # Responses (outputs)

        # Line 4 in Algorithm 1
        Y = (g_X - g_X.mean(axis=1, keepdims=True)) / np.sqrt(N - 1)

        # Line 5
        Omega = self.W.copy()
        Omega -= Omega.mean(axis=1, keepdims=True)
        Omega /= np.sqrt(N - 1)
        Omega.flat[:: Omega.shape[0] + 1] += 1  # Add identity in place

        # Line 6
        if n < N - 1:
            # There are fewer parameters than realizations. This means that the
            # system of equations is overdetermined, and we must solve a least
            # squares problem.

            # An alternative approach to producing A_i would be keeping the
            # returned value from the previous Newton iteration (call it X_i),
            # then computing:
            # A_i = (self.X_i - self.X_i.mean(axis=1, keepdims=True)) / np.sqrt(N - 1)
            A_i = self.A @ Omega
            S = sp.linalg.solve(
                Omega.T, np.linalg.multi_dot([Y, sp.linalg.pinv(A_i), A_i]).T
            ).T
        else:
            # The system is not overdetermined
            S = sp.linalg.solve(Omega.T, Y.T).T

        # Line 7
        H = S @ self.W + self.D - g_X

        # Line 8
        # Some asserts to remind us what the shapes of the matrices are
        assert self.W.shape == (N, N)
        assert S.shape == (m, N)
        assert self.C_dd.shape in [(m, m), (m,)]
        assert H.shape == (m, N)

        proposed_W: npt.NDArray[np.double] = self.inversion(
            W=self.W,
            step_length=step_length,
            S=S,
            C_dd=self.C_dd,
            H=H,
            C_dd_cholesky=self.C_dd_cholesky,
        )
        return proposed_W

    def propose_W_masked(
        self,
        responses: npt.NDArray[np.double],
        ensemble_mask: npt.NDArray[np.bool_],
        step_length: float = 0.5,
    ) -> npt.NDArray[np.double]:
        """Returns a proposal for W_i, without updating the internal W.

        This is an implementation of lines 4-8 in Algorithm 1.

        Parameters
        ----------
        responses : npt.NDArray[np.double]
            The model evaluated at X_i. In other words, responses = g(X_i).
            This is Y in the paper.
        step_length : float, optional
            Step length for Gauss-Newton. The default is 0.5.

        Returns
        -------
        W_i : npt.NDArray[np.double]
            A proposal for a new W in the algorithm.

        """
        Y = responses
        assert Y.ndim == 2
        assert Y.shape[0] == self.C_dd.shape[0]
        g_X = Y.copy()
        if ensemble_mask is not None:
            assert Y.shape[1] == ensemble_mask.sum()

        # Get shapes. Same notation as used in the paper.
        N = self.X.shape[1]  # Ensemble members in prior
        n = self.X.shape[0]  # Parameters (inputs)
        m = self.C_dd.shape[0]  # Responses (outputs)
        k = Y.shape[1]  # Active ensemble members

        # Line 4 in Algorithm 1.
        Y = (g_X - g_X.mean(axis=1, keepdims=True)) / np.sqrt(N - 1)

        # Line 5
        Omega = self.W.copy()
        Omega -= Omega.mean(axis=1, keepdims=True)
        Omega /= np.sqrt(N - 1)
        Omega.flat[:: Omega.shape[0] + 1] += 1  # Add identity in place
        Omega = Omega[:, ensemble_mask]

        # Line 6
        if n < k - 1:  # Here we use k instead of N
            # There are fewer parameters than realizations. This means that the
            # system of equations is overdetermined, and we must solve a least
            # squares problem.

            # An alternative approach to producing A_i would be keeping the
            # returned value from the previous Newton iteration (call it X_i),
            # then computing:
            # A_i = (self.X_i - self.X_i.mean(axis=1, keepdims=True)) / np.sqrt(N - 1)
            A_i = self.A @ Omega
            # S = sp.linalg.solve(
            #     Omega.T, np.linalg.multi_dot([Y, sp.linalg.pinv(A_i), A_i]).T
            # ).T
            ST, *_ = sp.linalg.lstsq(
                Omega.T, np.linalg.multi_dot([Y, sp.linalg.pinv(A_i), A_i]).T
            )
            S = ST.T

        else:
            # The system is not overdetermined
            # S = sp.linalg.solve(Omega.T, Y.T).T
            ST, *_ = sp.linalg.lstsq(Omega.T, Y.T)
            S = ST.T

        # Line 7
        H = S @ self.W[:, ensemble_mask] + self.D[:, ensemble_mask] - g_X

        # Line 8
        # Some asserts to remind us what the shapes of the matrices are
        assert self.W.shape == (N, N)
        assert S.shape == (m, N)
        assert self.C_dd.shape in [(m, m), (m,)]
        assert H.shape == (m, k)

        proposed_W: npt.NDArray[np.double] = self.inversion(
            W=self.W[:, ensemble_mask],
            step_length=step_length,
            S=S,
            C_dd=self.C_dd,
            H=H,
            C_dd_cholesky=self.C_dd_cholesky,
        )
        return proposed_W


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-v", "--maxfail=1"])
