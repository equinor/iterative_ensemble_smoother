"""
Contains (publicly available, but not officially supported) experimental
features of iterative_ensemble_smoother
"""
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy as sp

from iterative_ensemble_smoother import ESMDA
from iterative_ensemble_smoother.esmda import BaseESMDA
from iterative_ensemble_smoother.esmda_inversion import (
    empirical_cross_covariance,
    normalize_alpha,
)


def compute_cross_covariance_multiplier(
    *,
    alpha: float,
    C_D: npt.NDArray[np.double],
    D: npt.NDArray[np.double],
    Y: npt.NDArray[np.double],
) -> npt.NDArray[np.double]:
    """Compute transition matrix K such that:

        X + cov_XY @ transition_matrix

    In the notation of Emerick et al, recall that the update equation is:

        X_posterior = X_prior + cov_XY @ inv(C_DD + alpha * C_D) @ (D - Y)

    This function computes a transition matrix K, defined as:

        K := inv(C_DD + alpha * C_D) @ (D - Y)

    Note that this is not the most efficient way to compute the update equation,
    since cov_XY := center(X) @ center(Y) / (N - 1), and we can avoid creating
    this (huge) covariance matrix by performing multiplications right-to-left:

        cov_XY @ inv(C_DD + alpha * C_D) @ (D - Y)
        center(X) @ center(Y) / (N - 1) @ inv(C_DD + alpha * C_D) @ (D - Y)
        center(X) @ [center(Y) / (N - 1) @ inv(C_DD + alpha * C_D) @ (D - Y)]

    The advantage of forming K instead is that if we already have computed
    cov_XY, then we can apply the same K to a reduced number of rows (parameters)
    in cov_XY.
    """
    C_DD = empirical_cross_covariance(Y, Y)  # Only compute upper part

    # import matplotlib.pyplot as plt
    # plt.imshow((Y))
    # plt.show()

    # Arguments for sp.linalg.solve
    solver_kwargs = {
        "overwrite_a": True,
        "overwrite_b": True,
        "assume_a": "pos",  # Assume positive definite matrix (use cholesky)
        "lower": False,  # Only use the upper part while solving
    }

    # import matplotlib.pyplot as plt
    # plt.imshow((C_DD))
    # plt.show()

    # Compute K := sp.linalg.inv(C_DD + alpha * C_D) @ (D - Y)
    # by solving the system (C_DD + alpha * C_D) @ K = (D - Y)
    if C_D.ndim == 2:
        # C_D is a covariance matrix
        C_DD += alpha * C_D  # Save memory by mutating
    elif C_D.ndim == 1:
        # C_D is an array, so add it to the diagonal without forming diag(C_D)
        np.fill_diagonal(C_DD, C_DD.diagonal() + alpha * C_D)

    # import matplotlib.pyplot as plt
    # plt.imshow((C_DD))
    # plt.show()

    return sp.linalg.solve(C_DD, D - Y, **solver_kwargs)

    try:
        return sp.linalg.solve(C_DD, D - Y, **solver_kwargs)
    except sp.linalg.LinAlgError:
        ans, *_ = sp.linalg.lstsq(
            a=C_DD,
            b=D - Y,
            cond=None,
            overwrite_a=True,
            overwrite_b=True,
            check_finite=True,
        )
        return ans


class AdaptiveESMDA(BaseESMDA):
    def correlation_threshold(self, ensemble_size: int) -> float:
        """Decides whether or not to use user-defined or default threshold.

        Default threshold taken from luo2022,
        Continuous Hyper-parameter OPtimization (CHOP) in an ensemble Kalman filter
        Section 2.3 - Localization in the CHOP problem
        """
        return 3 / np.sqrt(ensemble_size)

    def adaptive_transition_matrix(self, Y, D, alpha):
        """Compute a transition matrix K, such that:

        X_posterior = X_prior + cov_XY @ K
        """
        assert D.shape == Y.shape

        # Compute an update matrix, independent of X
        return compute_cross_covariance_multiplier(alpha=alpha, C_D=self.C_D, D=D, Y=Y)

    def adaptive_assimilate(self, X, Y, transition_matrix, correlation_threshold=None):
        """Use X, or possibly a subset of variables (rows) in X, as well as Y,
        to compute the cross covariance matrix cov_XY.
        Then compute the update as:

            X + cov_XY @ transition_matrix
        """
        assert X.shape[1] == Y.shape[1] == transition_matrix.shape[1]
        assert Y.shape[0] == transition_matrix.shape[0]
        if correlation_threshold is None:
            correlation_threshold = self.correlation_threshold

        # Step 1: # Compute cross-correlation between parameters X and responses Y
        # Note: let the number of parameters be n and the number of responses be m.
        # This step requires both O(mn) computation and O(mn) storage, which is
        # larger than the O(n + m^2) computation used in ESMDA, which never explicitly
        # forms the cross-covariance matrix between X and Y. However, if we batch
        # X by parameter group (rows), the storage requirement decreases
        cov_XY = empirical_cross_covariance(X, Y)
        assert cov_XY.shape == (X.shape[0], Y.shape[0])
        stds_Y = np.std(Y, axis=1, ddof=1)
        stds_X = np.std(X, axis=1, ddof=1)

        # Compute the correlation matrix from the covariance matrix
        corr_XY = (cov_XY / stds_X[:, np.newaxis]) / stds_Y[np.newaxis, :]
        assert corr_XY.max() <= 1
        assert corr_XY.min() >= -1

        # Determine which elements in the cross covariance matrix that will
        # be set to zero
        thres = correlation_threshold(ensemble_size=X.shape[1])
        cov_XY[np.abs(corr_XY) < thres] = 0  # Set small values to zero
        print("  Entries in cov_XY set to zero", np.isclose(cov_XY, 0).sum())

        return X + cov_XY @ transition_matrix


if __name__ == "__main__":
    import time

    # =============================================================================
    # CREATE A PROBLEM - LINEAR REGRESSION
    # =============================================================================

    # Create a problem with g(x) = A @ x
    rng = np.random.default_rng(42)
    num_parameters = 100
    num_observations = 100
    num_ensemble = 80

    A = np.exp(rng.standard_normal(size=(num_observations, num_parameters)))

    def g(X):
        """Forward model."""
        return A @ X

    # Create observations
    x_true = np.linspace(-1, 1, num=num_parameters)
    observations = g(x_true) + rng.standard_normal(size=num_observations) / 10

    # Initial ensemble and covariance
    X = rng.normal(size=(num_parameters, num_ensemble))
    covariance = rng.triangular(0.1, 1, 1, size=num_observations)

    # Split the parameters into two groups of equal size
    num_groups = 10
    assert num_observations % num_groups == 0, "Num groups must divide parameters"
    group_size = num_parameters // num_groups
    parameters_groups = list(zip(*(iter(range(num_parameters)),) * group_size))
    assert len(parameters_groups) == num_groups

    # =============================================================================
    # SETUP ESMDA FOR LOCALIZATION AND SOLVE PROBLEM
    # =============================================================================
    alpha = normalize_alpha(np.ones(5))

    start_time = time.perf_counter()
    smoother = AdaptiveESMDA(covariance=covariance, observations=observations, seed=1)

    # Simulate realization that die
    living_mask = rng.choice(
        [True, False], size=(len(alpha), num_ensemble), p=[0.9, 0.1]
    )

    X_i = np.copy(X)
    for i, alpha_i in enumerate(alpha, 1):
        print(f"ESMDA iteration {i} with alpha_i={alpha_i}")

        # Run forward model
        Y_i = g(X_i)

        # We simulate loss of realizations due to compute clusters going down.
        # Figure out which realizations are still alive:
        alive_mask_i = np.all(living_mask[:i, :], axis=0)
        num_alive = alive_mask_i.sum()
        print(f"  Total realizations still alive: {num_alive} / {num_ensemble}")

        # Create noise D - common to this ESMDA update
        D_i = smoother.perturb_observations(
            size=(num_observations, num_alive), alpha=alpha_i
        )

        # Create transition matrix K, independent of X
        transition_matrix = compute_cross_covariance_multiplier(
            alpha=alpha_i, C_D=smoother.C_D, D=D_i, Y=Y_i[:, alive_mask_i]
        )

        # Loop over parameter groups and update
        for j, parameter_mask_j in enumerate(parameters_groups, 1):
            print(f"  Updating parameter group {j}/{len(parameters_groups)}")

            # Mask out rows in this parameter group, and columns of realization
            # that are still alive. This step simulates fetching from storage.
            mask = np.ix_(parameter_mask_j, alive_mask_i)

            # Update the relevant parameters and write to X (storage)
            X_i[mask] = smoother.adaptive_assimilate(
                X_i[mask],
                Y_i[:, alive_mask_i],
                transition_matrix,
                correlation_threshold=lambda ensemble_size: 0.5,
            )

        # import matplotlib.pyplot as plt
        # plt.title("X_i")
        # plt.imshow(X_i)
        # plt.show()

        print()

    print(f"ESMDA with localization - Ran in {time.perf_counter() - start_time} s")

    # =============================================================================
    # VERIFY RESULT AGAINST NORMAL ESMDA ITERATIONS
    # =============================================================================
    start_time = time.perf_counter()
    smoother = ESMDA(
        covariance=covariance,
        observations=observations,
        alpha=alpha,
        seed=1,
    )

    X_i2 = np.copy(X)
    for i in range(smoother.num_assimilations()):
        # Run simulations
        Y_i = g(X_i2)

        # We simulate loss of realizations due to compute clusters going down.
        # Figure out which realizations are still alive:
        alive_mask_i = np.all(living_mask[: i + 1, :], axis=0)
        num_alive = alive_mask_i.sum()

        X_i2[:, alive_mask_i] = smoother.assimilate(
            X_i2[:, alive_mask_i], Y_i[:, alive_mask_i]
        )

    # For this test to pass, correlation_threshold() should return <= 0
    print(
        "Norm difference between ESMDA with and without localization:",
        np.linalg.norm(X_i - X_i2),
    )
    assert np.allclose(X_i, X_i2, atol=1e-4)

    print(f"ESMDA without localization - Ran in {time.perf_counter() - start_time} s")

    print("------------------------------------------")


class RowScaling:
    # Illustration of how row scaling works, `multiply` is the important part
    # For the actual implementation, which is more involved, see:
    # https://github.com/equinor/ert/blob/84aad3c56e0e52be27b9966322d93dbb85024c1c/src/clib/lib/enkf/row_scaling.cpp
    def __init__(self, alpha=1.0):
        """Alpha is the strength of the update."""
        assert 0 <= alpha <= 1.0
        self.alpha = alpha

    def multiply(self, X, K):
        """Takes a matrix X and a matrix K and performs alpha * X @ K."""
        # This implementation merely mimics how RowScaling::multiply behaves
        # in the C++ code. It mutates the input argument X instead of returning.
        X[:, :] = X @ (K * self.alpha)


def ensemble_smoother_update_step_row_scaling(
    *,
    covariance: npt.NDArray[np.double],
    observations: npt.NDArray[np.double],
    X_with_row_scaling: List[Tuple[npt.NDArray[np.double], RowScaling]],
    Y: npt.NDArray[np.double],
    seed: Union[np.random._generator.Generator, int, None] = None,
    inversion: str = "exact",
    truncation: float = 1.0,
):
    """Perform a single ESMDA update (ES) with row scaling.
    The matrices in X_with_row_scaling WILL BE MUTATED.
    See the ESMDA class for information about input arguments.


    Explanation of row scaling
    --------------------------

    The ESMDA update can be written as:

        X_post = X_prior + X_prior @ K

    where K is a transition matrix. The core of the row scaling approach is that
    for each row i in the matrix X, we apply an update with strength alpha:

        X_post = X_prior + alpha * X_prior @ K
        X_post = X_prior @ (I + alpha * K)

    Clearly 0 <= alpha <= 1 denotes the 'strength' of the update; alpha == 1
    corresponds to a normal smoother update and alpha == 0 corresponds to no
    update. With the per row transformation of X the operation is no longer matrix
    multiplication but the pseudo code looks like:

        for i in rows:
            X_i_post = X_i_prior @ (I + alpha_i * K)

    See also original code:
        https://github.com/equinor/ert/blob/84aad3c56e0e52be27b9966322d93dbb85024c1c/src/clib/lib/enkf/row_scaling.cpp#L51

    """

    # Create ESMDA instance and set alpha=1 => run single assimilation (ES)
    smoother = ESMDA(
        covariance=covariance,
        observations=observations,
        seed=seed,
        inversion=inversion,
        alpha=1,
    )

    # Create transition matrix - common to all parameters in X
    transition_matrix = smoother.compute_transition_matrix(
        Y=Y, alpha=1, truncation=truncation
    )

    # The transition matrix K is a matrix such that
    #     X_posterior = X_prior + X_prior @ K
    # but the C++ code in ERT requires a transition matrix F that obeys
    #     X_posterior = X_prior @ F
    # To accomplish this, we add the identity to the transition matrix in place
    np.fill_diagonal(transition_matrix, transition_matrix.diagonal() + 1)

    # Loop over groups of rows (parameters)
    for X, row_scale in X_with_row_scaling:
        # In the C++ code, multiply() will transform the transition matrix F as
        #    F_new = F * alpha + I * (1 - alpha)
        # but the transition matrix F that we pass below is F := K + I, so:
        #    F_new = (K + I) * alpha + I * (1 - alpha)
        #    F_new = K * alpha + I * alpha + I - I * alpha
        #    F_new = K * alpha + I
        # And the update becomes : X_posterior = X_prior @ F_new
        # The update in the C++ code is equivalent to
        #    X_posterior = X_prior + alpha * X_prior @ K
        # if we had used the original transition matrix K that is returned from
        # ESMDA.compute_transition_matrix
        row_scale.multiply(X, transition_matrix)

    return X_with_row_scaling
