"""
Contains (publicly available, but not officially supported) experimental
features of iterative_ensemble_smoother
"""

import logging
from typing import List, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore

from iterative_ensemble_smoother import ESMDA
from iterative_ensemble_smoother.esmda_inversion import (
    singular_values_to_keep,
)
from iterative_ensemble_smoother.utils import (
    calc_max_number_of_layers_per_batch_for_distance_localization,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


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


class DistanceESMDA(ESMDA):
    def __init__(
        self,
        covariance: npt.NDArray[np.float64],
        observations: npt.NDArray[np.float64],
        alpha: Union[int, npt.NDArray[np.float64]] = 5,
        seed: Union[np.random._generator.Generator, int, None] = None,
    ) -> None:
        """Initialize DistanceESMDA instance.

        Parameters
        ----------
        covariance : np.ndarray
            Covariance matrix of the observations.
        observations : np.ndarray
            Array of observations.
        alpha : int or np.ndarray, optional
            Covariance inflation factor(s). Default is 5.
        seed : np.random.Generator or int or None, optional
            Random seed for reproducibility. Default is None.
        """
        # Initialize instance
        super().__init__(
            covariance=covariance, observations=observations, alpha=alpha, seed=seed
        )

        # Ensure self.X3 is initialized to None
        # Is set in prepare_assimilation and used in assimilate_batch
        # Is not used when using assimilate
        self.X3: npt.NDArray[np.float64] = None
        self.C_D = covariance

    def assimilate(
        self,
        *,
        X: npt.NDArray[np.float64],
        Y: npt.NDArray[np.float64],
        rho: npt.NDArray[np.float64],
        truncation: float = 0.99,
    ):
        """Calculate Ensemble Smoother update with distance-based localization.

        Implementation of algorithm described in Appendix B of Emerick's publication.
        Reference: Emerick, Journal of Petroleum Science and Engineering 136(3):219-239
                   DOI:10.1016/j.petrol.2016.01.029  (2016)

        Uses the RHO matrix to define the scaling/tapering function for reduction of
        observations influence on field parameters. This function implements all steps
        in the algorithm.

        Typical workflow:
          Create object of DistanceESMDA
          Loop over batches of field parameter values from a field parameter:
            Run 'assimilate' for each batch of field parameter values

        Parameters
        ----------
        X : np.ndarray
            Parameter matrix of shape (nparameters, nrealizations).
        Y : np.ndarray
            Response matrix with predictions of observations,
            shape (nobservations, nrealizations).
        rho : np.ndarray
            Localization matrix with scaling factors,
            shape (nparameters, nobservations).
        truncation : float, optional
            Threshold for how many singular values to include from SVD.
            Default is 0.99.

        Returns
        -------
        np.ndarray
            Posterior (updated) matrix with parameters,
            shape (nparameters, nrealizations).
        """

        N_n, N_e = Y.shape

        # Subtract the mean of every parameter, see Eqn (B.4)
        M_delta = X - np.mean(X, axis=1, keepdims=True)

        # Subtract the mean of every observation, see Eqn (B.5)
        D_delta = Y - np.mean(Y, axis=1, keepdims=True)

        # See Eqn (B.8)
        # Compute the diagonal of the inverse of S directly, without forming S itself.
        if self.C_D.ndim == 1:
            # If C_D is 1D, it's a vector of variances. S_inv_diag is 1/sqrt(variances).
            S_inv_diag = 1.0 / np.sqrt(self.C_D)
        else:
            # If C_D is 2D, extract its diagonal of variances, then compute S_inv_diag.
            S_inv_diag = 1.0 / np.sqrt(np.diag(self.C_D))

        # See Eqn (B.10)
        U, w, VT = sp.linalg.svd(S_inv_diag[:, np.newaxis] * D_delta)
        idx = singular_values_to_keep(w, truncation=truncation)
        N_r = min(N_n, N_e - 1, idx)  # Number of values in SVD to keep
        U_r, w_r = U[:, :N_r], w[:N_r]

        # See Eqn (B.12)
        # Calculate C_hat_D, the correlation matrix of measurement errors.
        # This is defined as C_hat_D = S^-1 * C_D * S^-1
        if self.C_D.ndim == 1:
            # If C_D is a 1D vector of variances,
            # it represents a diagonal matrix.
            # In this special case,
            # S_inv * C_D * S_inv simplifies to the identity matrix.
            C_hat_D = np.identity(N_n)
        else:  # C_D is a 2D matrix
            # This scales each ROW i of self.C_D by the scalar S_inv_diag[i].
            # This is numerically identical to the matrix multiplication S⁻¹ @ C_D
            C_hat_D_temp = S_inv_diag[:, np.newaxis] * self.C_D

            # This scales each COLUMN j of C_hat_D_temp by the scalar S_inv_diag[j].
            # This is numerically identical to the matrix multiplication (result) @ S⁻¹
            C_hat_D = C_hat_D_temp * S_inv_diag

        U_r_w_inv = U_r / w_r
        # See Eqn (B.13)
        R = (
            self.alpha[self.iteration]
            * (N_e - 1)
            * np.linalg.multi_dot([U_r_w_inv.T, C_hat_D, U_r_w_inv])
        )

        # See Eqn (B.14)
        H_r, Z_r = sp.linalg.eigh(R, driver="evr", overwrite_a=True)

        # See Eqn (B.18)
        _X = (S_inv_diag[:, np.newaxis] * U_r) * (1 / w_r) @ Z_r

        # See Eqn (B.19)
        L = np.diag(1.0 / (1.0 + H_r))

        # See Eqn (B.20)
        X1 = L @ _X.T
        # See Eqn (B.21)
        X2 = D_delta.T @ _X
        # See Eqn (B.22)
        X3 = X2 @ X1

        # See Eqn (B.23)
        K_i = M_delta @ X3

        # See Eqn (B.24)
        K_rho_i = rho * K_i

        D = self.perturb_observations(
            ensemble_size=N_e, alpha=self.alpha[self.iteration]
        )
        # See Eqn (B.25)
        X4 = K_rho_i @ (D - Y)

        self.iteration += 1

        # See Eqn (B.26)
        return X + X4

    def prepare_assimilation(
        self,
        Y: npt.NDArray[np.float64],
        truncation: float = 0.99,
        D: npt.NDArray[np.float64] = None,
    ):
        """
        The part of the algorithm that does not depend on the field parameters,
        but only on observations and responses are calculated here.
        No need to re-calculate this multiple times when running a loop to update
        batches of field parameters. This function implements all steps prior
        to the steps involving the field parameters and should be used as a
        preparation step once for each field parameter before running the
        loop over batches of field parameters where the field parameters
        are updated.

        Typical workflow:
          Create object of DistanceESMDA
          Prepare for update of field parameter by running 'prepare_assimilation'
          Loop over batches of field parameter values from a field parameter:
            Run 'assimilate_batch' for each batch of field parameter values

        Parameters
        ----------
        Y : np.ndarray
            Response matrix with predictions of observations,
            shape (nobservations, nrealizations).
        truncation : float, optional
            Threshold for how many singular values to include from SVD.
            Default is 0.99.
        D : np.ndarray or None, optional
            Matrix of perturbed observations, shape (nobservations, nrealizations).
            If None, the perturbed observations are drawn inside this function
            using assumption of independence between observation errors.
            Default is None.
        """

        N_n, N_e = Y.shape

        # Subtract the mean of every observation, see Eqn (B.5)
        D_delta = Y - np.mean(Y, axis=1, keepdims=True)

        # See Eqn (B.8)
        # Compute the diagonal of the inverse of S directly, without forming S itself.
        if self.C_D.ndim == 1:
            # If C_D is 1D, it's a vector of variances. S_inv_diag is 1/sqrt(variances).
            S_inv_diag = 1.0 / np.sqrt(self.C_D)
        else:
            # If C_D is 2D, extract its diagonal of variances, then compute S_inv_diag.
            S_inv_diag = 1.0 / np.sqrt(np.diag(self.C_D))

        # See Eqn (B.10)
        U, w, VT = sp.linalg.svd(S_inv_diag[:, np.newaxis] * D_delta)
        idx = singular_values_to_keep(w, truncation=truncation)
        N_r = min(N_n, N_e - 1, idx)  # Number of values in SVD to keep
        U_r, w_r = U[:, :N_r], w[:N_r]

        # See Eqn (B.12)
        # Calculate C_hat_D, the correlation matrix of measurement errors.
        # This is defined as C_hat_D = S^-1 * C_D * S^-1
        if self.C_D.ndim == 1:
            # If C_D is a 1D vector of variances,
            # it represents a diagonal matrix.
            # In this special case,
            # S_inv * C_D * S_inv simplifies to the identity matrix.
            C_hat_D = np.identity(N_n)
        else:  # C_D is a 2D matrix
            # This scales each ROW i of self.C_D by the scalar S_inv_diag[i].
            # This is numerically identical to the matrix multiplication S⁻¹ @ C_D
            C_hat_D_temp = S_inv_diag[:, np.newaxis] * self.C_D

            # This scales each COLUMN j of C_hat_D_temp by the scalar S_inv_diag[j].
            # This is numerically identical to the matrix multiplication (result) @ S⁻¹
            C_hat_D = C_hat_D_temp * S_inv_diag

        U_r_w_inv = U_r / w_r
        # See Eqn (B.13)
        R = (
            self.alpha
            * (N_e - 1)
            * np.linalg.multi_dot([U_r_w_inv.T, C_hat_D, U_r_w_inv])
        )

        # See Eqn (B.14)
        H_r, Z_r = sp.linalg.eigh(R, driver="evr", overwrite_a=True)

        # See Eqn (B.18)
        _X = (S_inv_diag[:, np.newaxis] * U_r) * (1 / w_r) @ Z_r

        # See Eqn (B.19)
        L = np.diag(1.0 / (1.0 + H_r))

        # See Eqn (B.20)
        X1 = L @ _X.T
        # See Eqn (B.21)
        X2 = D_delta.T @ _X

        # The matrices X3 and D with perturbed observations is saved
        # and reused in assimilation of each batch of field parameters

        # See Eqn (B.22)
        self.X3 = X2 @ X1

        # Observations with added perturbations
        if D is None:
            self.D = self.perturb_observations(ensemble_size=N_e, alpha=self.alpha)
        else:
            assert self.C_D.shape[0] == D.shape[0]
            self.D = D
        return

    def assimilate_batch(
        self,
        *,
        X_batch: npt.NDArray[np.float64],
        Y: npt.NDArray[np.float64],
        rho_batch: npt.NDArray[np.float64],
        D: npt.NDArray[np.float64] = None,
        truncation: float = 0.99,
    ):
        """Calculate Ensemble Smoother update for a batch of parameters.

        Implementation of algorithm described in Appendix B of Emerick's publication.
        Reference: Emerick, Journal of Petroleum Science and Engineering 136(3):219-239
                   DOI:10.1016/j.petrol.2016.01.029  (2016)

        Calculate Ensemble Smoother update with distance-based localization using the
        RHO matrix to define the scaling/tapering function for reduction of observations
        influence on field parameters. This function implements the update steps
        of the field parameters and requires that the initial steps in the algorithm
        is already run by using the function 'prepare_assimilation'.

        Parameters
        ----------
        X_batch : np.ndarray
            Parameter matrix of shape (nparameters, nrealizations).
        Y : np.ndarray
            Response matrix with predictions of observations,
            shape (nobservations, nrealizations).
        rho_batch : np.ndarray
            Localization matrix with scaling factors,
            shape (nparameters, nobservations).
        D : np.ndarray or None, optional
            Perturbed observations, shape (nobservations, nrealizations).
            If None, perturbations are simulated internally. Default is None.
        truncation : float, optional
            Threshold for how many singular values to include from SVD.
            Default is 0.99.

        Returns
        -------
        np.ndarray
            Posterior (updated) matrix with parameters,
            shape (nparameters, nrealizations).
        """
        if self.X3 is None:
            # Need to run preparation step
            assert Y.shape[0] == self.observations.shape[0]
            assert Y.shape[1] == X_batch.shape[1]
            self.prepare_assimilation(Y=Y, truncation=truncation, D=D)

        # Subtract the mean of every parameter, see Eqn (B.4)
        M_delta = X_batch - np.mean(X_batch, axis=1, keepdims=True)

        # See Eqn (B.23)
        K_i = M_delta @ self.X3

        # See Eqn (B.24)
        K_rho_i = rho_batch * K_i

        # See Eqn (B.25)
        X4 = K_rho_i @ (self.D - Y)

        # See Eqn (B.26)
        return X_batch + X4

    def update_params_2D(
        self,
        X: npt.NDArray[np.float64],
        Y: npt.NDArray[np.float64],
        rho_2D: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate posterior update with distance-based ESMDA for one 2D parameter.

        The RHO matrix is specified as input. Number of parameters for the field
        is nx*ny, dimensions of the rho_2D in addition to number of observations.

        Parameters
        ----------
        X : np.ndarray
            Matrix with prior realizations of all field parameters,
            shape (nparameters, nrealizations).
        Y : np.ndarray
            Matrix with response values for each observation for each realization,
            shape (nobservations, nrealizations).
        rho_2D : np.ndarray
            Localization matrix for distance-based correlation tapering.
            Shape (nx, ny, nobservations), where nx and ny are the
            dimensions of the 2D grid. Each element specifies the
            localization factor between a grid cell and an observation,
            with values in [0, 1]. A value of 1 indicates full correlation
            (no localization), while 0 indicates the parameter and
            observation are independent.

        Returns
        -------
        np.ndarray
            Posterior ensemble of field parameters,
            shape (nparameters, nrealizations).
        """
        # No update if no observations or responses
        if Y is None or Y.shape[0] == 0:
            # No update of the field parameters
            # Check if it necessary to make a copy or can we only return X?
            return X.copy()

        nx, ny, nobs = rho_2D.shape
        nparam = nx * ny
        nreal = X.shape[1]
        assert X.shape[0] == nparam, (
            f"Mismatch between X dimension {X.shape[0]} and nparam {nparam}"
        )

        assert nobs == Y.shape[0], (
            "Mismatch between Y matrix dimension for number of observation and rho_2D"
        )
        assert Y.shape[1] == nreal, (
            f"Mismatch between X dimension {Y.shape[1]} and nreal {nreal}"
        )
        assert self.observations.shape[0] == nobs

        # Skip assimilation if rho is all zeros (no localization effect)
        if np.count_nonzero(rho_2D) == 0:
            return X.copy()

        rho = rho_2D.reshape(nparam, nobs)
        return self.assimilate_batch(X_batch=X, Y=Y, rho_batch=rho)

    def update_params_3D(
        self,
        X: npt.NDArray[np.float64],
        Y: npt.NDArray[np.float64],
        rho_2D_slice: npt.NDArray[np.float64],
        nz: int,
        min_nbatch: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Calculate posterior update with distance-based ESMDA for one 3D parameter.

        The RHO for one layer of the 3D field parameter is input.
        This is copied to all other layers of RHO in each batch of grid parameter
        layers since only lateral distance is used when calculating distances.
        Number of parameters is nx*ny*nz where nx, ny is dimension of the rho_2D_slice
        in addition to number of observations.

        Parameters
        ----------
        X : np.ndarray
            Matrix with prior realizations of all field parameters,
            shape (nparameters, nrealizations).
        Y : np.ndarray
            Matrix with response values for each observation for each realization,
            shape (nobservations, nrealizations).
        rho_2D_slice : np.ndarray
            Localization matrix for distance-based correlation tapering
            for one horizontal layer (slice) of a 3D grid. Shape
            (nx, ny, nobservations), where nx and ny are the lateral
            dimensions of the grid. This same localization is applied to
            all vertical layers. Each element specifies the localization
            factor between a grid cell and an observation, with values
            in [0, 1]. A value of 1 indicates full correlation (no
            localization), while 0 indicates the parameter and observation
            are independent.
        nz : int, optional
            Number of grid layers for the 3D field parameter. Default is 1.
        min_nbatch : int, optional
            Minimum number of batches the field parameter is split into.
            Default is 1. Usually number of batches will be calculated based
            on available memory and the size of the field parameters,
            number of observations and realizations. The actual number of
            batches will be max(min_nbatch, min_number_of_batches_required).

        Returns
        -------
        np.ndarray
            Posterior ensemble of field parameters,
            shape (nx*ny*nz, nrealizations).
        """
        # No update if no observations or responses
        if Y is None or Y.shape[0] == 0:
            # No update of the field parameters
            # Check if it is it necessary to return a copy here?
            return X.copy()

        nx, ny, nobs = rho_2D_slice.shape

        nparam_per_layer = nx * ny
        nparam = nparam_per_layer * nz
        nreal = X.shape[1]
        assert X.shape[0] == nparam, (
            f"Mismatch between X dimension {X.shape[0]} and nparam {nparam}"
        )
        assert nobs == Y.shape[0], (
            f"Mismatch between number of observations in Rho_2D {nobs} and in Y matrix"
        )
        assert Y.shape[1] == nreal, (
            f"Mismatch between X dimension {Y.shape[1]} and nreal {nreal}"
        )

        X_3D = X.reshape(nx, ny, nz, nreal)

        # Check memory constraints and calculate how many grid layers of
        # field parameters is possible to update on one batch
        max_nlayers_per_batch = (
            calc_max_number_of_layers_per_batch_for_distance_localization(
                nx, ny, nz, nobs, nreal, bytes_per_float=8
            )
        )  # Use float64
        nlayer_per_batch = min(max_nlayers_per_batch, nz)
        nbatch = int(nz / nlayer_per_batch)

        # Number of batches is defined by available memory and wanted number of batches
        # Usually one should have as few batches as possible but sufficient to
        # be able to update a batch with the memory available.
        # It is possible to explicit require more batches than the minimum
        # number required by the memory constraint. Main use case for this
        # is probably only for unit testing to avoid having unit tests running
        # slow due to very big size of the field, the number of observations and
        # and number of realizations.
        nbatch = max(min_nbatch, nbatch)
        nlayer_per_batch = int(nz / nbatch)
        if nbatch * nlayer_per_batch == nz:
            logger.debug(f"Number of batches used: {nbatch}")
        else:
            logger.debug(f"Number of batches used: {nbatch + 1}")

        nparam_in_batch = (
            nparam_per_layer * nlayer_per_batch
        )  # For full sized batch of layers

        nlayer_last_batch = nz - nbatch * nlayer_per_batch

        # Initialize the X_post_3D
        X_post_3D = X_3D.copy()
        for batch_number in range(nbatch):
            start_layer_number = batch_number * nlayer_per_batch
            end_layer_number = start_layer_number + nlayer_per_batch

            X_batch = X_3D[:, :, start_layer_number:end_layer_number, :].reshape(
                (nparam_in_batch, nreal)
            )

            if np.count_nonzero(rho_2D_slice) != 0:
                # Copy rho calculated from one layer of 3D parameter into all layers for
                # current batch of layers.
                # Size of rho batch: (nx,ny,nlayer_per_batch,nobs)
                rho_3D_batch = np.zeros(
                    (nx, ny, nlayer_per_batch, nobs), dtype=np.float64
                )
                rho_3D_batch[:, :, :, :] = rho_2D_slice[:, :, np.newaxis, :]
                rho_batch = rho_3D_batch.reshape((nparam_in_batch, nobs))

                logger.debug(f"Assimilate batch number {batch_number}")
                X_post_batch = self.assimilate_batch(
                    X_batch=X_batch, Y=Y, rho_batch=rho_batch
                )

                X_post_3D[:, :, start_layer_number:end_layer_number, :] = (
                    X_post_batch.reshape(nx, ny, nlayer_per_batch, nreal)
                )
            else:
                X_post_batch = X_batch
                logger.debug(
                    f"Skip assimilate for batch number {batch_number} since rho is zero"
                )

        if nlayer_last_batch > 0:
            batch_number = nbatch
            start_layer_number = batch_number * nlayer_per_batch
            end_layer_number = start_layer_number + nlayer_last_batch
            nparam_in_last_batch = nparam_per_layer * nlayer_last_batch

            X_batch = X_3D[:, :, start_layer_number:end_layer_number, :].reshape(
                (nparam_in_last_batch, nreal)
            )

            if np.count_nonzero(rho_2D_slice) != 0:
                rho_3D_batch = np.zeros(
                    (nx, ny, nlayer_last_batch, nobs), dtype=np.float64
                )
                # Copy rho calculated from one layer of 3D parameter into all layers for
                # current batch of layers
                rho_3D_batch[:, :, :, :] = rho_2D_slice[:, :, np.newaxis, :]
                rho_batch = rho_3D_batch.reshape((nparam_in_last_batch, nobs))

                logger.debug(f"Assimilate batch number {batch_number}")
                X_post_batch = self.assimilate_batch(
                    X_batch=X_batch, Y=Y, rho_batch=rho_batch
                )

                X_post_3D[:, :, start_layer_number:end_layer_number, :] = (
                    X_post_batch.reshape(nx, ny, nlayer_last_batch, nreal)
                )
            else:
                X_post_batch = X_batch
                logger.debug(
                    f"Skip assimilate for batch number {batch_number} since rho is zero"
                )

        return X_post_3D.reshape(nparam, nreal)

    def update_params(
        self,
        X: npt.NDArray[np.float64],
        Y: npt.NDArray[np.float64],
        rho_input: npt.NDArray[np.float64],
        nz: int = 1,
        min_nbatch: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Calculate posterior update with distance-based ESMDA.

        Depending on the shape of rho_input, this method handles updates for
        parameters representing 1D, 2D, or 3D fields. X is always a 2D matrix
        of shape (nparameters, nrealizations), where nparameters corresponds to
        the flattened field (nx for 1D, nx*ny for 2D, nx*ny*nz for 3D).
        The shape of rho_input and the nz parameter determine how the field
        is structured.

        Parameters
        ----------
        X : np.ndarray
            Matrix with prior realizations of all field parameters,
            shape (nparameters, nrealizations).
        Y : np.ndarray
            Matrix with response values for each observation for each realization,
            shape (nobservations, nrealizations).
        rho_input : np.ndarray
            RHO matrix elements for localization.
            If rho_input has shape (nparam, nobs) and nz == 1,
            it is treated as a flat (1D) field and reshaped to (nparam, 1, nobs).
            If rho_input has shape (nx, ny, nobs) and nz == 1,
            parameters represent a 2D field with nparameters = nx*ny.
            If rho_input has shape (nx, ny, nobs) and nz > 1,
            parameters represent a 3D field with nparameters = nx*ny*nz.
        nz : int, optional
            Number of grid layers for the 3D field parameter. Default is 1.
        min_nbatch : int, optional
            Minimum number of batches the field parameter is split into.
            Default is 1. Usually number of batches will be calculated based
            on available memory and the size of the field parameters,
            number of observations and realizations. The actual number of
            batches will be max(min_nbatch, min_number_of_batches_required).

        Returns
        -------
        np.ndarray
            Posterior ensemble of field parameters,
            shape (nx*ny*nz, nrealizations).
        """
        if rho_input.ndim == 2 and nz == 1:
            # Treat as flat field: reshape (nparam, nobs) to (nparam, 1, nobs)
            return self.update_params_2D(X=X, Y=Y, rho_2D=rho_input[:, np.newaxis, :])
        if rho_input.ndim == 3 and nz == 1:
            return self.update_params_2D(X=X, Y=Y, rho_2D=rho_input)
        return self.update_params_3D(
            X=X,
            Y=Y,
            rho_2D_slice=rho_input,
            nz=nz,
            min_nbatch=min_nbatch,
        )


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
        ]
    )
