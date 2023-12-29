# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# ruff: noqa: E402

# %% [markdown]
# # Adaptive Localization
#
# In this example we run adaptive localization
# on a sparse Gauss-linear problem.
#
# - Each response is only affected by $3$ parameters.
# - This is represented by a tridiagonal matrix $A$ in the forward model $g(x) = Ax$.
# - The problem is Gauss-Linear, so in this case ESMDA will sample
#   the true posterior when the number of ensemble members (realizations) is large.
# - The sparse correlation structure will lead to spurious correlations, in the sense
#   that a parameter and response might appear correlated when in fact they are not.

# %% [markdown]
# ## Import packages

# %%
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from iterative_ensemble_smoother import ESMDA
from iterative_ensemble_smoother.esmda_inversion import empirical_cross_covariance
from iterative_ensemble_smoother.experimental import AdaptiveESMDA
from iterative_ensemble_smoother.utils import sample_mvnormal

# %% [markdown]
# ## Create problem data
#
# Some settings worth experimenting with:
#
# - Decreasing `prior_std=1` will pull the posterior solution toward zero.
# - Increasing `num_ensemble` will increase the quality of the solution.
# - Increasing `num_observations / num_parameters`
#   will increase the quality of the solution.

# %%
rng = np.random.default_rng(42)

# Dimensionality of the problem
num_parameters = 100
num_observations = 50
num_ensemble = 25
prior_std = 1

# Number of iterations to use in ESMDA
alpha = 1

# Effect size (coefficients in sparse mapping A)
effect_size = 1000

# %% [markdown]
# ## Create problem data - sparse tridiagonal matrix $A$

# %%
diagonal = effect_size * np.ones(min(num_parameters, num_observations))

# Create a tridiagonal matrix (easiest with scipy)
A = sp.sparse.diags(
    [diagonal, diagonal, diagonal],
    offsets=[-1, 0, 1],
    shape=(num_observations, num_observations),  # (num_observations, num_parameters),
    dtype=float,
).toarray()

A = np.tile(A, reps=(1, num_parameters // num_observations + 1))
A = A[:num_observations, :num_parameters]


# Create a tridiagonal matrix (easiest with scipy)
A = sp.sparse.diags(
    [diagonal, diagonal, diagonal],
    offsets=[-1, 0, 1],
    shape=(num_observations, num_parameters),
    dtype=float,
).toarray()

# We add some noise that is insignificant compared to the
# actual local structure in the forward model
A = A + rng.standard_normal(size=A.shape) * 0.001

plt.title("Linear mapping $A$ in forward model $g(x) = Ax$")
plt.imshow(A)
plt.xlabel("Parameters (inputs)")
plt.ylabel("Responses (outputs)")
plt.tight_layout()
plt.show()


# %% [markdown]
# - Below we draw prior realizations $X \sim N(0, \sigma)$.
# - The true parameter values used to generate observations are in the range $[-1, 1]$.
# - As the number of realizations (ensemble members) goes to infinity,
#   the correlation between the prior and the true parameter values converges to zero.
# - The correlation is zero for a finite number of realizations too,
#   but statistical noise might induce some spurious correlations between
#   the prior and the true parameter values.
#
# **In summary the baseline correlation is zero.**
# Anything we can do to increase the correlation above zero beats the baseline,
# which is using the prior (no update).
#
# The correlation coefficient does not take into account the uncertainty
# represeted in the posterior, only the mean posterior value is compared with
# the true parameter values.
# To compare distributions we could use the
# [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence),
# but we do not pursue this here.

# %%
def get_prior(num_parameters, num_ensemble, prior_std):
    """Sample prior from N(0, prior_std)."""
    return rng.normal(size=(num_parameters, num_ensemble)) * prior_std


def g(X):
    """Apply the forward model."""
    return A @ X


# Create observations: obs = g(x) + N(0, 1)
x_true = np.linspace(-1, 1, num=num_parameters)
observation_noise = rng.standard_normal(size=num_observations)  # N(0, 1) noise
observations = g(x_true) + observation_noise

# Initial ensemble X ~ N(0, prior_std) and diagonal covariance with ones
X = get_prior(num_parameters, num_ensemble, prior_std)

# Covariance matches the noise added to observations above
covariance = np.ones(num_observations)  # N(0, 1) covariance

# %% [markdown]
# ## Solve the maximum likelihood problem
#
# We can solve $Ax = b$, where $b$ are the observations,
# for the maximum likelihood estimate.
#
# Notice that unlike using a Ridge model,
# solving $Ax = b$ directly does not use any prior information.

# %%
x_ml, *_ = np.linalg.lstsq(A, observations, rcond=None)

plt.figure(figsize=(7, 3))
plt.scatter(np.arange(len(x_true)), x_true, label="True parameter values")
plt.scatter(np.arange(len(x_true)), x_ml, label="ML estimate (no prior)")
plt.xlabel("Parameter index")
plt.ylabel("Parameter value")
plt.grid(True, ls="--", zorder=0, alpha=0.33)
plt.legend()
plt.tight_layout()
plt.show()

# %%
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


def linear_l1_regression(U, Y):
    """
    Performs LASSO regression for each response in Y against predictors in U,
    constructing a sparse matrix of regression coefficients.

    The function scales features in U using standard scaling before applying
    LASSO, then re-scales the coefficients to the original scale of U. This
    extracts the effect of each feature in U on each response in Y, ignoring
    intercepts and constant terms.

    Parameters
    ----------
    U : np.ndarray
        2D array of predictors with shape (n, p).
    Y : np.ndarray
        2D array of responses with shape (n, m).

    Returns
    -------
    H_sparse : scipy.sparse.csc_matrix
        Sparse matrix (m, p) with re-scaled LASSO regression coefficients for
        each response in Y.

    Raises
    ------
    AssertionError
        If the number of samples in U and Y do not match, or if the shape of
        H_sparse is not (m, p).
    """
    # https://github.com/equinor/graphite-maps/blob/main/graphite_maps/linear_regression.py

    n, p = U.shape  # p: number of features
    n_y, m = Y.shape  # m: number of y responses

    # Assert that the first dimension of U and Y are the same
    assert n == n_y, "Number of samples in U and Y must be the same"

    scaler_u = StandardScaler()
    U_scaled = scaler_u.fit_transform(U)

    scaler_y = StandardScaler()
    Y_scaled = scaler_y.fit_transform(Y)

    # Loop over features
    coefs = []
    for j in range(m):
        y_j = Y_scaled[:, j]

        # Learn individual regularization and fit
        alphas = np.logspace(-8, 8, num=32)
        model_cv = LassoCV(alphas=alphas, fit_intercept=False, max_iter=10_000, cv=5)
        model_cv.fit(U_scaled, y_j)

        coef_scale = scaler_y.scale_[j] / scaler_u.scale_
        coefs.append(model_cv.coef_ * coef_scale)

    K = np.vstack(coefs)
    assert K.shape == (m, p)

    return K


Y = g(X)
K = linear_l1_regression(X.T, Y.T)
plt.imshow(K)
plt.show()


# %%
def solve_esmda(X, Y, covariance, observations, seed=1):
    """Solving using exact ESMDA computation."""

    # Create a smoother to get D
    smoother = ESMDA(
        covariance=covariance,
        observations=observations,
        alpha=1,
        seed=seed,
    )
    N = X.shape[1]  # Number of ensemble members (realizations)
    D = smoother.perturb_observations(ensemble_size=N, alpha=1.0)

    # Cross covariance and covariance
    cov_XY = empirical_cross_covariance(X, Y)
    cov_YY = empirical_cross_covariance(Y, Y)

    # Compute the update using the exact ESMDA approach
    K = cov_XY @ sp.linalg.inv(cov_YY + np.diag(covariance))
    return X + K @ (D - Y)


def solve_projected(X, Y, covariance, observations, seed=1):
    """Instead of using the covariance matrix, sample S ~ N(0, Cov),
    then use S @ S.T in place of it. This is a low rank approximation.

    In itsself, this method has little going for it. But it leads
    to other approaches since Y_noisy = Y + S can be defined.
    """

    # Create a smoother to get D
    smoother = ESMDA(
        covariance=covariance,
        observations=observations,
        alpha=1,
        seed=seed,
    )
    N = X.shape[1]
    D = smoother.perturb_observations(ensemble_size=N, alpha=1.0)

    # Instead of using the covariance matrix, sample from it
    S = sample_mvnormal(C_dd_cholesky=smoother.C_D_L, rng=smoother.rng, size=N)

    cov_XY = empirical_cross_covariance(X, Y)
    cov_YY = empirical_cross_covariance(Y, Y)

    # Compute the update
    # TODO: N-1 vs N here.
    K = cov_XY @ sp.linalg.inv(cov_YY + S @ S.T / (N-1))
    return X + K @ (D - Y)


def solve_lstsq(X, Y, covariance, observations, seed=1):
    """Since K =~ E(grad h^-1), we have K @ Y = X.
    Here we simply solve Y.T @ K.T = X.T for X."""

    # Create a smoother to get D
    smoother = ESMDA(
        covariance=covariance,
        observations=observations,
        alpha=1,
        seed=seed,
    )
    N = X.shape[1]
    D = smoother.perturb_observations(ensemble_size=N, alpha=1.0)

    # Instead of using the covariance matrix, sample from it
    S = sample_mvnormal(C_dd_cholesky=smoother.C_D_L, rng=smoother.rng, size=N)

    Y_noisy = Y + S

    # Compute the update by solving K @ Y = X
    K, *_ = sp.linalg.lstsq(Y_noisy.T, X.T)
    K = K.T

    return X + K @ (D - Y)


def solve_lasso_direct(X, Y, covariance, observations, seed=1):
    """Use lasso to solve for H, then use the update equation
    K := cov_XX @ H.T @ inv(H @ cov_XX @ H.T + Cov)
    """

    # Create a smoother to get D
    smoother = ESMDA(
        covariance=covariance,
        observations=observations,
        alpha=1,
        seed=seed,
    )
    N = X.shape[1]
    D = smoother.perturb_observations(ensemble_size=N, alpha=1.0)

    # Approximate forward model with H
    H = linear_l1_regression(X.T, Y.T)
    cov_XX = empirical_cross_covariance(X, X)

    # Compute the update
    K = cov_XX @ H.T @ sp.linalg.inv(H @ cov_XX @ H.T + np.diag(covariance))
    return X + K @ (D - Y)


def solve_lasso(X, Y, covariance, observations, seed=1):
    """Solve K @ Y = X using Lasso."""

    # Create a smoother to get D
    smoother = ESMDA(
        covariance=covariance,
        observations=observations,
        alpha=1,
        seed=seed,
    )
    N = X.shape[1]
    D = smoother.perturb_observations(ensemble_size=N, alpha=1.0)

    # Instead of using the covariance matrix, sample from it
    S = sample_mvnormal(C_dd_cholesky=smoother.C_D_L, rng=smoother.rng, size=N)

    Y_noisy = Y + S  # Seems to work better with Y instead of Y_noisy

    # Compute the update
    K = linear_l1_regression(Y_noisy.T, X.T)

    return X + K @ (D - Y)


def solve_adaptive_ESMDA(X, Y, covariance, observations, seed=1):
    """Solving using adaptive ESMDA with correlation threshold."""

    adaptive_smoother = AdaptiveESMDA(
        covariance=covariance,
        observations=observations,
        seed=seed,
    )

    X_i = np.copy(X)

    for i, alpha_i in enumerate([1], 1):
        # Perturb observations
        D_i = adaptive_smoother.perturb_observations(
            ensemble_size=X_i.shape[1], alpha=alpha_i
        )

        # Assimilate data
        X_i = adaptive_smoother.assimilate(
            X=X_i, Y=g(X_i), D=D_i, alpha=alpha_i, verbose=False
        )

    return X_i


# %%
plt.figure(figsize=(8, 4.5))

for function, label in zip(
    [
        solve_esmda,
        solve_projected,
        solve_lstsq,
        solve_lasso_direct,
        solve_lasso,
        solve_adaptive_ESMDA,
    ],
    [
        "ESMDA",
        "projected",
        "lstsq",
        "Lasso (direct)",
        "Lasso",
        "AdaptiveESMDA",
    ],
):
    print(label)
    # Loop over seeds and solve
    corrs = []
    posterior_means = []
    for seed in range(10):
        X_posterior = function(X, g(X), covariance, observations, seed=seed)
        x_posterior_mean = X_posterior.mean(axis=1)
        corr = sp.stats.pearsonr(x_true, x_posterior_mean).statistic
        corrs.append(corr)
        posterior_means.append(x_posterior_mean)

    posterior_mean = np.vstack(posterior_means).mean(axis=0)
    corr_mean, corr_std = np.mean(corrs), np.std(corrs)

    plt.scatter(
        np.arange(len(x_true)),
        posterior_mean,
        label=f"{label} (corr: {corr_mean:.2f} +- {corr_std:.2f})",
        alpha=0.6,
    )


corr = sp.stats.pearsonr(x_true, x_posterior_mean).statistic
plt.scatter(
    np.arange(len(x_true)),
    x_ml,
    label=f"ML solution (knowing h(x) = Ax) (corr: {corr:.2f})",
)
plt.scatter(
    np.arange(len(x_true)),
    x_true,
    label="True parameter values",
)

# plt.scatter(np.arange(len(x_true)), x_proj, label="Projected")
plt.xlabel("Parameter index")
plt.ylabel("Parameter value")
plt.ylim([-3, 3])
plt.grid(True, ls="--", zorder=0, alpha=0.33)
plt.legend()
plt.tight_layout()
plt.show()

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
