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
# on a linear sparse problem.
#
# - Each response is only affected by $3$ parameters.
# - This is represented by a tridiagonal matrix $A$ in the forward model $g(x) = Ax$.
# - The problem is Gauss-Linear, so in this case ESMDA will sample
#   the true posterior when the number of ensemble members (realizations) is large.

# %% [markdown]
# ## Import packages

# %%
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from iterative_ensemble_smoother import ESMDA
from iterative_ensemble_smoother.experimental import AdaptiveESMDA

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
alpha = 5

# %% [markdown]
# ## Create problem data - sparse tridiagonal matrix $A$

# %%
diagonal = np.ones(min(num_parameters, num_observations))

# Create a tridiagonal matrix (easiest with scipy)
A = sp.sparse.diags(
    [diagonal, diagonal, diagonal],
    offsets=[-1, 0, 1],
    shape=(num_observations, num_parameters),
    dtype=float,
).toarray()

# We add some noise that is insignificant compared to the
# actual local structure in the forward model
A = A + rng.standard_normal(size=A.shape) * 0.01

plt.title("Linear mapping $A$ in forward model $g(x) = Ax$")
plt.imshow(A)
plt.xlabel("Parameters (inputs)")
plt.ylabel("Responses (outputs)")
plt.tight_layout()
plt.show()


# %%
def g(X):
    """Apply the forward model."""
    return A @ X


# Create observations: obs = g(x) + N(0, 1)
x_true = np.linspace(-1, 1, num=num_parameters)
observation_noise = rng.standard_normal(size=num_observations)  # N(0, 1) noise
observations = g(x_true) + observation_noise

# Initial ensemble X ~ N(0, prior_std) and diagonal covariance with ones
X = rng.normal(size=(num_parameters, num_ensemble)) * prior_std

# Covariance matches the noise added to observations above
covariance = np.ones(num_observations)  # N(0, 1) covariance

# %% [markdown]
# ## Solve the maximum likelihood problem
#
# We can solve $Ax = b$, where $b$ is the observations,
# for the maximum likelihood estimate.
#
# Notice that unlike using a Ridge model,
# solving $Ax = b$ directly does not use any prior information.

# %%
x_ml, *_ = np.linalg.lstsq(A, observations, rcond=None)

plt.figure(figsize=(8, 3))
plt.scatter(np.arange(len(x_true)), x_true, label="True parameter values")
plt.scatter(np.arange(len(x_true)), x_ml, label="ML estimate (no prior)")
plt.xlabel("Parameter index")
plt.ylabel("Parameter value")
plt.grid(True, ls="--", zorder=0, alpha=0.33)
plt.legend()
plt.show()

# %% [markdown]
# ## Solve using ESMDA
#
# We crease an `ESMDA` instance and solve the Guass-linear problem.

# %%
smoother = ESMDA(
    covariance=covariance,
    observations=observations,
    alpha=alpha,
    seed=1,
)

X_i = np.copy(X)
for i, alpha_i in enumerate(smoother.alpha, 1):
    print(
        f"ESMDA iteration {i}/{smoother.num_assimilations()}"
        + f" with inflation factor alpha_i={alpha_i}"
    )
    X_i = smoother.assimilate(X_i, Y=g(X_i))


X_posterior = np.copy(X_i)

# %% [markdown]
# ## Plot and compare solutions
#
# Compare the true parameters with both the ML estimate
# from linear regression and the posterior means obtained using `ESMDA`.

# %%
plt.figure(figsize=(8, 3))
plt.scatter(np.arange(len(x_true)), x_true, label="True parameter values")
plt.scatter(np.arange(len(x_true)), x_ml, label="ML estimate (no prior)")
plt.scatter(
    np.arange(len(x_true)), np.mean(X_posterior, axis=1), label="Posterior mean"
)
plt.xlabel("Parameter index")
plt.ylabel("Parameter value")
plt.grid(True, ls="--", zorder=0, alpha=0.33)
plt.legend()
plt.show()

# %% [markdown]
# ## Solve using AdaptiveESMDA
#
# We crease an `AdaptiveESMDA` instance and solve the Guass-linear problem.

# %%
adaptive_smoother = AdaptiveESMDA(
    covariance=covariance,
    observations=observations,
    seed=1,
)

X_i = np.copy(X)

# Loop over alpha defined in ESMDA instance,
# the vector of inflation values alpha is then
# of the same size in AdaptiveESMDA and ESMDA,
# and it's correctly scaled
assert np.isclose(np.sum(1 / smoother.alpha), 1), "Incorrect scaling"
for i, alpha_i in enumerate(smoother.alpha, 1):
    print(
        f"AdaptiveESMDA iteration {i}/{len(smoother.alpha)}"
        + f" with inflation factor alpha_i={alpha_i}"
    )

    # Run forward model
    Y_i = g(X_i)

    # Perturb observations
    D_i = adaptive_smoother.perturb_observations(
        ensemble_size=X_i.shape[1], alpha=alpha_i
    )

    # Assimilate data
    X_i = adaptive_smoother.assimilate(X_i, Y=Y_i, D=D_i, alpha=alpha_i, verbose=False)


X_adaptive_posterior = np.copy(X_i)

# %%
plt.figure(figsize=(8, 3))
plt.scatter(np.arange(len(x_true)), x_true, label="True parameter values")
plt.scatter(np.arange(len(x_true)), x_ml, label="ML estimate (no prior)")
plt.scatter(
    np.arange(len(x_true)), np.mean(X_posterior, axis=1), label="Posterior mean"
)
plt.scatter(
    np.arange(len(x_true)),
    np.mean(X_adaptive_posterior, axis=1),
    label="Posterior mean (adaptive)",
)
plt.xlabel("Parameter index")
plt.ylabel("Parameter value")
plt.grid(True, ls="--", zorder=0, alpha=0.33)
plt.legend()
plt.show()

# %% [markdown]
# ## Correlations between true parameters and solution means
#
# - A more sophisticated way to measure goodness would be to use [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
# - Here we simply look at the correlations of the means.

# %%
for arr, label in zip(
    [
        np.mean(X, axis=1),
        x_ml,
        np.mean(X_posterior, axis=1),
        np.mean(X_adaptive_posterior, axis=1),
    ],
    ["Prior", "ML estimate", "Posterior mean", "Posterior mean (adaptive)"],
):
    corr = sp.stats.pearsonr(x_true, arr).statistic
    print(label, corr)

# %% [markdown]
# ## Run on several ensemble sizes and seeds

# %%
ENSEMBLE_SIZES = list(range(2, 51))
NUM_SEEDS = 10

# Store average correlation coefficients
ESMDA_corrs = []
AdaptiveESMDA_corrs = []

# Loop over increasingly large ensemble sizes
for ensemble_size in ENSEMBLE_SIZES:
    # Posteriors means for this size
    ESMDA_means = []
    AdaptiveESMDA_means = []

    # Loop over seeds used in ESMDA/AdaptiveESMDA,
    # which determine the perturbations of the observations.
    for seed in range(NUM_SEEDS):
        # Prior
        X = rng.normal(size=(num_parameters, ensemble_size)) * prior_std

        # ================ ESMDA ==============
        smoother = ESMDA(
            covariance=covariance,
            observations=observations,
            alpha=5,
            seed=None,
        )

        X_i = np.copy(X)
        for i, alpha_i in enumerate(smoother.alpha, 1):
            X_i = smoother.assimilate(X_i, Y=g(X_i))

        ESMDA_means.append(np.mean(X_i, axis=1))

        # ============ AdaptiveESMDA ============
        adaptive_smoother = AdaptiveESMDA(
            covariance=covariance,
            observations=observations,
            seed=None,
        )

        X_i = np.copy(X)

        for i, alpha_i in enumerate(smoother.alpha, 1):
            # Perturb observations
            D_i = adaptive_smoother.perturb_observations(
                ensemble_size=X_i.shape[1], alpha=alpha_i
            )

            # Assimilate data
            X_i = adaptive_smoother.assimilate(X_i, Y=g(X_i), D=D_i, alpha=alpha_i)

        AdaptiveESMDA_means.append(np.mean(X_i, axis=1))

    # Collect results for all runs of this size
    ESMDA_corr = np.mean(
        [sp.stats.pearsonr(x_true, arr).statistic for arr in ESMDA_means]
    )
    AdaptiveESMDA_corr = np.mean(
        [sp.stats.pearsonr(x_true, arr).statistic for arr in AdaptiveESMDA_means]
    )

    ESMDA_corrs.append(ESMDA_corr)
    AdaptiveESMDA_corrs.append(AdaptiveESMDA_corr)

# %%
# Compute correlations
title = "ESMDA vs AdaptiveESMDA on different ensemble sizes\n"
title += f"each point represents the average of {NUM_SEEDS} \n"
title += "runs with different seeds (perturbations of D)\n"
title += f"Parameters = {num_parameters}, Observations ="
title += f" {num_observations}, len(alpha) = {len(smoother.alpha)}"


plt.title(title)

plt.plot(ENSEMBLE_SIZES, ESMDA_corrs, label="ESMDA")
plt.plot(ENSEMBLE_SIZES, AdaptiveESMDA_corrs, label="AdaptiveESMDA")

plt.xlabel("Ensemble size")
plt.ylabel("Correlation coeff of \nposterior mean vs. true parameter vector")

plt.grid(True)
plt.legend()
plt.show()
