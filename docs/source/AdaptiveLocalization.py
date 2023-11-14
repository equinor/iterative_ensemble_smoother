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

# %% [markdown]
# # Imports

# %%
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge

from iterative_ensemble_smoother import ESMDA
from iterative_ensemble_smoother.esmda_inversion import normalize_alpha

# %%
# %#load_ext autoreload
# %#autoreload 2
# %%
from iterative_ensemble_smoother.experimental import AdaptiveESMDA

# %%
ALPHA_PLOT = 0.5  # For plotting

# %% [markdown]
# # Create problem - linear regression

# %%
# Create a problem with g(x) = A @ x
rng = np.random.default_rng(42)

# Increase both observations and ensemble members - extreme problems
num_parameters = 25
num_observations = 50
num_ensemble = 100

A = rng.standard_normal(size=(num_observations, num_parameters))


def g(X):
    """Forward model."""
    return A @ X


# Create observations
x_true = np.linspace(-1, 1, num=num_parameters)
observations = g(x_true) + rng.standard_normal(size=num_observations)

# Initial ensemble and covariance
X = rng.normal(size=(num_parameters, num_ensemble))

# Covariance matches the noise added to observations above
covariance = np.ones(num_observations)

# %% [markdown]
# # Solve linear regression problem using sklearn

# %%

model = Ridge(fit_intercept=False)
model.fit(A, observations)

plt.scatter(
    np.arange(len(x_true)),
    x_true,
    color="black",
    zorder=99,
    label="True parameter values",
)
plt.scatter(
    np.arange(len(x_true)), model.coef_, zorder=80, label="Linear regression coefs"
)
plt.legend()
plt.show()

# %% [markdown]
# # ESMDA

# %%
plt.scatter(
    np.arange(len(x_true)),
    x_true,
    color="black",
    zorder=99,
    label="True parameter values",
)
plt.scatter(
    np.arange(len(x_true)),
    model.coef_,
    zorder=80,
    label="Linear regression coefs",
    alpha=ALPHA_PLOT,
)


# =============================================================================
# SETUP ESMDA FOR LOCALIZATION AND SOLVE PROBLEM
# =============================================================================
smoother = ESMDA(
    covariance=covariance,
    observations=observations,
    alpha=5,
    seed=1,
)

X_i2 = np.copy(X)
for i in range(smoother.num_assimilations()):
    X_i2 = smoother.assimilate(X_i2, Y=g(X_i2))

    X_i_means = X_i2.mean(axis=1)

plt.scatter(
    np.arange(len(x_true)),
    X_i_means,
    zorder=i,
    label=f"ESMDA estimated parameters (iter={i+1})",
    alpha=ALPHA_PLOT,
)


plt.legend()
plt.show()

# %% [markdown]
# # ESMDA with adaptive localization

# %%
plt.scatter(
    np.arange(len(x_true)),
    x_true,
    color="black",
    zorder=99,
    label="True parameter values",
)
plt.scatter(
    np.arange(len(x_true)),
    model.coef_,
    zorder=80,
    label="Linear regression coefs",
    alpha=ALPHA_PLOT,
)
plt.scatter(
    np.arange(len(x_true)),
    X.mean(axis=1),
    zorder=80,
    label="Prior means",
    alpha=ALPHA_PLOT,
)

THRESHOLD = 0.5
plt.title(f"ESMDA with adaptive localization - threshold={THRESHOLD}")

# =============================================================================
# SETUP ESMDA FOR LOCALIZATION AND SOLVE PROBLEM
# =============================================================================
alpha = normalize_alpha(np.ones(5))

smoother = AdaptiveESMDA(covariance=covariance, observations=observations, seed=1)


X_i = np.copy(X)
for i, alpha_i in enumerate(alpha, 1):
    print(f"ESMDA iteration {i} with alpha_i={alpha_i}")

    # Run forward model
    Y_i = g(X_i)

    # Create noise D - common to this ESMDA update
    D_i = smoother.perturb_observations(size=Y_i.shape, alpha=alpha_i)

    # Create transition matrix K, independent of X
    transition_matrix = smoother.adaptive_transition_matrix(Y=Y_i, D=D_i, alpha=alpha_i)

    # Update the relevant parameters and write to X (storage)
    X_i = smoother.adaptive_assimilate(
        X_i,
        Y_i,
        transition_matrix,
        correlation_threshold=lambda ensemble_size: THRESHOLD,
    )
    X_i_means_adaptive = X_i.mean(axis=1)


# Finish the plot
plt.scatter(
    np.arange(len(x_true)),
    X_i_means,
    zorder=i,
    label=f"ESMDA estimated parameters (iter={i})",
    alpha=ALPHA_PLOT,
)
plt.scatter(
    np.arange(len(x_true)),
    X_i_means_adaptive,
    zorder=i,
    label=f"Adap. ESMDA estimated parameters (iter={i})",
    alpha=ALPHA_PLOT,
)


plt.legend()
plt.show()
