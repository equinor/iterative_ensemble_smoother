# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Estimating savings rate and deposit in a mutual fund using ESMDA
#
# In this notebook we use ESMDA to answer the question:
#
# - Given observations of someones mutual fund account, what is the likely yearly (1) interest rate and (2) deposit amount?
#
# We start with priors on these two parameters, then use the observations to assimilate data, updating the prior to a posterior.
#
# The purpose is primarily to demonstrate the API for ESMDA on a simple problem.
# %%
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
esmda = ies.ESMDA(
    # Covariances associated with every observation.
    # We assume a standard deviation of 10%, the square it
    # to obtain a diagonal covariance matrix.
    # Since the covariance must be positive we add 1 too
    C_D=1 + (observations * 0.1) ** 2,
    # The 1D vector of real-world observations
    observations=observations,
    # Number of steps
    alpha=1,
    # Random seed
    seed=1,
)


X_i = np.copy(X_prior)
for assimilation in range(esmda.num_assimilations()):
    X_i = esmda.assimilate(X_i, Y=G(X_i))

    plot_ensemble(
        observations,
        X_i,
        title=f"Posterior after assimilation step {assimilation+1} / {esmda.num_assimilations()}",
    )
    plt.show()

# %% [markdown]
# ## Create and run ESMDA - with 10 iterations

# %%
esmda = ies.ESMDA(
    C_D=1 + (observations * 0.1) ** 2, observations=observations, alpha=10, seed=1
)


X_i = np.copy(X_prior)
for assimilation in range(esmda.num_assimilations()):
    X_i = esmda.assimilate(X_i, Y=G(X_i))

    plot_ensemble(
        observations,
        X_i,
        title=f"Posterior after assimilation step {assimilation+1} / {esmda.num_assimilations()}",
    )
    plt.show()
