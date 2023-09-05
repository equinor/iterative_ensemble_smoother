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
# # Example: Estimating parameters of an anharmonic oscillator
#
# The anharnomic oscillator can be modelled by a non-linear partial differential
# equation as described in section 6.3.4 of the book [Fundamentals of Algorithms
# and Data Assimilation](https://www.amazon.com/Data-Assimilation-Methods-Algorithms-Applications/dp/1611974534) by Mark Asch, Marc Bocquet and Maëlle Nodet.
#
# -------------
#
# The discrete two-parameter model is:
#
# $$x_{k+1} - 2 x_{k} + x_{k-1} = \omega^2 x_{k} - \lambda^2 x_{k}^3$$
#
# with initial conditions $x_0 = 0$ and $x_1 = 1$.
#
# In other words we have the following recurrence relationship:
#
# $x_{k+1} = \omega^2 x_{k} - \lambda^2 x_{k}^3 + 2 x_{k} - x_{k-1}$
#
#
# -------------
# The state vector can be written as:
#
# $$
# \mathbf{u}_k = \begin{bmatrix}
# x_{k} \\
# x_{k-1}
# \end{bmatrix}
# \quad
# \mathcal{M}_{k+1} =
#  \begin{bmatrix}
# 2 + \omega^2 - \lambda^2 x_k^2 & -1 \\
# 1 & 0
# \end{bmatrix}
# $$
#
# so that $\mathbf{u}_{k+1} = \mathcal{M}_{k+1}(\mathbf{u}_k)$.
#
#

# %%
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import erf

rng = np.random.default_rng(12345)

plt.rcParams["figure.figsize"] = [8, 3]


# %%
def plot_result(A, response_x_axis):
    """Plot the anharmonic oscillator, as well as marginal
    and joint distributions for the parameters."""

    responses = forward_model(A, response_x_axis)

    fig, axes = plt.subplots(1, 2 + len(A), figsize=(9, 2.25))

    axes[0].plot(response_x_axis, responses, color="black", alpha=0.1)
    for ax, param, label in zip(axes[1:], A, ["omega", "lambda"]):
        ax.hist(param, bins="fd")
        ax.set_xlabel(label)

    axes[-1].scatter(A[0, :], A[1, :], s=5)

    fig.tight_layout()
    plt.show()


# %% [markdown]
# ## Setup

# %%
def generate_observations(K):
    """Run the model with true parameters, then generate observations."""
    # Evaluate using true parameter values on a fine grid with K sample points
    x = simulate_anharmonic(omega=3.5e-2, lmbda=3e-4, K=K)

    # Generate observations every `nobs` steps
    nobs = 50
    # Generate observation points [0, 50, 100, 150, ...] when `nobs` = 50
    observation_x_axis = np.arange(K // nobs) * nobs

    # Generate noisy observations, with value at least 5
    observation_errors = np.maximum(5, 0.2 * np.abs(x[observation_x_axis]))
    observation_values = rng.normal(loc=x[observation_x_axis], scale=observation_errors)

    return observation_values, observation_errors, observation_x_axis


def simulate_anharmonic(omega, lmbda, K):
    """Evaluate the anharmonic oscillator."""
    x = np.zeros(K)
    x[0] = 0
    x[1] = 1

    # Looping from 2 because we have initial conditions at k=0 and k=1.
    for k in range(2, K):
        x[k] = x[k - 1] * (2 + omega**2 - lmbda**2 * x[k - 1] ** 2) - x[k - 2]

    return x


def forward_model(A, response_x_axis):
    """Evaluate on each column (ensemble realization)."""
    return np.vstack(
        [simulate_anharmonic(*params, K=len(response_x_axis)) for params in A.T]
    ).T


response_x_axis = range(2500)
observation_values, observation_errors, observation_x_axis = generate_observations(
    len(response_x_axis)
)


# %% [markdown]
# ## Plot observations

# %%
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(
    np.arange(2500),
    simulate_anharmonic(omega=3.5e-2, lmbda=3e-4, K=2500),
    label="Truth",
    color="black",
)

ax.scatter(observation_x_axis, observation_values, label="Observations")

fig.tight_layout()
plt.legend()
plt.show()

# %% [markdown]
# ## Create and plot prior
#
#
# This section shows a "trick" for working with non-normal priors.
#
# Let $g$ be the forward model, so that $y = g(x)$.
# Assume for instance that $x$ must be a positive parameter.
# In that case we can for instance use an exponential prior on $x$.
# But if we use samples from an exponential prior directly in an ensemble smoother,
# there is no guarantee that the posterior samples will be positive.
#
# The trick is to let sample $x \sim \mathcal{N}(0, 1)$,
# then define a function $f$ that maps from standard normal to the
# exponential distribution.
# This funciton can be constructed by first mapping from standard normal to
# the interval $[0, 1)$ using the CDF, then mapping to exponential using the
# inverse CDF of the exponential distribution.
#
# - We send $x \sim \mathcal{N}(0, 1)$ into the ensemble smoother as before
# - We use the composite function $g(f(x))$ as our forward model instead of $g(x)$
# - When we plot the prior and posterior, we show $f(x)$ instead of $x$ directly

# %%
from scipy import stats

# Create Uniform prior distributions
realizations = 100
PRIORS = [stats.uniform(loc=0.025, scale=0.02), stats.uniform(loc=0.0002, scale=0.0002)]


# %%
def prior_to_standard_normal(A):
    """Map prior to U(0, 1), then use inverse of CDF to map to N(0, 1)."""
    return np.vstack(
        [stats.norm().ppf(prior.cdf(A_col)) for (A_col, prior) in zip(A, PRIORS)]
    )


def standard_normal_to_prior(A):
    """Reverse mapping."""
    return np.vstack(
        [prior.ppf(stats.norm().cdf(A_col)) for (A_col, prior) in zip(A, PRIORS)]
    )


# verify that mappings are invertible
A_uniform = np.vstack([prior.rvs(realizations) for prior in PRIORS])
assert np.allclose(
    standard_normal_to_prior(prior_to_standard_normal(A_uniform)), A_uniform
)
assert np.allclose(
    prior_to_standard_normal(standard_normal_to_prior(A_uniform)), A_uniform
)

# %%
# Create a standard normal prior and plot it in transformed space
A = rng.standard_normal(size=(2, realizations))
plot_result(standard_normal_to_prior(A), response_x_axis)

# %% [markdown]
# ## Update step

# %%
import numpy as np
import iterative_ensemble_smoother as ies

plot_result(standard_normal_to_prior(A), response_x_axis)

responses_before = forward_model(standard_normal_to_prior(A), response_x_axis)
Y = responses_before[observation_x_axis]

smoother = ies.ES()
smoother.fit(Y, observation_errors, observation_values)
new_A = smoother.update(A)

plot_result(standard_normal_to_prior(new_A), response_x_axis)

# %% [markdown]
# ## Iterative smoother

# %%
import numpy as np
import iterative_ensemble_smoother as ies


def iterative_smoother(A):
    A_current = np.copy(A)
    iterations = 4
    smoother = ies.SIES(seed=42)

    for _ in range(iterations):
        plot_result(standard_normal_to_prior(A_current), response_x_axis)

        responses_before = forward_model(
            standard_normal_to_prior(A_current), response_x_axis
        )
        Y = responses_before[observation_x_axis]

        smoother.fit(Y, observation_errors, observation_values)
        A_current = smoother.update(A_current)

    plot_result(standard_normal_to_prior(A_current), response_x_axis)


iterative_smoother(A)

# %% [markdown]
# ## ES-MDA (Multiple Data Assimilation - Ensemble Smoother)

# %%
import numpy as np
import iterative_ensemble_smoother as ies


def es_mda(A):
    A_current = np.copy(A)
    weights = [8, 4, 2, 1]
    length = sum(1.0 / x for x in weights)

    smoother = ies.ES()
    for weight in weights:
        plot_result(standard_normal_to_prior(A_current), response_x_axis)

        responses_before = forward_model(
            standard_normal_to_prior(A_current), response_x_axis
        )
        Y = responses_before[observation_x_axis]

        observation_errors_scaled = observation_errors * np.sqrt(weight * length)
        smoother.fit(Y, observation_errors_scaled, observation_values)
        A_current = smoother.update(A_current)

    plot_result(standard_normal_to_prior(A_current), response_x_axis)


es_mda(A)
