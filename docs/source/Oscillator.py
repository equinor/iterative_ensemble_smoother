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
def _generate_observations(K):
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
observation_values, observation_errors, observation_x_axis = _generate_observations(
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
# ## Plot prior

# %%
realizations = 100
priors = [(2.5e-2, 4.5e-2), (2.0e-4, 4.0e-4)]
A = np.vstack([rng.uniform(low, high, size=realizations) for (low, high) in priors])

plot_result(A, response_x_axis)

# %% [markdown]
# ## Update step

# %%
import numpy as np
import iterative_ensemble_smoother as ies

plot_result(A, response_x_axis)

responses_before = forward_model(A, response_x_axis)
Y = responses_before[observation_x_axis]

smoother = ies.ES()
smoother.fit(Y, observation_errors, observation_values)
new_A = smoother.update(A)

plot_result(new_A, response_x_axis)

# %% [markdown]
# ## Iterative smoother

# %%
import numpy as np
import iterative_ensemble_smoother as ies


def iterative_smoother():
    A_current = np.copy(A)
    iterations = 4
    smoother = ies.SIES(seed=42)

    for _ in range(iterations):
        plot_result(A_current, response_x_axis)

        responses_before = forward_model(A_current, response_x_axis)
        Y = responses_before[observation_x_axis]

        smoother.fit(Y, observation_errors, observation_values)
        A_current = smoother.update(A_current)

    plot_result(A_current, response_x_axis)


iterative_smoother()

# %% [markdown]
# ## ES-MDA (Multiple Data Assimilation - Ensemble Smoother)

# %%
import numpy as np
import iterative_ensemble_smoother as ies


def es_mda():
    A_current = np.copy(A)
    weights = [8, 4, 2, 1]
    length = sum(1.0 / x for x in weights)

    smoother = ies.ES()
    for weight in weights:
        plot_result(A_current, response_x_axis)

        responses_before = forward_model(A_current, response_x_axis)
        Y = responses_before[observation_x_axis]

        observation_errors_scaled = observation_errors * np.sqrt(weight * length)
        smoother.fit(Y, observation_errors_scaled, observation_values)
        A_current = smoother.update(A_current)
    plot_result(A_current, response_x_axis)


es_mda()
