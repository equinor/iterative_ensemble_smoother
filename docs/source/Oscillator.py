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
# # Estimating parameters of an anharmonic oscillator
#
# The anharnomic oscillator can be modelled by a non-linear partial differential
# equation as described in section 6.4.3 of the book Fundamentals of Algorithms
# and Data Assimilation by Mark Asch, Marc Bocquet and Maëlle Nodet.

# %%
# Simple plotting of forward-model with a single response and parameters
from matplotlib import pyplot as plt


def plot_result(A, response_x_axis, trans_func=lambda x: x, priors=[], title=None):
    responses = forward_model(A, priors, response_x_axis)
    plt.rcParams["figure.figsize"] = [12, 4]
    fig, axs = plt.subplots(1, 1 + len(A))
    if title:
        fig.suptitle(title)

    axs[0].plot(response_x_axis, responses)
    for i, param in enumerate(A):
        A_trans = np.array([trans_func(v, *priors[i]) for v in param])
        axs[i + 1].hist(A_trans, bins="fd")

    fig.tight_layout()
    plt.show()


# %% [markdown]
# ## Setup

# %%
# Oscilator example
import numpy as np
from math import sqrt
from scipy.special import erf


def _generate_observations(K):
    x = _evaluate(omega=3.5e-2, lmbda=3e-4, K=K)
    rng = np.random.default_rng(12345)
    nobs = 50
    obs_points = np.linspace(0, K, nobs, endpoint=False, dtype=int)

    obs_with_std = np.zeros(shape=(len(obs_points), 2))

    for obs_idx, obs_point in enumerate(obs_points):
        # Set observation error's standard deviation to some
        # percentage of the amplitude of x with a minimum of, e.g., 1.
        obs_std = max(1, 0.02 * abs(x[obs_point]))
        obs_with_std[obs_idx, 0] = x[obs_point] + rng.normal(loc=0.0, scale=obs_std)
        obs_with_std[obs_idx, 1] = obs_std
    return obs_with_std, obs_points


def _evaluate(omega, lmbda, K):
    x = np.zeros(K)
    x[0] = 0
    x[1] = 1

    # Looping from 2 because we have initial conditions at k=0 and k=1.
    for k in range(2, K - 1):
        M = np.array([[2 + omega**2 - lmbda**2 * x[k] ** 2, -1], [1, 0]])
        u = np.array([x[k], x[k - 1]])
        u = M @ u
        x[k + 1] = u[0]
        x[k] = u[1]

    return x


def uniform(x, min_x, max_x):
    y = 0.5 * (1 + erf(x / sqrt(2.0)))
    return y * (max_x - min_x) + min_x


def forward_model(A, prior, response_x_axis):
    responses = []
    for [omega, lmbda] in A.T:
        r = _evaluate(
            omega=uniform(omega, *prior[0]),
            lmbda=uniform(lmbda, *prior[1]),
            K=len(response_x_axis),
        )
        responses.append(r)
    return np.array(responses).T


response_x_axis = range(2500)
realizations = 100

observations, observation_x_axis = _generate_observations(len(response_x_axis))
observation_values = observations[:, 0]
observation_errors = observations[:, 1]

A = np.random.normal(0, 1, size=(2, realizations))

priors = [(2.5e-2, 4.5e-2), (2.0e-4, 4.0e-4)]
plot_result(A, response_x_axis, uniform, priors)


# %% [markdown]
# ## Update step

# %%
import numpy as np
import iterative_ensemble_smoother as ies

plot_result(A, response_x_axis, uniform, priors)

responses_before = forward_model(A, priors, response_x_axis)
Y = responses_before[observation_x_axis]

smoother = ies.ES()
smoother.fit(Y, observation_errors, observation_values)
new_A = smoother.update(A)

plot_result(new_A, response_x_axis, uniform, priors)

# %% [markdown]
# ## Iterative smoother

# %%
import numpy as np
import iterative_ensemble_smoother as ies


def iterative_smoother():
    A_current = np.copy(A)
    iterations = 4
    smoother = ies.SIES(seed=42)

    plot_result(A_current, response_x_axis, uniform, priors, title="Prior")

    for iteration in range(iterations):
        # Evaluate model
        responses_before = forward_model(A_current, priors, response_x_axis)
        Y = responses_before[observation_x_axis]

        smoother.fit(Y, observation_errors, observation_values)
        A_current = smoother.update(A_current)

        plot_result(
            A_current,
            response_x_axis,
            uniform,
            priors,
            title=f"IES iteration {iteration+1}",
        )


iterative_smoother()

# %% [markdown]
# ## ES-MDA (Ensemble Smoother - Multiple Data Assimilation)

# %%
smoother = ies.ESMDA(
    # Here C_D is a covariance matrix. If a 1D array is passed,
    # it is interpreted as the diagonal of the covariance matrix,
    # and NOT as a vector of standard deviations
    C_D=observation_errors**2,
    observations=observation_values,
    # The inflation factors used in ESMDA
    # They are scaled so that sum_i alpha_i^-1 = 1
    alpha=np.array([8, 4, 2, 1]),
    seed=42,
)


A_current = np.copy(A)

plot_result(A_current, response_x_axis, uniform, priors, title="Prior")

for iteration in range(smoother.num_assimilations()):
    # Evaluate the model
    responses_before = forward_model(A_current, priors, response_x_axis)
    Y = responses_before[observation_x_axis]

    # Assimilate data
    A_current = smoother.assimilate(A_current, Y)

    # Plot
    plot_result(
        A_current,
        response_x_axis,
        uniform,
        priors,
        title=f"ESMDA iteration {iteration+1}",
    )
