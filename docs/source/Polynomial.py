# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Example: Polynomial function
#
#
# The following is an example of history matching with the
# iterative_ensemble_smoother library.

# %% pycharm={"name": "#%%\n"}
# Simple plotting of forward-model with a single response and parameters
from matplotlib import pyplot as plt


def plot_result(
    A, response_x_axis, trans_func=lambda x: x, priors=None, show_params=False
):
    if priors is None:
        priors = []
    responses = forward_model(A, priors, response_x_axis)
    plt.rcParams["figure.figsize"] = [15, 4]
    figures = 1 + len(A) if show_params else 1
    fig, axs = plt.subplots(1, figures)

    if show_params:
        axs[0].plot(response_x_axis, responses)
        for i, param in enumerate(A):
            A_trans = np.array([trans_func(v, *priors[i]) for v in param])
            axs[i + 1].hist(A_trans, bins=10)
    else:
        axs.plot(response_x_axis, responses)
    plt.show()


# %% [markdown]
# ## Setup
#
# The setup contains a forward model (a second degree polynomial in this case),
# where the coefficents of the polynomial is the model parameters.
#
# There are 5 time steps t=0,1,2,3,4 and 3 observations at t=0,2,4.
#
# Before history matching, these observations are predicted by the forward
# model with the priors.
#
# The priors at t=0,2,4 are assumed uniform in [0,1], [0,2] and [0,5]
# respectively.
#
# As input to the history matching we have the observed values in
# `observation_values`. These would normally be historic measurements.
#
# The observed values have the measurement errors in `observation_errors`.
#
# A is populated with initial guesses for the parameters of the ensemble.

# %% pycharm={"name": "#%%\n"}
# Polynomial forward model with observations
import numpy as np
from scipy.special import erf
from math import sqrt


def uniform(x, min_x, max_x):
    y = 0.5 * (1 + erf(x / sqrt(2.0)))
    return y * (max_x - min_x) + min_x


def forward_model(A, priors, response_x_axis):
    responses = []
    for params in A.T:
        l = [uniform(x, *priors[i]) for i, x in enumerate(params)]
        response = [l[0] * x**2 + l[1] * x + l[2] for x in response_x_axis]
        responses.append(response)
    return np.array(responses).T


observation_values = np.array(
    [2.8532509308, 7.20311703432, 21.3864899107, 31.5145559347, 53.5676660405]
)

observation_errors = np.array([0.5 * (x + 1) for x, _ in enumerate(observation_values)])
observation_x_axis = [0, 2, 4, 6, 8]
response_x_axis = range(10)
realizations = 200
priors = [(0, 1), (0, 2), (0, 5)]
A = np.asfortranarray(np.random.normal(0, 1, size=(3, realizations)))
plot_result(A, response_x_axis, uniform, priors, True)

# %% [markdown]
# ## Update step

# %% pycharm={"name": "#%%\n"}
import numpy as np
import iterative_ensemble_smoother as ies

# Plot the initial guesses before the update step
plot_result(A, response_x_axis, uniform, priors, True)

responses_before = forward_model(A, priors, response_x_axis)
S = responses_before[observation_x_axis]
noise = np.random.rand(*S.shape)

# Update step
new_A = ies.ensemble_smoother_update_step(
    S, A, observation_errors, observation_values, noise
)

# Plot after update step
plot_result(new_A, response_x_axis, uniform, priors, True)

# %% [markdown]
# ## Iterative smoother

# %%
import numpy as np
from matplotlib import pyplot as plt
import iterative_ensemble_smoother as ies


def iterative_smoother():
    A_current = np.copy(A)
    iterations = 4
    obs_mask = [True] * len(observation_values)
    ens_mask = [True] * realizations
    smoother = ies.IterativeEnsembleSmoother(realizations)

    for i in range(iterations):
        plot_result(A_current, response_x_axis, uniform, priors, True)
        responses_before = forward_model(A_current, priors, response_x_axis)
        S = responses_before[observation_x_axis]
        noise = np.random.rand(*S.shape)
        A_current = smoother.update_step(
            S,
            A,
            observation_errors,
            observation_values,
            noise=noise,
            ensemble_mask=ens_mask,
            observation_mask=obs_mask,
        )
    plot_result(A_current, response_x_axis, uniform, priors, True)


iterative_smoother()

# %% pycharm={"name": "#%%\n"}
