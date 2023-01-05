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

# %% [markdown]
# # Example: Estimating parameters of an anharmonic oscillator
#
# This notebook covers quite a lot of ground.
# The overarching goal is to illustrate how ensemble smoothers can be used to estimate the parameters of a dynamical model, which in this case is the anharmonic oscillator.
# Anharmonic oscillators can be modelled by the following non-linear partial differential equation:
#
# $\frac{d^2x}{dt^2} - \Omega^2 x + \Lambda^2 x^3 = 0$
#
# Because the oscillator is non-linear, the signal will contain less information the further we are from the starting point.
# We will exploit this to show how adaptive localization can help us achieve a better fit to data.
#
# Outlier detection is included to handle failing realizations and failing observations.
# The prior is modeled by a uniform distribution to showcase how to use ensemble smoothers with non-normal distributions.
#
# TODO:
# - Add regression lines to parameter vs. response plots

# %% tags=["remove-input"]
## Import modules and set-up plotting

import numpy as np

from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [15, 4]
response_color = "#1b9e77"
obs_color = "#d95f02"
param_color = "#7570b3"

import pandas as pd
from scipy.special import erf

import iterative_ensemble_smoother as ies

# %% tags=["remove-input"]
## Choose random number seed

# It's worth playing around with the seed.

rng = np.random.default_rng(10)


# %% tags=["remove-input"]
## Numerical implementation of the anharmonic oscillator

# Based on section 6.3.4 of the book Data Assimilation: Methods, Algorithms, and Applications by Mark Asch, Marc Bocquet and Maëlle Nodet for details.


def anharmonic_oscillator(omega, lmbda, time_steps):
    """Numerical implementation of the anharmonic oscillator"""
    x = np.zeros(time_steps)
    x[0] = 0
    x[1] = 1

    # Looping from 2 because we have initial conditions at k=0 and k=1.
    for k in range(2, time_steps - 1):
        M = np.array([[2 + omega**2 - lmbda**2 * x[k] ** 2, -1], [1, 0]])
        u = np.array([x[k], x[k - 1]])
        u = M @ u
        x[k + 1] = u[0]
        x[k] = u[1]

    return x


# %% tags=["remove-input"]
## Set parameters and generate priors

# True parameter values are taken from asch2016.

ensemble_size = 100
num_params = 2
time_steps = 2000

# Parameter values that define the synthetic truth
omega = 3.5e-2
lmbda = 3e-4
param_ensemble = rng.standard_normal(size=(num_params, ensemble_size))
# IES needs its own prior ensemble because it updates parameters in-place
param_ensemble_ies = np.copy(param_ensemble)


# %% tags=["remove-input"]
## Define forward model

# The forward model runs the oscillator and does outlier detection.


def uniform(x, min_x, max_x):
    """Transform standard normal `x` to uniform"""
    y = 0.5 * (1 + erf(x / np.sqrt(2.0)))
    return y * (max_x - min_x) + min_x


def forward_model(param_ensemble, uniform_prior_limits, time_steps):
    """Runs dynamical model and other processing steps.

    Parameters
    ----------
    param_ensemble:
        Matrix with parameter priors of dimension
        (number of parameters by ensemble size)
        drawn from a standard normal distribution
    uniform_prior_limits:
        Limits used to convert standard normal
        to uniform
    """
    responses = []
    for [omega, lmbda] in param_ensemble.T:
        r = anharmonic_oscillator(
            omega=uniform(omega, *uniform_prior_limits[0]),
            lmbda=uniform(lmbda, *uniform_prior_limits[1]),
            time_steps=time_steps,
        )
        responses.append(r)

    responses = np.array(responses).T

    ens_mask = (
        np.abs(responses).max(axis=0) > 3 * np.abs(response_truth).max(axis=0).mean()
    )

    obs_mask = responses.var(axis=1) > 1e-6

    return (responses, ~ens_mask, obs_mask)


# %% tags=["remove-input"]
def plot_result(
    param_ensemble,
    responses,
    response_truth,
    time_steps,
    trans_func=lambda x: x,
    uniform_prior_limits=[],
):
    fig, axs = plt.subplots(1, 1 + len(param_ensemble))
    axs[0].set_xlabel("Time step")
    axs[0].set_ylabel("Response")
    axs[0].grid(linestyle="--")
    axs[0].plot(responses, color=response_color, alpha=0.05)
    axs[0].plot(response_truth, color=response_color, label="Truth")
    axs[0].legend()
    for i, param in enumerate(param_ensemble):
        A_trans = np.array([trans_func(v, *uniform_prior_limits[i]) for v in param])
        axs[i + 1].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        axs[i + 1].hist(A_trans, bins=10, color=param_color, alpha=0.5)
    axs[1].set_xlabel(f"Parameter $\Omega$")
    axs[2].set_xlabel(f"Parameter $\Lambda$")
    axs[1].axvline(
        omega, linestyle="--", linewidth=3, color=param_color, label="True $\Omega$"
    )
    axs[2].axvline(
        lmbda, linestyle="--", linewidth=3, color=param_color, label="True $\Lambda$"
    )
    axs[1].legend()
    axs[2].legend()
    return (fig, axs)


# %% [markdown]
# ## Run oscillator using true parameters and using priors
#
# This generates the synthetic truth which will be used to compare methods.
# Parameters are drawn from uniform distributions but might not look like it due to a small ensemble size.

# %% tags=["remove-input"]
response_truth = anharmonic_oscillator(omega, lmbda, time_steps)

# Try increasing the range from -something to +something to get more non-linear behaviour
# uniform_prior_limits = [(2.5e-2, 4.5e-2), (2.0e-4, 4.0e-4)]
# uniform_prior_limits = [(-2.5e-2, 4.5e-2), (-2.0e-4, 4.0e-4)]
uniform_prior_limits = [(0, 4.5e-2), (0, 4.0e-4)]
responses_prior, ens_mask, obs_mask = forward_model(
    param_ensemble, uniform_prior_limits, time_steps
)
responses_prior = responses_prior[:, ens_mask]

# Updating ensemble_size since the forward model may remove some of them
ensemble_size = ens_mask.sum()

# %% tags=["remove-input"]
fig, axs = plot_result(
    param_ensemble[:, ens_mask],
    responses_prior,
    response_truth,
    time_steps,
    uniform,
    uniform_prior_limits,
)
_ = fig.suptitle("Prior responses and parameter distributions")

# %% [markdown]
# ## Place sensors and generate observations
#
# Because of non-linearity, there's more information in signals closer to the zeroth time-step.

# %% tags=["remove-input"]
obs_points = np.array([10, 50, 100, 150, 1900])
mask_at_points = obs_mask[obs_points]

obs_errors = np.maximum(
    2 * np.ones(len(obs_points)), np.abs(0.15 * response_truth[obs_points])
)
df_observations = pd.DataFrame(
    {
        "Value": response_truth[obs_points] + rng.normal(scale=obs_errors),
        "Error": obs_errors,
    }
)

obs_values = df_observations["Value"].values
obs_errors = df_observations["Error"].values


# %% tags=["remove-input"]
def plot_responses_and_observations(
    response, response_truth, obs_values, obs_errors, obs_points
):
    fig, ax = plt.subplots()
    ax.set_title("True responses, realizations and observations with errors")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Response")
    ax.grid(linestyle="--")
    ax.plot(response, alpha=0.05, color=response_color)
    for obs_point in obs_points:
        ax.axvline(obs_point, linestyle="dashed", alpha=0.5, color=obs_color)
    ax.plot(response_truth, color=response_color, label="Truth")
    ax.scatter(obs_points, obs_values, marker="o", color=obs_color)
    ax.errorbar(
        obs_points,
        obs_values,
        yerr=obs_errors,
        linestyle="",
        capsize=8,
        color=obs_color,
    )
    _ = ax.legend()
    return (fig, ax)


_ = plot_responses_and_observations(
    responses_prior, response_truth, obs_values, obs_errors, obs_points
)


# %% [markdown]
# ## Scatter plots of parameters vs. responses
#
# As previously mentioned, there's more information in signals close to time `k=0`, which is illustrated here using scatter plots.
# The plots illustrate what is meant by non-linear, in that the relationships between responses and parameters are not straight lines.
# Note that due to small ensemble sizes, we might by chance produce relations that look significant, but are in fact not.
# These types of relations are known as `spurious correlations`.

# %% tags=["remove-input"]
def plot_params_vs_responses(
    param_ensemble,
    responses,
    obs_points,
    trans_func=lambda x: x,
    uniform_prior_limits=[],
):
    for i, obs_point in enumerate(obs_points):
        fig, ax = plt.subplots(nrows=1, ncols=2)
        fig.suptitle(f"Time-step k = {obs_point}")
        ax[0].grid(linestyle="--")
        ax[0].set_title(f"Parameters $\Omega$ vs. Responses")
        ax[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax[0].set_xlabel("Parameter $\Omega$")
        ax[0].set_ylabel(f"Response")
        ax[0].scatter(
            trans_func(param_ensemble[0, :], *uniform_prior_limits[0]),
            responses[obs_points, :][i, :],
            color=response_color,
        )
        ax[1].grid(linestyle="--")
        ax[1].set_title(f"Parameters $\Lambda$ vs. Responses")
        ax[1].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax[1].set_xlabel("Parameter $\Lambda$")
        ax[1].set_ylabel(f"Response")
        ax[1].scatter(
            trans_func(param_ensemble[1, :], *uniform_prior_limits[1]),
            responses[obs_points, :][i, :],
            color=response_color,
        )

        fig.tight_layout()


plot_params_vs_responses(
    param_ensemble[:, ens_mask],
    responses_prior,
    obs_points,
    uniform,
    uniform_prior_limits,
)

# %% [markdown]
# ## ES vs. ES with Adaptive Localization - RMSE Experiment
#
# We first place four sensors close to the zeroth time-step to get some good signal.
# We then run 1000 experiments where each experiment consists of adding another sensor somewhere after the 500th time-step.
# This sensor will have varying degrees of signal.
# Using results from 1000 experiments, we compare RMSE's of ES with and without localization.

# %% tags=["remove-input"]
correlation_threshold = 3 / np.sqrt(ensemble_size)


def ensemble_smoother_with_localization(
    param_ensemble, response_ensemble, obs_values, obs_errors
):
    param_loc = np.zeros_like(param_ensemble)
    num_parameters = param_ensemble.shape[0]
    ensemble_size = param_ensemble.shape[1]
    for i in range(num_parameters):
        responses_to_keep = (
            np.abs(np.corrcoef(param_ensemble[i, :], response_ensemble)[0, 1:])
            > correlation_threshold
        )
        _param_loc = ies.ensemble_smoother_update_step(
            response_ensemble[responses_to_keep, :],
            param_ensemble[i, :].reshape(1, ensemble_size),
            obs_errors[responses_to_keep],
            obs_values[responses_to_keep],
            noise=rng.standard_normal(
                size=(len(obs_errors[responses_to_keep]), ensemble_size)
            ),
        )
        param_loc[i, :] = _param_loc
    return param_loc


def rmse(param_estimates, param_truth, ensemble_size):
    return np.sqrt(np.sum((param_estimates - param_truth) ** 2) / ensemble_size)


num_experiments = 1000

df_results = pd.DataFrame(
    {
        "Method": pd.Series(dtype=str),
        "Experiment": pd.Series(dtype=int),
        "Omega_RMSE": pd.Series(dtype=float),
        "Lambda_RMSE": pd.Series(dtype=float),
    }
)

for experiment in range(num_experiments):
    # We put some sensors close to time step zero to get some good signal.
    obs_points_good = np.array([10, 50, 100, 150])
    # We also want some badly placed sensors in that they contain a lot of noise.
    obs_points_bad = rng.choice(np.arange(start=500, stop=time_steps), size=2)
    obs_points = np.append(obs_points_good, obs_points_bad)

    obs_errors = np.maximum(
        2 * np.ones(len(obs_points)), np.abs(0.15 * response_truth[obs_points])
    )
    obs_values = response_truth[obs_points] + rng.normal(scale=obs_errors)

    mask_at_points = obs_mask[obs_points]

    _responses = responses_prior[obs_points[mask_at_points], :]
    A_ES = ies.ensemble_smoother_update_step(
        _responses,
        param_ensemble[:, ens_mask],
        obs_errors[mask_at_points],
        obs_values[mask_at_points],
        noise=rng.standard_normal(size=(_responses.shape[0], _responses.shape[1])),
    )

    A_ES_loc = ensemble_smoother_with_localization(
        param_ensemble[:, ens_mask],
        _responses,
        obs_values[mask_at_points],
        obs_errors[mask_at_points],
    )

    df_results = pd.concat(
        [
            df_results,
            pd.DataFrame(
                {
                    "Method": ["ES", "ES_loc"],
                    "Experiment": [experiment, experiment],
                    "Omega_RMSE": [
                        rmse(
                            uniform(A_ES[0, :], *uniform_prior_limits[0]),
                            omega,
                            ensemble_size,
                        ),
                        rmse(
                            uniform(A_ES_loc[0, :], *uniform_prior_limits[0]),
                            omega,
                            ensemble_size,
                        ),
                    ],
                    "Lambda_RMSE": [
                        rmse(
                            uniform(A_ES[1, :], *uniform_prior_limits[1]),
                            omega,
                            ensemble_size,
                        ),
                        rmse(
                            uniform(A_ES_loc[1, :], *uniform_prior_limits[1]),
                            omega,
                            ensemble_size,
                        ),
                    ],
                }
            ),
        ]
    )

fig, ax = plt.subplots()
ax.set_title("RMSE of $\Omega$ with and without localization")
ax.set_xlabel("RMSE")
es = df_results.query("Method == 'ES'")
ax.hist(es["Omega_RMSE"], alpha=0.4, color=param_color, label="ES")
es_loc = df_results.query("Method == 'ES_loc'")
ax.hist(es_loc["Omega_RMSE"], alpha=0.9, color=param_color, label="ES_loc")
ax.legend()

fig, ax = plt.subplots()
ax.set_title("RMSE of $\Lambda$ with and without localization")
ax.set_xlabel("RMSE")
ax.hist(es["Lambda_RMSE"], alpha=0.4, color=param_color, label="ES")
ax.hist(es_loc["Lambda_RMSE"], alpha=0.9, color=param_color, label="ES_loc")
_ = ax.legend()


# %% [markdown]
# ## Ensemble Smoother with and without Adaptive Localization
#
# Using adaptive localization causes less overfitting.

# %% tags=["remove-input"]
def es_loc():
    fig, axs = plot_result(
        param_ensemble[:, ens_mask],
        responses_prior,
        response_truth,
        time_steps,
        uniform,
        uniform_prior_limits,
    )
    fig.suptitle("Prior responses and parameter distributions")

    param_ES_loc = ensemble_smoother_with_localization(
        param_ensemble[:, ens_mask],
        responses_prior[obs_points[mask_at_points], :],
        obs_values[mask_at_points],
        obs_errors[mask_at_points],
    )
    response_ES_loc, ens_mask_loc, obs_mask = forward_model(
        param_ES_loc, uniform_prior_limits, time_steps
    )

    fig, axs = plot_result(
        param_ES_loc[:, ens_mask_loc],
        response_ES_loc[:, ens_mask_loc],
        response_truth,
        time_steps,
        uniform,
        uniform_prior_limits,
    )
    _ = fig.suptitle("Posterior responses and parameter distributions")


es_loc()


# %% tags=["remove-input"]
def es():
    fig, axs = plot_result(
        param_ensemble[:, ens_mask],
        responses_prior,
        response_truth,
        time_steps,
        uniform,
        uniform_prior_limits,
    )
    fig.suptitle("Prior responses and parameter distributions")

    param_ES = ies.ensemble_smoother_update_step(
        responses_prior[obs_points[mask_at_points], :],
        param_ensemble[:, ens_mask],
        obs_errors[mask_at_points],
        obs_values[mask_at_points],
        noise=rng.standard_normal(
            size=(len(obs_points[mask_at_points]), ensemble_size)
        ),
    )

    response_ES, ens_mask_es, obs_mask = forward_model(
        param_ES, uniform_prior_limits, time_steps
    )

    fig, axs = plot_result(
        param_ES[:, ens_mask_es],
        response_ES[:, ens_mask_es],
        response_truth,
        time_steps,
        uniform,
        uniform_prior_limits,
    )
    _ = fig.suptitle("Posterior responses and parameter distributions")


es()


# %% [markdown]
# ## ES-MDA (Multiple Data Assimilation - Ensemble Smoother) with and without localization
#
# Note that the prior response plot may look strange because some realizations "blow-up" in that they produce amplitudes that are much larger than average.
# This is OK and should be handled by the outlier detection.

# %% tags=["remove-input"]
def es_mda(param_ensemble, weights, correlation_threshold, localization=True):
    param_ensemble = np.copy(param_ensemble)
    length = sum(1.0 / x for x in weights)

    responses, ens_mask, obs_mask = forward_model(
        param_ensemble, uniform_prior_limits, time_steps
    )
    param_ensemble = param_ensemble[:, ens_mask]

    fig, axs = plot_result(
        param_ensemble,
        responses,
        response_truth,
        time_steps,
        uniform,
        uniform_prior_limits,
    )
    fig.suptitle(f"Prior responses and parameter distributions")

    for i, weight in enumerate(weights):
        responses, ens_mask, obs_mask = forward_model(
            param_ensemble, uniform_prior_limits, time_steps
        )
        ensemble_size = ens_mask.sum()

        responses = responses[:, ens_mask]
        param_ensemble = param_ensemble[:, ens_mask]

        observation_errors_scaled = obs_errors * np.sqrt(weight * length)

        mask_at_points = obs_mask[obs_points]

        param_ensemble_loc = []
        for i in range(num_params):
            param_chunk = param_ensemble[i, :].reshape(1, ensemble_size)

            responses_to_keep = (
                np.abs(
                    np.corrcoef(param_chunk, responses[obs_points[mask_at_points]])[
                        0, 1:
                    ]
                )
                > correlation_threshold
            )
            responses_loc = responses[obs_points[responses_to_keep], :]

            noise = rng.standard_normal(
                size=(responses_loc.shape[0], responses_loc.shape[1])
            )
            _param_loc = ies.ensemble_smoother_update_step(
                responses_loc,
                param_chunk,
                observation_errors_scaled[responses_to_keep],
                obs_values[responses_to_keep],
                noise,
            )
            param_ensemble_loc.append(_param_loc)

        param_ensemble = np.vstack(param_ensemble_loc)

        fig, axs = plot_result(
            param_ensemble,
            responses,
            response_truth,
            time_steps,
            uniform,
            uniform_prior_limits,
        )
        fig.suptitle(f"ES-MDA iteration {i} with weights {weight}")


weights = [8, 4, 2, 1]
es_mda(param_ensemble, weights, correlation_threshold)


# %% [markdown]
# ## Iterative ensemble smoother (IES) without localization

# %% tags=["remove-input"]
def run_ies(param_ensemble_ies):
    responses_ies, ens_mask, obs_mask = forward_model(
        param_ensemble_ies, uniform_prior_limits, time_steps
    )

    ensemble_size = responses_ies.shape[1]

    fig, axs = plot_result(
        param_ensemble_ies[:, ens_mask],
        responses_ies[:, ens_mask],
        response_truth,
        time_steps,
        uniform,
        uniform_prior_limits,
    )
    fig.suptitle(f"Prior responses and parameter distributions")

    smoother = ies.IterativeEnsembleSmoother(ensemble_size)

    for i in range(4):
        mask_at_points = obs_mask[obs_points]
        _responses = responses_ies[obs_points[mask_at_points], :]
        noise = rng.standard_normal(size=(_responses.shape[0], _responses.shape[1]))
        param_ensemble_ies = smoother.update_step(
            _responses,
            param_ensemble_ies,
            obs_errors[mask_at_points],
            obs_values[mask_at_points],
            noise,
        )

        responses_ies, ens_mask, obs_mask = forward_model(
            param_ensemble_ies, uniform_prior_limits, time_steps
        )

        fig, axs = plot_result(
            param_ensemble_ies[:, ens_mask],
            responses_ies[:, ens_mask],
            response_truth,
            time_steps,
            uniform,
            uniform_prior_limits,
        )
        fig.suptitle(f"IES iteration {i}")


run_ies(param_ensemble_ies)
