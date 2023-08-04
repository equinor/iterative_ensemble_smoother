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
# # Fitting a polynomial with Gaussian priors
#
# We fit a simple polynomial with Gaussian priors, which is an example of a Gauss-linear problem for which the results obtained using Subspace Iterative Ensemble Smoother (SIES) tend to those obtained using Ensemble Smoother (ES).
# This notebook illustrated this property.
# %%
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)
rng = np.random.default_rng(12345)

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams.update({"font.size": 10})
from ipywidgets import interact
import ipywidgets as widgets

from p_tqdm import p_map

import iterative_ensemble_smoother as ies

# %% [markdown]
# ## Define synthetic truth and use it to create noisy observations

# %%
ensemble_size = 200


def poly(a, b, c, x):
    return a * x**2 + b * x + c


# True patameter values
a_t = 0.5
b_t = 1.0
c_t = 3.0

noise_scale = 0.01
x_observations = [0, 2, 4, 6, 8]
observations = [
    (
        rng.normal(loc=1, scale=noise_scale) * poly(a_t, b_t, c_t, x),
        noise_scale * poly(a_t, b_t, c_t, x),
        x,
    )
    for x in x_observations
]

d = pd.DataFrame(observations, columns=["value", "sd", "x"])
d = d.set_index("x")
num_obs = d.shape[0]

fig, ax = plt.subplots()
x_plot = np.linspace(0, 10, 50)
ax.set_title("Truth and noisy observations")
ax.set_xlabel("Time step")
ax.set_ylabel("Response")
ax.plot(x_plot, poly(a_t, b_t, c_t, x_plot))
ax.plot(d.index.get_level_values("x"), d.value.values, "o")
ax.grid()

# %% [markdown]
# ## Assume diagonal observation error covariance matrix and define perturbed observations

# %%
R = np.diag(d.sd.values**2)

E = rng.multivariate_normal(mean=np.zeros(len(R)), cov=R, size=ensemble_size).T
assert E.shape == (num_obs, ensemble_size)

D = d.value.values.reshape(-1, 1) + E

# %% [markdown]
# ## Define Gaussian priors

# %%
coeff_a = rng.normal(0, 1, size=ensemble_size)
coeff_b = rng.normal(0, 1, size=ensemble_size)
coeff_c = rng.normal(0, 1, size=ensemble_size)

X = np.vstack([coeff_a, coeff_b, coeff_c])

# %% [markdown]
# ## Run forward model in parallel

# %%
fwd_runs = p_map(
    poly,
    coeff_a,
    coeff_b,
    coeff_c,
    [np.arange(max(x_observations) + 1)] * ensemble_size,
    desc=f"Running forward model.",
)

# %% [markdown]
# ## Pick responses where we have observations

# %%
Y = np.array(
    [fwd_run[d.index.get_level_values("x").to_list()] for fwd_run in fwd_runs]
).T

assert Y.shape == (
    num_obs,
    ensemble_size,
), "Measured responses must be a matrix with dimensions (number of observations x number of realisations)"

# %% [markdown]
# ## Condition on observations to calculate posterior using both `ES` and `SIES`

# %%
X_ES_ert = X.copy()
Y_ES_ert = Y.copy()
smoother_es = ies.ES(seed=42)
smoother_es.fit(Y_ES_ert, d.sd.values, d.value.values)
X_ES_ert = smoother_es.update(X_ES_ert)

X_IES_ert = X.copy()
Y_IES_ert = Y.copy()
smoother_ies = ies.SIES(ensemble_size=ensemble_size, seed=42)
n_ies_iter = 7
for i in range(n_ies_iter):
    smoother_ies.fit(Y_IES_ert, d.sd.values, d.value.values)
    X_IES_ert = smoother_ies.update(X_IES_ert)

    _coeff_a, _coeff_b, _coeff_c = X_IES_ert

    _fwd_runs = p_map(
        poly,
        _coeff_a,
        _coeff_b,
        _coeff_c,
        [np.arange(max(x_observations) + 1)] * ensemble_size,
        desc=f"SIES ert iteration {i}",
    )

    Y_IES_ert = np.array(
        [fwd_run[d.index.get_level_values("x").to_list()] for fwd_run in _fwd_runs]
    ).T


# %% [markdown]
# ## Plots to compare results


# %%
def plot_posterior(ax, posterior, method):
    for i, param in enumerate(["a", "b", "c"]):
        ax[i].set_title(param)
        ax[i].hist(posterior[i, :], label=f"{method} posterior", alpha=0.5, bins="fd")
        ax[i].legend()

    fig.tight_layout()
    return ax


fig, ax = plt.subplots(nrows=3)
fig.set_size_inches(8, 8)

for i in range(3):
    ax[i].hist(X[i, :], label="prior", bins="fd")

ax[0].axvline(a_t, color="k", linestyle="--", label="truth")
ax[1].axvline(b_t, color="k", linestyle="--", label="truth")
ax[2].axvline(c_t, color="k", linestyle="--", label="truth")

plot_posterior(ax, X_ES_ert, method="ES ert")
_ = plot_posterior(ax, X_IES_ert, method=f"SIES ert ({n_ies_iter})")
