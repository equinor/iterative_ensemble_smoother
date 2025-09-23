# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Test Distance Based Localization on real cases

# %%
import numpy as np
import polars as pl
from ert.storage import open_storage

from iterative_ensemble_smoother.esmda import ESMDA
from iterative_ensemble_smoother.experimental import DistanceESMDA

SEED = 42

# %% [markdown]
# ## List all available experiments in storage

# %%
storage_path = "01_drogon_ahm/storage/"
storage_path = "/Users/FCUR/git/ert/test-data/ert/heat_equation/storage"
with open_storage(storage_path) as storage:
    [print(f"Experiment names: {x.name}") for x in storage.experiments]

# %% [markdown]
# ## Pick which experiment to analyse

# %%
experiment_name = "ensemble_smoother"

# %% [markdown]
# ## Load observations and responses from storage.
#
# Remove responses with zero standard deviation

# %%
with open_storage(storage_path, "r") as storage:
    ensemble = storage.get_experiment_by_name(experiment_name).get_ensemble_by_name(
        "default_0"
    )
    ensemble_size = ensemble.ensemble_size
    selected_obs = ensemble.experiment.observation_keys
    iens_active_index = np.array(ensemble.get_realization_list_with_responses())
    observations_and_responses = ensemble.get_observations_and_responses(
        selected_obs, iens_active_index
    )

response_cols = [str(i) for i in range(1, ensemble.ensemble_size)]
df_filtered = observations_and_responses.filter(
    pl.concat_list([pl.col(col) for col in response_cols])
    .list.eval(pl.element().std())
    .list.first()
    > 0
)

# %%
df_filtered

# %% [markdown]
# # Load parameters from storage

# %%
with open_storage(storage_path, "r") as storage:
    experiment = storage.get_experiment_by_name(experiment_name)
    ensemble = experiment.get_ensemble_by_name("default_0")
    groups = list(experiment.parameter_configuration.keys())

    # for group in groups:
    #    print(ensemble.load_parameters_numpy(group, [0, 1]))

    realizations = ensemble.get_realization_list_with_responses()
    cond = ensemble.load_parameters_numpy("COND", realizations)

# %% [markdown]
# ## Prepare response matrix

# %%
Y = df_filtered.select(
    pl.all().exclude(
        ["response_key", "index", "observation_key", "observations", "std"]
    )
).to_numpy()

# %% [markdown]
# ## Ensemble Smoother without Localization

# %%
X = cond

assert Y.shape[1] == X.shape[1]

observations = df_filtered["observations"].to_numpy()
C_D = np.diag(df_filtered["std"])

smoother_ESMDA = ESMDA(covariance=C_D, observations=observations, alpha=1, seed=SEED)

D = smoother_ESMDA.perturb_observations(ensemble_size=ensemble_size, alpha=1)

X_posterior_ESMDA = smoother_ESMDA.assimilate(X=X, Y=Y)

# %% [markdown]
# ## Ensemble Smoother with Distance Based Localization

# %%
smoother = DistanceESMDA(covariance=C_D, observations=observations, alpha=1, seed=SEED)

rho = np.ones(shape=(X.shape[0], Y.shape[0]))

X_posterior = smoother.assimilate(X=X, Y=Y, rho=rho)

# %%
