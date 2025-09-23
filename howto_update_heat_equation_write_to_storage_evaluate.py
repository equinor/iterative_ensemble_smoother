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
# # HOWTO - Update parameters of heat-equation, create new ensemble in storage and write the updated parameters to it
#
# **Steps:**
#
# - Run ensemble experiment using `heat_equation.ert` using `ert`. Close `ert`.
# - Run the code in the notebook that updates parameters using `ESMDA`, creates a new ensemble in the experiment created by `ert` in the first step, and writes updated parameters to it.
# - Re-open `ert`. Run `Evaluation ensemble` in `ert` using the new ensemble called `heat-posterior` as source.

# %%
import numpy as np
import polars as pl
from ert.storage import open_storage

from iterative_ensemble_smoother.esmda import ESMDA

SEED = 42

# %%
storage_path = "/Users/FCUR/git/ert/test-data/ert/heat_equation/storage"
experiment_name = "ensemble_experiment"

with open_storage(storage_path, "r") as storage:
    ensemble = storage.get_experiment_by_name(experiment_name).get_ensemble_by_name(
        "ensemble"
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

# Prepare response matrix
Y = df_filtered.select(
    pl.all().exclude(
        ["response_key", "index", "observation_key", "observations", "std"]
    )
).to_numpy()

observations = df_filtered["observations"].to_numpy()
C_D = np.power(df_filtered["std"], 2).to_numpy().ravel()

# %%
# Create empty ensemble for posterior
with open_storage(storage_path, "w") as storage:
    experiment = storage.get_experiment_by_name(experiment_name)
    ensemble = experiment.get_ensemble_by_name("ensemble")
    posterior_ensemble = experiment.create_ensemble(
        ensemble_size=ensemble_size, name="heat-posterior", prior_ensemble=ensemble
    )

for param_group in experiment.parameter_configuration:
    print(f"Updating {param_group}")
    smoother_ESMDA = ESMDA(
        covariance=C_D, observations=observations, alpha=1, seed=SEED
    )
    D = smoother_ESMDA.perturb_observations(ensemble_size=ensemble_size, alpha=1)

    with open_storage(storage_path, "w") as storage:
        experiment = storage.get_experiment_by_name(experiment_name)
        ensemble = experiment.get_ensemble_by_name("ensemble")
        realizations = ensemble.get_realization_list_with_responses()
        X = ensemble.load_parameters_numpy(param_group, realizations)

        assert Y.shape[1] == X.shape[1]

        X_posterior_ESMDA = smoother_ESMDA.assimilate(X=X, Y=Y)

        posterior_ensemble = experiment.get_ensemble_by_name("heat-posterior")
        posterior_ensemble.save_parameters_numpy(
            parameters=X_posterior_ESMDA,
            param_group=param_group,
            iens_active_index=iens_active_index,
        )

# %%
