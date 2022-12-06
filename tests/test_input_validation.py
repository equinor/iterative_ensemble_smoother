import iterative_ensemble_smoother as ies
import numpy as np
import pytest
import re


@pytest.mark.parametrize("method", ["ES", "IES"])
def test_that_bad_inputs_cause_nice_error_messages(method):
    ensemble_size = 10
    num_params = 5
    num_obs = 4
    A = np.ones(shape=(num_params, ensemble_size))
    Y = np.ones(shape=(num_obs, ensemble_size))
    obs_errors = np.ones(num_obs)
    obs_values = np.ones(num_obs)
    noise = np.zeros_like(Y)

    if method == "ES":
        smoother = ies.ensemble_smoother_update_step
    elif method == "IES":
        smoother = ies.IterativeEnsembleSmoother(ensemble_size).update_step

    with pytest.raises(
        ValueError,
        match="response_ensemble and parameter_ensemble must have the same number of columns",
    ):
        _ = smoother(Y[:, 1:], A, obs_errors, obs_values, noise=noise)

    with pytest.raises(
        ValueError,
        match="noise and response_ensemble must have the same number of columns",
    ):
        _ = smoother(Y, A, obs_errors, obs_values, noise=noise[:, 1:])

    with pytest.raises(
        ValueError,
        match="noise and response_ensemble must have the same number of rows",
    ):
        _ = smoother(Y, A, obs_errors, obs_values, noise=noise[1:, :])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "response_ensemble must be a matrix of size (number of responses by number of realizations)"
        ),
    ):
        _ = smoother(Y.ravel(), A, obs_errors, obs_values, noise)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "parameter_ensemble must be a matrix of size (number of parameters by number of realizations)"
        ),
    ):
        _ = smoother(Y, A.ravel(), obs_errors, obs_values, noise)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "observation_errors and observation_values must have the same number of elements"
        ),
    ):
        _ = smoother(Y, A, obs_errors[1:], obs_values, noise)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "observation_values must have the same number of elements as there are responses"
        ),
    ):
        _ = smoother(Y, A, obs_errors[1:], obs_values[1:], noise)
