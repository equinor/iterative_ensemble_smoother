from iterative_ensemble_smoother import ES, SIES
import numpy as np
import pytest
import re


def test_that_repr_can_be_created():
    smoother = SIES(100)
    smoother_repr = eval(repr(smoother))
    assert isinstance(smoother_repr, SIES)

    smoother = ES()
    smoother_repr = eval(repr(smoother))
    assert isinstance(smoother_repr, ES)


def test_that_bad_inputs_cause_nice_error_messages():
    ensemble_size = 10
    num_obs = 4
    Y = np.ones(shape=(num_obs, ensemble_size))
    obs_errors = np.ones(num_obs)
    obs_values = np.ones(num_obs)
    noise = np.zeros_like(Y)

    with pytest.raises(
        ValueError,
        match="noise and response_ensemble must have the same number of columns",
    ):
        _ = SIES(ensemble_size).fit(Y, obs_errors, obs_values, noise=noise[:, 1:])

    with pytest.raises(
        ValueError,
        match="noise and response_ensemble must have the same number of rows",
    ):
        _ = SIES(ensemble_size).fit(Y, obs_errors, obs_values, noise=noise[1:, :])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "response_ensemble must be a matrix of size (number of responses by number of realizations)"
        ),
    ):
        _ = SIES(ensemble_size).fit(Y.ravel(), obs_errors, obs_values, noise=noise)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "observation_errors and observation_values must have the same number of elements"
        ),
    ):
        _ = SIES(ensemble_size).fit(Y, obs_errors[1:], obs_values, noise=noise)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "observation_values must have the same number of elements as there are responses"
        ),
    ):
        _ = SIES(ensemble_size).fit(Y, obs_errors[1:], obs_values[1:], noise=noise)
