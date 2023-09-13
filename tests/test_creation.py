import re

import numpy as np
import pytest

from iterative_ensemble_smoother import ES, SIES, SiesInversionType
from iterative_ensemble_smoother.utils import validate_inputs


def test_that_repr_can_be_created():
    # smoother = SIES()
    # smoother_repr = eval(repr(smoother))
    # assert isinstance(smoother_repr, SIES)

    smoother = ES()
    smoother_repr = eval(repr(smoother))
    assert isinstance(smoother_repr, ES)


def test_that_bad_inputs_cause_nice_error_messages():
    ensemble_size = 10
    num_obs = 4
    num_params = 3
    param_ensemble = np.random.normal(size=(num_params, ensemble_size))
    Y = np.ones(shape=(num_obs, ensemble_size))
    obs_errors = np.ones(num_obs)
    obs_values = np.ones(num_obs)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "response_ensemble must be a matrix of size (number of responses by number of realizations)"
        ),
    ):
        _ = SIES(obs_errors, obs_values, ensemble_size).fit(Y.ravel())

    with pytest.raises(
        ValueError,
        match=re.escape(
            "observation_errors and observation_values must have the same number of elements"
        ),
    ):
        _ = SIES(obs_errors[1:], obs_values, ensemble_size).fit(Y)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "observation_values must have the same number of elements as there are responses"
        ),
    ):
        _ = SIES(obs_errors[1:], obs_values[1:], ensemble_size).fit(Y)

    with pytest.raises(
        ValueError,
        match="param_ensemble and response_ensemble must have the same number of columns",
    ):
        _ = SIES(obs_errors, obs_values, ensemble_size).fit(
            Y,
            param_ensemble=param_ensemble[:, : ensemble_size - 2],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "parameter_ensemble must be a matrix of size (number of parameters by number of realizations)"
        ),
    ):
        _ = SIES(obs_errors, obs_values, ensemble_size).fit(
            Y,
            param_ensemble=param_ensemble[0, :].ravel(),
        )


def test_validate_inversion_type() -> None:
    """Check that the inversion type is correctly checked."""
    ne = 100
    nobs = 50
    nx = 2000
    for _type in SiesInversionType.to_list():
        validate_inputs(_type, np.zeros((nobs, ne)), np.zeros(nobs), np.zeros((nx, ne)))
        # test with the string
        validate_inputs(
            _type.value, np.zeros((nobs, ne)), np.zeros(nobs), np.zeros((nx, ne))
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            '"random_str" is not a valid inversion type! It must be choosen '
            "among ['naive', 'exact', 'exact_r', 'subspace_re']"
        ),
    ):
        validate_inputs(
            "random_str", np.zeros((nobs, ne)), np.zeros(nobs), np.zeros((nx, ne))
        )


def test_that_nans_produced_due_to_outliers_in_responses_are_handled():
    # See: https://github.com/equinor/iterative_ensemble_smoother/issues/83
    # Creating response matrix with large outlier that will
    # lead to NaNs.
    response_ensemble = np.array([[1, 1, 1e19], [1, 10, 100]], dtype=float)
    obs_error = np.array([1, 2], dtype=float)
    obs_value = np.array([10, 20], dtype=float)
    smoother = ES()

    with pytest.raises(
        ValueError,
        match="Fit produces NaNs. Check your response matrix for outliers or use an inversion type with truncation.",
    ):
        smoother.fit(response_ensemble, obs_error, obs_value, inversion="exact")

    # Running with an inversion type that does truncation does not produce NaNs.
    param_ensemble = np.array([[1, 2, 3]], dtype=float)
    smoother.fit(
        response_ensemble,
        obs_error,
        obs_value,
        inversion="exact_r",
    )
    param_ensemble = smoother.update(param_ensemble)
    assert np.isnan(param_ensemble).sum() == 0


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-v"])
