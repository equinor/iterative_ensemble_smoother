from iterative_ensemble_smoother import SIES
import numpy as np
import pytest


def test_that_nans_produced_due_to_outliers_in_responses_are_handled():
    # See: https://github.com/equinor/iterative_ensemble_smoother/issues/83
    # Creating response matrix with large outlier that will
    # lead to NaNs.
    parameters = np.array([[1, 2, 3]], dtype=float)
    responses = np.array([[1, 1, 1e10], [1, 10, 100]], dtype=float)
    covariance = np.array([1, 2], dtype=float)
    observations = np.array([10, 20], dtype=float)

    smoother = SIES(
        parameters=parameters,
        covariance=covariance,
        observations=observations,
        inversion="exact",
    )

    # Exact inversion does not work
    with pytest.raises(
        np.linalg.LinAlgError,
        match="Matrix is singular.",
    ):
        smoother.sies_iteration(responses, step_length=1.0)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "-v"])
