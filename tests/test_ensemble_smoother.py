import io
from math import sqrt

import numpy as np
import pytest
from numpy import testing
from scipy.special import erf

import iterative_ensemble_smoother as ies
from iterative_ensemble_smoother.experimental import (
    ensemble_smoother_update_step_row_scaling,
)


# We fix the random seed in the tests for convenience
@pytest.fixture(autouse=True)
def fix_seed(seed=123456789):
    np.random.seed(seed)


def guassian_to_uniform(min_x, max_x):
    """Maps a standard guassian random variable
        to random variable, uniformly distributied between
        min_x and max_x.
    :param min_x: The lower bound on the returned value.
    :param max_x: The upper bound on the returned value.
    """

    def random_variable(x):
        """maps standard normal random variable x to
        uniform between min_x and max_x.
        """
        y = 0.5 * (1 + erf(x / sqrt(2.0)))
        return y * (max_x - min_x) + min_x

    return random_variable


# The following is an example of history matching with the
# iterative_ensemble_smoother library.

# The setup contains a forward model (a second degree polynomial in this case),
# where the coefficents of the polynomial is the model parameters.

# There are 5 time steps t=0,1,2,3,4 and 3 observations at t=0,2,4.
number_of_observations = 3
number_of_time_steps = 5
observation_times = np.array([0, 2, 4])
number_of_realizations = 10

# Before history matching, these observations are predicted by the forward
# model with the priors.


def forward_model(model_parameters):
    """Our :term:`forward_model` is s_0 * t**2 + s_1 * t + s_2 where s_0,
    s_1,s_2 is the model parameters and t is the time.
    """
    return np.array(
        [
            [
                sum(
                    parameter * time ** (2 - i)
                    for i, parameter in enumerate(model_parameters)
                )
            ]
            for time in range(number_of_time_steps)
        ],
    ).reshape((number_of_time_steps, number_of_realizations))


# The priors at t=0,2,4 are assumed uniform in [0,1], [0,2] and [0,5]
# respectively

priors = [
    guassian_to_uniform(0, 1),
    guassian_to_uniform(0, 2),
    guassian_to_uniform(0, 5),
]

# As input to the history matching we have the following
# observed values. These would normally be historic measurements.
observation_values = np.array([2.8532509308, 7.20311703432, 21.3864899107])
# The observed values have the following measurement errors
observation_errors = np.array([0.5 * (x + 1) for x, _ in enumerate(observation_values)])


@pytest.fixture
def initial_A():
    """Initial guess at parameters for the ensemble"""
    return np.random.normal(0, 1, size=(3, number_of_realizations))


@pytest.fixture
def initial_responses(initial_A):
    """The initial responses from the model"""
    return forward_model([prior(x) for prior, x in zip(priors, initial_A)])


@pytest.fixture
def initial_S(initial_responses):
    """S is the matrix of responses for each observation"""
    return initial_responses[observation_times]


def to_csv(nparray):
    buffer = io.StringIO()
    np.savetxt(
        buffer, np.round(nparray, 9), delimiter=",", newline="\n", encoding="utf8"
    )
    return buffer.getvalue()


def test_iterative_ensemble_smoother_update_step(snapshot, initial_A, initial_S):
    # performing an update step gives us a new A matrix with updated parameters
    # for the ensemble
    new_A = ies.ensemble_smoother_update_step(
        initial_S, initial_A, observation_errors, observation_values
    )
    assert new_A.shape == initial_A.shape
    assert new_A.dtype == initial_A.dtype
    snapshot.assert_match(
        to_csv(new_A), "test_iterative_ensemble_smoother_update_step.csv"
    )


class RowScaling:
    def multiply(self, A, _):
        return A


def test_ensemble_smoother_update_step_with_rowscaling(snapshot, initial_A, initial_S):
    ((new_A, _),) = ensemble_smoother_update_step_row_scaling(
        initial_S,
        [(initial_A, RowScaling())],
        observation_errors,
        observation_values,
    )
    assert new_A.shape == initial_A.shape
    assert new_A.dtype == initial_A.dtype
    snapshot.assert_match(to_csv(new_A), "test_ensemble_smoother_update_step.csv")


def test_ensemble_smoother_update_step(snapshot, initial_A, initial_S):
    # performing an update step gives us a new A matrix with updated parameters
    # for the ensemble
    new_A = ies.ensemble_smoother_update_step(
        initial_S, initial_A, observation_errors, observation_values
    )
    assert new_A.shape == initial_A.shape
    assert new_A.dtype == initial_A.dtype
    snapshot.assert_match(to_csv(new_A), "test_ensemble_smoother_update_step.csv")


def test_get_steplength():
    expected = [
        7.762203155904597862e-01,
        5.999999999999999778e-01,
        4.889881574842309675e-01,
        4.190550788976149521e-01,
        3.750000000000000000e-01,
        3.472470393710577197e-01,
        3.297637697244037436e-01,
        3.187499999999999778e-01,
        3.118117598427644355e-01,
        3.074409424311009276e-01,
    ]
    iterative_es = ies.IterativeEnsembleSmoother(0)
    steplengths = [iterative_es._get_steplength(i) for i in range(10)]
    testing.assert_array_equal(expected, steplengths)
