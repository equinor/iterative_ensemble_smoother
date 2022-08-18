from math import sqrt

import numpy as np
from scipy.special import erf

import iterative_ensemble_smoother as ies


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


def test_makeE():

    observation_values = np.array(
        [2.8532509308, 7.20311703432, 21.3864899107, 31.5145559347, 53.5676660405]
    )
    observation_errors = np.array(
        [0.5 * (x + 1) for x, _ in enumerate(observation_values)]
    )
    response_x_axis = range(10)
    priors = [(0, 1), (0, 2), (0, 5)]
    observation_x_axis = [0, 2, 4, 6, 8]
    realizations = 200
    A = np.asfortranarray(np.random.normal(0, 1, size=(3, realizations)))
    responses_before = forward_model(A, priors, response_x_axis)
    S = responses_before[observation_x_axis]
    noise = np.random.rand(*S.shape)
    assert ies.make_E(observation_errors, noise) is not None
