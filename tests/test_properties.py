import numpy as np
import pytest
import pandas as pd
from p_tqdm import p_map

from iterative_ensemble_smoother._ies import make_D
import iterative_ensemble_smoother as ies

rng = np.random.default_rng()

# The following tests follow the
# posterior properties described in
# https://ert.readthedocs.io/en/latest/theory/ensemble_based_methods.html#kalman-posterior-properties
a_true = 1.0
b_true = 5.0
number_of_parameters = 2
number_of_observations = 45


class LinearModel:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.size = 2

    @classmethod
    def random(cls):
        a_std = 2.0
        b_std = 2.0
        # Priors with bias
        a_bias = 0.5 * a_std
        b_bias = -0.5 * b_std

        return cls(
            rng.normal(a_true + a_bias, a_std),
            rng.normal(b_true + b_bias, b_std),
        )

    def eval(self, x):
        return self.a * x + self.b


@pytest.mark.parametrize("number_of_realizations", [100, 200])
def test_that_es_update_for_a_linear_model_follows_theory(number_of_realizations):
    true_model = LinearModel(a_true, b_true)

    ensemble = [LinearModel.random() for _ in range(number_of_realizations)]

    A = np.array(
        [
            [realization.a for realization in ensemble],
            [realization.b for realization in ensemble],
        ]
    )
    mean_prior = np.mean(A, axis=1)

    # We use time as the x-axis and observations are at
    # t=0,1,2...number_of_observations
    times = np.arange(number_of_observations)

    S = np.array([[realization.eval(t) for realization in ensemble] for t in times])

    # When observations != true model, then ml estimates != true parameters.
    # This gives both a more advanced and realistic test. Standard normal
    # N(0,1) noise is added to obtain this. The randomness ensures we are not
    # gaming the test. But the difference could in principle be any non-zero
    # scalar.
    observations = np.array(
        [true_model.eval(t) + np.random.normal(0.0, 1.0) for t in times]
    )

    # Leading to fixed Maximum likelihood estimate.
    # It will equal true values when observations are sampled without noise.
    # It will also stay the same over beliefs.
    mean_observations = np.mean(observations)
    times_mean = np.mean(times)
    times_square_sum = sum(np.square(times))
    a_maximum_likelihood = sum(
        t * (observations[t] - mean_observations) for t in times
    ) / (times_square_sum - times_mean * sum(times))
    b_maximum_likelihood = mean_observations - a_maximum_likelihood * times_mean
    maximum_likelihood = np.array([a_maximum_likelihood, b_maximum_likelihood])

    previous_mean_posterior = mean_prior

    # numerical precision tolerance
    epsilon = 1e-2

    # We iterate with an increased belief in the observations
    for error in [10000.0, 100.0, 10.0, 1.0, 0.1]:
        # An important point here is that we do not iteratively
        # update A, but instead, observations stay the same and
        # we increase our belief in the observations
        # As A is update inplace, we have to reset it.
        A = np.array(
            [
                [realization.a for realization in ensemble],
                [realization.b for realization in ensemble],
            ]
        )
        smoother = ies.ES()
        smoother.fit(
            S,
            np.full(observations.shape, error),
            observations,
        )
        A_posterior = smoother.update(A)
        mean_posterior = np.mean(A_posterior, axis=1)

        # All posterior estimates lie between prior and maximum likelihood estimate
        assert np.all(
            np.linalg.norm(mean_posterior - maximum_likelihood)
            - np.linalg.norm(mean_prior - maximum_likelihood)
            < epsilon
        )
        assert np.all(
            np.linalg.norm(mean_prior - mean_posterior)
            - np.linalg.norm(mean_prior - maximum_likelihood)
            < epsilon
        )

        # Posterior parameter estimates improve with increased trust in observations
        assert np.all(
            np.linalg.norm(mean_posterior - maximum_likelihood)
            - np.linalg.norm(previous_mean_posterior - maximum_likelihood)
            < epsilon
        )

        previous_mean_posterior = mean_posterior

    # At strong beliefs, we should be close to the maximum likelihood estimate
    assert np.all(
        np.linalg.norm(previous_mean_posterior - maximum_likelihood) < epsilon
    )


def test_that_sies_converges_to_es_in_gauss_linear_case():
    def poly(a, b, c, x):
        return a * x**2 + b * x + c

    ensemble_size = 200

    # True patameter values
    a_t = 0.5
    b_t = 1.0
    c_t = 3.0

    noise_scale = 0.1
    x_observations = [0, 2, 4, 6, 8]
    observations = [
        (
            poly(a_t, b_t, c_t, x)
            + rng.normal(loc=0, scale=noise_scale * poly(a_t, b_t, c_t, x)),
            noise_scale * poly(a_t, b_t, c_t, x),
            x,
        )
        for x in x_observations
    ]

    d = pd.DataFrame(observations, columns=["value", "sd", "x"])
    d = d.set_index("x")
    num_obs = d.shape[0]

    # Assume diagonal ensemble covariance matrix for the measurement perturbations.
    R = np.diag(d.sd.values**2)

    E = rng.multivariate_normal(mean=np.zeros(len(R)), cov=R, size=ensemble_size).T
    assert E.shape == (num_obs, ensemble_size)

    coeff_a = rng.normal(0, 1, size=ensemble_size)
    coeff_b = rng.normal(0, 1, size=ensemble_size)
    coeff_c = rng.normal(0, 1, size=ensemble_size)

    X = np.concatenate(
        (coeff_a.reshape(-1, 1), coeff_b.reshape(-1, 1), coeff_c.reshape(-1, 1)), axis=1
    ).T

    fwd_runs = p_map(
        poly,
        coeff_a,
        coeff_b,
        coeff_c,
        [np.arange(max(x_observations) + 1)] * ensemble_size,
        desc="Running forward model.",
    )

    # Pick responses where we have observations
    response_ensemble = np.array(
        [fwd_run[d.index.get_level_values("x").to_list()] for fwd_run in fwd_runs]
    ).T

    smoother_es = ies.ES()
    smoother_es.fit(response_ensemble, d.sd.values, d.value.values, noise=E)
    params_es = smoother_es.update(X)

    assert np.linalg.det(np.cov(X)) > np.linalg.det(np.cov(params_es))

    params_ies = X.copy()
    responses_ies = response_ensemble.copy()
    smoother_ies = ies.SIES(ensemble_size)
    for _ in range(10):
        smoother_ies.fit(responses_ies, d.sd.values, d.value.values, noise=E)
        params_ies = smoother_ies.update(params_ies)

        _coeff_a = params_ies[0, :]
        _coeff_b = params_ies[1, :]
        _coeff_c = params_ies[2, :]

        _fwd_runs = p_map(
            poly,
            _coeff_a,
            _coeff_b,
            _coeff_c,
            [np.arange(max(x_observations) + 1)] * ensemble_size,
            desc="SIES ert iteration",
        )

        assert np.linalg.det(np.cov(X)) > np.linalg.det(np.cov(params_ies))

        responses_ies = np.array(
            [fwd_run[d.index.get_level_values("x").to_list()] for fwd_run in _fwd_runs]
        ).T

    assert np.abs(
        (np.linalg.det(np.cov(params_es)) - np.linalg.det(np.cov(params_ies))) < 0.001
    )


var = 2.0


@pytest.mark.parametrize(
    "inversion,errors",
    [
        (ies.InversionType.EXACT, np.diag(np.array([var, var, var]))),
        (ies.InversionType.EXACT, np.array([np.sqrt(var), np.sqrt(var), np.sqrt(var)])),
        (ies.InversionType.EXACT_R, np.diag(np.array([var, var, var]))),
        (
            ies.InversionType.EXACT_R,
            np.array([np.sqrt(var), np.sqrt(var), np.sqrt(var)]),
        ),
        (ies.InversionType.SUBSPACE_RE, np.diag(np.array([var, var, var]))),
        (
            ies.InversionType.SUBSPACE_RE,
            np.array([np.sqrt(var), np.sqrt(var), np.sqrt(var)]),
        ),
    ],
)
def test_that_update_correctly_multiples_gaussians(inversion, errors):
    """
    NB! This test is potentially flaky because of finite ensemble size.

    Bayes' theorem states that
    p(x|y) is proportional to p(y|x)p(x)

    Assume p(x) is N(mu=0, Sigma=2I) and p(y|x) is N(mu=y, Sigma=2I).
    Multiplying these together (see 8.1.8 of the matrix cookbook) we get
    that p(x|y) is N(mu=y/2, Sigma=I).
    Note that Sigma is a covariance matrix.

    Here we use this property, and assume that the forward model is the identity
    to test analysis steps.
    """
    N = 1500
    nparam = 3

    A = rng.standard_normal(size=(nparam, N))
    A = np.linalg.cholesky(var * np.identity(nparam)) @ A
    # Assuming forward model is the identity
    Y = A

    obs_val = 10
    observation_values = np.array([obs_val, obs_val, obs_val])
    smoother = ies.SIES(N)
    smoother.fit(
        Y,
        errors,
        observation_values,
        inversion=inversion,
        step_length=1.0,
        truncation=1.0,
    )
    A_ES = smoother.update(A)

    for i in range(nparam):
        assert np.isclose(A_ES[i, :].mean(), obs_val / 2, rtol=0.15)

    assert (np.abs(np.cov(A_ES) - np.identity(nparam)) < 0.15).all()


def test_make_D():
    S = np.array([[2.0, 4.0], [6.0, 8.0]])
    E = np.array([[1.0, 2.0], [3.0, 4.0]])
    observation_values = np.array([1.0, 1.0])
    assert make_D(observation_values, E, S).tolist() == [
        [1.0 - 2 + 1.0, 2.0 - 4.0 + 1.0],
        [3.0 - 6.0 + 1.0, 4.0 - 8 + 1.0],
    ]


@pytest.mark.parametrize(
    "ensemble_size,num_params,linear",
    [
        pytest.param(100, 3, True, id="No projection because linear model"),
        pytest.param(
            100,
            3,
            False,
            id="Project because non-linear and num_params < ensemble_size - 1",
        ),
        pytest.param(3, 100, True, id="No projection because linear model"),
        pytest.param(
            3, 100, False, id="No projection because num_params > ensemble_size - 1"
        ),
        pytest.param(
            10, 10, True, id="No projection because num_params > ensemble_size - 1"
        ),
    ],
)
def test_that_global_es_update_is_identical_to_local(ensemble_size, num_params, linear):
    num_obs = num_params
    X = rng.normal(size=(num_params, ensemble_size))

    Y = X if linear else np.power(X, 2)

    observation_errors = rng.uniform(size=num_obs)
    observation_values = rng.normal(np.zeros(num_params), observation_errors)
    noise = rng.normal(size=(num_obs, ensemble_size))

    param_ensemble = None
    if not linear and num_params < ensemble_size - 1:
        param_ensemble = X

    smoother = ies.ES()
    smoother.fit(
        Y,
        observation_errors,
        observation_values,
        noise=noise,
        param_ensemble=param_ensemble,
    )
    X_ES_global = smoother.update(X)

    X_ES_local = np.zeros(shape=(num_params, ensemble_size))

    # First update a subset or batch of parameters at once
    # and and then loop through each remaining parameter and
    # update it separately.
    # This is meant to emulate a configuration where users
    # want to update, say, a field parameter separately from scalar parameters.
    batch = list(range(num_params // 3))
    smoother.fit(
        Y,
        observation_errors,
        observation_values,
        noise=noise,
        param_ensemble=param_ensemble,
    )
    X_ES_local[batch, :] = smoother.update(X[batch])

    for i in range(num_params // 3, num_params):
        smoother.fit(
            Y,
            observation_errors,
            observation_values,
            noise=noise,
            param_ensemble=param_ensemble,
        )
        X_ES_local[i, :] = smoother.update(X[i, :])

    assert np.isclose(X_ES_global, X_ES_local).all()


def test_that_ies_runs_with_failed_realizations():
    rng = np.random.default_rng()
    ensemble_size = 50
    num_params = 100
    num_responses = 5
    param_ensemble = rng.normal(size=(num_params, ensemble_size))
    response_ensemble = np.power(param_ensemble, 2)[:num_responses, :] + rng.normal(
        size=(num_responses, ensemble_size)
    )
    noise = rng.normal(size=(num_responses, ensemble_size))
    obs_values = rng.normal(size=num_responses)
    obs_errors = rng.normal(size=num_responses)
    ens_mask = np.array([True] * ensemble_size)
    ens_mask[10:] = False
    smoother = ies.SIES(ensemble_size)
    smoother.fit(
        response_ensemble[:, ens_mask],
        obs_errors,
        obs_values,
        noise=noise[:, ens_mask],
        ensemble_mask=ens_mask,
    )
    param_ensemble = smoother.update(param_ensemble[:, ens_mask])

    smoother.fit(
        response_ensemble[:, ens_mask],
        obs_errors,
        obs_values,
        noise=noise[:, ens_mask],
        ensemble_mask=ens_mask,
    )
    param_ensemble = smoother.update(param_ensemble)

    smoother.fit(
        response_ensemble[:, ens_mask],
        obs_errors,
        obs_values,
        noise=noise[:, ens_mask],
        ensemble_mask=ens_mask,
    )
    param_ensemble = smoother.update(param_ensemble)

    assert param_ensemble.shape == (num_params, ens_mask.sum())
