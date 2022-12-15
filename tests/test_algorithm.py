""" Tests for the underlying algorithm internals."""

import numpy as np

rng = np.random.default_rng()


from iterative_ensemble_smoother._ies import make_D
import iterative_ensemble_smoother as ies


def test_make_D():
    S = np.array([[2.0, 4.0], [6.0, 8.0]])
    E = np.array([[1.0, 2.0], [3.0, 4.0]])
    observation_values = np.array([1.0, 1.0])
    assert make_D(observation_values, E, S).tolist() == [
        [1.0 - 2 + 1.0, 2.0 - 4.0 + 1.0],
        [3.0 - 6.0 + 1.0, 4.0 - 8 + 1.0],
    ]


def test_that_global_es_update_is_identical_to_local():
    N = 10
    p = 50
    m = p
    A = rng.normal(size=(p, N))
    # Assume identity operator as forward model, i.e., Y = g(A) = A
    Y = A
    observation_errors = np.diag(rng.uniform(size=m))
    observation_values = rng.multivariate_normal(np.zeros(p), observation_errors)
    noise = rng.normal(size=(m, N))

    A_ES_global = ies.ensemble_smoother_update_step(
        Y, A, np.diag(observation_errors), observation_values, noise=noise
    )

    A_ES_local = np.zeros(shape=(p, N))

    for i in range(A.shape[0]):
        A_ES_local[i, :] = ies.ensemble_smoother_update_step(
            Y,
            A[i, :].reshape(1, N),
            np.diag(observation_errors),
            observation_values,
            noise=noise,
            projection=False,
        )

    assert np.isclose(A_ES_global, A_ES_local).all()


def test_that_global_ies_update_is_identical_to_local():
    N = 10
    p = 50
    m = p
    A = rng.normal(size=(p, N))
    # Assume identity operator as forward model, i.e., Y = g(A) = A
    Y = A
    observation_errors = np.diag(rng.uniform(size=m))
    observation_values = rng.multivariate_normal(np.zeros(p), observation_errors)
    noise = rng.normal(size=(m, N))

    A_local = A.copy()
    Y_local = A_local
    A_IES_global = ies.IterativeEnsembleSmoother(ensemble_size=N).update_step(
        Y,
        A,
        np.diag(observation_errors),
        observation_values,
        noise=noise,
        step_length=1.0,
    )

    # A is passed by reference and updated.
    assert A is A_IES_global

    A_IES_local = np.zeros(shape=(p, N))

    for i in range(A_local.shape[0]):
        A_IES_local[i, :] = ies.IterativeEnsembleSmoother(ensemble_size=N).update_step(
            Y_local,
            A_local[i, :].reshape(1, N).copy(),
            np.diag(observation_errors),
            observation_values,
            noise=noise,
            projection=False,
            step_length=1.0,
        )

    assert np.isclose(A_IES_global, A_IES_local).all()


def test_that_ies_runs_with_failed_realizations():
    """This used to cause Eigen to throw an `Assertion failed`
    that led to `Fatal Python error` which is not possible to catch in pytest.
    """
    rng = np.random.default_rng()
    ensemble_size = 50
    num_params = 100
    num_responses = 5
    param_ensemble = rng.normal(size=(num_params, ensemble_size))
    response_ensemble = np.power(param_ensemble[:num_responses, :], 2) + rng.normal(
        size=(num_responses, ensemble_size)
    )
    noise = rng.normal(size=(num_responses, ensemble_size))
    obs_values = rng.normal(size=num_responses)
    obs_errors = rng.normal(size=num_responses)
    ens_mask = np.array([True] * ensemble_size)
    ens_mask[10:] = False
    smoother = ies.IterativeEnsembleSmoother(ensemble_size)
    param_ensemble = smoother.update_step(
        response_ensemble[:, ens_mask],
        param_ensemble[:, ens_mask],
        obs_errors,
        obs_values,
        noise[:, ens_mask],
        ensemble_mask=ens_mask,
    )
    param_ensemble = smoother.update_step(
        response_ensemble[:, ens_mask],
        param_ensemble,
        obs_errors,
        obs_values,
        noise[:, ens_mask],
        ensemble_mask=ens_mask,
    )
    param_ensemble = smoother.update_step(
        response_ensemble[:, ens_mask],
        param_ensemble,
        obs_errors,
        obs_values,
        noise[:, ens_mask],
        ensemble_mask=ens_mask,
    )
    assert param_ensemble.shape == (num_params, ens_mask.sum())
