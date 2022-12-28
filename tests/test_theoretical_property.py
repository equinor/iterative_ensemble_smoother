import numpy as np
import pytest

rng = np.random.default_rng()

import iterative_ensemble_smoother as ies

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
def test_that_update_is_according_to_theory(inversion, errors):
    """
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

    # A is p(x)
    Sigma = var * np.identity(nparam)
    A = rng.multivariate_normal(mean=np.zeros(nparam), cov=Sigma, size=(N)).T
    # Assuming forward model is the identity
    Y = A

    obs_val = 10
    observation_values = np.array([obs_val, obs_val, obs_val])
    A_ES = ies.ensemble_smoother_update_step(
        Y, A, errors, observation_values, inversion=inversion, truncation=1.0
    )

    ens_mask = np.array([True] * N)
    ens_mask[:5] = False
    A_IES = ies.IterativeEnsembleSmoother(ensemble_size=N).update_step(
        Y[:, ens_mask],
        A[:, ens_mask],
        errors,
        observation_values,
        truncation=1.0,
        inversion=inversion,
        step_length=1.0,
        ensemble_mask=ens_mask,
    )

    for i in range(nparam):
        assert np.isclose(A_ES[i, :].mean(), obs_val / 2, rtol=0.1)
        assert np.isclose(A_IES[i, :].mean(), obs_val / 2, rtol=0.1)

    assert (np.abs(np.cov(A_ES) - np.identity(nparam)) < 0.15).all()
    assert (np.abs(np.cov(A_IES) - np.identity(nparam)) < 0.15).all()
