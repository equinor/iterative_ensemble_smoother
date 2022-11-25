import numpy as np

rng = np.random.default_rng()

from iterative_ensemble_smoother.utils import _compute_AA_projection


def test_that_svd_projection_is_same_as_pinv():
    realizations = 10
    parameters = 5
    X = rng.standard_normal(size=(parameters, realizations))
    A = X - X.mean(axis=1, keepdims=True)
    projection_pinv = np.linalg.pinv(A) @ A
    projection_svd = _compute_AA_projection(A)
    assert np.isclose(projection_pinv, projection_svd).all()
