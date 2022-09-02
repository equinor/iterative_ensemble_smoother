""" Tests for the underlying algorithm internals."""

import numpy as np

from iterative_ensemble_smoother._ies import make_D


def test_make_D():
    S = np.array([[2.0, 4.0], [6.0, 8.0]])
    E = np.array([[1.0, 2.0], [3.0, 4.0]])
    observation_values = np.array([1.0, 1.0])
    assert make_D(observation_values, E, S).tolist() == [
        [1.0 - 2 + 1.0, 2.0 - 4.0 + 1.0],
        [3.0 - 6.0 + 1.0, 4.0 - 8 + 1.0],
    ]
