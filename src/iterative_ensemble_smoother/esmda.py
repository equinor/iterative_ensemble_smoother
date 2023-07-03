#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp


def g(x):
    pass


def normalize_alpha(alpha):
    """Assure that sum_i (1/alpha_i) = 1

    Examples
    --------
    >>> alpha = np.arange(10) + 1
    >>> np.sum(1/normalize_alpha(alpha))
    1.0
    """
    factor = np.sum(1 / alpha)
    return alpha * factor


# Number of data assimilations
N_a = 10
alpha = np.arange(10)


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-v"])
