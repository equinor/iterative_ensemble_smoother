"""
Purpose
=======

**iterative_ensemble_smoother** is an open-source, pure python,
and object-oriented library that provides
a user friendly implementation of history matching algorithms
from :cite:t:`evensen2019efficient`.

The following functionalities are directly provided on module-level.

Classes
=======

.. autosummary::
   :toctree: _autosummary

   SIES
   ESMDA

Functions
=========

.. autosummary::
   :toctree: _autosummary

    steplength_exponential

"""
try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version", "unknown commit")

from iterative_ensemble_smoother.esmda import ESMDA
from iterative_ensemble_smoother.sies import SIES
from iterative_ensemble_smoother.utils import steplength_exponential

__all__ = [
    "SIES",
    "ESMDA",
    "steplength_exponential",
]
