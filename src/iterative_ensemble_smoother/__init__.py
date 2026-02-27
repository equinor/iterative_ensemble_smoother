"""
Purpose
=======

**iterative_ensemble_smoother** is an open-source, pure python,
and object-oriented library that provides
a user friendly implementation of history matching algorithms.

The following functionalities are directly provided on module-level.

Classes
=======

.. autosummary::
   :toctree: _autosummary

   ESMDA
   AdaptiveESMDA
   LocalizedESMDA

"""

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version", "unknown commit")

from iterative_ensemble_smoother.esmda import ESMDA
from iterative_ensemble_smoother.esmda_adaptive import AdaptiveESMDA
from iterative_ensemble_smoother.esmda_localized import LocalizedESMDA

__all__ = ["ESMDA", "AdaptiveESMDA", "LocalizedESMDA"]
