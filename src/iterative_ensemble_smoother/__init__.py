""" Implementation of the iterative ensemble smoother history matching algorithms
from Evensen et al. "Efficient Implementation of an Iterative Ensemble Smoother
for Data Assimilation and Reservoir History Matching"
https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full
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
