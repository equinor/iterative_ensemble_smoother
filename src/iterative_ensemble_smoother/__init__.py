""" Implementation of the iterative ensemble smoother history matching algorithms
from Evensen et al. "Efficient Implementation of an Iterative Ensemble Smoother
for Data Assimilation and Reservoir History Matching"
https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full
"""
try:
    from ._version import version as __version__  # type: ignore
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version", "unknown commit")
from iterative_ensemble_smoother._iterative_ensemble_smoother import ES, SIES
from iterative_ensemble_smoother.esmda import ESMDA

__all__ = [
    "ES",
    "SIES",
    "ESMDA",
]
