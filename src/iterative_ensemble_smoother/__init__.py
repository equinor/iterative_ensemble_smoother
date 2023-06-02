""" Implementation of the iterative ensemble smoother history matching algorithms
from Evensen et al. "Efficient Implementation of an Iterative Ensemble Smoother
for Data Assimilation and Reservoir History Matching"
https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full
"""

from iterative_ensemble_smoother._ies import InversionType
from iterative_ensemble_smoother._iterative_ensemble_smoother import (
    ES,
    SIES,
)

__all__ = [
    "ES",
    "SIES",
    "InversionType",
]

__version__ = "0.2.0"
