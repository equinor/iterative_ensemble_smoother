""" Implementation of the iterative ensemble smoother history matching algorithms
from Evensen et al. "Efficient Implementation of an Iterative Ensemble Smoother
for Data Assimilation and Reservoir History Matching"
https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full
"""

from ._ensemble_smoother import ensemble_smoother_update_step
from ._ies import InversionType
from ._iterative_ensemble_smoother import IterativeEnsembleSmoother

__all__ = [
    "ensemble_smoother_update_step",
    "IterativeEnsembleSmoother",
    "InversionType",
]
