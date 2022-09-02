""" Implementation of the iterative ensemble smoother history matching algorithms
from Evensen, G. "Analysis of iterative ensemble smoothers for solving inverse
problems." for details about the algorithm.

"""
from ._ensemble_smoother import ensemble_smoother_update_step
from ._ies import InversionType
from ._iterative_ensemble_smoother import IterativeEnsembleSmoother

__all__ = [
    "ensemble_smoother_update_step",
    "IterativeEnsembleSmoother",
    "InversionType",
]
