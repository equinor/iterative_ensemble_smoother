.. glossary::
    history matching
        History matching is a process of refining a forward model by looking
        at new evidence: observations.
    prior
         A prior  is a probability distribution of a value before some new
         evidence is taken into account. See priorwiki_.

    prediction
        A prediction is a vector of real numbers of size equal to the number
        of observation.

    forward model
         A forward model maps model parameters to predicted measurements
         (:term:`prediction`). See See for instance evensen1_ and evensen2_ .

    error function
        The error function (see erfwiki_), or erf, is a function that for a
        normal distribution with a standard derivation s and expected value
        0, has the property that erf(a / (s*sqrt(2))) is the probability that
        the error of a single measurement lies between -a and +a.

_erfwiki: https://en.wikipedia.org/wiki/Error_function
_priorwiki: https://en.wikipedia.org/wiki/Prior_probability
_evensen1: `Evensen, Geir. "Analysis of iterative ensemble smoothers for solving inverse problems." Computational Geosciences 22.3 (2018): 885-908. <https://link.springer.com/article/10.1007/s10596-018-9731-y>`_
_evensen2: `Evensen, Geir, et al. "Efficient implementation of an iterative ensemble smoother for data assimilation and reservoir history matching." Frontiers in Applied Mathematics and Statistics 5 (2019): 47. <https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full>`_
