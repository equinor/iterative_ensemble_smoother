glossary
========

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

.. _erfwiki: https://en.wikipedia.org/wiki/Error_function
.. _priorwiki: https://en.wikipedia.org/wiki/Prior_probability
.. _evensen1: https://link.springer.com/article/10.1007/s10596-018-9731-y
.. _evensen2: https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full
