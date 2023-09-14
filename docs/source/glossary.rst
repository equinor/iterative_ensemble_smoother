Glossary
========

.. glossary::

    history matching
        History matching is a process of refining a forward model by looking
        at new evidence: observations.

    prior
         A prior_ is a probability distribution of a value before some new
         evidence is taken into account.

    prediction
        A prediction is a vector of real numbers of size equal to the number
        of observation.

    forward model
         A forward model maps model parameters to predicted measurements
         (:term:`prediction`). See, for instance, :cite:t:`evensen2018analysis` and
         :cite:t:`evensen2019efficient`.

    error function
        The error function, or erf_, is a function that for a
        normal distribution with a standard derivation s and expected value
        0, has the property that erf(a / (s*sqrt(2))) is the probability that
        the error of a single measurement lies between -a and +a.

.. _erf: https://en.wikipedia.org/wiki/Error_function
.. _prior: https://en.wikipedia.org/wiki/Prior_probability
