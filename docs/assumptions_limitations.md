# Assumptions and limitations


Data assimilation algorithms like ES (Ensemble Smoother) and ESMDA (Ensemble Smoother with Multiple Data Assimilation) are used in complicated domains such as atmospheric physics and petroleum reservoirs.
This might lead some to believe that they are very complex and make few assumptions.
The exact opposite is true: they are in many ways very simple and impose extremely restrictive assumptions.
This does not mean that they are not useful, not that assumptions have to be met before they can be used in practise.
However, it does mean that practictioners should know how they work and what the drawbacks are.

## Lesson 1: The algorithms are simple because the models are complex

In almost all Bayesian statistics, where we have a prior over model parameters and update it with observed data to obtain a posterior, MCMC (Markov Chain Monte Carlo) is the preferred method.
However, in reservoir models we use algorithms like ES and ESMDA.
How come?
The reason is that when models are complex, the method must be simple:

- Most statistical models are fast to compute, a reservoir simulator $`f`$ is expensive and slow
- Most statistical models are differentiable, while a reservoir simulator $`f`$ is not (it is considered a "black-box")


## Lesson 2: Few samples lead to uncertain results



In statistics it is not uncommon to draw 1000 or even 10,000 samples from the posterior distribution





A reservoir model

Here the models are typically fast to compute, differentiable and not wildly non-linear.
It's not uncommon to draw 1000 or even 10,000 samples from the posterior distribution, and while there are pitfalls with MCMC the guarantee is that those samples do reflect the posterior.

A reservoir model is not fast to compute, it is not differentiable and is is more non-linear than most statistical models.
This changes the approach: MCMC is no longer viable since the models take so long to evaluate.
Instead algorithmms like ES and ESMDA must be used.

ES and ESMDA:

- Assume that 


