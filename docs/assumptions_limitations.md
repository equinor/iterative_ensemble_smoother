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
The reason is that when running complex models, the method must remain simple for the overall inference to remain tractable:

- Most statistical models are fast to compute, a reservoir simulator $`f`$ is expensive and slow
- Most statistical models are differentiable, while a reservoir simulator $`f`$ is not (it is considered a "black-box")

Algorithms like ES an ESMDA are simple in the sense that they are derived assuming a Gauss-Linear model.
Both of the assumptions (1) linearity of the model $`f`$ and (2) Gaussian noise are untrue.
More on that later.

## Lesson 2: Few samples lead to uncertain results

In general statistics it is not uncommon to draw 1000 or even 10,000 samples from the posterior distribution.
In reservoir models each function evaluation is expensive, so we have to make do with far fewer samples.

This can be an issue even is small, simple problems.
Suppose A and B are two variables that are uniformly distributed.
What is the expected value of their product, i.e. E[A * B] ?

## Lesson 3: Data assimilation will one-shot linear models

The figure below shows a linear model $`f(x_1, x_2) = x_1 + x_2`$, where $`x_1`$ and $`x_2`$ are input parameters.
Our prior belief is a normal distribution around the point $`(x_1, x_2) = (1, 1)`$ and the scalar observation is $`y = f(x_1, x_2) = 0`$.
ES and ESMDA answer the question "_How can our prior belief be reconsiled with the observation?_".

The answer depends on the noise associated with the observation $`y = 0`$.
With high noise, the answer moves the prior distribution "half-way" toward the line $`y = f(x_1, x_2) = 0`$.
With low noise, the answer moves the prior distribution all the way toward the line $`y = f(x_1, x_2) = 0`$, but retains uncertainty along that line.

![](linear_model_obs_noise.png)

Either way, with Gaussian noise and a linear model $`f`$ the answer is correct -- this is the _only_ model where the answer is correct.








-----------------





A reservoir model

Here the models are typically fast to compute, differentiable and not wildly non-linear.
It's not uncommon to draw 1000 or even 10,000 samples from the posterior distribution, and while there are pitfalls with MCMC the guarantee is that those samples do reflect the posterior.

A reservoir model is not fast to compute, it is not differentiable and is is more non-linear than most statistical models.
This changes the approach: MCMC is no longer viable since the models take so long to evaluate.
Instead algorithmms like ES and ESMDA must be used.

ES and ESMDA:

- Assume that 


