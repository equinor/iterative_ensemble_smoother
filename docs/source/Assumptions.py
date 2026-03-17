# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# ruff: noqa: E402

# %% [markdown]
# # Assumptions and limitations
#
# Data assimilation algorithms like ES (Ensemble Smoother) and ESMDA
# (Ensemble Smoother with Multiple Data Assimilation) are used in complicated
# domains such as atmospheric physics and petroleum reservoirs.
# This might lead you to believe that they are complex, sophisticated
# algorithms that work well on a wide variety of problems.
# The opposite is true: they are very simple and impose
# extremely restrictive assumptions.
# Practitioners should know how ESMDA works have an an intuition for how it fails.
# Although the assumptions used to derive ES and ESMDA are never met in practice,
# the algorithms are still useful.
#
# As a reminder, the problem that ESMDA solves is this: given a prior over model
# parameters as well as observed outcomes, adjust the prior in the direction
# of the observed outcomes.
# This is a bit like optimization ("which inputs would produce these outputs?"),
# but ESMDA aims to capture uncertainty and mix the prior with the
# observations to obtain a posterior.


# %% [markdown]
# ## Lesson 1: The algorithms are simple because the models are complex
#
# MCMC (Markov Chain Monte Carlo) is used in most statistical applications where
# the goal is to sample a posterior distribution.
# This is state of the art, but in geology we use ES and ESMDA, not MCMC.
# How come?
#
# The reason is that since the models are complex (slow to evaluate and black-box),
# the method must remain simple for the overall
# inference to have any chance of success:
#
# - Most statistical models are fast to evaluate, a reservoir simulator is slow
# - Most statistical models are differentiable,
#   while a reservoir simulator is not (it is considered "black-box")
#
# Algorithms like ES an ESMDA are simple in the sense that their theoretical
# foundation rests on an assumption that is never met in reality:
# a Gauss-Linear model.
# Both of the assumptions (1) linearity of the model $f$ and
# (2) Gaussian noise are untrue in practice.
#
# The figure below shows the Gauss-linear case, where the ESMDA solution
# corresponds to the theoretical solution in the limit of many samples (realizations).
#

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from iterative_ensemble_smoother import ESMDA

prop_cycle = plt.rcParams["axes.prop_cycle"]
COLORS = prop_cycle.by_key()["color"]
rng = np.random.default_rng(42)


# A simple linear model
def f(x):
    return np.sum(x, keepdims=True)
    # return np.abs(np.sum(x, keepdims=True))**2 + x[0] * 1


def F(X):
    return np.array([f(x_i) for x_i in X.T]).T


x1 = np.linspace(-1, 2, 100)
x2 = np.linspace(-1, 2, 100)
X1, X2 = np.meshgrid(x1, x2)

# Evaluate f on the grid
Z = np.array(
    [
        [f(np.array([x1_val, x2_val]))[0] for x1_val, x2_val in zip(row1, row2)]
        for row1, row2 in zip(X1, X2)
    ]
)


fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 2.5))
axes = iter(axes.ravel())
fig.suptitle("A linear model with different levels of observation noise", y=0.925)

realizations = 200
X_prior = 1 + 0.2 * rng.normal(size=(2, realizations))
observations = np.zeros(1)

for i, covariance_factor in enumerate([0.1, 0.01, 0.001]):
    ax = next(axes)
    ax.set_title(f"Covariance = {covariance_factor}")
    covariance = np.diag([1]) * covariance_factor

    # All contours in gray with labels
    cs = ax.contour(X1, X2, Z, levels=10, cmap="Greys", alpha=0.7, vmin=-20, vmax=4)
    if i == 0:
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    # Highlight f(x) = 0 line
    cs0 = ax.contour(X1, X2, Z, levels=[0], colors="black", linewidths=2)
    if i == 0:
        ax.clabel(cs0, inline=True, fontsize=8, fmt="%.1f")

    ax.scatter(*X_prior, label="Prior", s=10)

    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    ax.grid(True, ls="--", alpha=0.5)

    esmda = ESMDA(covariance, observations, alpha=1, seed=rng)
    X = X_prior.copy()
    for _ in range(esmda.num_assimilations()):
        esmda.prepare_assimilation(Y=F(X))
        X = esmda.assimilate_batch(X=X)

    ax.scatter(*X, label="Posterior", s=10)
    if i == 0:
        ax.legend()


fig.tight_layout()
# plt.savefig("linear_model_obs_noise.png", dpi=200)
plt.show()

# %% [markdown]
# In this example, two inputs (parameters) go into the
# model $f$ and there is one output.
# Several inputs could produce the observed value (black line).
# ES (one iteration of ESMDA) moves the prior toward the parameter
# configuration that explains the observed values.
# The size of the movement here depends on the observation noise,
# which is an algorithm parameter.
# In the rest of this document we'll set the observation noise to a low number.

# %% [markdown]
# ## Lesson 2: Few samples lead to uncertain results
#
# In most statistical models it's common to draw 1000 or even 10,000 samples
# (realizations) from the posterior distribution.
# In reservoir models each function evaluation (running the simulator) is expensive,
# so we have to make do with far fewer samples.
#
# This can be an issue even in small, simple problems:
# Suppose $A$ and $B$ are two uniform variables.
# What is the expected value of their product, i.e., $\mathbb{E}[AB]$ ?
#
# The answer is 1/4, and if we use 25 samples to
# estimate this quantity we get 0.266 on the first try.
# Pretty good!
# However, if we re-seed the random number generator and
# try again we get 0.200 as the result.
# A third seed produces 0.239, a fourth seed 0.226, etc.
#
# With less than 25 samples the results are even worse.
# In fact, the uncertainty (standard deviation) decreases asymptotically
# like $1/\sqrt{n}$, where $n$ is the number of samples.
# The asymptotic result holds for _any_ quantity that you wish to estimate,
#  but the constant differs depending on exactly what quantity you estimate.
# In the book Statistical Rethinking (section 9.5.1), McElreath writes:
#
# > If all you want are posterior means, it doesn't take
# > many samples at all to get very good estimates.
# > Even a couple hundred samples will do.
# > But if you care about the exact shape in the extreme tails of the posterior,
# > the 99th percentile or so, then you'll need many more.
#
# McElreath says a few hundred will do, and in most books and papers
# at least a thousand samples are used.
# The figure below shows the estimation of $\mathbb{E}[AB]$ as
# a function of the number of samples.
# Each done is one simulation study using $n$ samples.

# %%
n = 25
print(f"Computing E[A * B] with n={n} samples")
for experiment in range(5):
    A = rng.uniform(size=n)
    B = rng.uniform(size=n)
    E_AB = np.mean(A * B)
    print(f"  E[A * B] = {E_AB:.3f}")


plt.figure(figsize=(5, 3.0))
plt.title(r"Error decreases like $1/\sqrt{n}$")
for samples in range(1, 51):
    obs = np.mean(np.prod(rng.uniform(size=(2, samples, 10)), axis=0), axis=0)
    x = samples + rng.uniform(-0.5, 0.25, size=len(obs))
    plt.scatter(x, obs, color="black", s=5, alpha=0.8)


x = np.linspace(1, 51, num=2**10)
sigma = np.sqrt(7 / 144)  # Var[A*B] = E[A] * E[B] - E[A*B]

plt.plot(x, 0.25 + sigma / np.sqrt(x), color=COLORS[0], lw=2, alpha=0.8)
plt.plot(x, 0.25 - sigma / np.sqrt(x), color=COLORS[0], lw=2, alpha=0.8)

plt.xlabel("Number of samples $n$")
plt.ylabel("E[AB] : Expected value\nof product of uniforms")
plt.grid(True, ls="--", alpha=0.4)
plt.tight_layout()
# plt.savefig("sample_estimation_sqrt.png", dpi=200)

plt.show()


# %% [markdown]
# Even if ES and ESMDA were perfect algorithms that correctly sampled the
# posterior (they are not), we would still be bound by this law of statistics.

# %% [markdown]
# ## Lesson 3: Marginal distributions hide high-dimensional information
#
# Summary statistics like the expected value summarize
# information by collapsing samples to a single value.
# One remedy is to plot and inspect all samples, using for instance a histogram.
#
# However, histograms do not tell us anything about high
# dimensional phenomena such as correlations or other structure.
# The figure below shows three data set with identical marginals
# (therefore also identical summary statistics: mean, standard deviation, etc.).


# %%
def swap(X):
    X = X.copy()
    i, j = rng.choice(range(X.shape[1]), replace=False, size=2)
    assert i != j
    X[0, i], X[0, j] = X[0, j], X[0, i]
    return X


n = 100

lhs = sp.stats.qmc.LatinHypercube(d=2, rng=rng)
X = lhs.random(n).T


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(5.2, 2.2))
fig.suptitle("Three multivariate samples with identical (uniform) marginals", y=0.9)

# Plot uniform samples
ax1.scatter(*X.copy(), s=10)


best = sp.stats.pearsonr(*X).statistic
for iteration in range(999):
    X_proposed = swap(X)
    if sp.stats.pearsonr(*X_proposed).statistic > best:
        best = sp.stats.pearsonr(*X_proposed).statistic
        X = X_proposed


ax2.scatter(*X.copy(), s=10)


def error(X):
    return sp.spatial.distance.cdist(X, np.array([[0.5, 0.5]])).mean()


X = X.T
best = error(X)
for iteration in range(700):
    X_proposed = swap(X.T).T
    if error(X_proposed) > best:
        best = error(X_proposed)
        X = X_proposed


ax3.scatter(*X.T, s=10)

fig.tight_layout()
# plt.savefig("identical_marginals.png", dpi=200)
plt.show()


# %% [markdown]
# Plotting only reveals relationships in one dimension and two dimensions.
# In high dimensions it is hard to study the relationships between variables.

# %% [markdown]
# ## Lesson 4: ESMDA tends to deal with non-linearities better than ES
#
# Above we saw that ES and ESMDA are derived from the Gauss-linear case.
# The idea behind ESMDA is that several
# iterations can help deal with non-linearities.
#
# Here is a weakly non-linear problem is two dimensions.
# The first iteration takes us part of the way to the posterior,
# and the second iteration takes us closer.
# The true, analytical posterior is the intersection between the
# black line and the gaussian represented by the samples.
#


# %%
def plot_esmda(
    forward_model,
    iterations=2,
    seed=42,
    title=None,
    covar_scale=0.001,
    realizations=99,
    cov=None,
):
    """
    Plot ESMDA iterations for a 2D parameter space.
    """
    rng = np.random.default_rng(seed)

    # Vectorized version of forward model
    def F(X):
        return np.array([forward_model(x_i) for x_i in X.T]).T

    # Set up grid for contours
    x1 = np.linspace(-1, 2, 100)
    x2 = np.linspace(-1, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Evaluate forward_model on the grid
    Z = np.array(
        [
            [
                forward_model(np.array([x1_val, x2_val]))[0]
                for x1_val, x2_val in zip(row1, row2)
            ]
            for row1, row2 in zip(X1, X2)
        ]
    )

    # Number of columns = iterations + 1 (for prior)
    num_cols = iterations + 1
    fig_width = 2.5 * num_cols
    fig, axes = plt.subplots(
        1, num_cols, sharex=True, sharey=True, figsize=(fig_width, 2.5)
    )

    # Handle single subplot case
    if num_cols == 1:
        axes = np.array([axes])

    axes_iter = iter(axes.ravel())
    if title:
        fig.suptitle(title, y=0.925)

    # Set up ESMDA
    if cov is None:
        cov = np.eye(2) * 0.3
    X = rng.multivariate_normal([1, 1], cov=cov, size=realizations).T

    observations = np.zeros(1)
    covariance = np.diag([1]) * covar_scale

    esmda = ESMDA(covariance, observations, alpha=iterations, seed=rng)

    # Generate text labels dynamically
    texts = ["Prior"] + [f"Iteration {i + 1}" for i in range(iterations)]
    texts[-1] += " (posterior)"

    # Blue for first, orange for last
    colors = COLORS[:1] + COLORS[2:]
    colors[iterations] = COLORS[1]

    for i in range(esmda.num_assimilations() + 1):
        ax = next(axes_iter)
        ax.set_title(texts[i])
        ax.set_xlim([-1, 2])
        ax.set_ylim([-1, 2])
        ax.grid(True, ls="--", alpha=0.5)

        # All contours in gray with labels
        cs = ax.contour(X1, X2, Z, levels=10, cmap="Greys", alpha=0.7, vmin=-20, vmax=4)
        if i == 0:
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

        # Highlight f(x) = 0 line
        cs0 = ax.contour(X1, X2, Z, levels=[0], colors="black", linewidths=2)
        if i == 0:
            ax.clabel(cs0, inline=True, fontsize=8, fmt="%.1f")

        ax.scatter(*X, label="Prior" if i == 0 else "_nolegend_", s=10, color=colors[i])

        if i == iterations:
            break

        esmda.prepare_assimilation(Y=F(X))
        X = esmda.assimilate_batch(X=X)
        print("mean(X)", X.mean(axis=1))

        if i == 0:
            ax.legend()

    fig.tight_layout()
    return fig, axes


def forward_model(x):
    return np.abs(np.sum(x, keepdims=True)) ** 2 + x[0] * 5


# If the model is non-linear, two iterations of ESMDA
# will ensure that it converges, while one is not enough.
fig, axes = plot_esmda(
    forward_model,
    iterations=2,
    seed=42,
    title="A non-linear model: $f(x_1, x_2) = (x_1 + x_2)^2 + 5 x_1$",
    realizations=250,
    covar_scale=0.01,
)
# plt.savefig("non_linear_model_two_iters.png", dpi=200)
plt.show()


# %% [markdown]
# The first iteration above only takes us part-way because when we
# linearize a quadratic function, the linear approximation is
# a lower bound (the function is convex).
# With a concave function, such as a square-root, the opposite
# phenomenon occurs: ESMDA overshoots it the
# first iteration and corrects in the second.
#


# %%
def forward_model(x):
    summed = np.sum(x, keepdims=True)
    return np.abs(summed) ** 0.5 * np.sign(np.sum(x)) + x[0]


fig, axes = plot_esmda(
    forward_model,
    iterations=2,
    seed=42,
    title=r"A non-linear model: $f(x_1, x_2) ="
    r"\operatorname{sign}(x_1 + x_2)\sqrt{| x_1 + x_2 |} + x_1$",
    realizations=250,
    covar_scale=0.001,
    cov=[[0.2, 0], [0, 0.2]],
)
# plt.savefig("non_linear_model_overshoot.png", dpi=200)
plt.show()


# %% [markdown]
# In the figure above we observe that the posterior we obtain does not
# match the analytical answer.
# The analytical answer is the intersection between the black line
# and the gaussian represented by the samples.
# ESMDA places samples too high up on the line, while the true
# posterior has more probability around the bendy part of the black line.
# The reason is that ESMDA has a propensity to move along the
# major covariance axes in each iteration,
# so in the second iteration it prefers to move up.
# More on this in the next lesson!

# %% [markdown]
# ## Lesson 5: The update direction is determined by gradient, covariance and more
#
# In all examples above, ESMDA behaves a bit like
# optimization because it follows the gradient.
# However, ESMDA is not an optimization algorithm and
# should not be thought of as such.
# If we were solving an optimization problem we would not use ESMDA:
# optimization routines are better for optimization (duh!) -
# sampling a 2D function hundreds of times to optimize it should not be needed.
#
# ESMDA uses gradients, but it is also influenced by the covariance
# in the current ensemble members (the samples).
# This is shown in the figure below, where the update does not go to
# the origin (which is the point on the line closest to the prior mean).
# This result matches the theoretical posterior distribution
# (this problem is Gauss-linear and ESMDA solves it correctly).


# %%
def forward_model(x):
    return np.sum(x, keepdims=True)


# When the prior is not spherical,
# it "moves" in the direction of the principal axes.
# This corresponds to the analytical answer is the gauss-linear case
fig, axes = plot_esmda(
    forward_model,
    iterations=2,
    seed=42,
    title="A linear model: $f(x_1, x_2) = x_1 + x_2$",
    realizations=250,
    cov=[[0.5, 0], [0, 0.1]],
    covar_scale=0.01,
)
# plt.savefig("linear_model_ellipse_prior.png", dpi=200)
plt.show()


# %% [markdown]
# ## Lesson 6: Updates can oscillate, and more iterations is not always better
#
# Even on simple non-linear problems, ESMDA can produce
# embarressingly bad posteriors.
# After one iteration the spherical samples below contract to an ellipse,
# which influences the update direction.
# This produces oscillations that lead to posterior estimates that are
# worse (in expected value) for some parameters compared to what we began with.
# By tweaking parameters it's possible to produce strong oscillations,
# even in two dimensions with very many samples.
# Adding observation noise can help mitigate this
# effect by regularizing the updates,
# but at the cost of using a model we might not believe in.
#


# %%
def forward_model(x):
    return np.sum((x - np.array([-0.5, 1])) ** 2, keepdims=True) - 0.001


# On a simple quadratic, an optimization algorithm would immediately
# find the minimum. However, ESMDA struggles. It moves slowly
# due to the non-linearity, and as the ensemble adapts to the shape
# it tends to become unstable and oscillates.
fig, axes = plot_esmda(
    forward_model,
    iterations=3,
    seed=42,
    title="A quadratic model: $f(x_1, x_2) = (x_1 + 0.5)^2 + (x_2-1)^2$",
    covar_scale=0.01,
    realizations=500,
)
# plt.savefig("quadratic_model.png", dpi=200)
plt.show()


# %% [markdown]
# In short, even with hundreds of samples on a
# 2D problem that is a simple quadratic,
# running ESMDA can be worse than not running it.
#

# %% [markdown]
# ## Lesson 7: In high dimensions, everything is worse
#
# So far we studied two dimensions, with hundreds of samples.
# We've seen that on even weakly non-linear problems ESMDA produces
# posterior samples that are not correct.
# We've seen that running several iterations can help on non-linear problems,
# but it can also be worse than running a single iteration.
#
# In high dimensions, when the ratio of samples
# to dimensions is low, everything is worse:
#
# - The samples are likely ellipse-like (randomly correlated) in some direction,
#   because there are so many directions. This means ESMDA favors
#   updates in those random directions.
# - Estimating the gradient, which ESMDA implicitly does when it
#   computes cross-covariance, becomes harder.
#
# Both of these is due to spurious correlations.
# In addition to this, more parameters means more chance that some of the
# samples from the prior distributions do not match the theoretical distributions.
#

# %% [markdown]
# ### A high-dimensional, linear problem
#
# In high dimensions, we cannot visualize ESMDA any longer.
# To set the stage, we create a 100-dimensional linear problem
# $f(x_1, x_2, x_3, \ldots) = \sum_{i=1}^{100} x_i$,
# observe $y=0$ and place a prior on $x_i \sim N(\mu=1, \sigma=0.3)$.
#
# Then we count how many parameters improve,
# across 1000 experiments with different seeds:
#
# |   realizations |   perc_moved_correct |   post_dist_over_prior_dist |
# |---------------:|---------------------:|----------------------------:|
# |             10 |                 0.23 |                        3.5  |
# |             25 |                 0.37 |                        2.07 |
# |             50 |                 0.52 |                        1.41 |
# |            100 |                 0.69 |                        0.98 |
# |            200 |                 0.85 |                        0.69 |
# |            500 |                 0.98 |                        0.44 |
# |           1000 |                 1    |                        0.31 |
#
# A parameter is moved in the correct direction if its ESMDA posterior mean is
# closer to the true posterior mean than the ESMDA prior mean was.
# Note that the prior mean is subject to sampling randomness too: the mean of the
# normal distribution we sample from is not always close to the mean of the samples.
#
# The posterior distance over the prior distance measures
# if the mean moves closer to the true posterior.
#
# In summary:
#
# - With 50 realizations we have a 50% chance of improving parameters
#   by running ESMDA on this simple, linear example.
# - With 100 realizations we have a 50% chance of the posterior
#   mean moving in the correct direction.
#
# More iterations will not help either: with three ESMDA iterations
# the results are more or less exactly the same.

# %% [markdown]
# ### A high-dimensional, quadratic problem
#
# We create a 100-dimensional quadratic problem
# $f(x_1, x_2, x_3, \ldots) = \sum_{i=1}^{100} x_i^2$, observe $y=0$
# and place a prior on $x_i \sim N(\mu=1, \sigma=0.3)$.
#
# Here are the results from running a single ESMDA iteration:
#
# |   realizations |   perc_moved_correct |   post_dist_over_prior_dist |
# |---------------:|---------------------:|----------------------------:|
# |             10 |                 0.42 |                        1.86 |
# |             25 |                 0.61 |                        1.17 |
# |             50 |                 0.74 |                        0.87 |
# |            100 |                 0.84 |                        0.71 |
# |            200 |                 0.92 |                        0.61 |
# |            500 |                 0.99 |                        0.55 |
# |           1000 |                 1    |                        0.52 |
#
#
# With three iterations the results are worse, not better:
#
# |   realizations |   perc_moved_correct |   post_dist_over_prior_dist |
# |---------------:|---------------------:|----------------------------:|
# |             10 |                 0.12 |                        6.94 |
# |             25 |                 0.39 |                        2    |
# |             50 |                 0.67 |                        1.01 |
# |            100 |                 0.84 |                        0.7  |
# |            200 |                 0.89 |                        0.64 |
# |            500 |                 0.88 |                        0.65 |
# |           1000 |                 0.93 |                        0.59 |

# %% [markdown]
# ## Summary
#
# ESMDA and ES are pretty simple algorithms: the crudely move realizations in
# one direction, attempting to balance the prior with the observations.
# The algorithm is derived for the Gauss-linear case, and only holds in
# the limiting case of many samples.
# Furthermore, anything that is not Gaussian and linear
# comes with absolutely no guarantees.
#
# Visually we have seen that ESMDA does manage to crudely deal with some
# simple non-linearities (weakly non-linear, quadratic) problems in some sense.
# But we have also seen that it does not solve 2D non-linear problems,
# even with hundreds of samples or unlimited number of ESMDA iterations.
# No 2D problem shown here had posteriors that matched
# the analytical ones (expect the linear problems).
# In high dimensions another set of issues crop up and
# confound the understanding of the algorithm.
# Since a reservoir simulator is most certainly non-linear, with multiple minima,
# very high dimensional, etc - great care should be taken so
# that we do not end up studying random noise.


# %%
def moved_in_right_direction(realizations, iterations, seed, linear=True):
    # Create a high-dimensional problem
    dimensions = 100

    if linear:

        def forward_model(x):
            return np.sum(x, keepdims=True)  # + 1e-3 * np.sum(x**2, keepdims=True)
    else:

        def forward_model(x):
            return np.sum(x**2, keepdims=True)

    # Vectorized version of forward model
    def F(X):
        return np.array([forward_model(x_i) for x_i in X.T]).T

    covariance = np.ones(1) * 0.01
    observations = np.ones(1) * 0

    esmda = ESMDA(covariance, observations, alpha=iterations, seed=seed)

    X_prior = 1 + 0.3 * rng.random(size=(dimensions, realizations))
    assert np.all(np.mean(X_prior, axis=1) > 0)

    true_posterior = np.mean(X_prior, axis=1) * 0

    X = X_prior.copy()
    for _ in range(esmda.num_assimilations()):
        esmda.prepare_assimilation(Y=F(X))
        X = esmda.assimilate_batch(X=X)

    prior_mean = np.mean(X_prior, axis=1)
    posterior_mean = np.mean(X, axis=1)

    # import matplotlib.pyplot as plt
    # plt.scatter(prior_mean, posterior_mean)
    # plt.show()

    perc_moved = np.mean(np.abs(posterior_mean) < np.abs(prior_mean))
    relative_dist = np.linalg.norm(posterior_mean - true_posterior) / np.linalg.norm(
        prior_mean - true_posterior
    )
    return perc_moved, relative_dist


rows = []

seeds = iter(range(9999))
for realizations in [10, 25, 50, 100, 200, 500, 1000]:
    results = []
    results_dist = []
    for experiment in range(1000):
        perc_moved, dist = moved_in_right_direction(
            realizations, iterations=3, seed=next(seeds), linear=False
        )
        results.append(perc_moved)
        results_dist.append(dist)

        # print(realizations, experiment, dist)

    print(
        f"Realizations: {realizations}"
        f"Avg moved: {np.mean(results):.2f}"
        f"Avt dist: {np.mean(results_dist):.3f}"
    )

    rows.append(
        {
            "realizations": realizations,
            "percent_moved_in_correct_direction": np.mean(results),
            "posterior_distance_over_prior_distance": np.mean(results_dist),
        }
    )


print(pd.DataFrame(rows).round(2).to_markdown(index=False))
