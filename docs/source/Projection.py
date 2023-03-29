# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams.update({'font.size': 14})
develop = False
plt.rcParams['text.usetex'] = True

import itertools

from typing import Callable
import numpy.typing as npt


# %%
def subspace_ies(
    X: npt.NDArray[np.double], 
    D: npt.NDArray[np.double],
    g: Callable[[npt.NDArray[np.double]], float],
    linear: bool = False,
    step_size: float = 0.3, 
    iterations: int = 2
) -> npt.NDArray[np.double]:
    """Updates ensemble of parameters according to the Subspace 
    Iterative Ensemble Smoother (Evensen 2019).
    
    :param X: sample from prior parameter distribution, i.e. ensemble.
    :param D: observations perturbed with noise having observation uncertainty.
    :param g: the forward model g:Re**parameter_size -> R^m
    :param linear: if g is a linear forward model.
    :param step_size: the step size of an ensemble-weight update at each iteration.
    :param iterations: number of iterations in the udpate algorithm.
    """
    parameters, realizations = X.shape
    projection = not linear and parameters < realizations
    #if(projection):
        #print("Using projection matrix")
    m = D.shape[0]
    W = np.zeros((realizations, realizations))
    Xi = X.copy()
    I = np.identity(realizations)
    centering_matrix = (
        I - 
        np.ones((realizations,realizations))/realizations
    )/np.sqrt(realizations-1)
    E = D @ centering_matrix
    for i in range(iterations):
        gXi = np.array([g(parvec) for parvec in Xi.T]).reshape(realizations,m).T
        Y = gXi @ centering_matrix
        if projection:
            Ai = Xi @ centering_matrix ## NB: Using Xi not X
            projection_matrix = np.linalg.pinv(Ai) @ Ai
            Y = Y @ projection_matrix
        Ohmega = I + W @ centering_matrix
        S = np.linalg.solve(Ohmega.T, Y.T).T
        H = S @ W + D - gXi
        W = W - step_size * (
            W - S.T @
            np.linalg.inv(S@S.T + E@E.T) @
            H)
        Xi = X @ (I + W/np.sqrt(realizations-1))
        #print(np.mean(Xi, axis=1))
    return Xi


# %%
def g(x: npt.NDArray[np.double]) -> float:
    # x is a scalar
    x1 = x[0]
    return x1 + 0.2*x1**3


# %%
def loss_function(xj, xj_prior, dj, Cxx, Cdd, g):
    # Equation 10 in Evensen 2019
    return 0.5 * (
        (xj-xj_prior).T @ np.linalg.inv(Cxx) @ (xj-xj_prior) + 
        (g(xj)-dj).T @ np.linalg.inv(Cdd) @ (g(xj)-dj)
    )


# %%
res = []
for _ in range(1000):
    # Generate data
    #rng = np.random.default_rng(12345)
    realizations = 30
    # sample x
    x1 = -1.0
    x1_sd = 1.0
    x = np.array([x1])
    bias = 1
    # Define prior
    X = np.array([
        np.random.normal(x[0]+bias, x1_sd, size=(1,realizations)), 
    ]).reshape(1,realizations)

    # Define observations with very little uncertainty
    d_sd = 1.0
    D = np.random.normal(g(x), d_sd, size=(1,realizations))
    
    # Find solutions with and without projection
    iterations = 1
    Xi_proj = subspace_ies(X, D, g, False, 0.3, iterations) # with projection
    Xi_no_proj = subspace_ies(X, D, g, True, 0.3, iterations) # without projection
    
    centering_matrix = (
        np.identity(realizations) - 
        np.ones((realizations,realizations))/realizations
    )/np.sqrt(realizations-1)
    A = X @ centering_matrix
    E = D @ centering_matrix
    Cxx = A @ A.T
    #Cdd = E @ E.T
    Cdd = np.diag(np.array([d_sd for _ in range(m)]))
    
    loss_proj = [loss_function(Xi_proj[:,j], X[:,j], D[:,j], Cxx, Cdd, g) for j in range(realizations)]
    loss_no_proj = [loss_function(Xi_no_proj[:,j], X[:,j], D[:,j], Cxx, Cdd, g) for j in range(realizations)]
    
    # is projection solution better?
    res.append(np.sum(loss_proj) < np.sum(loss_no_proj))

# %%
all(res)

# %%
# Find solutions with and without projection
iterations = 1
Xi_proj = subspace_ies(X, D, g, False, 0.3, iterations) # with projection
Xi_no_proj = subspace_ies(X, D, g, True, 0.3, iterations) # without projection

# %%
centering_matrix = (
    np.identity(realizations) - 
    np.ones((realizations,realizations))/realizations
)/np.sqrt(realizations-1)
A = X @ centering_matrix
E = D @ centering_matrix
Cxx = A @ A.T
Cdd = E @ E.T

# %%
loss_proj = [loss_function(Xi_proj[:,j], X[:,j], D[:,j], Cxx, Cdd, g) for j in range(realizations)]
loss_no_proj = [loss_function(Xi_no_proj[:,j], X[:,j], D[:,j], Cxx, Cdd, g) for j in range(realizations)]

# %%
print(np.sum(loss_proj))
print(np.sum(loss_no_proj))
# is projection solution better?
np.sum(loss_proj) < np.sum(loss_no_proj)

# %%
plot = plt.figure()
plt.scatter(loss_proj, loss_no_proj)
plt.show()


# %%
# implementation using ies
class NonLinearModel:
    """Example 5.1 from Evensen 2019"""

    def __init__(self, x):
        self.x = x
        
    @classmethod
    def g(cls, x):
        beta = 0.2
        return x[0] + beta * x[0]**3

    @classmethod
    def simulate_prior(cls, x_prior_mean, x_prior_sd):
        return cls(
            np.array([rng.normal(x_prior_mean, x_prior_sd)]),
        )

    def eval(self):
        return self.g(self.x)


# %%
true_model = NonLinearModel(np.array([1.7]))
print(true_model.x)
true_model.eval()
NonLinearModel.g(np.array([1.6]))

# %%
res = []
for _ in range(100):
    N=100
    x_true = -1.0
    x_sd = 1.0
    bias = 0.5

    # define observations
    m = 1  # number of observations
    d_sd = 1.0
    true_model = NonLinearModel(np.array([x_true]))
    d = np.array([true_model.eval() + np.random.normal(0.0, d_sd) for _ in range(m)])
    d_sd = np.full(d.shape, d_sd)
    Cdd = np.diag(d_sd**2)
    noise_standard_normal = rng.standard_normal(size=(m, N))
    D = d + np.linalg.cholesky(Cdd) @ noise_standard_normal

    # define prior ensemble
    ensemble = [NonLinearModel.simulate_prior(x_true + bias, x_sd) for _ in range(N)]

    X_prior = np.array(
        [realization.x for realization in ensemble],
    ).reshape(m,N)
    Y = np.array([realization.eval() for realization in ensemble]).reshape(m,N)

    import iterative_ensemble_smoother as ies
    # Property holds for small step-size and one iteration.
    # Likely also holds for infinite iterations, or at convergence,
    # but then for infinitessimal stepsize
    step_length = 0.1
    # find solutions with and without projection
    model_projection = ies.SIES(N)
    model_projection.fit(
        Y,
        d_sd,
        d,
        truncation=1.0,
        step_length=step_length,
        param_ensemble=X_prior,
    )
    X_posterior_projection = model_projection.update(X_prior)
    model_no_projection = ies.SIES(N)
    model_no_projection.fit(
        Y,
        d_sd,
        d,
        truncation=1.0,
        step_length=step_length,
    )
    X_posterior_no_projection = model_no_projection.update(X_prior)

    # evaluate solutions through loss functions. Equation 10 of Evensen 2019
    def loss_function(xj, xj_prior, dj, Cxx, Cdd, g):
        # Equation 10 in Evensen 2019
        return 0.5 * (
            (xj - xj_prior).T @ np.linalg.inv(Cxx) @ (xj - xj_prior)
            + (g(xj) - dj).T @ np.linalg.inv(Cdd) @ (g(xj) - dj)
        )

    # Assert projection solution better than no-projection
    # need perturbed observations dj
    # but D matrix is "hidden"?
    # SIES creates R and observations_errors through _create_errors()
    # observation_errors will be observation standard deviations, 1d array (diagonal of cov)
    # R will be Correlation matrix with ones on diagonal
    centering_matrix = (np.identity(N) - np.ones((N, N)) / N) / np.sqrt(N - 1)
    A = X_prior @ centering_matrix
    Cxx = A @ A.T
    #Y_posterior_projection = np.array([NonLinearModel.g(xj) for xj in X_posterior_projection.T]).reshape(m,N)
    #Y_posterior_no_projection = np.array([NonLinearModel.g(xj) for xj in X_posterior_no_projection.T]).reshape(m,N)

    loss_proj = [
        loss_function(
            X_posterior_projection[:, j],
            X_prior[:, j],
            D[:, j],
            Cxx,
            Cdd,
            NonLinearModel.g,
        )
        for j in range(N)
    ]
    loss_no_proj = [
        loss_function(
            X_posterior_no_projection[:, j],
            X_prior[:, j],
            D[:, j],
            Cxx,
            Cdd,
            NonLinearModel.g,
        )
        for j in range(N)
    ]
    res.append(np.sum(loss_proj) < np.sum(loss_no_proj))

# %%
all(res)

# %%
