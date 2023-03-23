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
    Cdd = E @ E.T
    
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
m = 2  # number of observations
d_sd = 0.5
d = np.array([np.random.normal(0.0, d_sd) for _ in range(m)])
errors = np.full(d.shape, d_sd)
errors

# %%
R = np.diag(errors**2)

# %%
obs_sd = np.sqrt(R.diagonal())

# %%
(R.T / R.diagonal()).T

# %%
R

# %%
d.reshape(-1,1)

# %%
rng = np.random.default_rng()
m = 2
N=5
Z = rng.standard_normal(size=(m, N))
I = np.identity(N)
Ones_NxN = np.ones((N,N))
Ones_Nx1 = np.ones((N,1))
mu = Z @ Ones_Nx1 / N
Z

# %%
Z @ (I - Ones_NxN/N) # centering with mean

# %%
mu

# %%
Z - mu

# %%
(Z-mu) @ (Z-mu).T / (N-1) # empirical cov matrix

# %%
pert_var = ((Z-mu) * (Z-mu)) @ Ones_Nx1

# %%
pert_var

# %%
((Z-mu) * (Z-mu))

# %%

# %%

# %%
mu @ np.ones((1,N))

# %%
