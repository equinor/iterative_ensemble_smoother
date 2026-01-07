import numpy as np
import scipy as sp
from iterative_ensemble_smoother import ESMDA
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n = 10
print(f"Computing E[A * B] with n={n} samples")
for experiment in range(5):
    A = rng.uniform(size=n)
    B = rng.uniform(size=n)
    E_AB = np.mean(A * B)
    print(f"  E[A * B] = {E_AB:.3f}")
    
    
# =================================================

def swap(X):
    X = X.copy()
    i, j = rng.choice(range(X.shape[1]), replace=False, size=2)
    assert i != j
    X[0, i], X[0, j] = X[0, j], X[0, i]
    return X
    


n = 100

lhs = sp.stats.qmc.LatinHypercube(d=2, rng=rng)
X = lhs.random(n).T


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True,
                                    figsize=(5.2, 2.2))
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
plt.show()
# =================================================
    
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
Z = np.array([[f(np.array([x1_val, x2_val]))[0] 
               for x1_val, x2_val in zip(row1, row2)] 
              for row1, row2 in zip(X1, X2)])


fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 2.5))
axes = iter(axes.ravel())
fig.suptitle("A linear model with different levels of observation noise", y=0.925)

realizations = 100
X_prior = 1 + 0.3 * rng.normal(size=(2, realizations))
observations = np.zeros(1)

texts = ["High", "Medium", "Low"]
for i, covariance_factor in enumerate([0.1, 0.01, 0.001]):
    
    ax = next(axes)
    ax.set_title(texts[i])
    covariance = np.diag([1]) * covariance_factor
    
    # All contours in gray with labels
    cs = ax.contour(X1, X2, Z, levels=10, cmap='Greys', alpha=0.7, vmin=-20, vmax=4)
    if i == 0:
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    # Highlight f(x) = 0 line
    cs0 = ax.contour(X1, X2, Z, levels=[0], colors='black', linewidths=2)
    if i == 0:
        ax.clabel(cs0, inline=True, fontsize=8, fmt='%.1f')

    ax.scatter(*X_prior, label='Prior', s=10)
    
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    ax.grid(True, ls="--", alpha=0.5)

    esmda = ESMDA(covariance, observations, alpha=1, seed=rng)
    X = X_prior.copy()
    for _ in range(esmda.num_assimilations()):
        X = esmda.assimilate(X, F(X))
        
    ax.scatter(*X, label='Posterior', s=10)
    if i == 0:
        ax.legend()
    
    
fig.tight_layout()


# =================================================
    
# A simple linear model
def f(x):
    #x1, x2 = x
    
    # return np.array([x1**2 + x2**2])
    return np.abs(np.sum(x, keepdims=True))**2 + x[0] * 5
def F(X):
    return np.array([f(x_i) for x_i in X.T]).T


x1 = np.linspace(-1, 2, 100)
x2 = np.linspace(-1, 2, 100)
X1, X2 = np.meshgrid(x1, x2)

# Evaluate f on the grid
Z = np.array([[f(np.array([x1_val, x2_val]))[0] 
               for x1_val, x2_val in zip(row1, row2)] 
              for row1, row2 in zip(X1, X2)])


fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 2.5))
axes = iter(axes.ravel())
fig.suptitle("A non-linear model", y=0.925)

realizations = 100
X = 1 + 0.3 * rng.normal(size=(2, realizations))
observations = np.zeros(1)
covariance = np.diag([1]) * 0.001
esmda = ESMDA(covariance, observations, alpha=2, seed=rng, inversion="exact")

texts = ["Prior", "One iteration", "Two iterations"]
for i in range(esmda.num_assimilations() + 1):

    ax = next(axes)
    ax.set_title(texts[i])
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    ax.grid(True, ls="--", alpha=0.5)

    
    # All contours in gray with labels
    cs = ax.contour(X1, X2, Z, levels=10, cmap='Greys', alpha=0.7, vmin=-20, vmax=4)
    if i == 0:
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    # Highlight f(x) = 0 line
    cs0 = ax.contour(X1, X2, Z, levels=[0], colors='black', linewidths=2)
    if i == 0:
        ax.clabel(cs0, inline=True, fontsize=8, fmt='%.1f')

    ax.scatter(*X, label='Prior', s=10)
    
    if i == 2:
        break
    
    X = esmda.assimilate(X, F(X))
    print("mean(X)", X.mean(axis=1))
        
    if i == 0:
        ax.legend()
    
    
fig.tight_layout()