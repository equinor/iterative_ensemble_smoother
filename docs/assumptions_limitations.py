import numpy as np
import scipy as sp
from iterative_ensemble_smoother import ESMDA
import matplotlib.pyplot as plt

prop_cycle = plt.rcParams["axes.prop_cycle"]
COLORS = prop_cycle.by_key()["color"]


rng = np.random.default_rng(42)

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
plt.savefig("sample_estimation_sqrt.png", dpi=200)

plt.show()


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
plt.savefig("identical_marginals.png", dpi=200)
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
Z = np.array(
    [
        [f(np.array([x1_val, x2_val]))[0] for x1_val, x2_val in zip(row1, row2)]
        for row1, row2 in zip(X1, X2)
    ]
)


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
        X = esmda.assimilate(X, F(X))

    ax.scatter(*X, label="Posterior", s=10)
    if i == 0:
        ax.legend()


fig.tight_layout()
plt.savefig("linear_model_obs_noise.png", dpi=200)
plt.show()


# =================================================
def plot_esmda(forward_model, iterations=2, seed=42, title=None, covar_scale=0.001):
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
    realizations = 99
    X = 1 + 0.3 * rng.normal(size=(2, realizations))
    observations = np.zeros(1)
    covariance = np.diag([1]) * covar_scale
    esmda = ESMDA(
        covariance, observations, alpha=iterations, seed=rng, inversion="exact"
    )

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

        X = esmda.assimilate(X, F(X))
        print("mean(X)", X.mean(axis=1))

        if i == 0:
            ax.legend()

    fig.tight_layout()
    return fig, axes


def forward_model(x):
    return np.abs(np.sum(x, keepdims=True)) ** 2 + x[0] * 5


fig, axes = plot_esmda(
    forward_model,
    iterations=2,
    seed=42,
    title="A non-linear model: $f(x_1, x_2) = (x_1 + x_2)^2 + 5 x_1$",
)
plt.show()

fig, axes = plot_esmda(
    forward_model,
    iterations=1,
    seed=42,
    title="A non-linear model: $f(x_1, x_2) = (x_1 + x_2)^2 + 5 x_1$",
)
plt.savefig("non_linear_model_overshoot.png", dpi=200)
plt.show()


def forward_model(x):
    summed = np.sum(x, keepdims=True)

    return np.abs(summed) ** 0.5 * np.sign(np.sum(x)) + x[0]


fig, axes = plot_esmda(
    forward_model,
    iterations=2,
    seed=42,
    title=r"A non-linear model: $f(x_1, x_2) = \operatorname{sign}(x_1 + x_2)\sqrt{| x_1 + x_2 |} + x_1$",
)
plt.show()

fig, axes = plot_esmda(
    forward_model,
    iterations=1,
    seed=42,
    title=r"A non-linear model: $f(x_1, x_2) = \operatorname{sign}(x_1 + x_2)\sqrt{| x_1 + x_2 |} + x_1$",
)
plt.savefig("non_linear_model_overshoot.png", dpi=200)
plt.show()


def forward_model(x):
    return np.sum((x - np.array([-0.5, 1])) ** 2, keepdims=True) - 0.001


def forward_model(x):
    return (np.sum(x, keepdims=True) - 0) ** 2


fig, axes = plot_esmda(forward_model, iterations=1, seed=42, title="A non-linear model")

fig, axes = plot_esmda(forward_model, iterations=2, seed=42, title="A non-linear model")

fig, axes = plot_esmda(
    forward_model, iterations=5, seed=42, title="A non-linear model", covar_scale=1e-8
)

for ax in axes:
    ax.plot([-1, 1], [1, -1], color="black")


1 / 0


def forward_model(x):
    return np.sum((x - np.array([-0.5, 1])) ** 2, keepdims=True) - 0.1


fig, axes = plot_esmda(forward_model, iterations=1, seed=42, title="A non-linear model")
fig, axes = plot_esmda(forward_model, iterations=2, seed=42, title="A non-linear model")
fig, axes = plot_esmda(forward_model, iterations=3, seed=42, title="A non-linear model")


def forward_model(x):
    return np.sum((x - np.array([0.5, 1])) ** 2, keepdims=True) - 0.5


fig, axes = plot_esmda(forward_model, iterations=1, seed=42, title="A non-linear model")
fig, axes = plot_esmda(forward_model, iterations=2, seed=42, title="A non-linear model")
