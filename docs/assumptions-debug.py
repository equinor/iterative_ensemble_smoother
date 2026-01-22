
import numpy as np
import scipy as sp
from iterative_ensemble_smoother import ESMDA
import matplotlib.pyplot as plt

prop_cycle = plt.rcParams["axes.prop_cycle"]
COLORS = prop_cycle.by_key()["color"]


rng = np.random.default_rng(42)





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
    realizations = 100
    X = 1 + 0.3 * rng.normal(size=(2, realizations))
    observations = np.zeros(1)
    covariance = np.diag([1]) * covar_scale
    esmda = ESMDA(
        covariance, observations, 
        alpha=iterations, 
        seed=rng, inversion="exact"
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
    return np.sum((x - np.array([0, 0])) ** 2, keepdims=True) - 0.001

# def forward_model(x):
#     return np.sum((x - np.array([-0.5, 1])), keepdims=True)


#fig, axes = plot_esmda(forward_model, iterations=1, seed=42, title="A non-linear model")
#fig, axes = plot_esmda(forward_model, iterations=2, seed=42, title="A non-linear model")
fig, axes = plot_esmda(forward_model, iterations=3, seed=42, title="A non-linear model",
                       covar_scale=0.001)

#fig, axes = plot_esmda(forward_model, iterations=8, seed=42, title="A non-linear model")
