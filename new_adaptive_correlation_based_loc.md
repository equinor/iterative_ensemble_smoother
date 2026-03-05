## 2026: Why indexing matters in adaptive localization, and why tapering is better than hard thresholding

Femke Vossepoel, Geir Evensen and Peter Jan Van Leeuwen published a paper in 2025 titled:
`Adaptive Correlation- and Distance Based Localization for Iterative Ensemble Smoothers in a Couple Nonlinear Multiscale Model`.
In it they describe a method similar to our adaptive correlation based localization method.
One key aspect is that they invert only a subset of responses and observations when updating a parameter (indexing), which is also what our implementation does.
They also introduce tapering via variance inflation, which our implementation does not currently do.
Below are two simple analytical examples that illustrate (1) why indexing is the correct approach, and (2) why tapering is preferable to hard thresholding.

Let's set up a system with **1 parameter** (state variable $x$) and **2 observations** ($y_1$ and $y_2$).

* Assume $y_1$ is a "local" observation (physically close to $x$).
* Assume $y_2$ is a "far" observation that we want to localize away (we want it to have zero impact on $x$).

### The Setup

Let's define the terms in your matrices:

* **Innovations:** $\Delta = \begin{pmatrix} \delta_1 \\ \delta_2 \end{pmatrix}$
* **State-Obs Covariance ($XY^T$):** $\begin{pmatrix} c_1 & c_2 \end{pmatrix}$ where $c_1 = Cov(x,y_1)$ and $c_2 = Cov(x,y_2)$.
* **Obs-Obs Covariance ($YY^T$):** $\begin{pmatrix} v_1 & c_{12} \\ c_{12} & v_2 \end{pmatrix}$ where $v_1, v_2$ are variances, and $c_{12} = Cov(y_1, y_2)$. Because of limited ensemble size, $c_{12}$ is usually non-zero (spurious correlation).
* **Obs Error ($R$):** $\begin{pmatrix} r_1 & 0 \\ 0 & r_2 \end{pmatrix}$
* **Localization ($\rho$):** Because we want to cut off $y_2$, our localization array for the state variable $x$ is $\rho = \begin{pmatrix} 1 & 0 \end{pmatrix}$.

Let $S = YY^T + R = \begin{pmatrix} v_1 + r_1 & c_{12} \\ c_{12} & v_2 + r_2 \end{pmatrix}$.
The determinant of this matrix is $|S| = (v_1+r_1)(v_2+r_2) - c_{12}^2$.

---

### Method 1: No Indexing

The update formula is: **$\Delta x = (\rho \circ XY^T) S^{-1} \Delta$**

**Step 1: Localize the State-Obs Covariance**
We multiply $XY^T$ by $\rho$ element-wise:
$\rho \circ XY^T = \begin{pmatrix} 1 \cdot c_1 & 0 \cdot c_2 \end{pmatrix} = \begin{pmatrix} c_1 & 0 \end{pmatrix}$
*(Notice how we successfully zeroed out the direct link to $y_2$)*.

**Step 2: Invert the Global $S$ Matrix**
Using the analytical formula for a 2x2 inverse:
$S^{-1} = \frac{1}{|S|} \begin{pmatrix} v_2 + r_2 & -c_{12} \\ -c_{12} & v_1 + r_1 \end{pmatrix}$

**Step 3: Multiply $(\rho \circ XY^T)$ by $S^{-1}$**
$\begin{pmatrix} c_1 & 0 \end{pmatrix} \frac{1}{|S|} \begin{pmatrix} v_2 + r_2 & -c_{12} \\ -c_{12} & v_1 + r_1 \end{pmatrix} = \frac{1}{|S|} \begin{pmatrix} c_1(v_2+r_2) & -c_1 c_{12} \end{pmatrix}$

**Step 4: Multiply by the Innovations to get the final update**
$\Delta x = \frac{1}{|S|} \begin{pmatrix} c_1(v_2+r_2) & -c_1 c_{12} \end{pmatrix} \begin{pmatrix} \delta_1 \\ \delta_2 \end{pmatrix}$

**Analytical Result (No Indexing):**


Here is the simplified version:

$$\Delta x = \frac{c_1(v_2+r_2)}{|S|} \delta_1 - \frac{c_1 c_{12}}{|S|} \delta_2$$

**The Problem:** The update still depends on the far observation's innovation ($\delta_2$) and variance ($v_2$), which it should not. This happens because the spurious correlation ($c_{12}$) in the unlocalized $YY^T$ matrix allows their influence to leak into the local update.

---

### Method 2: Indexing (Local Analysis — our implementation)

In this method (which is what our implementation uses), we look at $\rho$, see that $y_2$ is a "far" observation (weight 0), and we physically delete it from all matrices *before* doing any math.

We are left with 1 parameter and 1 local observation ($y_1$).

* New Innovation: $\delta_1$
* New $XY^T = c_1$
* New $YY^T = v_1$
* New $R = r_1$

Your formula becomes: **$\Delta x = XY^T (YY^T + R)^{-1} \Delta$**

**Analytical Result (Indexing):**


$$\Delta x = c_1 (v_1 + r_1)^{-1} \delta_1 = \frac{c_1}{v_1 + r_1} \delta_1$$

---

### The Conclusion

Compare the two results:

1. **No Indexing:** $\Delta x = \frac{c_1(v_2+r_2)}{|S|} \delta_1 - \frac{c_1 c_{12}}{|S|} \delta_2$
2. **Indexing:** $\Delta x = \frac{c_1}{v_1 + r_1} \delta_1$

By extracting the subset of local observations (indexing), $y_2$ is completely removed. There is no $\delta_2$, and the spurious correlation $c_{12}$ vanishes.

If you do not index, the inversion of the global $YY^T$ matrix mixes the observations together. The local observation "absorbs" the error from the far observation, and your mathematical update gets contaminated.

### Tapering

Femke uses variance inflation as a method of tapering:

"Thus, to ensure smooth updates, we introduce tapering by infating the errors of the most distant observations"

Our implementation of adaptive correlation based localization uses hard thresholding without tapering.

---

## Analytical example: Hard thresholding vs Tapering (2 parameters, 2 observations)

The previous example showed why indexing matters. This example assumes we use **Method 2 (indexing)** and shows why **tapering** is better than **hard thresholding** (binary 0/1 localization weights).

### The Setup

Consider a system with **2 parameters** ($x_1$, $x_2$) and **2 observations** ($y_1$, $y_2$).

* $x_1$ and $x_2$ are **spatially close neighbors** in the parameter field.
* $y_1$ is close to both parameters (high correlation with both).
* $y_2$ is at a moderate distance such that the sample correlations with the two parameters are almost the same, straddling the threshold $\tau$:
  * $|\hat{\text{corr}}(x_1, y_2)|$ is **just above** $\tau$ → hard threshold keeps it.
  * $|\hat{\text{corr}}(x_2, y_2)|$ is **just below** $\tau$ → hard threshold removes it.

Define:

* **Cross-covariance:** $XY^T = \begin{pmatrix} c_{11} & c_{12} \\ c_{21} & c_{22} \end{pmatrix}$ where $c_{ij} = \widehat{Cov}(x_i, y_j)$.
* **$S = YY^T + R = \begin{pmatrix} s_{11} & s_{12} \\ s_{12} & s_{22} \end{pmatrix}$**
* **Innovations:** $\Delta = \begin{pmatrix} \delta_1 \\ \delta_2 \end{pmatrix}$

**Assumption:** Since $x_1$ and $x_2$ are spatial neighbors, we assume their covariances with each observation are nearly identical: $c_{11} \approx c_{21}$ and $c_{12} \approx c_{22}$.

With indexing, each parameter $x_i$ builds its own **local analysis** by selecting only the observations with non-zero localization weight, then solves an independent update problem.

---

### Hard Thresholding

With a threshold $\tau$, the localization weight is binary:

$$\rho_{ij}^{\text{hard}} = \begin{cases} 1 & \text{if } |\hat{\text{corr}}(x_i, y_j)| \geq \tau \\ 0 & \text{if } |\hat{\text{corr}}(x_i, y_j)| < \tau \end{cases}$$

Given our setup ($x_1$ barely above threshold for $y_2$, $x_2$ barely below):

$$\rho^{\text{hard}} = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}$$

**Update for $x_1$** — active set: $\{y_1, y_2\}$ (both observations kept):

$$\Delta x_1 = \begin{pmatrix} c_{11} & c_{12} \end{pmatrix} \begin{pmatrix} s_{11} & s_{12} \\ s_{12} & s_{22} \end{pmatrix}^{-1} \begin{pmatrix} \delta_1 \\ \delta_2 \end{pmatrix}$$

This is a full **2-observation system**.

**Update for $x_2$** — active set: $\{y_1\}$ ($y_2$ is physically removed):

$$\Delta x_2 = c_{21} \cdot s_{11}^{-1} \cdot \delta_1 = \frac{c_{21}}{s_{11}} \delta_1$$

This is a **1-observation system**.

**The problem:** Despite $x_1$ and $x_2$ being spatial neighbors with nearly identical covariance structures, their updates are computed from **structurally different linear systems** — one uses a 2×2 matrix inversion, the other a scalar inversion. The 2×2 inversion decorrelates $y_1$ and $y_2$, which changes the effective weight given to $\delta_1$ (and adds an update from $\delta_2$). An infinitesimal difference in sample correlation — one barely above $\tau$, the other barely below — causes a **finite jump** in the updated field between neighboring parameters.

---

### Tapering via Variance Inflation

Instead of a binary keep/discard decision, we assign each parameter–observation pair a continuous weight $\alpha_{ij} \in [0,\, 1]$ based on the sample correlation. A simple choice is a linear ramp between two thresholds $\tau_0$ and $\tau_1$:

$$\alpha_{ij} = \begin{cases} 1 & \text{if } |\hat{\text{corr}}(x_i, y_j)| \geq \tau_1 \\ \displaystyle\frac{|\hat{\text{corr}}(x_i, y_j)| - \tau_0}{\tau_1 - \tau_0} & \text{if } \tau_0 < |\hat{\text{corr}}(x_i, y_j)| < \tau_1 \\ 0 & \text{if } |\hat{\text{corr}}(x_i, y_j)| \leq \tau_0 \end{cases}$$

Observations with $\alpha_{ij} = 0$ are removed (indexing). For the remaining observations with $\alpha_{ij} \in (0,\, 1]$, variance inflation implements tapering by replacing the observation error $r_j$ with $r_j / \alpha_{ij}$. As $\alpha_{ij} \to 0$, the inflated error $r_j / \alpha_{ij} \to \infty$, which smoothly suppresses the observation's influence — approaching the same result as physically removing it.

In our example, both $y_1$ weights are 1 (high correlation). For $y_2$, since $x_1$ and $x_2$ have nearly the same sample correlation, they get nearly equal weights: $\alpha_{12} \approx \alpha_{22}$. Denote these $\alpha_1$ and $\alpha_2$ for brevity.

**Update for $x_1$** — inflated error $\tilde{s}_{22}^{(1)} = v_2 + r_2/\alpha_1$:

$$\Delta x_1 = \begin{pmatrix} c_{11} & c_{12} \end{pmatrix} \begin{pmatrix} s_{11} & s_{12} \\ s_{12} & \tilde{s}_{22}^{(1)} \end{pmatrix}^{-1} \begin{pmatrix} \delta_1 \\ \delta_2 \end{pmatrix}$$

**Update for $x_2$** — inflated error $\tilde{s}_{22}^{(2)} = v_2 + r_2/\alpha_2$:

$$\Delta x_2 = \begin{pmatrix} c_{21} & c_{22} \end{pmatrix} \begin{pmatrix} s_{11} & s_{12} \\ s_{12} & \tilde{s}_{22}^{(2)} \end{pmatrix}^{-1} \begin{pmatrix} \delta_1 \\ \delta_2 \end{pmatrix}$$

Since $\alpha_1 \approx \alpha_2$, we have $\tilde{s}_{22}^{(1)} \approx \tilde{s}_{22}^{(2)}$, which means both parameters solve **nearly identical** 2×2 systems. Combined with $c_{11} \approx c_{21}$ and $c_{12} \approx c_{22}$, the updates are nearly equal: $\Delta x_1 \approx \Delta x_2$.

---

### Comparison

|  | **Hard Thresholding** | **Tapering (Variance Inflation)** |
|---|---|---|
| System for $x_1$ | 2×2 (both observations) | 2×2 (inflated error on $y_2$) |
| System for $x_2$ | 1×1 (only $y_1$) | 2×2 (slightly more inflated error on $y_2$) |
| $\Delta x_1 \approx \Delta x_2$? | **No** — structurally different systems | **Yes** — nearly identical systems |
| Spatial continuity | **Broken** at the threshold boundary | **Preserved** |

Hard thresholding with indexing introduces **artificial discontinuities** in the updated parameter field: two spatially adjacent parameters, whose sample correlations with $y_2$ differ by an infinitesimal amount straddling the threshold, end up solving structurally different update equations. Tapering via variance inflation ensures that both parameters solve the same type of system with nearly identical inputs, preserving the physical continuity we expect in the parameter field.
