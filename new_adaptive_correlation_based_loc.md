## 2026: Potential bug in our implementation of adaptive localization

Femke Vossepoel, Geir Evensen and Peter Jan Van Leeuwen published a paper in 2025 titled:
`Adaptive Correlation- and Distance Based Localization for Iterative Ensemble Smoothers in a Couple Nonlinear Multiscale Model`.
In it they describe a method similar to our adaptive correlation based localization method, but not identical.
One difference is that they invert a subset of responses and observations when updating a parameter.
In our implementation, we always use all responses and observations.
We believe that their new method is better and here is a simple analytical example that illustrates why.

Let's set up a system with **1 parameter** (state variable $x$) and **2 observations** ($y_1$ and $y_2$).

* Assume $y_1$ is a "local" observation (physically close to $x$).
* Assume $y_2$ is a "far" observation that we want to localize away (we want it to have zero impact on $x$).

### The Setup

Let's define the terms in your matrices:

* **Innovations:**

$$\Delta = \begin{pmatrix} \delta_1 \\\\ \delta_2 \end{pmatrix}$$

* **State-Obs Covariance** ($XY^T$):

$$\begin{pmatrix} c_1 & c_2 \end{pmatrix}$$

  where $c_1 = Cov(x,y_1)$ and $c_2 = Cov(x,y_2)$.

* **Obs-Obs Covariance** ($YY^T$):

$$\begin{pmatrix} v_1 & c_{12} \\\\ c_{12} & v_2 \end{pmatrix}$$

  where $v_1, v_2$ are variances, and $c_{12} = Cov(y_1, y_2)$. Because of limited ensemble size, $c_{12}$ is usually non-zero (spurious correlation).

* **Obs Error** ($R$):

$$\begin{pmatrix} r_1 & 0 \\\\ 0 & r_2 \end{pmatrix}$$

* **Localization** ($\rho$): Because we want to cut off $y_2$, our localization array for the state variable $x$ is:

$$\rho = \begin{pmatrix} 1 & 0 \end{pmatrix}$$

Let:

$$S = YY^T + R = \begin{pmatrix} v_1 + r_1 & c_{12} \\\\ c_{12} & v_2 + r_2 \end{pmatrix}$$

The determinant of this matrix is $\lvert S \rvert = (v_1+r_1)(v_2+r_2) - c_{12}^2$.

---

### Method 1: No Indexing (Our Current Implementation)

The update formula is: **$\Delta x = (\rho \circ XY^T) S^{-1} \Delta$**

**Step 1: Localize the State-Obs Covariance**
We multiply $XY^T$ by $\rho$ element-wise:

$$\rho \circ XY^T = \begin{pmatrix} 1 \cdot c_1 & 0 \cdot c_2 \end{pmatrix} = \begin{pmatrix} c_1 & 0 \end{pmatrix}$$

*(Notice how we successfully zeroed out the direct link to $y_2$)*.

**Step 2: Invert the Global $S$ Matrix**
Using the analytical formula for a 2x2 inverse:

$$S^{-1} = \frac{1}{\lvert S \rvert} \begin{pmatrix} v_2 + r_2 & -c_{12} \\\\ -c_{12} & v_1 + r_1 \end{pmatrix}$$

**Step 3: Multiply $(\rho \circ XY^T)$ by $S^{-1}$**

$$\begin{pmatrix} c_1 & 0 \end{pmatrix} \frac{1}{\lvert S \rvert} \begin{pmatrix} v_2 + r_2 & -c_{12} \\\\ -c_{12} & v_1 + r_1 \end{pmatrix} = \frac{1}{\lvert S \rvert} \begin{pmatrix} c_1(v_2+r_2) & -c_1 c_{12} \end{pmatrix}$$

**Step 4: Multiply by the Innovations to get the final update**

$$\Delta x = \frac{1}{\lvert S \rvert} \begin{pmatrix} c_1(v_2+r_2) & -c_1 c_{12} \end{pmatrix} \begin{pmatrix} \delta_1 \\\\ \delta_2 \end{pmatrix}$$

**Analytical Result (No Indexing):**


Here is the simplified version:

$$\Delta x = \frac{c_1(v_2+r_2)}{\lvert S \rvert} \delta_1 - \frac{c_1 c_{12}}{\lvert S \rvert} \delta_2$$

**The Problem:** The update still depends on the far observation's innovation ($\delta_2$) and variance ($v_2$), which it should not. This happens because the spurious correlation ($c_{12}$) in the unlocalized $YY^T$ matrix allows their influence to leak into the local update.

---

### Method 2: Indexing (Local Analysis)

In this method, we look at $\rho$, see that $y_2$ is a "far" observation (weight 0), and we physically delete it from all matrices *before* doing any math.

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

1. **No Indexing:** $\Delta x = \frac{c_1(v_2+r_2)}{\lvert S \rvert} \delta_1 - \frac{c_1 c_{12}}{\lvert S \rvert} \delta_2$
2. **Indexing:** $\Delta x = \frac{c_1}{v_1 + r_1} \delta_1$

By extracting the subset of local observations (indexing), $y_2$ is completely removed. There is no $\delta_2$, and the spurious correlation $c_{12}$ vanishes.

If you do not index, the inversion of the global $YY^T$ matrix mixes the observations together. The local observation "absorbs" the error from the far observation, and your mathematical update gets contaminated.

### Tapering

Femke uses variance inflation as a method of tapering:

"Thus, to ensure smooth updates, we introduce tapering by infating the errors of the most distant observations"

Our implementation of adaptive correlation based localization does not do any form of tapering.
