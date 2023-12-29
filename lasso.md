# Lasso

Notes on the Lasso approach for

## Notation and preliminaries

In this section we establish notation.

- Let $X \in \mathbb{R}^{n \times N}$ be a parameter ensemble matrix.
There are $n$ parameters (rows) and $N$ realizations in the ensemble (columns).
- Let $Y \in \mathbb{R}^{m \times N}$ be a response ensemble matrix.
- The forward model is $h: \mathbb{R}^n \to \mathbb{R}^n$.
- The observational noise has distribution $\mathcal{N}(0, \Sigma_\epsilon)$.

To add observational noise to the observations, we sample $D \sim \mathcal{N}(d_\text{obs}, \Sigma_\epsilon)$, where $D \in \mathbb{R}^{m \times N}$.
Each column in D is drawn from the multivariate normal.
We take the Cholesky factorization of the covariance and write $\Sigma_\epsilon = L_\epsilon  L_\epsilon^T$.
Note that each column in $D$ can be computed as $d_\text{obs} + L_\epsilon z$, where $z \sim \mathcal{N}(0, 1)$.

## Update equation

Our starting point for the update equation will be from the ESMDA paper.

The ESMDA update equation, with inflation factor $\alpha=1$, is

$$X_\text{posterior} = X + \text{cov}(X, Y) (\text{cov}(Y, Y) + \Sigma_\epsilon)^{-1} (D - Y),$$

where the empirical cross covariance matrix $\text{cov}(X, Y) = c(X) c(Y)^T / (N - 1)$, and $c(X)$ centers each row in X by subtracting the mean.

The term $K := \text{cov}(X, Y) (\text{cov}(Y, Y) + \Sigma_\epsilon)^{-1}$ is the Kalman gain matrix.
Notice that in this expression we estimate both the covariance matrix and the cross covariance matrix using few realizations $N$.

## The Kalman gain

We will now transform the Kalman gain matrix into a linear regression.

We switch notations and set $X := c(X)$ and $Y := c(Y)$, meaning that we assume that each row (variable) has been centered by subtracting the mean.

Writing out the empirical covariances, the Kalman gain becomes
$$
K = \frac{X Y^T}{N - 1} \left(
\frac{Y Y^T}{N - 1} + \Sigma_\epsilon
\right)^{-1}.
$$

### Adding sampled noise

Let $S \sim \mathcal{N}(0, \Sigma_\epsilon)$.
We can write $S S^T / (N - 1) \approx \Sigma_\epsilon$, which is a low rank approximation to the observation covariance matrix.
Notice that the observation covariance matrix is typically diagonal, has shape $m \times m$, whereas $S S^T$ has rank $N \ll m$.

Replacing $\Sigma_\epsilon$ with $S S^T / (N - 1)$,
we can approximate $K$ as
$$
K \approx X Y^T \left(
Y Y^T + S S^T
\right)^{-1}.
$$

If we assume that $YS^T \approx SY^T \approx 0$, then the middle terms in $(Y +S)( Y + S)^T = Y Y^T + Y S^T + SY^T  + SS^T$ vanish.
This is reasonable in the limit of large $N$ since the rows in $S$ are uncorrelated with the rows in $Y$.

Define $Y_\text{noisy} := Y + S$, then
$$
K \approx X Y^T \left(
Y_\text{noisy} Y_\text{noisy}^T
\right)^{+}.
$$
Finally, assume that $X S^T \approx 0$ so that $Y \approx Y_\text{noisy}$, then
$$
K = X Y_\text{noisy}^{+}
$$
and we solve a least squares problem $Y_\text{noisy}^T K^T = X^T$ for $K$.
If we use Ridge or Lasso, the solution decomposes since in general $A[x_1 | x_2] = [Ax_1 | Ax_2] = [y_1 | y_2]$ and we can consider each row of $X$ independently.
We can find $K$ by solving a Ridge or Lasso problem.

### Comment: Ridge and the Kalman gain

Solving $Y^T K^T = X^T$ with Ridge produces the Normal equations that define $K$.

## Comment: Another Lasso approach

The Kalman gain can also be written as
$$K = \Sigma_{x} \hat{H}^T (\hat{H}\Sigma_{x}\hat{H}^T + \Sigma_{\epsilon})^{-1}$$
which suggests another possible way to compute $K$:

1. Estimate $\hat{H}$ using Lasso or similar.
2. Estimate $\Sigma_{x}$ as $\text{cov}(X, X)$.

This is likely unfeasible since $\text{cov}(X, X)$ becomes huge.

## Comment: Covariances with linear forward model

Here we show two facts that can be proven using the definition of covariance, see e.g. sections 6.2.2 and 8.2.1 in [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).

Assume that $y = h(x) = H x + c$, then

$$\text{cov}(x, y ) = \text{cov}(x, x) H^T.$$

Similarity, if $y = h(x) = H x + c$, then

$$\text{cov}(y, y ) = H \text{cov}(x, x) H^T.$$

## References

- [Issue 6599 on the Lasso in ERT](https://github.com/equinor/ert/issues/6599)
- [Presentation: Ensemblized linear least squares (LLS)](https://ncda-fs.web.norce.cloud/WS2023/raanes.pdf)
- [Paper: Ensemble Smoother with Multiple Data Assimilation](http://dx.doi.org/10.1016/j.cageo.2012.03.011)
- [Paper: Ensemble transport smoothing. Part I: Unified framework](https://arxiv.org/pdf/2210.17000.pdf)
