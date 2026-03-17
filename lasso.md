# Lasso

Notes on the Lasso approach for regularizing the forward model.

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
$$K = \frac{X Y^T}{N - 1} \left(
\frac{Y Y^T}{N - 1} + \Sigma_\epsilon
\right)^{-1}.$$

### Adding sampled noise

Let $R \sim \mathcal{N}(0, \Sigma_\epsilon)$.
If we assume the mean is known, Bessel's correction is not needed and we can write $R R^T / N \approx \Sigma_\epsilon$.
We sample from $\mathcal{N}(0, \Sigma_\epsilon)$ then estimate the covariance $\Sigma_\epsilon$ using the empirical covariance.

Define $S = R \sqrt{(N-1) / N}$, then

$$\Sigma_\epsilon \approx
\frac{1}{N} R R^T
=
\frac{1}{N-1} \frac{N-1}{N} R R^T
= \frac{S S^T}{N-1}
$$
Notice that this is a low rank approximation to the observation covariance matrix.
The observation covariance matrix $\Sigma_\epsilon$ is typically diagonal, has shape $m \times m$, whereas $S S^T$ has rank $N \ll m$.

Replacing $\Sigma_\epsilon$ with $S S^T / (N - 1)$,
we can approximate $K$ as
$$K \approx X Y^T \left(
Y Y^T + S S^T
\right)^{-1}.$$

If we assume that $YS^T \approx SY^T \approx 0$, then the middle terms in $(Y +S)( Y + S)^T = Y Y^T + Y S^T + SY^T  + SS^T$ vanish.
This is reasonable in the limit of large $N$ since the rows in $S$ are uncorrelated with the rows in $Y$.

Define $Y_\text{noisy} := Y + S$, then

$$K \approx X Y^T \left(
Y_\text{noisy} Y_\text{noisy}^T
\right)^{+}.$$

Finally, assume that $X S^T \approx 0$ so that $Y \approx Y_\text{noisy}$, then

$$K = X Y_\text{noisy}^{+}$$

and we solve a least squares problem $Y_\text{noisy}^T K^T = X^T$ for $K$.
If we use Ridge or Lasso, the solution decomposes and we can consider each row of $X$ independently.
This is because in the equation $Y_\text{noisy}^T K^T = X^T$ the matrix $K^T$ acts on each row vector in $Y_\text{noisy}^T$ independently to produce each row vector in $X^T$.
We can find $K$ by solving a Ridge or Lasso problem.

### Comment: Kalman gain and the forward model

Regularizing $K$ using e.g. Ridge or Lasso makes sense if we believe that forward model is localized, in the sense that each output is determined primarily by a few inputs (sparsity).

A rough sketch of why $K \sim \mathbb{E} \nabla h^{-1}$

Since $K \approx X Y^{+}$ holds for centered matrices, if we change notation to uncentered matrices we have
$$K \approx \frac{X - \mathbb{E} X}{Y - \mathbb{E} Y}$$
rearranging we see that this is a second order Taylor expansion.

Therefore it makes sense to regularize $K$ if we believe in a sparse forward model.

### Comment: Ridge and the Kalman gain

Solving $Y^T K^T = X^T$ with Ridge produces the Normal equations that define $K$.
To be concrete, minimizing $\lVert Y^T K^T - X^T \rVert_2^2 + \alpha \lVert D K \rVert_2^2$ with respect to $K$ yields the equation
$$\left( Y Y^T + \alpha D D^T\right) K = X Y^T.$$
This is, up to scaling, exactly the equation for the kalman gain $K$ that we started with from the ESMDA paper.
See equations (119) and (132) in the matrix cookbook for the matrix calculus needed to differentiate the loss function.

### Comment: Another Lasso approach

The Kalman gain can also be written as

$$K = \Sigma_{x} \hat{H}^T (\hat{H}\Sigma_{x}\hat{H}^T + \Sigma_{\epsilon})^{-1}$$

which suggests another possible way to compute $K$:

1. Estimate $\hat{H}$ using Lasso or similar.
2. Estimate $\Sigma_{x}$ as $\text{cov}(X, X)$.

This is likely unfeasible since $\text{cov}(X, X)$ becomes huge.

### Comment: Covariances with linear forward model

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