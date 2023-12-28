# Lasso

## Notation and preliminaries

Let $X \in \mathbb{R}^{n \times N}$ be a parameter ensemble matrix.
There are $n$ parameters (rows) and $N$ realizations in the ensemble (columns).

Let $Y \in \mathbb{R}^{m \times N}$ be a response ensemble matrix.
The forward model is $h: \mathbb{R}^n \to \mathbb{R}^n$.

Assume that observational noise is generated as $D \sim \mathcal{N}(d_\text{obs}, \Sigma_\epsilon)$, where $D \in \mathbb{R}^{m \times N}$.

## Update equation

We will start with the ESMDA update equation, and work our way to the linear regression update equation.

The ESMDA update equation, with inflation factor $\alpha=1$, is

$$X_\text{posterior} = X + \operatorname{cov}(X, Y) (\operatorname{cov}(Y, Y) + \Sigma_\epsilon)^{-1} (D - Y)$$
where $\operatorname{cov}(X, Y) = c(X) c(Y)^T / (N - 1)$ and $c(X)$ centers each row in X by subtracting the mean.

The term $K := \operatorname{cov}(X, Y) (\operatorname{cov}(Y, Y) + \Sigma_\epsilon)^{-1}$ is the Kalman gain.

## The Kalman gain

We will now transform the kalman gain matrix into a linear regression.
We switch notations and set $X := c(X)$ and $Y := c(Y)$, meaning that we assume that each row (variable) has been centered.

Then we have
$$
K = \frac{X Y^T}{N - 1} \left(
\frac{Y Y^T}{N - 1} + \Sigma_\epsilon
\right)^{-1}.
$$

Since $D D^T / (N - 1) \approx \Sigma_\epsilon$, we can approximate $K$ as
$$
K = X Y^T \left(
Y Y^T + D D^T
\right)^{-1}.
$$

If we assume that $YD^T \approx DY^T \approx 0$, then the middle terms in $(Y +D)( Y + D)^T = Y Y^T + Y D^T + DY^T  + DD^T$ vanish.
Define $Y_\text{noisy} := Y + D$, then
$$
K = X Y^T \left(
Y_\text{noisy} Y_\text{noisy}^T
\right)^{+}.
$$
Finally, assume that $X D^T \approx 0$, then
$$
K = X Y_\text{noisy}^{+}.
$$
We can find $K$ by solving the optimization problem
$$
\lVert Y^T K^T - X^T \rVert^2
$$
using e.g. Ridge og Lasso.


## Solution approaches

### SVD

Since $K (D - Y)$














The ES update equation is

$$X_\text{posterior} = X + K (D - Y)$$

where

$$K = \Sigma_{x} \hat{H}^T (\hat{H}\Sigma_{x}\hat{H}^T + \Sigma_{\epsilon})^{-1}$$
$$K = \operatorname{cov}(X, Y) (\operatorname{cov}(Y, Y) + \Sigma_{\epsilon})^{-1}$$

## Covariances with linear forward model

Assume that $y = h(x) = H x + c$, then

$$\operatorname{cov}(x, y ) = \operatorname{cov}(x, x) H^T.$$

Similarity, if $y = h(x) = H x + c$, then

$$\operatorname{cov}(y, y ) = H \operatorname{cov}(x, x) H^T.$$

## Variations on the update equation


## References

- [Ensemblized linear least squares (LLS)](https://ncda-fs.web.norce.cloud/WS2023/raanes.pdf)





------------------




- test
- test

$`a=b`$

`$$a=b$$`

$$`a=b`$$

```math
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
```
