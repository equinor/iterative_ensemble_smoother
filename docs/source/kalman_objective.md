# Kalman-type ensemble approaches

How to think about Kalman-type ensemble based data assimilation:

- What does Kalman-type ensemble methods do?
- What is their ideal objective?
- How does the objective behave under statistical estimation?
- How to think about surrogate objectives?
- What is the modelling setup and what information is known a-priori?
- List common methods and discuss them in the context of the above.

What is _not_ provided, is an evaluation-comparison between methods.
Only how we _think_ they will behave, given what we know about statistics and our objective.
From the contents, it should however be clear how a simulation study could be constructed to compare methods.


## Gaussian conditional transport

Let $x$ and $y$ be random variables jointly multivariate Gaussian with mean and covariance

$$
\begin{bmatrix}
x \\
y
\end{bmatrix}
\sim \mathcal{N}\left(\begin{bmatrix}
\mu_x \\
\mu_y
\end{bmatrix}, \begin{bmatrix}
\Sigma_{x} & \Sigma_{xy} \\
\Sigma_{yx} & \Sigma_{y}
\end{bmatrix}\right)
$$

Define the "Kalman gain" as

$$
K = \Sigma_{xy}\Sigma_{y}^{-1}.
$$

Then, a multivariate sample $(x_i,y_i)$ is _transported_ to a sample from the conditional
$p(x|y)$, having observed $y$ as such, via the formula 

$$
x_i + K(y-y_i) \sim p(x|y).
$$

Note that $x|y$ retains Gaussianity as

$$
x | y \sim \mathcal{N}(\mu_x + K(y - \mu_y), \Sigma_{x} - K \Sigma_{yx}).
$$

For any distribution having a bijection between (the informative parts of) 
$x$ and $y$, there exist a similar mapping, transporting a sample $(x,y)$ to $x|y$.
The EnKF and ES variants follows from writing $y=d=Hx+\epsilon$ where 
$\epsilon\sim \mathcal{N}(0,\Sigma_{\epsilon})$, thus $\Sigma_y=H\Sigma_xH^T+\Sigma_{\epsilon}$.
Note that $y$ is common to use as $y=Hx$ but this is not the case in the preliminaries here.
A point of confusion is that it is indeed _not_ the observation $d$ that is "perturbed" with noise,
but rather, $d$ has variance $H\Sigma_xH^T+\Sigma_{\epsilon}$ and an $\epsilon$ is then added to the response $Hx$ to correctly sample from this distribution of $d$.
Because both $(d-d_i)$ and $d_i=Hx_i+\epsilon_i$ are additive, and $\epsilon$ is symmetric about 0, it does not matter if we add the noise to $d$ or to $Hx$ in the expression.
If any of these points were different, it would be evident that we are not perturbing observations.
Thus, it is incorrect and slightly confusing to say that we perturb the observations.
It is correct to say that we sample from the distribution of $d$, accounting from both prior uncertainty _and_ observation uncertainty.

Note that the EnKF and ES is intended to work (and indeed do) when $d=h(x)+\epsilon$
for some non-linear $h$.
The joint system is then _not_ Gaussian, and we have no $H$ and no immediate $K$.
It turns out that the Gaussian conditional transport makes sense in a general setting,
where the true distribution on $(x,y)$ is unknown and must be (implicitly) estimated from data.
When approximating it with the two
first moments (mean and covariance) seeking to minimize information loss, we arrive
at the Gaussian.
This could be expected by remembering that the Gaussian is the maximum entropy distribution when the two first moments are known. It does not encode any other information in the distribution.
The Gaussian transport using the Kalman gain above then follows.

The details of minimizing information loss, and how to compare estimates of $K$ (belonging to different models) are explained below.


## Minimize model information-loss: KLD

We seek to estimate the optimal transport function, that takes a sample $(x,y)$ to a conditional marginal sample $x|y$.
Such a transport implies a joint model-distribution on $(x,y)$.
The objective is to learn the transport, corresponding to a model distribution on $(x,y)$, say $Q$, from a training dataset generated from the true distribution, say $P$.
"Optimal" transport to a posterior sample therefore needs an objective on how far away the model distribution is from the true distribution, in particular when the true underlying distribution is unknown.

We want to minimize information loss by modelling data $(x,y)$,
drawn from true data generating distribution $P$, with the "model" distribution $Q$.
The $Q$ minimizing Kullback-Leibler-Divergence (KLD) is the distribution that does this.
The KLD is given by

$$
D_{KL}(P \parallel Q) = \int \int p(x, y) \log\left(\frac{p(x, y)}{q(x, y)}\right) dx dy.
$$

Note two formulas:

1. Information is additive, and we may decompose the KLD as 
the KLD over the marginal and the expected conditional KLD w.r.t. the variable considered in the marginal.
The optimization can be done disjoint if marginal and conditional marginal depend upon disjoint parameter-sets.

$$
D_{KL}(P(x, y) \parallel Q(x, y)) = D_{KL}(P(y) \parallel Q(y)) + E_{P(y)}\left[D_{KL}(P(x | y) \parallel Q(x | y))\right]
$$

2. The relative Kullback-Leibler Divergence (KLD) between two distributions 
$P$ and $Q$ is given by:

$$
D_{KL}(P \parallel Q) = E_P[\log(P)] - E_P[\log(Q)]
$$

Dropping the first term, which is constant with respect to the model 
$Q$, and retaining the negative of the latter term gives us the objective of maximizing the likelihood (or minimizing the negative log-likelihood) of $Q$:

$$
-E_P[\log(Q)]
$$

This is the core for all of maximum likelihood estimation, information criteria, and regression and supervised learning using a negative log-likelihood as its loss function.

### Why EnKF and ES works so well

Let $(x,y)\sim P$ possibly non-Gaussian and assume we have a finite dataset to infer a model $Q$ from.
Arguably, the most important aspects to encode in $Q$, without any other knowledge, are 
the two first moments of $P$.
Having access to the sample covariances of
$\Sigma_{xy}$
and $\Sigma_{y}^{-1}$
after (maximum likelihood when $n>p$) estimation (minimizing empiricla KLD), the distribution $Q$ minimizing KLD towards $P$ is the Gaussian.
Then the Gaussian conditional transport function using Kalman gain follows.
Therefore EnKF and ES are not just something that works in the linear-Gaussian case.

### Optimizing empirical KLD: Training-loss VS. test-loss

The objective is the KLD from $Q$ to $P$.
If we have minimized the empirical KLD, w.r.t. a parametrization $\theta$ of $Q$ using a training dataset of $(x,y)$,
then evaluation of the empirical KLD using the optimized/fitted $\hat{\theta}$ (i.e. training loss) is no longer an unbiased estimator of
the KLD objective (but using a new test loss would be).
In particular, the bias is a random variable, but its expectation is positive, so that the training loss is expected to be smaller than the expected test loss.
Furthermore, the expectation of the bias is an increasing function of the dimensionality of $\theta$, or more precisely the number of degrees of freedom in the statistical estimation / parametrization of $Q$.
The expected bias is decreasing in ensemble/sample size.

To alleviate this we can either adjust for the expectation in the bias, or just pass through a test-set not seen in the fitting of $\theta$.
The latter does not require advanced use of theory (information criteria), and is an "obvious" thing to do.
Thus, using a test dataset, we would like to evaluate different models $Q$, performing slightly different transports.


## Objectives for Kalman-type ensembles

When constraining $Q$ to only encode information on the two first moments, that are learned from data, the KLD
approach yields a Gaussian distribution $Q$ even when $P$ is non-Gaussian.
We call methods using the Kalman-gain in Gaussian transport from prior to posterior as "Kalman-type" ensemble based data assimilation methods.
They implicitly seek to minimize information loss in this transport.
Different Kalman-type modelling approaches essentially yields different estimates of the Kalman-gain, $K$.
The following discuss how to evaluate (on some objective, using unseen test-data)
which Kalman-gains are superior to others.
This turns out to be non-trivial for several common applications like the EnKF/ES due to implicit singular covariance estimates.


### Joint-Gaussian
The KLD approach suggests $-E_P[\log(Q)]$ as the objective of fitting $Q$.
Taking $Q$ to be multivariate Gaussian, we arrive at the negative log-likelihood for a sample

$$
-\log p(x, y) = \frac{1}{2} \log \left| \begin{bmatrix}
\Sigma_{x} & \Sigma_{xy} \\
\Sigma_{yx} & \Sigma_{y}
\end{bmatrix} \right| + \frac{k}{2} \log(2\pi) + \frac{1}{2} \begin{bmatrix}
x - \mu_x \\
y - \mu_y
\end{bmatrix}^T \begin{bmatrix}
\Sigma_{x} & \Sigma_{xy} \\
\Sigma_{yx} & \Sigma_{y}
\end{bmatrix}^{-1} \begin{bmatrix}
x - \mu_x \\
y - \mu_y
\end{bmatrix}
$$

and the negative log-likelihood 

$$
-\frac{1}{n}\sum_i \log(p(x_i,y_i))
$$

as an estimator of $-E_P[\log(Q)]$, suitable for fitting procedures.
It may also be used for evaluation of different models, as long as the sample $(x_i,y_i)$ is not used to optimize $\theta$ parametrizing $Q$.
This means we may use the same formula, but with a different test-set of samples.
This is a natural objective if explicit estimates of $\Sigma_{x}$, $\Sigma_{xy}$, and $\Sigma_{y}$ are computed in creating
an estimate $K=\Sigma_{xy}\Sigma_{y}^{-1}$.
Importantly, it allows encoding information or structure of how $\Sigma_{x}$, $\Sigma_{xy}$, and $\Sigma_{y}$ should look like,
for instance  $\Sigma_y=H\Sigma_xH^T+\Sigma_{\epsilon}$. Then only $\Sigma_x$ and $H$ needs estimation if they should be unknown.
It could also be that $\Sigma_x$ is diagonal, or is a function of some $\theta$ in a smaller dimensional space.
We would then require fewer degrees of freedom to fit $\Sigma_x(\theta)$.

Note that if covariance estimates are singular (i.e. EnKF when $p>n$) then this objective cannot be used due to the log-determinant and covariance inverse.
Furthermore, we often only have access to the final estimate of the Kalman-gain, not the full covariance structure.

### Gaussian conditional marginal

The joint KLD is a natural objective to evaluate model performance on.
However, multiple (Kalman) ensemble methods only produce an estiamte $K$ either explicit or explicit,
without specifying the full covariance structure.
In such cases, the following objective, that still aligns with the joint-likelihood, is a natural objective.
It evaluates the model posterior using samples from the optimal posterior, and averaging over
the conditioning variable $y$.

Putting the two points of the joint KLD together, we obtain a regression objective

$$
-E_{P(y)}\left[E_{P(x|y)}\left[ \log(Q(x|y)) \right]  \right].
$$

In spoken language, this means the negative log-likelihood for the conditional,
and the conditioning variable $y$ is sampled from its true distribution and then averaged over.

A Kalman-type method estimating a model Kalman-gain, say $\hat{K}$, transports samples $(x_i,y_i)$
according to 
$x_i^\ast =x_i + \hat{K} (y-y_i)$ which is Gaussian with mean and covariance

$$
\mu_{x^\ast}(y, \hat{K}) = \mu_x + \hat{K}(y-\mu_y)
$$

$$
\Sigma_{x^\ast}(\hat{K}) = \Sigma_{x} + \hat{K}\Sigma_y \hat{K}^T - 2 \Sigma_{xy}\hat{K}^T
$$

When $\hat{K}=K=\Sigma_{xy}\Sigma_y^{-1}$ the expression simplifies to the afformentioned posterior.
This conditional marginal is $Q(x|y)$.

To take the two expectations w.r.t. $P(y)$ and $P(x|y)$ we evaluate the negative log density
averaged over a test data where $y\sim P(y)$ and $x\sim P(x|y)$.
This is "just" an ordinary sample from $P(x,y)$.
Thus evaluating

$$
-\frac{1}{n}\sum_{i=1}^n \log q(x_i|y_i;\hat{K}) = \frac{1}{2n}\sum_{i=1}^n
(x_i - \mu_{x^\ast}(y, \hat{K}))^T \Sigma_{x^\ast}(\hat{K})^{-1} (x_i - \mu_{x^\ast}(y,\hat{K})) +  \log |\Sigma_{x^\ast}(\hat{K})| +  k\log(2\pi)
$$

approximates and indeed converges to the expected relative KLD on the conditional.
Therefore it can be used to evaluate different Kalman-gain estimates, used for different transport.

Note that $\mu_{x^\ast}(y)$ and $\Sigma_{x^\ast}$ uses the true underlying covariance structure, in addition
to $\hat{K}$.
This makes sense, because $\hat{K}$ has only been used to transport samples.
To evaluate in this way does however require knowledge of these quantities, which is likely or often unknown.
Otherwise they would be used in the creation of $\hat{K}$ to be exactly $K$.
This evaluation therefore makes sense in synthetic cases where the prior is exactly known or simulated from,
or where it can be simulated to create estimates of dependence independent of training data.

Using the estimates from the training data and modelling resulting in $\hat{K}$ yields
a model where more is assumed, employed and also evaluated in the criterion, with regards to structure.
It is more close to "believing" in the Gaussian, not just moment-estimation as an approximation and then used for updating.
It is slightly too strict in terms of how what Kalman-type methods do (usage of $\hat{K}$ in transport).
Note also that the estimated posterior covariance simplifies when using the same estimates used for $\hat{K}$, but the determinant is given by

```math
|\hat{\Sigma}_{x} - \hat{K} \hat{\Sigma}_{yx}|=
|(I-\hat{K}\hat{H})\hat{\Sigma}_{x}|=
|(I-\hat{K}\hat{H})||\hat{\Sigma}_{x}|
```

so if the prior covariance estimate is singular $|\hat{\Sigma}_x|=0$
(e.g. if using the sample covariance in place of $\Sigma_x$ and $p>n$)
then so is the estimated posterior covariance of $x|y$.
This highlights problems with using estimates from the training data associate $\hat{K}$.
E.g. the Ensemble Smoother would not be possible to evaluate.


### Generalized least squares objective

If $X-KY$ has correlated residuals with a known covariance, then the objective
yielding the _best_ (meaning minimum variance) unbiased (BLUE) estimator when minimized is the generalized least squares (GLS) objective

$$
\min_{\beta} (Y - X\beta)^T \Omega^{-1} (Y - X\beta)
$$

so $\Omega=\Sigma_{x^\ast}(K)$
and when this is known a-priori then $\hat{\beta}$ is the BLUE estimate of $K$.
This is typically solved with weighted least squares methods.

Unfortunately, $\Sigma_{x^\ast}(K)$ is generally unknown and must in some parts be estimated.
Then, $\log |\Sigma_{x^\ast}(K)|$ should be included in the objective for accounting for this estimation and penalizing certainty.
One then once again arrives at the relevant parts of the negative log-likelihood for $p(x|y)$.


### Least squares objective

If $K-KY$ satisfies the Gauss-Markov conditions, which it does when we assume that $(x,y)$ is Gaussian and $x|y$ has a diagonal posterior covariance,
then the least squares objective provides the BLUE estimator when minimized

$$
\min_{\beta} (Y - X\beta)^T (Y - X\beta)
$$

Because of uncorrelated errors, the problem is separable and each dimension may be optimized individually in one-dimensional regressions.
The LS objective corresponds to the GLS and more generally the Gaussian NLL when the assumptions are met.

In general, the linear least squares estimator (LLS) is unbiased, so it converges to $K$ in our case, but it is not the BLUE estimator unless the errors indeed are uncorrelated.
Note however that when $\Sigma_{x^\ast}(K)$ must be estimated and we employ the Gaussian $p(x,y)$ NLL and then solve for the Kalman gain,
assuming $n>p$ so the MLE exists, then this indeed also produces the LLS estimator of $K$.
The rationale is that because $\Sigma_{x^\ast}(K)$ is unknown and estimated, this offsets the GLS part of the objective when also optimizing the log-determinant of $\Sigma_{x^\ast}(K)$.
It is offset in the exact way so that we arrive at LLS, which we _know_ is inefficient (but unbiased).

It is possible to now err and reason that due to arriving at the same stimator, the LS and Gaussian-NLL objectives are equivalent.
This is, however false.
It is rather that in this particular case, when we have failed to inform of structure in dependence, 
then the Gaussian-NLL arrives at the same inefficient LLS estimator.
We know that the LLS estimator is inefficient when Gauss-Markov conditions are not satisfied.
And we know that this is because the LS objective then does not appropriately target dependence.
This is an additional point towards searching for better estimators of $K$ in suitable smaller dimensional spaces, but using the general likelihood approach.


### Summary on how to evaluate different methods.

- If covariances used to create $\hat{K}$ exists, and are SPD, then use the NLL with test-data. This is the most efficient way to discriminate methods.
- If only the $\hat{K}$ is known from an assimilation algorithm, then evaluate methods by the expected relative KLD on the conditional marginal. Using known covariances in a synthetic experiment.
- If only the $\hat{K}$ is known. And access to test-data from an unknown data-generating-process is sought to evaluate over: Use the LS objective. This is less efficient, but seems to be the only thing one then may do.


## Development of Kalman-type methods

The goal of this section is to introduce different Kalman-type ensemble based approaches, and to discuss their properties in terms of objective functions.
This is challenging for two reasons:

1. Methods have historically been developed with statistical considerations of an infinite ensemble size.
2. The KLD objective on learning a Gaussian transport function, and the bias of the training loss, showcases that it is properties of convergence on the path to the asymptotic case that is important to discuss when comparing methods.

Because point 2. has not historically been considered in point 1, common methods employed at $p>n$ are not easy to discuss.
It is symptoms of poor performance in point 2 (ensemble collapse and spurious correlations) that have led to many of the developments in Kalman-type ensemble based methods.
It is however better to evaluate point 2 directly, and then discuss the symptoms of poor behaviour in context of this.
The previous sections show that asymptotically different objectives provide the same estimator.
This does however not mean that different objectives are equally "good".

> The expected KLD (over $y\sim P(y)$ ) of $Q(x|y)$ to $P(x|y)$ that the updated ensemble is sampled from, is the goal.

Then GLS and LS is consistent with this under specific conditions, but may be easier to apply generally.
But are generally less efficient in evaluating method performance (in particular the LS objective).

Evaluation of methods should be done using, preferably a large, test dataset.
The following do not do this, but attempts at discussing and motivating methods through knowledge of 
statistical methodology, in context of point 2, conditioned on the sizes of $p$ and $n$.
This involves drawing on knowledge from information criteria, the bias-variance trade-off, and regularization techniques.
No definite answers on what is the best method is given here.
The goal is to provide a feeling and intuition for how the methods will perform, how they should be appropriately evaluated, and guidance towards developing new methods in terms of optimizing point 2 above.


### Modelling setup

The structures of an ensemble based data assimilation problem is now defined.
We now change notation slightly.
Let $x$ be sampled from some distribution with finite two first moments.
Define $y=h(x)$, possibly non-linear, and $d=y+\epsilon$ where $\epsilon\sim N(0,\Sigma_{\epsilon})$
and $\Sigma_{\epsilon})$ is assumed diagonal.

We have an observation vector, say $d^\ast$.
We have a sample of $x$'s, say $x_i$, that we pass through $y$ to get a corresponding of $y_i$'s, and then 
sample some $\epsilon_i$'s appropriately so that we have samples $(x_i,y_i,d_i)$.
The goal is to use these samples to learn the best possible $\hat{K}$ to transport the samples $(x_i,d_i)$ to a sample, sampled from a distribution as close as posbiele to $x\sim p(x_i|d_i)$.

Structure of the problem can be
- $d=y+\epsilon$
- Knowledge of $\Sigma_{\epsilon}$
- Knowledge of $\Sigma_{x}$
- Knowledge of the structure of $\Sigma_{x}^{-1}$ (encoding Markov properties)
- Knowledge of $h$, e.g. direct observations encoded by $H$ with zeroes and ones.

Encoding such information allows us to target the information in the samples $(x_i, y_i, d_i)$ towards information that is unknown and therefore must be learnt to obtain an estimate $\hat{K}$.
Generally, encoding such information makes the bias of the traning loss smaller, which is good.

A final consideration is to add regularization of objectives.
This induces bias in estimators, but reduces variance (bias-variance trade-off).
Overall, this can lead to improved performance on test-loss, which is our goal.
This therefore provides guidance in developing estimates of K.


### Ensemble smoother

The Ensemble Smoother (ES) is developed through 
1. Sample covariance matrices $`\hat{\Sigma}_{xy}`$, $`\hat{\Sigma}_{y}`$ converges to the population quantities at an infinite ensemble size. 
2. Find $`\hat{\Sigma}_{d}=\hat{\Sigma}_y + \Sigma_{\epsilon}`$ which is guaranteed SPD.
3. Solve $`\hat{K}=\hat{\Sigma}_{xy}\hat{\Sigma}_{d}^{-1}`$

- Some structure is employed, namely that $d=y+\epsilon$ and knowledge of the noise-covariance.
This is good.
- Any knowledge of structure in $\Sigma_{x}$ or $h$, or even the exact quantity, is not used.
Neither is regularization techniques to improve on the sample-covariance matrices.

The likely consequence is overfitting to training set (the ensemble), and thus spurious correlations and ensemble collapse at large dimensions.

Notice that the ES solution of using sample covariances can be found as the solution of the MLE when $n>p$ and also the LS regression from $x\to y$.
The sample cross-covariance $`\hat{\Sigma}_{xy}`$ is the sample-covariance $`\hat{\Sigma}_{x}`$ multiplied with the LLS estimate $\hat{H}$ for $Y-HX$.


### Adaptive localization: correlation-based model selection

Recognizing spurious correlations. 
Recognizing sampling distribution of spurious correlations.
Do automatic model selection between LS regressions in $x$ and $y$.
Improves estimates Kalman gain. Less degrees of freedom used.


### Distance based localization

Recognizing spurious correlations.
Recognizing that effects are based on distance.
So update encoded in Kalman gain should be zero if $d_j$ and $x_k$ are far away from another.
- Set to zero or smooth down (kernel/tapering) effects based on distance


### Linear least squares

The LLS Kalman gain estimate (NORCE slides) is 

$$
\hat{K} = X^TD(D^TD)^{-1}
$$

It is both a solution from fiding the MLE estimates (if $n>p$) for the full covariance matrix on $(x,d)$ using samples $(x_i,d_i)$,
or from solving the LS objective on $X-KD$.
As noted earlier, these produce the same estimator here.

- Does not use any prior knowledge of structure of the problem 
- No regularization of the likelihood objective or LS objective.

Since this estimator has the propertis of LLS, we know it is unbiased asymptotically.
But, when the Gauss-Markov conditions are not satisfied, which they are _not_ due to prior correlation in the prior of $x$, 
then the estimator is not the minimum variance estimator.

It provides insight into that estimators of $\hat{K}$ can be produced through the LS objective on $X-KD$, which can be separated on the dimensions of $x$.

In a comparison to e.g. ES, the difference lies in its uninformed (implied) sample estimate $\hat{\Sigma}_d$. 
The discrepancy from ES, and a poorer estimate, increaess in the dimension of $d$, and thus the number of obervations.


### LASSO without structure

Using the same objective as for LLS, i.e. LS, which produces inefficient but unbiased estimators, can be used with other linear regression techniques.
In particular LASSO, better solving the bias-variance tradeoff than LLS, comes to mind, due to the explainability of a sparse estimate $`\hat{K}_{lasso}`$.
If both $`{\Sigma_{x}}`$, $H$, and $`\Sigma_{\epsilon}`$ are sparse, then $K$ should also be sparse.
LASSO is ideally suited to learn this sparsity from data.
The downside is that this sparsity is often known a-priori, except perhaps for the structure of $H$, but is not used to inform the regression here.
As such, it suffers from the same problem as LLS compared to ES, but to a much smaller extent.
It likely benefits compared to ES when $\Sigma_x$ is sparse.

Thus LASSO without structure makes sense (perhaps, this should be tested on data(!) as for everything) over ES (and always over LLS) when
- The dimensions of the problem is not enormous (thousands, not fields).
- There are more than a few parameters (ES is good when this number is _very_ small).
- The Kalman gain can be well approximated with a sparse matrix (almost a necessity: prior on parameters is sparse or diagonal).
- One seeks an human-understandable model.

Derivations with scaling should maybe be produced, again.


### Using all structure: EnIF

Ideally we would like to encode all the structure that is known a-priori.
Furthermore, we prefer doing the bias-variance trade-off over purely BLUE estimators, because evaluation on test-data is our target.
EnIF allows encoding all (or none of) the information 

- $d=y+\epsilon$
- $\Sigma_{\epsilon}$ is known
- $H$ is possibly known
- $\Sigma_x$ is possibly known, or at least non-zeroes $\Sigma_x^{-1}$ are known

If non-zeroes $\Sigma_x^{-1}$ are not known, they provide a powerful parsimoneous model-selection tool even when 
$\Sigma_x^{-1}$ is dense.

Solutions of $H$ is found through LASSO on $Y-HX$.
The LLS solution here is likely BLUE due to conditional independence $y|x$ (at least on linear maps), so the objective is efficient in producing estimators.
There are possible inefficiencies when $h$ is non-linear, and ideally it should be optimized jointly with covariances in the MLE.
Employing LASSO ensures sparsity of $\hat{H}$ and a better bias-variance trade-off on LS.
Note that any variance in $y$ that is unexplained in the regressions on $x$ should be added to $\Sigma_\epsilon$ as variance not accounted for by $x$ and $H$.
