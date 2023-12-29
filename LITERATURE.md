# Literature

A list of literature that we find useful for Data Assimilation.
Could be moved to `docs/source` eventually.

## Papers

### SIES (Subspace Iterative Ensemble Smoother)

An optimization approach.
The observations are perturbed once, and we use the Gauss-Newton algorithm to minimize a cost function.
The proposed algorithm has state across iterations in a matrix $W$, which makes it more challenging to reason about compared to ESMDA.

- (2019) [**Efficient Implementation of an Iterative Ensemble Smoother for Data Assimilation and Reservoir History Matching**](https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full) by Evensen et al

### ESMDA (Ensemble Smoother with Multiple Data Assimilation)

The idea behind ESMDA is to inflate the covariance of the observations and update several times.
In the Gauss-Linear case, this makes no difference.
The hope is that for non-linear forward models, it's better to make many small updates rather than one big update.
Just like with SIES, there are no theoretical guarantees.
One difference between ESMDA and SIES is that ESMDA perturbs in each iteration, whereas SIES perturbs once.
The 2013 paper is the main paper, but the others are related too.

- (2012) [History matching time-lapse seismic data using the ensemble Kalman filter with multiple data assimilations](https://link.springer.com/article/10.1007/s10596-012-9275-5) by Emerick et al
- (2013) [**Ensemble smoother with multiple data assimilation**](https://www.sciencedirect.com/science/article/abs/pii/S0098300412000994) by Emerick et al
- (2016) [An Adaptive Ensemble Smoother With Multiple Data Assimilation for Assisted History Matching](https://doi.org/10.2118/173214-PA) by Emerick et al

### Ensemble Smoothers and related

Some papers that might be useful to have a look at.

- (2012) [Levenberg–Marquardt forms of the iterative ensemble smoother for efficient history matching and uncertainty quantification](https://link.springer.com/article/10.1007/s10596-013-9351-5) by Chen et al
- (2014) [Randomize-Then-Optimize: A Method for Sampling from Posterior Distributions in Nonlinear Inverse Problems](https://epubs.siam.org/doi/10.1137/140964023) by Bardsley et al
- (2018) [Analysis of iterative ensemble smoothers for solving inverse problems](https://link.springer.com/article/10.1007/s10596-018-9731-y) by Evensen
- (2023) [Review of ensemble gradients for robust optimisation](https://arxiv.org/abs/2304.12136) by Raanes et al

### Covariance regularization

- (2014) [Nonparametric Stein-type Shrinkage Covariance Matrix Estimators in High-Dimensional Settings](https://arxiv.org/abs/1410.4726) by Touloumis
- (2022) [GraphSPME: Markov Precision Matrix Estimation and Asymptotic Stein-Type Shrinkage](https://arxiv.org/abs/2205.07584) by Lunde et al

### Ensemble Information Filter and related

- (unpublished) An Ensemble Information Filter: Retrieving Markov-Information from the SPDE discretization by Lunde et al
- (2022) [Ensemble transport smoothing. Part I: Unified framework](https://arxiv.org/abs/2210.17000) by Ramgraber et al
- (2022) [Ensemble transport smoothing. Part II: Nonlinear updates](https://arxiv.org/abs/2210.17435) by Ramgraber et al

## Books

### Preliminaries

- (2006) [Pattern Recognition and Machine Learning](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738) by Bishop
    - In particular, see Section 2.3.3: Bayes' theorem for Gaussian variables

### Data Assimilation

- (2014) [Data Assimilation: The Ensemble Kalman Filter](https://www.amazon.com/Data-Assimilation-Ensemble-Kalman-Filter/dp/3642424767/) by Evensen
- (2022) [Data Assimilation Fundamentals: A Unified Formulation of the State and Parameter Estimation Problem](https://www.amazon.com/Data-Assimilation-Fundamentals-Formulation-Environment/dp/3030967085/) by Evensen
