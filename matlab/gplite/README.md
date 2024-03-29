# gplite
Lite Gaussian process (GP) regression toolbox

The main structure of the toolbox is that:
- `gplite_train` fits a GP model to some data (by either optimizing the GP hyperparameters or sampling from the posterior of GP hyperparamers), returning a gp struct.
- `gplite_post` is a main GP function that computes the posterior GP given some data, for given GP hyperparameters.
- During training, the main function used is `gplite_nlZ` which computes the GP log marginal likelihood (possibly adding a prior term to it) for given GP hyperparameters.
  - The log marginal likelihood is the quantity which is either maximized or used as target for MCMC sampling.
  - `gplite_hypprior` computes the prior over hyperparameters (for now, only Gaussian and Student t priors are supported).
- `gplite_post` and `gplite_nlZ` use very similar computations, which are in a private function (in the private folder) `gplite_core`.
- Given a trained `gp` struct, `gplite_pred` is used to make predictions of mean and variance at new input points.

The GP models in `gplite` are all GP regression with Gaussian observation noise, which affords analytical inference.
- `gplite_covfun` stores a small number of covariance functions for the GP. The standard covariance function is the squared exponential.
- `gplite_meanfun` stores a large number of different mean functions for the GP.
- `gplite_noisefun` includes a few different options for the observation noise.

The data used to create a GP model are:
- `X` a `N-by-D` matrix of training inputs (`N` inputs of dimension `D`)
- `y` a `N-by-1` vector of observed function values at the training inputs
- `s2` a `N-by-1` vector of estimated noise value (variance) at the training inputs (this can be omitted, in which case the noise at the input locations is unknown)
