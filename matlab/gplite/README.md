# gplite
Lite Gaussian process regression toolbox

The main structure of the toolbox is that:
- `gplite_train` fits a GP model to some data (by either optimizing the GP hyperparameters or sampling from the posterior of GP hyperparamers), returning a gp struct.
- `gplite_post` is a main GP function that computes the posterior GP given some data, for given GP hyperparameters.
- During training, the main function used is `gplite_nlZ` which computes the GP log marginal likelihood (possibly adding a prior term to it) for given GP hyperparameters.
  - The log marginal likelihood is the quantity which is either maximized or used as target for MCMC sampling.
- `gplite_post` and `gplite_nlZ` use very similar computations, which are in a private function (in the private folder) `gplite_core`.
- Given a trained `gp` struct, `gplite_pred` is used to make predictions of mean and variance at new input points.
