import numpy as np
import gpyreg as gpr

from scipy.stats import norm

# Create toy training data for the GP
np.random.seed(1234)
N = 20
D = 2
x_data = np.random.uniform(low = -3, high = 3, size = (N, D))
y_data = np.reshape(np.sin(np.sum(x_data, 1)) + norm.ppf(np.random.random_sample(size = x_data.shape[0]), scale=0.1), (-1, 1))  # np.random.normal(scale = 0.1, size = x_data.shape[0])

# Define the GP model

# the likelihood function is assumed to be Gaussian with variance
# sigma_const**2 + s2 where s2 is a user-provided array of 
# observation noise variance at each training input (if s2 is not
# provided, it is assumed to be 0)

# sigma_const is generally fit to the data (noise_constant_fit)

gp = gpr.GP(
    D = D,
    covariance = gpr.covariance_functions.SquaredExponential(),
    mean = gpr.mean_functions.ConstantMean(),
    noise = gpr.noise_functions.GaussianNoise([1, 0, 0])
)

# Define the priors of the GP hyperparameters (supported priors
# are 'gaussian', 'studentt', 'smoothbox', 'smoothbox_studentt')
gp_priors = {'covariance_log_outputscale' :
    ('studentt', (0, np.log(10), 3)),
    'covariance_log_lengthscale' :
    ('gaussian', (np.log(np.std(x_data)), np.log(10))),
    'noise_log_scale' : 
    ('gaussian', (np.log(1e-3), 1.0)),
    'mean_const' :
    ('smoothbox', (np.min(y_data), np.max(y_data), 1.0))
}

# Assign the hyperparameter priors to the gp model
gp.set_priors(gp_priors)

# Define the GP training options.
gp_train = {'n_samples' : 10}

# Train the GP
gp.fit(
    hyp = np.loadtxt('hyp2.txt', delimiter=',', usecols=range(10)),
    x = x_data,
    y = y_data,
    options = gp_train
)

# Create test point regular grid (400-by-D) points
xx, yy = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))
x_star = np.array((xx.ravel(), yy.ravel())).T

# Predictive latent mean and variance of the gp at test points
# ("latent" means that we do not add observation noise)
_, _, fmu, fs2 = gp.predict(x_star, add_noise = False)

# Plot the GP
gp.plot()

assert(False)

# Update the GP by adding some extra points

x_new = np.random.uniform(low = -5, high = 5, size = (N, D))
y_new = np.sin(np.sum(x_new, 1)) + np.random.normal(scale = 0.1, size = x_new.shape[0])

# This function updates the training data and (usally) the GP posterior but does not 
# retrain the GP hyperparameters - it also fills in the auxiliary data that might have
# been stripped out.
gp.update(x_new = x_new, y_new = y_new, compute_posterior = False)

# In the case above we did not recompute the posterior as we are 
# anyhow retraining the GP right after

# Retrain the GP (the data are already inside the GP object, and
# include both the original data and the new data)
gp.train(options = gp_train)
