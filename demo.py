import numpy as np
from scipy.stats import norm

import gpyreg as gpr

# Create example data in 1D
np.random.seed(1234)
N = 31
D = 1
X = -5 + np.random.rand(N, 1) * 10
s2 = 0.05 * np.exp(0.5 * X)
y = np.sin(X) + np.sqrt(s2) * norm.ppf(np.random.random_sample(X.shape))
y[y < 0] = -np.abs(3 * y[y < 0])**2

# Load hyperparameters from file
hyp = np.loadtxt('hyp.txt', delimiter=',', usecols=range(10)) 

# Define the GP model
gp = gpr.GP(
    D = D,
    covariance = gpr.covariance_functions.Matern(3),
    mean = gpr.mean_functions.NegativeQuadratic(),
    noise = gpr.noise_functions.GaussianNoise([1, 0, 0]),
    s2 = s2
)

# Define the priors of the GP hyperparameters (supported priors
# are 'gaussian', 'studentt', 'smoothbox', 'smoothbox_studentt')
gp_priors = {
    'noise_log_scale' : 
    ('gaussian', (np.log(1e-3), 1.0)),
}

# Assign the hyperparameter priors to the gp model
gp.set_priors(gp_priors)

# Define the GP training options.
gp_train = {'n_samples' : 10}

# Train the GP
gp.fit(
    hyp = hyp,
    x = X,
    y = y,
    options = gp_train
)

x_star = np.reshape(np.linspace(-15, 15, 200), (-1, 1))
ymu, ys2, fmu, fs2 = gp.predict(x_star)

# Plot the GP
gp.plot()
