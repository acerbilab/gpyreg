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
y[y < 0] = -np.abs(3 * y[y < 0]) ** 2

# Define the GP model
gp = gpr.GP(
    D=D,
    covariance=gpr.covariance_functions.Matern(degree=3),
    mean=gpr.mean_functions.NegativeQuadratic(),
    noise=gpr.noise_functions.GaussianNoise(
        constant_add=True, user_provided_add=True
    ),
)

# Define the priors of the GP hyperparameters (supported priors
# are 'gaussian', 'student_t', 'smoothbox', 'smoothbox_student_t')
gp_priors = {
    "covariance_log_lengthscale": None,
    "covariance_log_outputscale": None,
    "mean_const": None,
    "mean_location": None,
    "mean_log_scale": None,
    "noise_log_scale": ("student_t", (np.log(1e-3), 1.0, 7)),
}

# Assign the hyperparameter priors to the gp model
gp.set_priors(gp_priors)

# Define the GP training options.
gp_train = {"n_samples": 10}

# Train the GP
gp.fit(X=X, y=y, s2=s2, options=gp_train)

x_star = np.reshape(np.linspace(-15, 15, 200), (-1, 1))
fmu, fs2 = gp.predict(x_star, add_noise=False)

# Plot the GP
gp.plot()
