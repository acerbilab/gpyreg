import numpy as np
from scipy.stats import norm

import gaussian_process as g
import covariance_functions 
import mean_functions
import noise_functions

# Create example data in 1D
np.random.seed(1234)
N = 31
X = -5 + np.random.rand(N, 1) * 10
s2 = 0.05 * np.exp(0.5 * X)
y = np.sin(X) + np.sqrt(s2) * norm.ppf(np.random.random_sample(X.shape))
y[y < 0] = -np.abs(3 * y[y < 0])**2

# Pick GP "parameters".
cov_fun = covariance_functions.Matern(3)
mean_fun = mean_functions.NegativeQuadratic()
noise_fun = noise_functions.PlaceholderNoise([1, 0, 0]) 
# Load hyperparameters from file
hyp = np.loadtxt('hyp.txt', delimiter=',', usecols=range(10)) 

gp = g.GP(X, y, s2, hyp, cov_fun, mean_fun, noise_fun)

x_star = np.reshape(np.linspace(-15, 15, 200), (-1, 1))
# ymu, ys2, fmu, fs2 = gp.predict(x_star)

gp.plot()
