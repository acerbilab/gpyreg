"""Shared utilities for testing."""

import numdifftools as nd
import numpy as np


def partial(f, x0_orig, x0_i, i):
    """Evaluate function with one parameter varied."""
    x0 = x0_orig.copy()
    x0[i] = x0_i
    return f(x0)


def compute_gradient(f, x0):
    """Compute numerical gradient using numdifftools."""
    num_grad = np.zeros(x0.shape)

    for i in range(0, np.size(x0)):
        f_i = lambda x0_i: partial(f, x0, x0_i, i)
        tmp = nd.Derivative(f_i)(x0[i])
        num_grad[i] = tmp

    return num_grad


def check_grad(f, grad, x0):
    """Compare analytical and numerical gradients."""
    analytical_grad = grad(x0)
    numerical_grad = compute_gradient(f, x0)
    return np.abs(analytical_grad - numerical_grad)


def gauss_hermite_quadrature_reference(gp, mu, sigma, n_nodes=80):
    """Integrate the latent posterior of a GP against N(mu, diag(sigma^2)).

    Returns the mean and variance of the integral for each hyperparameter
    sample, computed by tensor-product Gauss-Hermite quadrature of
    ``gp.predict_full`` with ``add_noise=False``, which is the quantity
    ``GP.quad`` estimates. ``mu`` and ``sigma`` are scalars or arrays of
    length ``gp.D``; ``n_nodes`` is the number of nodes per dimension.
    """
    D = gp.D
    mu = np.broadcast_to(np.asarray(mu, dtype=float), (D,))
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (D,))
    t, w = np.polynomial.hermite.hermgauss(n_nodes)
    nodes = np.stack(
        [g.ravel() for g in np.meshgrid(*([t] * D), indexing="ij")], axis=1
    )
    weights = np.prod(
        np.stack(
            [g.ravel() for g in np.meshgrid(*([w] * D), indexing="ij")],
            axis=1,
        ),
        axis=1,
    ) / np.sqrt(np.pi) ** D
    x_star = mu + np.sqrt(2) * sigma * nodes
    f_mu, f_cov = gp.predict_full(x_star, add_noise=False)
    F = weights @ f_mu
    F_var = np.array(
        [weights @ f_cov[:, :, s] @ weights for s in range(f_mu.shape[1])]
    )
    return F, F_var
