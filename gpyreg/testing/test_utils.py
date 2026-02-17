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
