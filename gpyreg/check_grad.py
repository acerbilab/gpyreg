import numpy as np
import scipy as sp

def __partial(f, x0_orig, x0_i, i):
    x0 = x0_orig.copy()
    x0[i] = x0_i
    return f(x0)

def __compute_gradient(f, x0):
    num_grad = np.zeros(x0.shape)

    for i in range(0, np.size(x0)):
        f_i = lambda x0_i : __partial(f, x0, x0_i, i)
        num_grad[i] = sp.misc.derivative(f_i, x0[i], dx=np.finfo(float).eps**(1/5.0), order=5)

    return num_grad

def check_grad(f, grad, x0):
    # print(x0)
    analytical_grad = grad(x0)
    # print("Analytical: ", analytical_grad)
    numerical_grad = __compute_gradient(f, x0)
    # print("Numerical:", numerical_grad)
    return np.sum(np.abs(analytical_grad - numerical_grad))
