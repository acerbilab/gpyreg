import numpy as np
import pytest
from gpyreg.covariance_functions import AbstractKernel, Matern, SquaredExponential, RationalQuadraticARD

def test_squared_exponential_compute_sanity_checks():
    squared_expontential = SquaredExponential()
    D = 3
    N = 20
    X = np.ones((N, D))

    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(D + 2)
        squared_expontential.compute(hyp, X)
    assert (
        "Expected 4 covariance function hyperparameters"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones((D + 1, 1))
        squared_expontential.compute(hyp, X)
    assert (
        "Covariance function output is available only for"
        in execinfo.value.args[0]
    )

def test_sqr_exp_kernel_gradient():
    sqr_exp = SquaredExponential()
    D = 3
    N = 20
    X = np.ones((N, D))
    hyp = np.ones(D + 1)
    _test_kernel_gradient_(sqr_exp, X, hyp)

def test_matern_compute_sanity_checks():
    matern = Matern(3)
    D = 3
    N = 20
    X = np.ones((N, D))

    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(D + 2)
        matern.compute(hyp, X)
    assert (
        "Expected 4 covariance function hyperparameters"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones((D + 1, 1))
        matern.compute(hyp, X)
    assert (
        "Covariance function output is available only for"
        in execinfo.value.args[0]
    )

def test_matern_invalid_degree():
    for degree in [0, 2, 4, 6]:
        with pytest.raises(ValueError) as execinfo:
            Matern(degree)
        assert (
            "Only degrees 1, 3 and 5 are supported for the"
            in execinfo.value.args[0]
        )

def test_matern_kernel_gradient():
    matern_fun = Matern(3)
    D = 3
    N = 20
    X = np.ones((N, D))
    hyp = np.ones(D + 1)

    _test_kernel_gradient_(matern_fun, X, hyp)

def test_rational_quad_ard_checks():
    rq_ard = RationalQuadraticARD()
    D = 3
    N = 20
    X = np.ones((N, D))

    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(D + 3)
        rq_ard.compute(hyp, X)
    assert (
        "Expected 5 covariance function hyperparameters"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones((D + 2, 1))
        rq_ard.compute(hyp, X)
    assert (
        "Covariance function output is available only for"
        in execinfo.value.args[0]
    )

def test_simple_rational_quad_ard():
    rq_ard = RationalQuadraticARD()
    D = 3
    N = 20
    X = np.ones((N, D))
    hyp = np.ones(D + 2)
    print(hyp.size)
    res = rq_ard.compute(hyp, X)
    eps = 0.001
    assert (np.all(res == res[0, 0]) and np.abs(res[0, 0] - 7.389) < eps)

def test_rqard_kernel_gradient():
    rq_ard = RationalQuadraticARD()
    D = 3
    N = 20
    X = np.ones((N, D))
    hyp = np.ones(D + 2)
    _test_kernel_gradient_(rq_ard, X, hyp)

def _test_kernel_gradient_(kernel_fun: AbstractKernel, X0:np.ndarray, hyp, h=1e-3, eps=1e-3):
    K, dK = kernel_fun.compute(hyp, X0, compute_grad=True)

    hyp_new = hyp.copy()
    finite_diff = np.zeros((K.shape[0], K.shape[1], len(hyp)))

    for idx, h_p in enumerate(hyp.squeeze()):
        hyp_new[idx] = h_p + 2.0 * h
        f_2h = kernel_fun.compute(hyp_new, X0)
        hyp_new[idx] = h_p
        
        hyp_new[idx] = h_p + h
        f_h = kernel_fun.compute(hyp_new, X0)
        hyp_new[idx] = h_p
        
        hyp_new[idx] = h_p - h
        f_neg_h = kernel_fun.compute(hyp_new, X0)
        hyp_new[idx] = h_p

        hyp_new[idx] = h_p - 2*h
        f_neg_2h = kernel_fun.compute(hyp_new, X0)

        finite_diff[:, :, idx] = -f_2h + 8.0 * f_h - 8.0 * f_neg_h + f_neg_2h
        finite_diff[:, :, idx] = finite_diff[:, :, idx] / (12 * h)
    
    assert np.all(np.abs(finite_diff - dK) <= eps)