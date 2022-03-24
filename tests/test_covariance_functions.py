import numpy as np
import pytest
from gpyreg.covariance_functions import Matern, SquaredExponential, RationalQuadraticARD

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


def test_kernel_gradient(kernel_fun=None, x0=None):
    pass