import numpy as np
import pytest

from gpyreg.mean_functions import ConstantMean, NegativeQuadratic, ZeroMean


def test_constant_mean_compute_sanity_checks():
    constantmean = ConstantMean()
    D = 3
    N = 20
    X = np.ones((N, D))

    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(D + 2)
        constantmean.compute(hyp, X)
    assert "Expected 1 mean function hyperparameters" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones((1, 1))
        constantmean.compute(hyp, X)
    assert (
        "Mean function output is available only for" in execinfo.value.args[0]
    )


def test_negative_quadratic_compute_sanity_checks():
    negative_quadratic = NegativeQuadratic()
    D = 3
    N = 20
    X = np.ones((N, D))

    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(D + 2)
        negative_quadratic.compute(hyp, X)
    assert "Expected 7 mean function hyperparameters" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones((7, 1))
        negative_quadratic.compute(hyp, X)
    assert (
        "Mean function output is available only for" in execinfo.value.args[0]
    )


def test_zero_mean_compute_sanity_checks():
    zeromean = ZeroMean()
    D = 3
    N = 20
    X = np.ones((N, D))

    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(D + 2)
        zeromean.compute(hyp, X)
    assert "Expected 0 mean function hyperparameters" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones((0, 0))
        zeromean.compute(hyp, X)
    assert (
        "Mean function output is available only for" in execinfo.value.args[0]
    )
