import numpy as np
import pytest
from scipy.stats import (
    beta,
    expon,
    multivariate_normal,
    multivariate_t,
    norm,
    uniform,
)

from gpyreg.slice_sample import SliceSampler

options = {"display": "off", "diagnostics": True}
threshold = 0.1


def test_multiple_runs():
    state = np.random.get_state()

    np.random.seed(1234)
    slicer1 = SliceSampler(norm.logpdf, np.array([0.5]), options=options)
    res1 = slicer1.sample(300)

    np.random.seed(1234)
    slicer2 = SliceSampler(norm.logpdf, np.array([0.5]), options=options)
    res2 = slicer2.sample(100, burn=100)
    res3 = slicer2.sample(100)
    res4 = slicer2.sample(100)

    np.random.set_state(state)

    assert np.all(
        res1["samples"]
        == np.concatenate((res2["samples"], res3["samples"], res4["samples"]))
    )


# The following tests can fail with some small probability.


def test_normal():
    slicer = SliceSampler(norm.logpdf, np.array([0.5]), options=options)
    samples = slicer.sample(20000)["samples"]

    assert np.abs(norm.mean() - np.mean(samples)) < threshold
    assert np.abs(norm.var() - np.var(samples)) < threshold


def test_normal_step_out():
    new_options = options = {
        "display": "off",
        "diagnostics": True,
        "step_out": True,
    }
    slicer = SliceSampler(norm.logpdf, np.array([0.5]), options=new_options)
    samples = slicer.sample(20000)["samples"]

    assert np.abs(norm.mean() - np.mean(samples)) < threshold
    assert np.abs(norm.var() - np.var(samples)) < threshold


def test_normal_mixture():
    p = 0.7
    rv1 = norm(0, 1)
    rv2 = norm(6, 2)
    pdf = lambda x: p * rv1.pdf(x) + (1 - p) * rv2.pdf(x)
    logpdf = lambda x: np.log(pdf(x))  # if pdf(x) > np.spacing(0) else -np.inf
    slicer = SliceSampler(logpdf, np.array([0.5]), options=options)
    samples = slicer.sample(20000)["samples"]

    assert np.abs((1 - p) * 6 - np.mean(samples)) < threshold
    # plt.scatter(samples , pdf(samples))
    # plt.show()


def test_exponential():
    slicer = SliceSampler(
        expon.logpdf, np.array([0.5]), LB=0.0, options=options
    )
    samples = slicer.sample(20000)["samples"]

    # plt.scatter(samples, expon.pdf(samples))
    # plt.show()
    assert np.abs(expon.mean() - np.mean(samples)) < threshold
    assert np.abs(expon.var() - np.var(samples)) < threshold


def test_uniform():
    slicer = SliceSampler(
        uniform.logpdf, np.array([0.5]), LB=0.0, UB=1.0, options=options
    )
    samples = slicer.sample(20000)["samples"]

    assert np.abs(uniform.mean() - np.mean(samples)) < threshold
    assert np.abs(uniform.var() - np.var(samples)) < threshold


def test_beta():
    a, b = 2.31, 0.627
    rv = beta(a, b)
    slicer = SliceSampler(
        rv.logpdf, np.array([0.5]), LB=0.0, UB=1.0, options=options
    )
    samples = slicer.sample(20000)["samples"]

    assert np.abs(rv.mean() - np.mean(samples)) < threshold
    assert np.abs(rv.var() - np.var(samples)) < threshold


def test_multivariate_normal():
    mean = np.array([0.68, 0.6, 0.4])
    cov = np.array(
        [[1.58, 0.96, -1.2], [0.96, 2.17, -1.725], [-1.2, -1.725, 1.85]]
    )
    rv = multivariate_normal(mean, cov)
    slicer = SliceSampler(
        rv.logpdf, np.array([0.5, -0.5, 1.0]), options=options
    )
    samples = slicer.sample(20000)["samples"]

    assert np.all(np.abs(mean - np.mean(samples, axis=0)) < threshold)
    # assert np.all(np.abs(cov - np.cov(samples.T)) < threshold)


def test_multivariate_t():
    x = [1.0, -0.5]
    loc = [[2.1, 0.3], [0.3, 1.5]]
    rv = multivariate_t(x, loc, df=3)
    slicer = SliceSampler(rv.logpdf, np.array([0.5, 0.5]), options=options)
    samples = slicer.sample(20000)["samples"]

    assert np.all(np.abs(x - np.mean(samples, axis=0)) < threshold)


def test_init_sanity_checks():
    """
    Just some basic tests to check for incorrect input for __init__.
    """
    x = [1.0, -0.5]
    loc = [[2.1, 0.3], [0.3, 1.5]]
    rv = multivariate_t(x, loc, df=3)
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2, 2)))
    assert "initial point x0 needs to be a scalar" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2)), LB=np.zeros((2, 2)))
    assert "LB and UB need to be None, scalars" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(
            rv.logpdf, np.zeros((2)), LB=np.zeros((2, 2)), UB=np.ones((2, 2))
        )
    assert "LB and UB need to be None, scalars" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2)), UB=np.zeros((2, 2)))
    assert "LB and UB need to be None, scalars" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2)), LB=1, UB=0)
    assert "UB need to be equal or greater than" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2)), widths=-1, UB=0)
    assert (
        "The widths vector needs to be all positive real numbers"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2)), widths=1 + 2j, UB=0)
    assert (
        "The widths vector needs to be all positive real numbers"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2)), LB=1, UB=2)
    assert (
        "The initial starting point X0 is outside the bounds"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2)), LB=-2, UB=-1)
    assert (
        "The initial starting point X0 is outside the bounds"
        in execinfo.value.args[0]
    )


def test_init_logger():
    mean = np.ones(3)
    cov = np.eye(3)
    rv = multivariate_normal(mean, cov)
    options = {"display": "off"}
    slicer = SliceSampler(rv.logpdf, np.ones(3), options=options)
    assert slicer.logger.getEffectiveLevel() == 30  # WARNING
    options = {"display": "summary"}
    slicer = SliceSampler(rv.logpdf, np.ones(3), options=options)
    assert slicer.logger.getEffectiveLevel() == 20  # INFO
    options = {"display": "full"}
    slicer = SliceSampler(rv.logpdf, np.ones(3), options=options)
    assert slicer.logger.getEffectiveLevel() == 10  # DEBUG


def test_sample_sanity_checks():
    """
    Just some basic tests to check for incorrect input for sample.
    """
    mean = np.ones(3)
    cov = np.eye(3)
    rv = multivariate_normal(mean, cov)
    slicer = SliceSampler(rv.logpdf, np.ones(3))
    with pytest.raises(ValueError) as execinfo:
        slicer.sample(3, thin=-1)
    assert (
        "The thinning factor option needs to be a positive integer"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        slicer.sample(3, thin=np.ones((3, 3)))
    assert (
        "The thinning factor option needs to be a positive integer"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        slicer.sample(3, burn=-1)
    assert (
        "burn-in samples option needs to be a non-negative"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        slicer.sample(3, burn=np.ones((3, 3)))
    assert (
        "burn-in samples option needs to be a non-negative"
        in execinfo.value.args[0]
    )
    slicer.x0 = slicer.x0 * np.NaN
    with pytest.raises(ValueError) as execinfo:
        slicer.sample(3)
    assert (
        "The initial starting point X0 needs to evaluate to a"
        in execinfo.value.args[0]
    )
