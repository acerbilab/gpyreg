"""Tests of the bounds the model components recommend from a training set."""

import warnings

import numpy as np
import pytest

import gpyreg as gpr
from gpyreg.covariance_functions import (
    Matern,
    RationalQuadraticARD,
    SquaredExponential,
    _target_spread,
)
from gpyreg.isotropic_covariance_functions import (
    MaternIsotropic,
    SquaredExponentialIsotropic,
)
from gpyreg.mean_functions import ConstantMean, NegativeQuadratic
from gpyreg.noise_functions import GaussianNoise

EQUAL_TARGET = 1.3


def _inputs(N=12, D=2, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, (N, D))


def _equal_targets(N=12):
    return np.full((N, 1), EQUAL_TARGET)


@pytest.mark.parametrize(
    "component",
    [
        SquaredExponential(),
        Matern(3),
        RationalQuadraticARD(),
        SquaredExponentialIsotropic(),
        MaternIsotropic(3),
        ConstantMean(),
        NegativeQuadratic(),
        GaussianNoise(constant_add=True),
    ],
    ids=lambda component: type(component).__name__,
)
def test_equal_targets_give_usable_bounds(component):
    """Targets that are all equal have a range of zero, and the bounds
    built from its logarithm are infinite: the output scale of a kernel
    gets ``(-inf, -inf)``, which the optimizer cannot take. Such a training
    set is given a range of one instead, with a warning, and every bound
    is then finite and in order."""
    X = _inputs()
    y = _equal_targets(X.shape[0])

    with pytest.warns(UserWarning, match="all equal"):
        info = component.get_bounds_info(X, y)

    for key in ("LB", "UB", "PLB", "PUB", "x0"):
        assert np.all(np.isfinite(info[key])), key
    assert np.all(info["LB"] <= info["PLB"])
    assert np.all(info["PLB"] <= info["PUB"])
    assert np.all(info["PUB"] <= info["UB"])
    assert np.all(info["LB"] <= info["x0"])
    assert np.all(info["x0"] <= info["UB"])


@pytest.mark.parametrize(
    "y", [[[0.3], [np.nan]], [[np.inf], [np.inf]]], ids=["nan", "inf"]
)
def test_targets_without_a_range_are_not_called_equal(y):
    """Targets with a NaN among them, or all the same infinity, have a NaN
    range, which is returned as it is: they get neither the warning that
    the targets are all equal nor the unit range."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        height, __ = _target_spread(np.array(y))
    assert np.isnan(height)
    assert not any("all equal" in str(w.message) for w in caught)


def test_fit_on_equal_targets_completes():
    """A log joint that is flat over the training set gives equal targets,
    and the fit went no further than the first optimizer call, which
    answered the pair ``(-inf, -inf)`` with a ``KeyError``."""
    X = _inputs()
    y = _equal_targets(X.shape[0])
    gp = gpr.GP(
        D=X.shape[1],
        covariance=SquaredExponential(),
        mean=NegativeQuadratic(),
        noise=GaussianNoise(constant_add=True),
    )

    with pytest.warns(UserWarning, match="all equal"):
        hyp, _, _ = gp.fit(
            X,
            y,
            options={"n_samples": 2, "init_N": 64},
            rng=np.random.default_rng(1),
        )

    assert np.all(np.isfinite(hyp))


def _columns_apart(N=20, seed=2):
    """Inputs whose three columns differ in location and in spread."""
    location = np.array([0.0, 10.0, -50.0])
    spread = np.array([0.5, 2.0, 20.0])
    rng = np.random.default_rng(seed)
    return location + spread * rng.uniform(-1, 1, (N, location.size))


@pytest.mark.parametrize(
    "kernel",
    [SquaredExponential(), Matern(3), RationalQuadraticARD()],
    ids=lambda kernel: type(kernel).__name__,
)
def test_length_scales_start_at_the_spread_of_their_column(kernel):
    """The starting value of a length scale is the logarithm of the
    standard deviation of its own column of the training inputs, as the
    bounds beside it are built from that column's width. It was the
    standard deviation over all the entries of ``X``, one number for
    every dimension, which mixes the locations of the columns with their
    spreads."""
    X = _columns_apart()
    N, D = X.shape
    y = np.random.default_rng(3).normal(size=(N, 1))

    info = kernel.get_bounds_info(X, y)

    np.testing.assert_array_equal(
        info["x0"][:D], np.log(np.std(X, axis=0, ddof=1))
    )


def test_negative_quadratic_bounds_are_per_column():
    """The location of the negative quadratic mean has, in each
    dimension, a hard box, a plausible box and a starting value built
    from the minimum, the maximum and the median of that column of the
    training inputs, and its scale has them built from the column's
    width and standard deviation. Each was one number for every
    dimension, from the statistics over all the entries of ``X``."""
    X = _columns_apart()
    N, D = X.shape
    y = np.random.default_rng(3).normal(size=(N, 1))
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    width = high - low
    location = slice(1, 1 + D)
    scale = slice(1 + D, 1 + 2 * D)

    info = NegativeQuadratic().get_bounds_info(X, y)

    assert_equal = np.testing.assert_array_equal
    assert_equal(info["LB"][location], low - 0.5 * width)
    assert_equal(info["UB"][location], high + 0.5 * width)
    assert_equal(info["PLB"][location], low)
    assert_equal(info["PUB"][location], high)
    assert_equal(info["x0"][location], np.median(X, axis=0))
    assert_equal(info["LB"][scale], np.log(width) + np.log(1e-6))
    assert_equal(info["UB"][scale], np.log(width) + 3)
    assert_equal(info["PLB"][scale], np.log(width) + 0.5 * np.log(1e-6))
    assert_equal(info["PUB"][scale], np.log(width))
    assert_equal(info["x0"][scale], np.log(np.std(X, axis=0, ddof=1)))
