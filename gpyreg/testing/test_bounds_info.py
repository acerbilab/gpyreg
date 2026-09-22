"""Tests of the bounds the model components recommend from a training set."""

import numpy as np
import pytest

import gpyreg as gpr
from gpyreg.covariance_functions import (
    Matern,
    RationalQuadraticARD,
    SquaredExponential,
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
