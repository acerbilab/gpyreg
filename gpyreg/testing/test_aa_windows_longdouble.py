"""Diagnostic, not for merging: long-double inputs on gpyreg main's kernels
and on SciPy's distance functions, to find where Windows' heap corruption
(0xc0000374) comes from."""
import sys

import numpy as np
import pytest
import scipy
from scipy.spatial.distance import cdist, pdist, squareform

from gpyreg.covariance_functions import (
    Matern,
    RationalQuadraticARD,
    SquaredExponential,
)


def test_platform():
    print(sys.version, np.__version__, scipy.__version__)
    print("longdouble", np.finfo(np.longdouble), np.dtype(np.longdouble).itemsize)
    print("g == d:", np.dtype("g") == np.dtype("d"))
    scipy.show_config()


@pytest.mark.parametrize("step", range(6))
def test_scipy_distances_on_long_double(step):
    rng = np.random.default_rng(step)
    a = rng.normal(size=(40, 3)).astype(np.longdouble)
    for _ in range(20):
        if step % 3 == 0:
            d = cdist(a, a, "sqeuclidean")
        elif step % 3 == 1:
            d = cdist(a, a)
        else:
            d = squareform(pdist(a))
        junk = [np.empty(1000 * (k + 1)) for k in range(50)]
        del junk
    assert d.shape == (40, 40)


@pytest.mark.parametrize(
    "kernel", [SquaredExponential(), Matern(1), Matern(3), Matern(5), RationalQuadraticARD()]
)
def test_main_kernels_on_long_double(kernel):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(40, 3)).astype(np.longdouble)
    for seed in range(3):
        hyp = np.random.default_rng(seed).normal(scale=0.5, size=kernel.hyperparameter_count(3))
        with np.errstate(all="ignore"):
            kernel.compute(hyp, X, compute_grad=True)
            kernel.compute(hyp, X)
            kernel.compute(hyp, X, rng.normal(size=(9, 3)))
        junk = [np.empty(1000 * (k + 1)) for k in range(50)]
        del junk
