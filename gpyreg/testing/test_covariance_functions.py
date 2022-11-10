import numpy as np
import pytest

from gpyreg.covariance_functions import (
    AbstractKernel,
    Matern,
    RationalQuadraticARD,
    SquaredExponential,
)


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
    diag_cov = np.eye(N) * (0.2)
    X = (np.random.multivariate_normal(np.zeros(N), diag_cov, D)).T
    hyp_D = D + 1
    diag_cov = np.eye(hyp_D) * (0.2)
    hyp = np.random.multivariate_normal(np.zeros(hyp_D), diag_cov)
    _test_kernel_gradient_(sqr_exp, hyp, X)


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
    diag_cov = np.eye(N) * (0.2)
    X = (np.random.multivariate_normal(np.zeros(N), diag_cov, D)).T
    hyp_D = D + 1
    diag_cov = np.eye(hyp_D) * (0.2)
    hyp = np.random.multivariate_normal(np.zeros(hyp_D), diag_cov)

    _test_kernel_gradient_(matern_fun, hyp, X)


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


def test_cov_rational_quad_ard():
    rq_ard = RationalQuadraticARD()
    X = np.array([[0.343, 0.967, 0.724]]).T
    hyp = np.array([0.5, 0.6, 0.4])
    res = rq_ard.compute(hyp, X)
    assert np.all(
        np.array([res[1, 0] - 3.3201, res[1, 0] - 3.0958, res[2, 0] - 3.2334])
        < 1e-5
    ) and np.allclose(res, res.T)


def test_simple_rational_quad_ard():
    rq_ard = RationalQuadraticARD()
    D = 3
    N = 20
    X = np.ones((N, D))
    hyp = np.ones(D + 2)
    res = rq_ard.compute(hyp, X)
    assert np.allclose(res[0, 0], np.array([7.389])) and np.allclose(
        res, res.T
    )


def test_rqard_kernel_gradient():
    D = 3
    N = 20
    diag_cov = np.eye(N) * (0.2)
    X = (np.random.multivariate_normal(np.zeros(N), diag_cov, D)).T
    hyp_D = D + 2
    diag_cov = np.eye(hyp_D) * (0.2)
    hyp = np.random.multivariate_normal(np.zeros(hyp_D), diag_cov)

    rq_ard = RationalQuadraticARD()
    _test_kernel_gradient_(rq_ard, hyp, X)


def _test_kernel_gradient_(
    kernel_fun: AbstractKernel,
    hyp,
    X: np.ndarray,
    X_star: np.ndarray = None,
    h=1e-5,
    eps=1e-4,
):
    """
    Test the gradient of the kernel function via the Five-point stencil difference method (https://en.wikipedia.org/wiki/Five-point_stencil).

    Parameters
    ----------
    kernel_fun : AbstractKernel

    X : ndarray, shape (N, D)

    hyp : ndarray, shape (cov_N,)
        A 1D array of hyperparameters, where ``cov_N`` is
        the number of hyperparameters.
    h: float
        Grid spacing.
    eps: float
        Error tolerance.
    """

    K, dK = kernel_fun.compute(hyp, X, X_star, compute_grad=True)

    hyp_new = hyp.copy()
    finite_diff = np.zeros((K.shape[0], K.shape[1], len(hyp)))

    for idx, h_p in enumerate(hyp.squeeze()):
        hyp_new[idx] = h_p + 2.0 * h
        f_2h = kernel_fun.compute(hyp_new, X, X_star)
        hyp_new[idx] = h_p

        hyp_new[idx] = h_p + h
        f_h = kernel_fun.compute(hyp_new, X, X_star)
        hyp_new[idx] = h_p

        hyp_new[idx] = h_p - h
        f_neg_h = kernel_fun.compute(hyp_new, X, X_star)
        hyp_new[idx] = h_p

        hyp_new[idx] = h_p - 2 * h
        f_neg_2h = kernel_fun.compute(hyp_new, X, X_star)

        finite_diff[:, :, idx] = -f_2h + 8.0 * f_h - 8.0 * f_neg_h + f_neg_2h
        finite_diff[:, :, idx] = finite_diff[:, :, idx] / (12 * h)

    assert np.all(np.abs(finite_diff - dK) <= eps)


test_simple_rational_quad_ard()
