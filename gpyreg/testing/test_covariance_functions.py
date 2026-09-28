import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

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
    X_star = np.zeros((N, D))

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
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(D + 1)
        squared_expontential.compute(hyp, X, X_star, compute_grad=True)
    assert (
        "X_star should be None when compute_grad is True."
        in execinfo.value.args[0]
    )


@pytest.mark.parametrize("seed", [0, 3, 42])
def test_sqr_exp_kernel_gradient(seed):
    rng = np.random.RandomState(seed)
    sqr_exp = SquaredExponential()
    D = 3
    N = 20
    diag_cov = np.eye(N) * (0.2)
    X = (rng.multivariate_normal(np.zeros(N), diag_cov, D)).T
    hyp_D = D + 1
    diag_cov = np.eye(hyp_D) * (0.2)
    hyp = rng.multivariate_normal(np.zeros(hyp_D), diag_cov)
    _test_kernel_gradient_(sqr_exp, hyp, X)


def test_matern_compute_sanity_checks():
    matern = Matern(3)
    D = 3
    N = 20
    X = np.ones((N, D))
    X_star = np.zeros((N, D))

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
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(D + 1)
        matern.compute(hyp, X, X_star, compute_grad=True)
    assert (
        "X_star should be None when compute_grad is True."
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


@pytest.mark.parametrize("degree", [1, 3, 5])
@pytest.mark.parametrize("seed", [0, 3, 42])
def test_matern_kernel_gradient(seed, degree):
    rng = np.random.RandomState(seed)
    matern_fun = Matern(degree)
    D = 3
    N = 20
    diag_cov = np.eye(N) * (0.2)
    X = (rng.multivariate_normal(np.zeros(N), diag_cov, D)).T
    hyp_D = D + 1
    diag_cov = np.eye(hyp_D) * (0.2)
    hyp = rng.multivariate_normal(np.zeros(hyp_D), diag_cov)

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


@pytest.mark.parametrize("seed", [0, 3, 42])
def test_rqard_kernel_gradient(seed):
    rng = np.random.RandomState(seed)
    D = 3
    N = 20
    diag_cov = np.eye(N) * (0.2)
    X = (rng.multivariate_normal(np.zeros(N), diag_cov, D)).T
    hyp_D = D + 2
    diag_cov = np.eye(hyp_D) * (0.2)
    hyp = rng.multivariate_normal(np.zeros(hyp_D), diag_cov)

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

    finite_diff = np.zeros((K.shape[0], K.shape[1], len(hyp)))

    for idx, h_p in enumerate(hyp.squeeze()):
        # Differentiate at the original point, not the previous stencil's
        # final -2*h perturbation of another hyperparameter.
        hyp_new = hyp.copy()
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


def test_rational_quad_ard_plausible_upper_bounds():
    """The plausible upper bound of the output scale is the range of the
    targets and the shape's is 5, its own hard upper bound. The shape's
    line wrote into the output scale's slot, which left the shape's
    plausible upper bound at infinity and lost the output scale's."""
    rq_ard = RationalQuadraticARD()
    rng = np.random.default_rng(0)
    D = 3
    X = rng.uniform(-1.0, 1.0, (20, D))
    y = rng.normal(size=(20, 1))

    info = rq_ard.get_bounds_info(X, y)

    assert np.all(np.isfinite(info["PUB"]))
    assert info["PUB"][D] == np.log(np.max(y) - np.min(y))
    assert info["PUB"][D + 1] == 5.0


@pytest.mark.parametrize(
    "kernel",
    [RationalQuadraticARD(), Matern(1), Matern(3), Matern(5)],
    ids=lambda kernel: type(kernel).__name__
    + str(getattr(kernel, "degree", "")),
)
@pytest.mark.parametrize("D", [1, 4])
def test_length_scale_gradients_equal_their_per_dimension_formula(kernel, D):
    """Each length scale's gradient equals, to the last bit, the direct
    formula for its dimension: ``sf2 * M ** (-alpha - 1) * Ki`` for the
    rational-quadratic kernel, ``sf2 * (df(t) * exp(-t)) * Ki`` for Matern,
    zero where ``Ki`` is. Two inputs share their first coordinate, which
    puts a zero of that gradient off the diagonal; at D = 1 they coincide,
    where Matern's d=1 factor is infinite."""
    rng = np.random.default_rng(D)
    N = 30
    X = rng.normal(size=(N, D))
    X[1, 0] = X[0, 0]
    hyp = rng.normal(scale=0.5, size=kernel.hyperparameter_count(D))
    ell = np.exp(hyp[0:D])
    sf2 = np.exp(2 * hyp[D])

    _, dK = kernel.compute(hyp, X, compute_grad=True)

    with np.errstate(all="ignore"):
        for i in range(D):
            if isinstance(kernel, RationalQuadraticARD):
                alpha = np.exp(hyp[D + 1])
                tmp = squareform(pdist(X @ np.diag(1.0 / ell), "sqeuclidean"))
                M = 1 + 0.5 * tmp / alpha
                Ki = squareform(
                    pdist(
                        np.reshape(1.0 / ell[i] * X[:, i], (-1, 1)),
                        "sqeuclidean",
                    )
                )
                expected = sf2 * M ** (-alpha - 1) * Ki
            else:
                scale = np.sqrt(kernel.degree)
                tmp = squareform(pdist(X @ np.diag(scale / ell)))
                Ki = squareform(
                    pdist(
                        np.reshape(scale / ell[i] * X[:, i], (-1, 1)),
                        "sqeuclidean",
                    )
                )
                expected = np.where(
                    Ki > 0, sf2 * (kernel.df(tmp) * np.exp(-tmp)) * Ki, 0.0
                )
            assert np.array_equal(dK[:, :, i], expected)


@pytest.mark.parametrize(
    "kernel",
    [SquaredExponential(), Matern(3), RationalQuadraticARD()],
    ids=lambda kernel: type(kernel).__name__,
)
def test_kernel_refuses_a_gradient_of_the_diagonal(kernel):
    """The gradient is the gradient of the full covariance matrix. Asked
    for the diagonal and the gradient at once the kernels returned an
    (N, 1) value beside an (N, N, cov_N) gradient, a pair that means
    nothing."""
    D = 3
    X = np.ones((20, D))
    hyp = np.ones(kernel.hyperparameter_count(D))

    with pytest.raises(ValueError) as execinfo:
        kernel.compute(hyp, X, compute_diag=True, compute_grad=True)
    assert "cannot both be True" in execinfo.value.args[0]
