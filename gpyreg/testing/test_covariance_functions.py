import copy
import pickle

import numdifftools as nd
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


_ARD_KERNELS = {
    "SquaredExponential": lambda periods: SquaredExponential(periods=periods),
    "Matern1": lambda periods: Matern(1, periods=periods),
    "Matern3": lambda periods: Matern(3, periods=periods),
    "Matern5": lambda periods: Matern(5, periods=periods),
    "RationalQuadraticARD": lambda periods: RationalQuadraticARD(
        periods=periods
    ),
}

_each_ard_kernel = pytest.mark.parametrize(
    "make_kernel", list(_ARD_KERNELS.values()), ids=list(_ARD_KERNELS)
)

# Two periodic dimensions around a non-periodic one. The periods and the
# first inputs are exact in binary: the second input lies exactly one
# period from the first along each periodic dimension and shares its other
# coordinate, and the fourth lies one period from the third along the
# first dimension only.
_PERIODS = np.array([1.5, np.inf, 0.75])


def _periodic_inputs(seed, N=12):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(N, 3))
    X[0] = [0.25, 0.5, -0.125]
    X[1] = [1.75, 0.5, -0.875]
    X[2] = [-0.5, 0.25, 0.375]
    X[3, 0] = 1.0
    return X


def _hyperparameters(kernel, seed, D=3):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.5, size=kernel.hyperparameter_count(D))


@_each_ard_kernel
@pytest.mark.parametrize("d", [0, 2])
@pytest.mark.parametrize("k", [1, -2, 5])
def test_periodic_kernel_repeats_with_the_period(make_kernel, d, k):
    """Shifting one set of inputs by a whole number of periods along a
    periodic dimension leaves the kernel as it is."""
    kernel = make_kernel(_PERIODS)
    X = _periodic_inputs(0)
    hyp = _hyperparameters(kernel, 1)
    X_shifted = X.copy()
    X_shifted[:, d] += k * _PERIODS[d]

    K = kernel.compute(hyp, X, X)

    assert np.allclose(
        kernel.compute(hyp, X, X_shifted), K, rtol=1e-12, atol=1e-14
    )


@_each_ard_kernel
def test_inputs_a_period_apart_are_the_same_input(make_kernel):
    """Two inputs a whole number of periods apart along every periodic
    dimension, and equal along the others, have the covariance of an input
    with itself, and the gradient of the length scales is zero there."""
    kernel = make_kernel(_PERIODS)
    X = _periodic_inputs(1)
    hyp = _hyperparameters(kernel, 2)

    K, dK = kernel.compute(hyp, X, compute_grad=True)

    assert K[0, 1] == K[0, 0]
    assert np.all(dK[0, 1, 0:3] == 0.0)
    assert dK[2, 3, 0] == 0.0


@_each_ard_kernel
def test_infinite_periods_give_the_kernel_without_periods(make_kernel):
    """Periods that are all infinite leave no dimension periodic: the
    kernel stores none, and computes what the kernel without periods
    computes, to the last bit."""
    kernel = make_kernel(np.full(3, np.inf))
    reference = make_kernel(None)
    X = _periodic_inputs(2)
    X_star = np.random.default_rng(3).normal(size=(5, 3))
    hyp = _hyperparameters(kernel, 4)

    assert kernel.periods is None
    K, dK = kernel.compute(hyp, X, compute_grad=True)
    K_ref, dK_ref = reference.compute(hyp, X, compute_grad=True)
    assert np.array_equal(K, K_ref)
    assert np.array_equal(dK, dK_ref)
    assert np.array_equal(
        kernel.compute(hyp, X, X_star), reference.compute(hyp, X, X_star)
    )
    assert np.array_equal(
        kernel.compute(hyp, X, compute_diag=True),
        reference.compute(hyp, X, compute_diag=True),
    )


@_each_ard_kernel
def test_large_periods_approach_the_kernel_without_periods(make_kernel):
    """The squared chord tends to the squared difference as the period
    grows: with periods a million times the spread of the inputs, the
    kernel and its gradient are those without periods to about 1e-12."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(15, 3))
    X_star = rng.normal(size=(6, 3))
    spread = np.ptp(np.vstack((X, X_star)), axis=0)
    kernel = make_kernel(1e6 * spread)
    reference = make_kernel(None)
    hyp = _hyperparameters(kernel, 6)

    K, dK = kernel.compute(hyp, X, compute_grad=True)
    K_ref, dK_ref = reference.compute(hyp, X, compute_grad=True)

    assert np.allclose(K, K_ref, rtol=1e-9, atol=0.0)
    assert np.allclose(
        dK, dK_ref, rtol=1e-9, atol=1e-12 * np.max(np.abs(dK_ref))
    )
    assert np.allclose(
        kernel.compute(hyp, X, X_star),
        reference.compute(hyp, X, X_star),
        rtol=1e-9,
        atol=0.0,
    )


@_each_ard_kernel
def test_huge_periods_stay_finite(make_kernel):
    """A period near the largest float, far beyond any length scale, gives
    finite values, the kernel's variance on the diagonal among them: the
    chord is formed before it is scaled, so that no product of the period
    and the scale overflows."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(5, 2))
    kernel = make_kernel([1e308, np.inf])
    hyp = _hyperparameters(kernel, 2, D=2)
    hyp[0] = -5.0  # a length scale of e^-5
    K, dK = kernel.compute(hyp, X, compute_grad=True)
    assert np.all(np.isfinite(K)) and np.all(np.isfinite(dK))
    assert np.allclose(np.diag(K), np.exp(2 * hyp[2]), rtol=1e-12, atol=0)


@_each_ard_kernel
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_periodic_kernel_gradient(make_kernel, seed):
    """The analytic gradient matches a numerical one, with periodic and
    non-periodic dimensions, at inputs that include a pair exactly one
    period apart, where Matern's d=1 factor is infinite."""
    kernel = make_kernel(_PERIODS)
    X = _periodic_inputs(seed)
    hyp = _hyperparameters(kernel, seed + 10)

    _, dK = kernel.compute(hyp, X, compute_grad=True)
    numerical = nd.Jacobian(lambda h: kernel.compute(h, X).ravel())(hyp)

    assert np.all(np.isfinite(dK))
    assert np.allclose(dK, numerical.reshape(dK.shape), rtol=1e-6, atol=1e-8)


@_each_ard_kernel
def test_periodic_kernel_diagonal_and_cross_covariance(make_kernel):
    """The diagonal and the cross-covariance are the corresponding parts
    of the full covariance matrix, which is exactly symmetric."""
    kernel = make_kernel(_PERIODS)
    X = _periodic_inputs(7)
    X_star = np.random.default_rng(8).uniform(-3.0, 3.0, size=(5, 3))
    hyp = _hyperparameters(kernel, 9)
    N = X.shape[0]

    K = kernel.compute(hyp, np.vstack((X, X_star)))

    assert np.array_equal(K, K.T)
    assert np.array_equal(
        kernel.compute(hyp, X, compute_diag=True), np.diag(K)[0:N, None]
    )
    assert np.allclose(
        kernel.compute(hyp, X, X_star), K[0:N, N:], rtol=1e-14, atol=0.0
    )


@_each_ard_kernel
@pytest.mark.parametrize(
    "periods, message",
    [
        ([1.0, np.nan], "positive number"),
        ([1.0, 0.0], "positive number"),
        ([-1.0, np.inf], "positive number"),
        ([1.0, -np.inf], "positive number"),
        ([[1.0, 2.0]], "one-dimensional"),
        (2.0, "one-dimensional"),
        ([], "one-dimensional"),
    ],
)
def test_kernel_refuses_invalid_periods(make_kernel, periods, message):
    with pytest.raises(ValueError) as execinfo:
        make_kernel(periods)
    assert message in execinfo.value.args[0]


@_each_ard_kernel
def test_kernel_refuses_periods_of_another_dimension(make_kernel):
    kernel = make_kernel([1.0, np.inf])
    X = np.zeros((4, 3))
    hyp = np.zeros(kernel.hyperparameter_count(3))

    with pytest.raises(ValueError) as execinfo:
        kernel.compute(hyp, X)
    assert "has 2 periods" in execinfo.value.args[0]


@_each_ard_kernel
def test_periods_are_fixed_constants(make_kernel):
    """The periods are stored as a float copy and add no hyperparameter."""
    periods = np.array([3, 1])
    kernel = make_kernel(periods)
    reference = make_kernel(None)
    X = np.array([[0.0, 1.0], [2.0, -1.0]])
    y = np.array([[0.0], [1.0]])

    periods[0] = 7
    assert kernel.periods.dtype == float
    assert np.array_equal(kernel.periods, [3.0, 1.0])
    assert kernel.hyperparameter_count(2) == reference.hyperparameter_count(2)
    assert kernel.hyperparameter_info(2) == reference.hyperparameter_info(2)
    bounds, bounds_ref = (
        kernel.get_bounds_info(X, y),
        reference.get_bounds_info(X, y),
    )
    for key, value in bounds_ref.items():
        assert np.array_equal(bounds[key], value)


@_each_ard_kernel
def test_kernel_without_a_periods_attribute_is_not_periodic(make_kernel):
    """A kernel unpickled from gpyreg 1.3.3 or earlier has no ``periods``
    of its own, and takes the class's ``None``."""
    kernel = make_kernel(None)
    del kernel.periods
    reference = make_kernel(None)
    X = _periodic_inputs(11)
    hyp = _hyperparameters(kernel, 12)

    assert kernel.periods is None
    K, dK = kernel.compute(hyp, X, compute_grad=True)
    K_ref, dK_ref = reference.compute(hyp, X, compute_grad=True)
    assert np.array_equal(K, K_ref)
    assert np.array_equal(dK, dK_ref)


@_each_ard_kernel
def test_a_copy_and_a_pickle_keep_the_periods(make_kernel):
    """A copy of a periodic kernel keeps its periods, and so does a pickle
    of the kernels that pickle (Matern's functions of its degree do not)."""
    kernel = make_kernel(_PERIODS)
    X = _periodic_inputs(13)
    hyp = _hyperparameters(kernel, 14)
    copies = [copy.deepcopy(kernel)]
    if not isinstance(kernel, Matern):
        copies.append(pickle.loads(pickle.dumps(kernel)))

    for copied in copies:
        assert np.array_equal(copied.periods, _PERIODS)
        assert np.array_equal(copied.compute(hyp, X), kernel.compute(hyp, X))
