import numpy as np
import pytest

from gpyreg.covariance_functions import (
    AbstractKernel,
    Matern,
    SquaredExponential,
)
from gpyreg.isotropic_covariance_functions import (
    AbstractIsotropicKernel,
    MaternIsotropic,
    SquaredExponentialIsotropic,
)


def test_squared_exponential_isotropic_compute_sanity_checks():
    squared_expontential = SquaredExponentialIsotropic()
    D = 3
    N = 20
    X = np.ones((N, D))
    X_star = np.zeros((N, D))

    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(D + 2)
        squared_expontential.compute(hyp, X)
    assert (
        "Expected 2 covariance function hyperparameters"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones((2, 1))
        squared_expontential.compute(hyp, X)
    assert (
        "Covariance function output is available only for"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(2)
        squared_expontential.compute(hyp, X, X_star, compute_grad=True)
    assert (
        "X_star should be None when compute_grad is True."
        in execinfo.value.args[0]
    )


def test_sqr_exp_iso_kernel_gradient():
    sqr_exp = SquaredExponentialIsotropic()
    D = 3
    N = 20
    diag_cov = np.eye(N) * (0.2)
    X = (np.random.multivariate_normal(np.zeros(N), diag_cov, D)).T
    hyp_D = 2
    diag_cov = np.eye(hyp_D) * (0.2)
    hyp = np.random.multivariate_normal(np.zeros(hyp_D), diag_cov)
    _test_kernel_gradient_(sqr_exp, hyp, X)


def test_matern_isotropic_compute_sanity_checks():
    matern = MaternIsotropic(3)
    D = 3
    N = 20
    X = np.ones((N, D))
    X_star = np.zeros((N, D))

    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(D + 2)
        matern.compute(hyp, X)
    assert (
        "Expected 2 covariance function hyperparameters"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones((2, 1))
        matern.compute(hyp, X)
    assert (
        "Covariance function output is available only for"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(2)
        matern.compute(hyp, X, X_star, compute_grad=True)
    assert (
        "X_star should be None when compute_grad is True."
        in execinfo.value.args[0]
    )


def test_matern_isotropic_invalid_degree():
    for degree in [0, 2, 4, 6]:
        with pytest.raises(ValueError) as execinfo:
            MaternIsotropic(degree)
        assert (
            "Only degrees 1, 3 and 5 are supported for the"
            in execinfo.value.args[0]
        )


def test_matern_isotropic_kernel_gradient():
    matern_fun = MaternIsotropic(3)
    D = 3
    N = 20
    diag_cov = np.eye(N) * (0.2)
    X = (np.random.multivariate_normal(np.zeros(N), diag_cov, D)).T
    hyp_D = 2
    diag_cov = np.eye(hyp_D) * (0.2)
    hyp = np.random.multivariate_normal(np.zeros(hyp_D), diag_cov)

    _test_kernel_gradient_(matern_fun, hyp, X)


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


def test_matern_isotropic_against_anisotropic():
    N = 10
    M = 5
    for degree in [1, 3, 5]:
        D = np.random.randint(1, 11)

        # Isotropic kernel:
        matern_iso = MaternIsotropic(degree)
        n_hyp_iso = matern_iso.hyperparameter_count(D)
        hyp_iso = np.random.normal(size=n_hyp_iso)

        # Anisotropic kernel with equal length scales:
        matern = Matern(degree)
        n_hyp = matern.hyperparameter_count(D)
        hyp = np.zeros(n_hyp)
        hyp[0:-1] = hyp_iso[0]
        hyp[-1] = hyp_iso[1]

        # Test both on random data:
        X = np.random.normal(size=(N, D))
        X_star = np.random.normal(size=(M, D))

        K1_iso = matern_iso.compute(hyp_iso, X)
        K1 = matern.compute(hyp, X)
        assert np.allclose(K1_iso, K1), f"degree {degree}"

        K2_iso, dK2_iso = matern_iso.compute(hyp_iso, X, compute_grad=True)
        K2, dK2 = matern.compute(hyp, X, compute_grad=True)
        dK2 = np.dstack(
            [dK2[:, :, 0:-1].sum(axis=2, keepdims=True), dK2[:, :, [-1]]]
        )
        assert np.allclose(K2_iso, K2), f"degree {degree}"
        assert np.allclose(dK2_iso, dK2, equal_nan=True), f"degree {degree}"
        assert np.allclose(K2_iso, K1_iso), f"degree {degree}"

        K3_iso = matern_iso.compute(hyp_iso, X, X_star)
        K3 = matern.compute(hyp, X, X_star)
        assert np.allclose(K3_iso, K3), f"degree {degree}"


def test_squared_exponential_isotropic_against_anisotropic():
    N = 10
    M = 5
    D = np.random.randint(1, 11)

    # Isotropic kernel:
    sqexp_iso = SquaredExponentialIsotropic()
    n_hyp_iso = sqexp_iso.hyperparameter_count(D)
    hyp_iso = np.random.normal(size=n_hyp_iso)

    # Anisotropic kernel with equal length scales:
    sqexp = SquaredExponential()
    n_hyp = sqexp.hyperparameter_count(D)
    hyp = np.zeros(n_hyp)
    hyp[0:-1] = hyp_iso[0]
    hyp[-1] = hyp_iso[1]

    # Test both on random data:
    X = np.random.normal(size=(N, D))
    X_star = np.random.normal(size=(M, D))

    K1_iso = sqexp_iso.compute(hyp_iso, X)
    K1 = sqexp.compute(hyp, X)
    assert np.allclose(K1_iso, K1)

    K2_iso, dK2_iso = sqexp_iso.compute(hyp_iso, X, compute_grad=True)
    K2, dK2 = sqexp.compute(hyp, X, compute_grad=True)
    dK2 = np.dstack(
        [dK2[:, :, 0:-1].sum(axis=2, keepdims=True), dK2[:, :, [-1]]]
    )
    assert np.allclose(K2_iso, K2)
    assert np.allclose(dK2_iso, dK2)
    assert np.allclose(K2_iso, K1_iso)

    K3_iso = sqexp_iso.compute(hyp_iso, X, X_star)
    K3 = sqexp.compute(hyp, X, X_star)
    assert np.allclose(K3_iso, K3)
