import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.linalg
import scipy.special
import scipy.stats
from scipy.integrate import quad

import gpyreg as gpr
from gpyreg.testing.test_utils import (
    check_grad,
    gauss_hermite_quadrature_reference,
)


@pytest.mark.filterwarnings(
    """ignore:Matplotlib is currently using agg:UserWarning"""
)
def test_empty_gp():
    N = 20
    D = 2

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    # Test that temporary_data dict exists (required for e.g. pyvbmc)
    assert isinstance(gp.temporary_data, dict)

    # Test that lower and upper bounds are set appropriately.
    bounds = gp.get_bounds()
    assert np.all(np.isnan(bounds["covariance_log_lengthscale"]))
    assert np.all(np.isnan(bounds["covariance_log_lengthscale"]))
    assert np.all(np.isnan(bounds["covariance_log_outputscale"]))
    assert np.all(np.isnan(bounds["covariance_log_outputscale"]))
    assert np.all(np.isnan(bounds["noise_log_scale"]))
    assert np.all(np.isnan(bounds["noise_log_scale"]))
    assert np.all(np.isnan(bounds["mean_const"]))
    assert np.all(np.isnan(bounds["mean_const"]))
    assert np.all(np.isnan(bounds["mean_location"]))
    assert np.all(np.isnan(bounds["mean_location"]))
    assert np.all(np.isnan(bounds["mean_log_scale"]))
    assert np.all(np.isnan(bounds["mean_log_scale"]))

    # Test that hyperparameter priors are set appropriately.
    prior = gp.get_priors()
    assert prior["covariance_log_lengthscale"] is None
    assert prior["covariance_log_outputscale"] is None
    assert prior["noise_log_scale"] is None
    assert prior["mean_const"] is None
    assert prior["mean_location"] is None
    assert prior["mean_log_scale"] is None

    # Come up with some hyperparameters.
    cov_N = gp.covariance.hyperparameter_count(D)
    mean_N = gp.mean.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()
    hyp = np.random.standard_normal(size=(3, cov_N + noise_N + mean_N))
    hyp[:, D] *= 0.2
    hyp[:, D + 1 : D + 1 + noise_N] *= 0.3

    # Set GP to have them.
    gp.update(hyp=hyp)

    # Test that we can call prediction functions etc.
    # with a GP that only has hyperparameters.
    xx, yy = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))
    x_star = np.array((xx.ravel(), yy.ravel())).T
    gp.predict_full(x_star, add_noise=True)
    gp.predict_full(x_star, add_noise=False)

    gp.predict(x_star, add_noise=True)
    gp.predict(x_star, add_noise=False)

    y_star = np.zeros((400, 1))
    gp.predict(x_star, y_star, return_lpd=True, add_noise=False)
    gp.predict(x_star, y_star, return_lpd=True, add_noise=True)

    # gp.quad(0, 1, compute_var=True)

    # gp.plot()


@pytest.mark.filterwarnings(
    """ignore:Matplotlib is currently using agg:UserWarning"""
)
def test_random_function():
    N = 20
    D = 2
    X = np.random.standard_normal(size=(N, D))

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(1),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    cov_N = gp.covariance.hyperparameter_count(D)
    mean_N = gp.mean.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()

    N_s = np.random.randint(1, 3)
    hyp = np.random.standard_normal(size=(N_s, cov_N + noise_N + mean_N))
    hyp[:, D] *= 0.2
    hyp[:, D + 1 : D + 1 + noise_N] *= 0.3

    gp.update(hyp=hyp)
    y = gp.random_function(X)
    gp.update(X_new=X, y_new=y)

    # gp.plot()

    X_new = np.random.standard_normal(size=(10, D))
    y_new = gp.random_function(X_new)

    # Test return_lpd:
    __, __, lpd = gp.predict(
        X_new, y_new, return_lpd=True, add_noise=False, separate_samples=False
    )
    assert lpd.shape == (10, 1)
    __, __, lpd = gp.predict(
        X_new, y_new, return_lpd=True, add_noise=False, separate_samples=True
    )
    assert lpd.shape == (10, N_s)
    __, __, lpd = gp.predict(
        X_new, y_new, return_lpd=True, add_noise=True, separate_samples=False
    )
    assert lpd.shape == (10, 1)
    __, __, lpd = gp.predict(
        X_new, y_new, return_lpd=True, add_noise=True, separate_samples=True
    )
    assert lpd.shape == (10, N_s)

    gp.update(X_new=X_new, y_new=y_new)

    # gp.plot(delta_y=5, max_min_flag=False)


def test_getters_setters():
    N = 20
    D = 2
    X = np.random.uniform(low=-3, high=3, size=(N, D))
    y = np.sin(np.sum(X, 1)) + np.random.normal(scale=0.1, size=N)

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    hyp_dict_list = gp.get_hyperparameters()
    assert len(hyp_dict_list) == 1

    hyp_dict = hyp_dict_list[0]
    assert np.all(np.isnan(hyp_dict["covariance_log_lengthscale"]))
    assert np.all(np.isnan(hyp_dict["covariance_log_outputscale"]))
    assert np.all(np.isnan(hyp_dict["noise_log_scale"]))
    assert np.all(np.isnan(hyp_dict["mean_const"]))

    assert np.all(np.isnan(gp.get_hyperparameters(as_array=True)))

    gp_priors_mistaken = {
        "covariance_log_outputscal": (
            "smoothbox_student_t",
            (-0.5, 0.5, np.log(10), 3),
        ),
        "covariance_log_lengthscale": (
            "student_t",
            (np.log(np.std(X, ddof=1)), np.log(10), 3),
        ),
        "noise_log_scale": ("gaussian", (np.log(1e-3), 1.0)),
        "mean_const": ("smoothbox", (np.min(y), np.max(y), 1.0)),
    }

    with pytest.raises(ValueError):
        gp.set_priors(gp_priors_mistaken)

    gp_priors = {
        "covariance_log_outputscale": (
            "smoothbox_student_t",
            (-0.5, 0.5, np.log(10), 3),
        ),
        "covariance_log_lengthscale": (
            "student_t",
            (np.log(np.std(X, ddof=1)), np.log(10), 3),
        ),
        "noise_log_scale": ("gaussian", (np.log(1e-3), 1.0)),
        "mean_const": ("smoothbox", (np.min(y), np.max(y), 1.0)),
    }

    gp.set_priors(gp_priors)

    prior1 = gp.get_priors()

    gp_bounds_mistaken = {
        "covariance_log_outputscal": (-np.inf, np.inf),
        "covariance_log_lengthscale": (-np.inf, np.inf),
        "noise_log_scale": (-np.inf, np.inf),
        "mean_const": (-np.inf, np.inf),
    }

    with pytest.raises(ValueError):
        gp.set_bounds(gp_bounds_mistaken)

    hyp_arr = np.array(
        [[-0.4630094, -0.78566179, -0.2209450, -7.2947503, 0.03713608]]
    )
    hyp = gp.hyperparameters_to_dict(hyp_arr)
    gp.set_hyperparameters(hyp)
    assert np.all(gp.get_hyperparameters(as_array=True) == hyp_arr)

    gp.set_hyperparameters(hyp_arr)
    assert np.all(gp.get_hyperparameters(as_array=True) == hyp_arr)

    gp_train = {"n_samples": 10}
    hyp, _, _ = gp.fit(X=X, y=y, options=gp_train)

    assert np.all(gp.get_hyperparameters(as_array=True) == hyp)

    hyp_dict_list = gp.get_hyperparameters()
    for i, hyp_dict in enumerate(hyp_dict_list):
        assert np.all(hyp_dict["covariance_log_lengthscale"] == hyp[i, 0:2])
        assert np.all(hyp_dict["covariance_log_outputscale"] == hyp[i, 2])
        assert np.all(hyp_dict["noise_log_scale"] == hyp[i, 3])
        assert np.all(hyp_dict["mean_const"] == hyp[i, 4])

    prior2 = gp.get_priors()

    bounds = gp.get_bounds()
    assert np.all(
        bounds["covariance_log_lengthscale"][0] == gp.lower_bounds[0:2]
    )
    assert np.all(
        bounds["covariance_log_outputscale"][0] == gp.lower_bounds[2]
    )
    assert np.all(bounds["noise_log_scale"][0] == gp.lower_bounds[3])
    assert np.all(bounds["mean_const"][0] == gp.lower_bounds[4])

    assert np.all(
        bounds["covariance_log_lengthscale"][1] == gp.upper_bounds[0:2]
    )
    assert np.all(
        bounds["covariance_log_outputscale"][1] == gp.upper_bounds[2]
    )
    assert np.all(bounds["noise_log_scale"][1] == gp.upper_bounds[3])
    assert np.all(bounds["mean_const"][1] == gp.upper_bounds[4])


@pytest.mark.filterwarnings(
    """ignore:Matplotlib is currently using agg:UserWarning"""
)
def test_cleaning():
    N = 20
    D = 2
    X = np.random.uniform(low=-3, high=3, size=(N, D))
    y = np.reshape(
        np.sin(np.sum(X, 1)) + np.random.normal(scale=0.1, size=N), (-1, 1)
    )

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    gp.temporary_data["foo"] = "bar"
    assert len(gp.temporary_data) == 1

    gp_train = {"n_samples": 10}
    hyps, _, _ = gp.fit(X=X, y=y, options=gp_train)

    posteriors = copy.deepcopy(gp.posteriors)

    gp.clean()

    assert len(gp.temporary_data) == 0

    for i in range(0, 10):
        assert np.all(posteriors[i].hyp == gp.posteriors[i].hyp)
        assert gp.posteriors[i].alpha is None
        assert gp.posteriors[i].sW is None
        assert gp.posteriors[i].L is None
        assert gp.posteriors[i].L_chol is None
        assert gp.posteriors[i].sn2_mult is None
        assert gp.posteriors[i].sl is None

    gp.update(compute_posterior=True)

    for i in range(0, 10):
        assert np.all(posteriors[i].hyp == gp.posteriors[i].hyp)
        assert np.all(posteriors[i].alpha == gp.posteriors[i].alpha)
        assert np.all(posteriors[i].sW == gp.posteriors[i].sW)
        assert np.all(posteriors[i].L == gp.posteriors[i].L)
        assert posteriors[i].L_chol == gp.posteriors[i].L_chol
        assert posteriors[i].sn2_mult == posteriors[i].sn2_mult

    # gp.plot()


@pytest.mark.filterwarnings(
    """ignore:Matplotlib is currently using agg:UserWarning"""
)
def test_gp_gradient_computations():
    N = 20
    D = 2
    X = np.random.standard_normal(size=(N, D))

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    cov_N = gp.covariance.hyperparameter_count(D)
    mean_N = gp.mean.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()

    N_s = np.random.randint(1, 3)
    hyp = np.random.standard_normal(size=(N_s, cov_N + noise_N + mean_N))
    hyp[:, D] *= 0.2
    hyp[:, D + 1 : D + 1 + noise_N] *= 0.3

    gp.update(hyp=hyp, compute_posterior=False)
    y = gp.random_function(X)

    gp.update(X_new=X, y_new=y)

    hyp0 = hyp[0, :]

    # Check GP marginal likelihood gradient computation.
    f = lambda hyp_: gp._GP__compute_nlZ(hyp_, False, False)
    f_grad = lambda hyp_: gp._GP__compute_nlZ(hyp_, True, False)[1]
    assert np.all(
        np.isclose(
            check_grad(
                f,
                f_grad,
                hyp0 * np.exp(0.1 * np.random.uniform(size=hyp0.size)),
            ),
            0.0,
            atol=1e-6,
        )
    )

    # Check GP hyperparameters log prior gradient computation.
    hyp1 = hyp0 * np.exp(0.1 * np.random.uniform(size=hyp0.size))
    prior_types = np.random.permutation(range(0, 5))
    for i in range(0, cov_N + mean_N + noise_N):
        prior_type = prior_types[i]
        if prior_type == 1:  # 'gaussian'
            gp.hyper_priors["mu"][i] = np.random.standard_normal()
            gp.hyper_priors["sigma"][i] = np.exp(np.random.standard_normal())
            gp.hyper_priors["df"][i] = 0
        elif prior_type == 2:  #'student_t'
            gp.hyper_priors["mu"][i] = np.random.standard_normal()
            gp.hyper_priors["sigma"][i] = np.random.standard_normal()
            gp.hyper_priors["df"][i] = np.exp(np.random.standard_normal())
        elif prior_type == 3:  # 'smoothbox'
            gp.hyper_priors["a"][i] = -3
            gp.hyper_priors["b"][i] = 3
            gp.hyper_priors["sigma"][i] = np.random.standard_normal()
            gp.hyper_priors["df"][i] = 0
        elif prior_type == 4:  # 'smoothbox_student_t'
            gp.hyper_priors["a"][i] = -3
            gp.hyper_priors["b"][i] = 3
            gp.hyper_priors["sigma"][i] = np.random.standard_normal()
            gp.hyper_priors["df"][i] = np.exp(np.random.standard_normal())
        else:  # None
            pass

    # Manual changes to hyper priors requires us to call this
    gp._GP__recompute_normalization_constants()

    f = lambda hyp_: gp._GP__compute_log_priors(hyp_, False)
    f_grad = lambda hyp_: gp._GP__compute_log_priors(hyp_, True)[1]
    assert np.all(
        np.isclose(
            check_grad(f, f_grad, hyp1),
            0.0,
            atol=1e-6,
        )
    )

    # Test rank-1 update.
    idx = int(np.ceil(X.shape[0] / 2))
    gp1 = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    gp1.update(X_new=X[0:idx, :], y_new=y[0:idx], hyp=hyp)

    for i in range(idx, N):
        gp1.update(X_new=X[i : i + 1, :], y_new=y[i : i + 1])

    # These should be exactly the same.
    assert np.all(gp.X == gp1.X)
    assert np.all(gp.y == gp1.y)
    assert np.all(gp.posteriors[0].hyp == gp1.posteriors[0].hyp)

    # These only approximately the same I think.
    assert np.all(np.isclose(gp.posteriors[0].alpha, gp1.posteriors[0].alpha))
    assert np.all(np.isclose(gp.posteriors[0].sW, gp1.posteriors[0].sW))
    assert np.all(np.isclose(gp.posteriors[0].L, gp1.posteriors[0].L))
    assert np.isclose(gp.posteriors[0].sn2_mult, gp1.posteriors[0].sn2_mult)
    assert gp.posteriors[0].L_chol and gp1.posteriors[0].L_chol

    # Test getting and setting hyperparameters.
    hyp_dict = gp.get_hyperparameters()
    gp1.set_hyperparameters(hyp_dict)

    assert np.all(
        np.isclose(
            gp.get_hyperparameters(as_array=True),
            gp1.get_hyperparameters(as_array=True),
        )
    )

    # Test printing
    print(gp)

    # Test plotting
    # gp.plot()


def test_update_one_point_with_new_hyperparameters():
    # Appending a single observation takes a rank-one shortcut that extends
    # the existing posteriors. Replacement hyperparameters passed in the
    # same call must instead trigger a full recomputation with them.
    N = 12
    D = 2
    rng = np.random.default_rng(3)
    X = rng.uniform(-3, 3, size=(N, D))
    y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2])

    def make_gp():
        return gpr.GP(
            D=D,
            covariance=gpr.covariance_functions.SquaredExponential(),
            mean=gpr.mean_functions.ConstantMean(),
            noise=gpr.noise_functions.GaussianNoise(constant_add=True),
        )

    # Layout: [log ell (D), log sf, log sn, m0].
    hyp_old = np.array([[0.0, 0.0, 0.0, np.log(0.1), 0.0]])
    hyp_new = np.array([[0.3, -0.2, 0.5, np.log(0.05), 0.7]])

    gp = make_gp()
    gp.update(X_new=X[:-1], y_new=y[:-1], hyp=hyp_old)
    gp.update(X_new=X[-1:], y_new=y[-1:], hyp=hyp_new)

    gp_ref = make_gp()
    gp_ref.update(X_new=X, y_new=y, hyp=hyp_new)

    assert np.array_equal(gp.posteriors[0].hyp, hyp_new[0])
    for attr in ("alpha", "sW", "L"):
        value = getattr(gp.posteriors[0], attr)
        value_ref = getattr(gp_ref.posteriors[0], attr)
        assert np.array_equal(value, value_ref)
    assert gp.posteriors[0].sn2_mult == gp_ref.posteriors[0].sn2_mult
    assert gp.posteriors[0].L_chol == gp_ref.posteriors[0].L_chol

    x_star = rng.uniform(-3, 3, size=(5, D))
    f_mu, f_s2 = gp.predict(x_star)
    f_mu_ref, f_s2_ref = gp_ref.predict(x_star)
    assert np.array_equal(f_mu, f_mu_ref)
    assert np.array_equal(f_s2, f_s2_ref)


@pytest.mark.parametrize(
    "case", ["provided_with_s2_new", "provided_without_s2_new", "rectified"]
)
def test_rank_one_update_with_heteroskedastic_noise(case):
    # A single appended observation extends the Cholesky factor with the
    # noise scale the factor was built with, so the result matches a full
    # recomputation also when the training noise varies across points.
    N = 15
    D = 2
    rng = np.random.default_rng(6)
    X = rng.uniform(-3, 3, size=(N, D))
    y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2])

    if case == "rectified":
        noise_kwargs = {
            "constant_add": True,
            "rectified_linear_output_dependent_add": True,
        }
        # [log ell (D), log sf, log sn, y threshold, log multiplier, m0]
        hyp = np.array([[0.0, 0.0, 0.0, np.log(0.1), 0.5, 0.0, 0.0]])
        s2 = None
    else:
        noise_kwargs = {
            "constant_add": True,
            "user_provided_add": True,
            "scale_user_provided": True,
        }
        # [log ell (D), log sf, log sn, log s2 multiplier, m0]
        hyp = np.array([[0.0, 0.0, 0.0, np.log(0.1), 0.2, 0.0]])
        s2 = rng.uniform(0.01, 0.5, size=(N, 1))

    def make_gp():
        return gpr.GP(
            D=D,
            covariance=gpr.covariance_functions.SquaredExponential(),
            mean=gpr.mean_functions.ConstantMean(),
            noise=gpr.noise_functions.GaussianNoise(**noise_kwargs),
        )

    s2_old = None if s2 is None else s2[:-1]
    s2_last = s2[-1:] if case == "provided_with_s2_new" else None
    gp = make_gp()
    gp.update(X_new=X[:-1], y_new=y[:-1], s2_new=s2_old, hyp=hyp)
    gp.update(X_new=X[-1:], y_new=y[-1:], s2_new=s2_last)

    # Reference: everything at once. A point appended without s2_new has
    # zero user-provided variance.
    s2_ref = None if s2 is None else s2.copy()
    if case == "provided_without_s2_new":
        s2_ref[-1] = 0.0
    gp_ref = make_gp()
    gp_ref.update(X_new=X, y_new=y, s2_new=s2_ref, hyp=hyp)

    if s2 is not None:
        assert np.array_equal(gp.s2, s2_ref)
    post, post_ref = gp.posteriors[0], gp_ref.posteriors[0]
    assert post.L_chol and post_ref.L_chol
    assert np.allclose(post.alpha, post_ref.alpha, rtol=1e-10, atol=1e-12)
    # sW is uniform and consistent with the stored scale, and the factor
    # reproduces the same unscaled matrix K + sn2_mult * diag(sn2).
    assert np.allclose(post.sW, 1 / np.sqrt(post.sl))
    assert np.allclose(
        post.L.T @ post.L * post.sl,
        post_ref.L.T @ post_ref.L * post_ref.sl,
        rtol=1e-10,
        atol=1e-12,
    )

    x_star = rng.uniform(-3, 3, size=(5, D))
    f_mu, f_s2 = gp.predict(x_star)
    f_mu_ref, f_s2_ref = gp_ref.predict(x_star)
    assert np.allclose(f_mu, f_mu_ref, rtol=1e-10, atol=1e-12)
    assert np.allclose(f_s2, f_s2_ref, rtol=1e-10, atol=1e-12)

    # Bayesian quadrature normalizes its solves by the scale the stored
    # factor carries, which after a rank-one update need not be the
    # minimum of the enlarged training noise, so the integral and its
    # variance agree with those of the full recomputation.
    F, F_var = gp.quad(0.0, 1.0, compute_var=True)
    F_ref, F_var_ref = gp_ref.quad(0.0, 1.0, compute_var=True)
    assert np.allclose(F, F_ref, rtol=1e-10, atol=1e-12)
    assert np.allclose(F_var, F_var_ref, rtol=1e-8, atol=1e-12)
    assert np.all(F_var > np.spacing(1))


def test_rank_one_update_without_stored_noise_scale():
    # Posteriors pickled by earlier versions have no ``sl`` attribute; the
    # rank-one update then recovers the scale from ``sW``.
    N = 12
    D = 1
    rng = np.random.default_rng(15)
    X = np.reshape(np.linspace(-2, 2, N), (-1, 1))
    y = np.sin(X) + 0.1 * rng.standard_normal((N, 1))
    hyp = np.array([[0.0, 0.0, np.log(0.1), 0.0]])

    def make_gp():
        return gpr.GP(
            D=D,
            covariance=gpr.covariance_functions.SquaredExponential(),
            mean=gpr.mean_functions.ConstantMean(),
            noise=gpr.noise_functions.GaussianNoise(constant_add=True),
        )

    gp = make_gp()
    gp.update(X_new=X[:-1], y_new=y[:-1], hyp=hyp)
    del gp.posteriors[0].sl
    gp.update(X_new=X[-1:], y_new=y[-1:])

    gp_ref = make_gp()
    gp_ref.update(X_new=X, y_new=y, hyp=hyp)
    f_mu, f_s2 = gp.predict(X)
    f_mu_ref, f_s2_ref = gp_ref.predict(X)
    assert np.allclose(f_mu, f_mu_ref, rtol=1e-10, atol=1e-12)
    assert np.allclose(f_s2, f_s2_ref, rtol=1e-10, atol=1e-12)


def test_update_aligns_user_provided_noise():
    # The stored s2 always has one row per training input: points without
    # a supplied variance get zero, whichever side of the update lacks it.
    N = 10
    D = 1
    rng = np.random.default_rng(12)
    X = np.reshape(np.linspace(-2, 2, N), (-1, 1))
    y = np.sin(X)
    s2 = rng.uniform(0.01, 0.2, size=(N, 1))
    hyp = np.array([[0.0, 0.0, np.log(0.1), 0.0]])

    def make_gp():
        return gpr.GP(
            D=D,
            covariance=gpr.covariance_functions.SquaredExponential(),
            mean=gpr.mean_functions.ConstantMean(),
            noise=gpr.noise_functions.GaussianNoise(
                constant_add=True, user_provided_add=True
            ),
        )

    gp = make_gp()
    gp.update(X_new=X[:5], y_new=y[:5], hyp=hyp)
    assert gp.s2 is None
    gp.update(X_new=X[5:8], y_new=y[5:8], s2_new=s2[5:8])
    gp.update(X_new=X[8:], y_new=y[8:])

    s2_expected = s2.copy()
    s2_expected[:5] = 0.0
    s2_expected[8:] = 0.0
    assert np.array_equal(gp.s2, s2_expected)

    gp_ref = make_gp()
    gp_ref.update(X_new=X, y_new=y, s2_new=s2_expected, hyp=hyp)
    f_mu, f_s2 = gp.predict(X)
    f_mu_ref, f_s2_ref = gp_ref.predict(X)
    assert np.array_equal(f_mu, f_mu_ref)
    assert np.array_equal(f_s2, f_s2_ref)


def test_split_update():
    N = 20
    D = 2
    X = np.random.standard_normal(size=(N, D))
    s2 = np.full((N, 1), 0.05)

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(user_provided_add=True),
    )

    cov_N = gp.covariance.hyperparameter_count(D)
    mean_N = gp.mean.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()

    N_s = np.random.randint(1, 3)
    hyp = np.random.standard_normal(size=(N_s, cov_N + noise_N + mean_N))
    hyp[:, D] *= 0.2
    hyp[:, D + 1 : D + 1 + noise_N] *= 0.3

    gp.update(hyp=hyp, compute_posterior=False)
    y = gp.random_function(X)

    gp.update(X_new=X, y_new=y, s2_new=s2, compute_posterior=True)

    gp1 = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(user_provided_add=True),
    )

    gp1.update(
        X_new=X[0:10, :],
        y_new=y[0:10],
        s2_new=s2[0:10, :],
        hyp=hyp,
        compute_posterior=True,
    )
    gp1.update(
        X_new=X[10:, :],
        y_new=y[10:],
        s2_new=s2[10:, :],
        hyp=hyp,
        compute_posterior=True,
    )

    # These should be exactly the same.
    assert np.all(gp.X == gp1.X)
    assert np.all(gp.y == gp1.y)
    assert np.all(gp.posteriors[0].hyp == gp1.posteriors[0].hyp)

    # These only approximately the same I think.
    assert np.all(np.isclose(gp.posteriors[0].alpha, gp1.posteriors[0].alpha))
    assert np.all(np.isclose(gp.posteriors[0].sW, gp1.posteriors[0].sW))
    assert np.all(np.isclose(gp.posteriors[0].L, gp1.posteriors[0].L))
    assert np.isclose(gp.posteriors[0].sn2_mult, gp1.posteriors[0].sn2_mult)
    assert gp.posteriors[0].L_chol and gp1.posteriors[0].L_chol


@pytest.mark.filterwarnings(
    """ignore:Matplotlib is currently using agg:UserWarning"""
)
def test_quadrature_without_noise():
    f = lambda x: np.exp(-((x - 0.35) ** 2 / (2 * 0.01))) + np.sin(10 * x) / 3
    f_p = lambda x: f(x) * scipy.stats.norm.pdf(x, scale=0.1)
    N = 50
    D = 1
    X = np.linspace(-2.5, 2.5, N)
    y = f(X)

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ZeroMean(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True,
        ),
    )

    gp_train = {"n_samples": 0}
    gp.fit(
        X=np.reshape(X, (-1, 1)), y=np.reshape(y, (-1, 1)), options=gp_train
    )

    F_true = scipy.integrate.quad(f_p, -np.inf, np.inf)[0]

    mu_N = 1000
    x_star = np.reshape(np.linspace(-10, 10, mu_N), (-1, 1))
    f_mu, f_cov = gp.predict_full(x_star, add_noise=False)

    F_predict = 0
    for i in range(0, mu_N):
        F_predict += f_mu[i, 0] * scipy.stats.norm.pdf(x_star[i], scale=0.1)
    F_predict *= 20 / mu_N

    pdf_tmp = np.reshape(scipy.stats.norm.pdf(x_star, scale=0.1), (-1, 1))
    tmp = np.dot(pdf_tmp, pdf_tmp.T)
    F_var_predict = np.sum(np.sum(f_cov[:, :, 0] * tmp)) * (20 / mu_N) ** 2

    F_bayes, F_var_bayes = gp.quad(0, 0.1, compute_var=True)

    assert np.abs(F_var_bayes - F_var_predict) < 0.00001
    assert np.abs(F_bayes - F_predict) < 0.0001
    assert np.abs(F_true - F_bayes) < 0.0001
    assert np.abs(F_true - F_predict) < 0.0001

    F_bayes_2, F_var_bayes_2 = gp.quad(0.5, 0.4, compute_var=True)

    # Test that we can compute multiple quadratures easily.
    F_bayes_total, F_var_bayes_total = gp.quad(
        np.array([[0], [0.5]]), np.array([[0.1], [0.4]]), compute_var=True
    )
    assert np.isclose(F_bayes[0, 0], F_bayes_total[0, 0])
    assert np.isclose(F_bayes_2[0, 0], F_bayes_total[1, 0])
    assert np.isclose(F_var_bayes[0, 0], F_var_bayes_total[0, 0])
    assert np.isclose(F_var_bayes_2[0, 0], F_var_bayes_total[1, 0])

    # gp.plot()


@pytest.mark.parametrize(
    "noise_kwargs, s2_scale",
    [
        ({"constant_add": True}, None),
        ({"constant_add": True, "user_provided_add": True}, 1.0),
        (
            {
                "constant_add": True,
                "user_provided_add": True,
                "scale_user_provided": True,
            },
            0.5,
        ),
    ],
)
def test_quadrature_with_noise_matches_numerical_integration(
    noise_kwargs, s2_scale
):
    # Fixed hyperparameters, so quad and the numerical reference integrate
    # exactly the same posterior. User-provided noise makes the training
    # noise heteroskedastic, which exercises the normalization of the
    # Cholesky factor inside quad.
    rng = np.random.default_rng(11)
    N = 30
    D = 1
    X = np.reshape(np.linspace(-3, 3, N), (-1, 1))
    y = np.sin(X) + 0.1 * rng.standard_normal((N, 1))
    if s2_scale is None:
        s2 = None
    else:
        s2 = s2_scale * rng.uniform(0.1, 2.0, (N, 1))

    noise = gpr.noise_functions.GaussianNoise(**noise_kwargs)
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=noise,
    )
    # Three hyperparameter samples, each [log ell, log sf, noise..., m0].
    extra_noise_N = noise.hyperparameter_count() - 1
    hyp = np.array(
        [
            [0.0, 0.0, np.log(0.1)] + [0.2] * extra_noise_N + [0.1],
            [np.log(0.7), np.log(1.5), np.log(0.3)]
            + [-0.3] * extra_noise_N
            + [-0.2],
            [np.log(1.3), np.log(0.8), np.log(0.05)]
            + [0.0] * extra_noise_N
            + [0.4],
        ]
    )
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp)

    mu, sigma = 0.4, 0.6
    F_ref, F_var_ref = gauss_hermite_quadrature_reference(gp, mu, sigma)

    F, F_var = gp.quad(mu, sigma, compute_var=True, separate_samples=True)
    assert F.shape == F_var.shape == (1, hyp.shape[0])
    assert np.allclose(F[0], F_ref, rtol=1e-9, atol=1e-13)
    assert np.allclose(F_var[0], F_var_ref, rtol=1e-9, atol=1e-13)

    # Averaging over samples: mean of the means, and mean of the variances
    # plus the variance of the means.
    F_avg, F_var_avg = gp.quad(mu, sigma, compute_var=True)
    assert F_avg.shape == F_var_avg.shape == (1, 1)
    assert np.isclose(F_avg[0, 0], np.mean(F_ref), rtol=1e-9, atol=1e-13)
    assert np.isclose(
        F_var_avg[0, 0],
        np.mean(F_var_ref) + np.var(F_ref, ddof=1),
        rtol=1e-9,
        atol=1e-13,
    )


def test_quadrature_with_noise_fitting():
    # Fit a GP to noisy observations, check quad against numerical
    # integration of the fitted posterior, and compare the Bayesian
    # quadrature estimate with the integral of the underlying function.
    rng = np.random.default_rng(1234)
    N = 500
    D = 1
    s2_constant = 0.01
    X = np.reshape(np.linspace(-15, 15, N), (-1, 1))
    s2 = np.full(X.shape, s2_constant)

    y = np.sin(X) + np.sqrt(s2) * rng.standard_normal(X.shape)
    y[y < 0] = -(np.abs(3 * y[y < 0]) ** 2)

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True,
            user_provided_add=True,
            scale_user_provided=True,
            rectified_linear_output_dependent_add=True,
        ),
    )

    gp_train = {"n_samples": 10}
    gp.fit(X=X, y=y, s2=s2, options=gp_train, rng=rng)

    # Per hyperparameter sample, quad must agree with numerical integration
    # of the fitted posterior, including the variance.
    F_ref, F_var_ref = gauss_hermite_quadrature_reference(gp, 0, 0.1)
    F_s, F_var_s = gp.quad(0, 0.1, compute_var=True, separate_samples=True)
    assert np.allclose(F_s[0], F_ref, rtol=1e-8, atol=1e-13)
    assert np.allclose(F_var_s[0], F_var_ref, rtol=1e-8, atol=1e-13)

    F_bayes, F_bayes_var = gp.quad(0, 0.1, compute_var=True)

    def f(x):
        y = np.sin(x)
        if y < 0:
            return -(np.abs(3 * y) ** 2)
        return y

    f_p = lambda x: f(x) * scipy.stats.norm.pdf(x, scale=0.1)

    F_true = scipy.integrate.quad(f_p, -np.inf, np.inf)[0]

    assert np.abs(F_true - F_bayes) < 0.1


@pytest.mark.filterwarnings(
    """ignore:Matplotlib is currently using agg:UserWarning"""
)
def test_fitting_with_fixed_bounds():
    N = 20
    D = 1
    X = np.reshape(np.linspace(-10, 10, N), (-1, 1))
    y = 1 + np.sin(X)

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(3),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    gp_bounds = {
        "covariance_log_outputscale": (-np.inf, np.inf),
        "covariance_log_lengthscale": (-np.inf, np.inf),
        "noise_log_scale": (-np.inf, np.inf),
        "mean_const": (0.5, 0.5),
    }

    gp_priors = {
        "covariance_log_outputscale": None,
        "covariance_log_lengthscale": None,
        "noise_log_scale": ("gaussian", (np.log(1e-3), 1.0)),
        "mean_const": None,
    }

    gp.set_priors(gp_priors)
    gp.set_bounds(gp_bounds)

    assert gp.get_bounds() == gp_bounds

    hyp, _, _ = gp.fit(X=X, y=y)

    assert np.all(hyp[:, 3] == 0.5)

    # gp.plot()


def test_setting_bounds():
    N = 20
    D = 2
    X = np.reshape(np.linspace(-10, 10, N), (-1, 2))
    y = 1 + np.sum(np.sin(X), 1)

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(3),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    gp_bounds = {
        "covariance_log_outputscale": (-np.inf, 1.0),
        "covariance_log_lengthscale": (-2.0, np.inf),
        "noise_log_scale": (-np.inf, np.inf),
        "mean_const": (-4.0, 4.0),
    }

    gp_priors = {
        "covariance_log_outputscale": None,
        "covariance_log_lengthscale": None,
        "noise_log_scale": ("gaussian", (np.log(1e-3), 1.0)),
        "mean_const": None,
    }

    # Test setting all bounds manually:
    lower_bounds = np.array([-2.0, -2.0, -np.inf, -np.inf, -4.0])
    upper_bounds = np.array([np.inf, np.inf, 1.0, np.inf, 4.0])
    gp.set_priors(gp_priors)
    gp.set_bounds(gp_bounds)
    hyp, _, _ = gp.fit(X=X, y=y)
    assert np.all(gp.lower_bounds == lower_bounds)
    # Make sure fitting doesn't undo the set bounds:
    hyp, _, _ = gp.fit(X=X, y=y)
    assert np.all(gp.lower_bounds == lower_bounds)
    assert np.all(gp.upper_bounds == upper_bounds)

    # Test setting all bounds automatically (by default)
    gp.set_bounds(None)
    assert np.all(np.isnan(gp.lower_bounds))
    assert np.all(np.isnan(gp.upper_bounds))
    hyp, _, _ = gp.fit(X=X, y=y)
    default_lower_bounds = np.concatenate(
        [
            gp.covariance.get_bounds_info(gp.X, gp.y)["LB"],
            gp.noise.get_bounds_info(gp.X, gp.y)["LB"],
            gp.mean.get_bounds_info(gp.X, gp.y)["LB"],
        ]
    )
    default_upper_bounds = np.concatenate(
        [
            gp.covariance.get_bounds_info(gp.X, gp.y)["UB"],
            gp.noise.get_bounds_info(gp.X, gp.y)["UB"],
            gp.mean.get_bounds_info(gp.X, gp.y)["UB"],
        ]
    )
    assert np.all(gp.lower_bounds == default_lower_bounds)
    assert np.all(gp.upper_bounds == default_upper_bounds)

    # Test setting some bounds to defaults, via set_bounds.
    # Bounds with value ``None`` should map to default values. Other bounds
    # should stay the same.
    gp_bounds = {
        "covariance_log_outputscale": None,
        "covariance_log_lengthscale": (-2.0, np.inf),
        "noise_log_scale": None,
        "mean_const": (-4.0, 4.0),
    }
    gp.set_bounds(gp_bounds)
    mask = np.array([False, False, True, True, False])
    assert np.all(np.isnan(gp.lower_bounds[mask]))
    assert np.all(np.isnan(gp.upper_bounds[mask]))
    assert np.all(gp.lower_bounds[~mask] == lower_bounds[~mask])
    assert np.all(gp.upper_bounds[~mask] == upper_bounds[~mask])
    gp.fit(X, y)
    assert np.all(gp.lower_bounds[mask] == default_lower_bounds[mask])
    assert np.all(gp.upper_bounds[mask] == default_upper_bounds[mask])
    assert np.all(gp.lower_bounds[~mask] == lower_bounds[~mask])
    assert np.all(gp.upper_bounds[~mask] == upper_bounds[~mask])

    gp_bounds = {
        "covariance_log_outputscale": (-np.inf, 1.0),
        "covariance_log_lengthscale": None,
        "noise_log_scale": (-np.inf, np.inf),
        "mean_const": None,
    }
    gp.set_bounds(gp_bounds)
    mask = np.array([False, False, True, True, False])
    assert np.all(np.isnan(gp.lower_bounds[~mask]))
    assert np.all(np.isnan(gp.upper_bounds[~mask]))
    assert np.all(gp.lower_bounds[mask] == lower_bounds[mask])
    assert np.all(gp.upper_bounds[mask] == upper_bounds[mask])
    gp.fit(X, y)
    assert np.all(gp.lower_bounds[~mask] == default_lower_bounds[~mask])
    assert np.all(gp.upper_bounds[~mask] == default_upper_bounds[~mask])
    assert np.all(gp.lower_bounds[mask] == lower_bounds[mask])
    assert np.all(gp.upper_bounds[mask] == upper_bounds[mask])

    # Test setting some bounds to defaults, via gp.fit():
    lower_bounds = np.array([-2.0, np.nan, -np.inf, np.nan, -4.0])
    upper_bounds = np.array([np.nan, np.inf, np.nan, np.inf, np.nan])
    fit_options = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    hyp, _, _ = gp.fit(X=X, y=y, options=fit_options)
    # Bounds should follow user-provided options, where not nan:
    mask = np.isnan(lower_bounds)
    assert np.all(gp.lower_bounds[~mask] == lower_bounds[~mask])
    assert np.all(gp.upper_bounds[mask] == upper_bounds[mask])
    # Bounds should follow defaults, where nan:
    assert np.all(gp.lower_bounds[mask] == default_lower_bounds[mask])
    assert np.all(gp.upper_bounds[~mask] == default_upper_bounds[~mask])


def test_fitting_options():
    N = 20
    D = 1
    X = np.reshape(np.linspace(-10, 10, N), (-1, 1))
    y = 1 + np.sin(X)

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    gp_train_1 = {"opts_N": 0}
    gp_train_2 = {"n_samples": 0}
    gp_train_3 = {"init_N": 0}
    gp_train_4 = {"opts_N": 0, "n_samples": 0}
    gp_train_5 = {"n_samples": 0, "init_N": 0}
    gp_train_6 = {"opts_N": 0, "init_N": 0}
    gp_train_7 = {"opts_N": 0, "n_samples": 0, "init_N": 0}
    gp_train_8 = {"init_N": 1}

    # Test that all these at least can be run in a row.
    gp.fit(X=X, y=y, options=gp_train_1)
    gp.fit(X=X, y=y, options=gp_train_2)
    gp.fit(X=X, y=y, options=gp_train_3)
    gp.fit(X=X, y=y, options=gp_train_4)
    gp.fit(X=X, y=y, options=gp_train_5)
    gp.fit(X=X, y=y, options=gp_train_6)
    gp.fit(X=X, y=y, options=gp_train_7)
    gp.fit(X=X, y=y, options=gp_train_8)


def test_fitting():
    N = 500
    D = 1
    X = np.reshape(np.linspace(-10, 10, N), (-1, 1))

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(5),
        mean=gpr.mean_functions.ZeroMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    cov_N = gp.covariance.hyperparameter_count(D)
    mean_N = gp.mean.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()

    N_s = 1
    hyp = np.random.standard_normal(size=(N_s, cov_N + noise_N + mean_N))
    hyp[:, D] *= 0.3
    hyp[:, D + 1 : D + 1 + noise_N] *= 0.3

    gp.update(hyp=hyp, compute_posterior=False)
    y = gp.random_function(X, add_noise=True)
    gp.update(X_new=X, y_new=y, hyp=hyp, compute_posterior=True)

    gp1 = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(5),
        mean=gpr.mean_functions.ZeroMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    gp_train = {"n_samples": 0}
    hyp2, _, _ = gp1.fit(X=X, y=y, options=gp_train)

    assert np.all(np.abs(hyp - hyp2)[0] < 0.5)

    assert (
        np.abs(gp.log_likelihood(hyp[0, :]) - gp.log_likelihood(hyp2[0, :]))
        < 20
    )


def test_get_recommended_bounds_no_bounds_set():
    D = 3
    d = 1 + 2 * np.random.randint(0, 3)
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(d),
        mean=gpr.mean_functions.ZeroMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    with pytest.raises(ValueError) as execinfo:
        gp.get_recommended_bounds()
    assert "GP does not have X or y set!" in execinfo.value.args[0]
    gp.X = 1
    with pytest.raises(ValueError) as execinfo:
        gp.get_recommended_bounds()
    assert "GP does not have X or y set!" in execinfo.value.args[0]
    gp.X = None
    gp.y = 1
    with pytest.raises(ValueError) as execinfo:
        gp.get_recommended_bounds()
    assert "GP does not have X or y set!" in execinfo.value.args[0]


def test_set_hyperparameters_wrong_shape():
    D = 3
    d = 1 + 2 * np.random.randint(0, 3)
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(d),
        mean=gpr.mean_functions.ZeroMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    with pytest.raises(ValueError) as execinfo:
        gp.set_hyperparameters(np.ones((1, 20)))
    assert (
        "Input hyperparameter array is the wrong shape!"
        in execinfo.value.args[0]
    )


def test_hyperparameters_to_dict_wrong_shape():
    D = 3
    d = 1 + 2 * np.random.randint(0, 3)
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(d),
        mean=gpr.mean_functions.ZeroMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    with pytest.raises(ValueError) as execinfo:
        gp.hyperparameters_to_dict(np.ones((1, 20)))
    assert (
        "Input hyperparameter array is the wrong shape!"
        in execinfo.value.args[0]
    )


def test_hyperparameters_from_dict_single_dict():
    D = 3
    d = 1 + 2 * np.random.randint(0, 3)
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(d),
        mean=gpr.mean_functions.ZeroMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    hyper_dict = gp.get_hyperparameters()[0]
    gp.hyperparameters_from_dict(hyper_dict)
    for key in hyper_dict.keys():
        assert np.all(
            np.array_equal(
                gp.get_hyperparameters()[0][key],
                hyper_dict[key],
                equal_nan=True,
            )
        )


def test_quad_not_squared_exponential():
    D = 3
    d = 1 + 2 * np.random.randint(0, 3)
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(d),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    with pytest.raises(ValueError) as execinfo:
        gp.quad(0, 0.1, compute_var=True)
    assert (
        "Bayesian quadrature only supports the squared exponential"
        in execinfo.value.args[0]
    )


def test_predict_lpd():
    D = 3
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(user_provided_add=True),
    )
    # This noise function has no hyperparameter of its own: its variance is
    # the user-provided one plus a nugget of ``eps``. So a row is
    # [log ell (D), log sf, m0, mode location (D), log scale (D)]. The two
    # samples differ, so that neither is the average over them.
    hyp = np.array(
        [
            [
                # Covariance
                0.0,
                0.0,
                0.0,  # log ell
                1.0,  # log sf
                # Mean
                -(D / 2) * np.log(2 * np.pi),  # MVN mode
                0.0,
                0.0,
                0.0,  # Mode location
                0.0,
                0.0,
                0.0,  # log scale
            ],
            [
                # Covariance
                0.3,
                -0.2,
                0.1,  # log ell
                0.5,  # log sf
                # Mean
                -1.0,  # MVN mode
                0.2,
                -0.1,
                0.4,  # Mode location
                0.3,
                0.3,
                0.3,  # log scale
            ],
        ]
    )
    gp.update(hyp=hyp)

    X_star = np.arange(-9, 9).reshape((-1, 3))
    offset = np.random.normal(size=(6, 1))
    y_star = (
        scipy.stats.multivariate_normal.logpdf(
            X_star, mean=np.zeros((D,))
        ).reshape(-1, 1)
        + offset
    )
    s2_star = np.linspace(0.1, 0.7, 6).reshape((-1, 1))

    # The log predictive density always carries the observation noise,
    # here the user-provided variance, whichever variance ``add_noise``
    # selects for the returned ``s2``.
    f_mu, f_s2, lpd = gp.predict(
        X_star, y_star, s2_star=s2_star, return_lpd=True
    )
    assert np.allclose(
        lpd,
        scipy.stats.norm.logpdf(
            y_star, loc=f_mu, scale=np.sqrt(s2_star + f_s2)
        ),
    )
    __, s2_with_noise, lpd2 = gp.predict(
        X_star, y_star, s2_star=s2_star, return_lpd=True, add_noise=True
    )
    assert np.all(lpd2 == lpd)
    assert np.allclose(s2_with_noise, f_s2 + s2_star)

    # Per sample it is the density of that sample's own Gaussian.
    f_mu_s, y_s2_s, lpd3 = gp.predict(
        X_star,
        y_star,
        s2_star=s2_star,
        return_lpd=True,
        add_noise=True,
        separate_samples=True,
    )
    assert np.allclose(
        lpd3,
        scipy.stats.norm.logpdf(y_star, loc=f_mu_s, scale=np.sqrt(y_s2_s)),
    )
    assert not np.allclose(lpd3[:, 0], lpd3[:, 1])

    # Averaged over the samples it is the density of the Gaussian that
    # carries the mean and the variance of the mixture, and not the average
    # of the per-sample densities.
    mu_bar = np.mean(f_mu_s, 1, keepdims=True)
    var_bar = np.mean(y_s2_s, 1, keepdims=True) + np.var(
        f_mu_s, axis=1, ddof=1, keepdims=True
    )
    assert np.allclose(f_mu, mu_bar)
    assert np.allclose(
        lpd,
        scipy.stats.norm.logpdf(y_star, loc=mu_bar, scale=np.sqrt(var_bar)),
    )
    assert not np.allclose(lpd, np.mean(lpd3, 1, keepdims=True))


def test__str__and__repr__():
    # 1-D:
    N = 20
    D = 1
    X = np.reshape(np.linspace(-10, 10, N), (-1, 1))
    y = 1 + np.sum(np.sin(X), 1)

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(3),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    gp_bounds = {
        "covariance_log_outputscale": (np.nan, np.nan),
        "covariance_log_lengthscale": (np.nan, np.nan),
        "noise_log_scale": (np.nan, np.nan),
        "mean_const": (0.5, 0.5),
    }

    gp_priors = {
        "covariance_log_outputscale": None,
        "covariance_log_lengthscale": None,
        "noise_log_scale": ("gaussian", (np.log(1e-3), 1.0)),
        "mean_const": None,
    }

    gp.set_priors(gp_priors)
    gp.set_bounds(gp_bounds)
    hyp, _, _ = gp.fit(X=X, y=y)

    str_ = gp.__str__()
    assert "Covariance function: Matern" in str_
    repr_ = gp.__repr__()
    assert (
        "self.covariance = <gpyreg.covariance_functions.Matern object at "
        in repr_
    )
    assert "self.lower_bounds = [-10.8" in repr_

    # 2-D:
    N = 20
    D = 2
    X = np.reshape(np.linspace(-10, 10, N), (-1, 2))
    y = 1 + np.sum(np.sin(X), 1)

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(3),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    gp_bounds = {
        "covariance_log_outputscale": (np.nan, np.nan),
        "covariance_log_lengthscale": (np.nan, np.nan),
        "noise_log_scale": (np.nan, np.nan),
        "mean_const": (0.5, 0.5),
    }

    gp_priors = {
        "covariance_log_outputscale": None,
        "covariance_log_lengthscale": None,
        "noise_log_scale": ("gaussian", (np.log(1e-3), 1.0)),
        "mean_const": None,
    }

    gp.set_priors(gp_priors)
    gp.set_bounds(gp_bounds)
    hyp, _, _ = gp.fit(X=X, y=y)

    str_ = gp.__str__()
    assert "Covariance function: Matern" in str_
    repr_ = gp.__repr__()
    assert (
        "self.covariance = <gpyreg.covariance_functions.Matern object at "
        in repr_
    )
    assert "self.lower_bounds = [-10.8" in repr_


def test_convert_shapes():
    D = 3
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(user_provided_add=True),
    )
    N = 10
    X = np.ones((N, D))
    y = np.ones(N)
    s2 = np.ones(N)
    X, y, s2 = gp._convert_shapes(X, y, s2)
    assert X.shape == (N, D) and y.shape == (N, 1) and s2.shape == (N, 1)
    s2 = None
    X, y, s2 = gp._convert_shapes(X, y, s2)
    assert X.shape == (N, D) and y.shape == (N, 1) and s2 is None
    s2 = 1
    X, y, s2 = gp._convert_shapes(X, y, s2)
    assert (
        X.shape == (N, D)
        and y.shape == (N, 1)
        and np.allclose(s2, np.ones((N, 1)))
    )
    with pytest.raises(AttributeError):
        gp._convert_shapes(None, y, s2)
    gp.X = X
    y = np.ones(N)
    s2 = np.ones(N)
    X, y, s2 = gp._convert_shapes(None, y, s2)
    assert X is None and y.shape == (N, 1) and s2.shape == (N, 1)


@pytest.mark.parametrize("trans", [0, 1, 2])
@pytest.mark.parametrize("order", ["C", "F"])
def test_solve_triangular_matches_scipy(order, trans):
    """The direct LAPACK call is bit-identical to scipy's wrapper for both
    memory layouts of the factor (scipy solves the transposed system for a
    C-ordered factor; the helper applies the same rule)."""
    from gpyreg.gaussian_process import _solve_triangular

    rng = np.random.default_rng(0)
    for N, k in [(5, 1), (60, 8), (200, 3)]:
        A = rng.standard_normal((N, N))
        A = A @ A.T + N * np.eye(N)
        L = np.array(scipy.linalg.cholesky(A), order=order)
        B = rng.standard_normal((N, k))
        expected = scipy.linalg.solve_triangular(
            L, B, trans=trans, check_finite=False
        )
        assert np.array_equal(_solve_triangular(L, B, trans=trans), expected)
    with pytest.raises(scipy.linalg.LinAlgError):
        _solve_triangular(np.zeros((3, 3)), np.ones((3, 1)))


def test_predict_mean_fallback_without_batched_method():
    """A mean function without ``compute_batched`` (a user-defined or an
    unpickled old one) takes the per-sample loop and gives the same
    prediction as the batched form."""

    class PlainNegativeQuadratic(gpr.mean_functions.NegativeQuadratic):
        compute_batched = None

    N, D, N_s = 30, 3, 4
    rng = np.random.default_rng(1)
    X = rng.standard_normal((N, D))
    y = rng.standard_normal((N, 1))
    hyp_N = D + 1 + 1 + (1 + 2 * D)  # SE + constant noise + quadratic mean
    hyp = rng.standard_normal((N_s, hyp_N))
    outputs = []
    for mean in (
        gpr.mean_functions.NegativeQuadratic(),
        PlainNegativeQuadratic(),
    ):
        gp = gpr.GP(
            D=D,
            covariance=gpr.covariance_functions.SquaredExponential(),
            mean=mean,
            noise=gpr.noise_functions.GaussianNoise(constant_add=True),
        )
        gp.update(X_new=X, y_new=y, compute_posterior=False)
        gp.set_hyperparameters(hyp, compute_posterior=True)
        outputs.append(gp.predict(X[:7], separate_samples=True))
    assert np.array_equal(outputs[0][0], outputs[1][0])
    assert np.array_equal(outputs[0][1], outputs[1][1])


@pytest.mark.parametrize("trained", [False, True])
@pytest.mark.parametrize(
    "mean_cls",
    [
        gpr.mean_functions.ZeroMean,
        gpr.mean_functions.ConstantMean,
        gpr.mean_functions.NegativeQuadratic,
    ],
)
def test_predict_preserves_overridden_mean_compute(mean_cls, trained):
    """An inherited batched method must not hide a custom scalar mean."""

    class OffsetMean(mean_cls):
        def compute(self, hyp, X, compute_grad=False):
            result = super().compute(hyp, X, compute_grad)
            offset = 3 * X[:, 0]
            if compute_grad:
                return result[0] + offset, result[1]
            return result + offset

    gp = gpr.GP(
        D=1,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=OffsetMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    X = np.array([[-1.0], [0.0], [1.0]])
    hyp = np.zeros((2, 3 + gp.mean.hyperparameter_count(1)))
    hyp[:, 2] = -1
    hyp[1] += 0.2
    if trained:
        gp.update(X, np.array([[1.0], [2.0], [3.0]]), hyp=hyp)
    else:
        gp.set_hyperparameters(hyp)
    actual = gp.predict(X, separate_samples=True)[0]
    expected = gp.predict_full(X)[0]
    assert np.array_equal(actual, expected)


def test_predict_custom_batched_mean_inheritance():
    """Compatible batched overrides work through inheritance and mixins."""

    class OffsetMean(gpr.mean_functions.ConstantMean):
        def compute(self, hyp, X, compute_grad=False):
            result = super().compute(hyp, X, compute_grad)
            if compute_grad:
                return result[0] + X[:, 0], result[1]
            return result + X[:, 0]

    class BatchedOffsetMean(OffsetMean):
        def compute_batched(self, hyp, X):
            self.batched_calls += 1
            return hyp[:, 0] + X[:, :1]

    class BothMethodsMean(OffsetMean):
        compute = OffsetMean.compute
        compute_batched = BatchedOffsetMean.compute_batched

    class InheritedBatchedMean(BatchedOffsetMean):
        pass

    class OverrideAgainMean(BatchedOffsetMean):
        def compute(self, hyp, X, compute_grad=False):
            result = super().compute(hyp, X, compute_grad)
            if compute_grad:
                return result[0] + 2, result[1]
            # The original per-sample path also accepted column vectors.
            return (result + 2)[:, None]

    class UnrelatedMixin:
        def compute_batched(self, hyp, X):
            raise AssertionError("Unrelated batched mean must not be used")

    class MixinMean(UnrelatedMixin, OffsetMean):
        pass

    for mean_cls, calls in (
        (BatchedOffsetMean, 1),
        (BothMethodsMean, 1),
        (InheritedBatchedMean, 1),
        (OverrideAgainMean, 0),
        (MixinMean, 0),
    ):
        mean = mean_cls()
        mean.batched_calls = 0
        gp = gpr.GP(
            D=1,
            covariance=gpr.covariance_functions.SquaredExponential(),
            mean=mean,
            noise=gpr.noise_functions.GaussianNoise(constant_add=True),
        )
        gp.set_hyperparameters(np.array([[0, 0, -1, 2], [0, 0, -1, 3]]))
        X = np.array([[-1.0], [0.0], [1.0]])
        assert np.array_equal(
            gp.predict(X, separate_samples=True)[0], gp.predict_full(X)[0]
        )
        assert mean.batched_calls == calls


@pytest.mark.parametrize("noise_variance", [1e-8, 1e-4])
@pytest.mark.parametrize("per_point_noise", [False, True])
def test_float32_kernel_preserves_diagonal_noise(
    noise_variance, per_point_noise
):
    """Small diagonal noise survives both Cholesky parametrizations."""

    class Float32Kernel(gpr.covariance_functions.SquaredExponential):
        def compute(self, *args, **kwargs):
            result = super().compute(*args, **kwargs)
            if isinstance(result, tuple):
                return tuple(value.astype(np.float32) for value in result)
            return result.astype(np.float32)

    gp = gpr.GP(
        D=1,
        covariance=Float32Kernel(),
        mean=gpr.mean_functions.ZeroMean(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=per_point_noise
        ),
    )
    X = np.array([[0.0], [0.0], [1.0]])
    y = np.array([[1.0], [1.0], [0.0]])
    hyp = np.array([0, 0, 0.5 * np.log(noise_variance)], dtype=np.float32)
    s2 = noise_variance * np.array([[0.0], [0.5], [1.0]])
    if not per_point_noise:
        s2 = None

    # Reference the original matrix expression, including dtype promotion.
    K = gp.covariance.compute(hyp[:2], X)
    sn2 = gp.noise.compute(hyp[2:], X, y, s2)
    if np.min(sn2) >= 1e-6:
        sl = np.min(sn2)
        diagonal = np.eye(3) if np.isscalar(sn2) else np.diag(sn2.ravel() / sl)
        A = K / sl + diagonal
    else:
        sl = 1
        A = K + (sn2 * np.eye(3) if np.isscalar(sn2) else np.diag(sn2.ravel()))
    L = scipy.linalg.cholesky(A, check_finite=False)
    alpha = (
        scipy.linalg.solve_triangular(
            L,
            scipy.linalg.solve_triangular(L, y, trans=1, check_finite=False),
            check_finite=False,
        )
        / sl
    )
    expected_nlz = (
        (y.T @ (alpha / 2))[0, 0]
        + np.sum(np.log(np.diag(L)))
        + 3 * np.log(2 * np.pi * sl) / 2
    )

    gp.update(X, y, s2, hyp=hyp[None, :])
    assert gp.posteriors[0].sn2_mult == 1
    assert gp.posteriors[0].L.dtype == np.float64
    assert gp.log_likelihood(hyp) == -expected_nlz


def _small_gp_with_priors(seed=3):
    rng = np.random.default_rng(seed)
    N, D = 25, 2
    X = rng.standard_normal((N, D))
    y = np.sin(X).sum(1, keepdims=True) + 0.1 * rng.standard_normal((N, 1))
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    hyp = 0.3 * rng.standard_normal((1, 3 * D + 3))
    gp.update(X_new=X, y_new=y, hyp=hyp, compute_posterior=True)
    names = list(gp.get_bounds().keys())
    priors = {
        names[0]: (
            "student_t",
            (np.zeros(D), np.full(D, 1.0), np.full(D, 3.0)),
        ),
        names[1]: ("gaussian", (np.zeros(1), np.ones(1))),
        names[2]: (
            "smoothbox",
            (np.array([-3.0]), np.array([-1.0]), np.array([0.5])),
        ),
        names[3]: (
            "smoothbox_student_t",
            (
                np.array([-1.0]),
                np.array([1.0]),
                np.array([0.5]),
                np.array([4.0]),
            ),
        ),
        names[4]: ("gaussian", (np.zeros(D), np.full(D, 2.0))),
        names[5]: None,
    }
    bounds = {
        n: (np.full(np.size(v[0]), -6.0), np.full(np.size(v[0]), 6.0))
        for n, v in gp.get_bounds().items()
    }
    gp.set_bounds(bounds)
    gp.set_priors(priors)
    return gp, hyp[0]


def test_log_likelihood_and_posterior_gradients():
    """``compute_grad=True`` returns ``(value, gradient)`` with the gradient
    of the value returned without it (the two public wrappers used to apply
    the unary minus to the returned tuple and raise)."""
    gp, hyp = _small_gp_with_priors()
    for value_fn in (gp.log_likelihood, gp.log_posterior):
        value, grad = value_fn(hyp, compute_grad=True)
        assert value == value_fn(hyp)
        assert grad.shape == hyp.shape
        assert np.all(
            check_grad(value_fn, lambda h: value_fn(h, True)[1], hyp) < 1e-5
        )


def test_prior_mask_cache_follows_priors_and_bounds():
    """The cached hyperprior masks are dropped whenever the priors, the
    bounds or the prior's ``df`` change, so ``log_posterior`` always
    matches a GP that never cached (and an object without the attribute,
    as an old pickle, builds it on first use)."""
    gp, hyp = _small_gp_with_priors()
    lp0 = gp.log_posterior(hyp)
    fresh, _ = _small_gp_with_priors()
    assert lp0 == fresh.log_posterior(hyp)
    if hasattr(gp, "_prior_cache"):
        del gp._prior_cache
    assert gp.log_posterior(hyp) == lp0
    # priors change
    priors = gp.get_priors()
    names = list(priors)
    priors[names[1]] = ("gaussian", (np.array([0.5]), np.array([0.2])))
    gp.set_priors(priors)
    lp1 = gp.log_posterior(hyp)
    fresh.set_priors(priors)
    assert lp1 != lp0 and lp1 == fresh.log_posterior(hyp)
    # bounds change (normalization constants)
    bounds = gp.get_bounds()
    bounds[names[1]] = (np.array([-1.0]), np.array([1.0]))
    gp.set_bounds(bounds)
    lp2 = gp.log_posterior(hyp)
    fresh.set_bounds(bounds)
    assert lp2 != lp1 and lp2 == fresh.log_posterior(hyp)
    # the df fill at the top of fit (NaN -> df_base) changes the masks
    gp.hyper_priors["df"][:] = np.nan
    gp._prior_cache = None
    fresh.hyper_priors["df"][:] = np.nan
    fresh._prior_cache = None
    gp.fit(options={"n_samples": 0, "init_N": 0, "opts_N": 0})
    fresh.fit(options={"n_samples": 0, "init_N": 0, "opts_N": 0})
    assert gp.log_posterior(hyp) == fresh.log_posterior(hyp)


def test_squared_exponential_symmetric_kernel_matrix():
    """``compute(X)`` equals ``squareform(pdist(X / ell))`` bit for bit,
    is exactly symmetric and has ``sf2`` on the diagonal."""
    from scipy.spatial.distance import pdist, squareform

    rng = np.random.default_rng(5)
    cov = gpr.covariance_functions.SquaredExponential()
    for N, D in [(1, 2), (7, 1), (40, 3), (120, 9)]:
        X = rng.standard_normal((N, D))
        hyp = 0.3 * rng.standard_normal(D + 1)
        K = cov.compute(hyp, X)
        ell, sf2 = np.exp(hyp[:D]), np.exp(2 * hyp[D])
        expected = sf2 * np.exp(-squareform(pdist(X / ell, "sqeuclidean")) / 2)
        assert np.array_equal(K, expected)
        assert np.array_equal(K, K.T)
        assert np.array_equal(np.diag(K), np.full(N, sf2))
        assert np.array_equal(
            cov.compute(hyp, X, compute_diag=True), np.full((N, 1), sf2)
        )


def test_fit_cholesky_reuse_is_exact(monkeypatch):
    """The sampler's objective reuses the Cholesky factor when only a
    mean-function hyperparameter moved; a fit with the reuse on reproduces
    a fit with it off bit for bit under the same seed."""
    import gpyreg.gaussian_process as gpmod

    results = []
    for reuse in (True, False):
        monkeypatch.setattr(gpmod, "_REUSE_CHOLESKY", reuse)
        gp, _ = _small_gp_with_priors(seed=11)
        hyp, _, res = gp.fit(
            options={
                "n_samples": 6,
                "thin": 2,
                "burn": 6,
                "init_N": 24,
                "opts_N": 1,
                "init_method": "rand",
            },
            rng=np.random.default_rng(2026),
        )
        results.append((hyp, res["samples"], np.asarray(res["f_vals"])))
    for a, b in zip(results[0], results[1]):
        assert np.array_equal(a, b)


def test_gradient_path_never_uses_the_cache():
    """A cache whose key matches but whose factor is wrong must not reach
    the gradient objective (it needs the kernel derivatives the reused
    block would skip)."""
    gp, hyp = _small_gp_with_priors(seed=12)
    cov_N = gp.covariance.hyperparameter_count(gp.D)
    noise_N = gp.noise.hyperparameter_count()
    N = gp.X.shape[0]
    reference = gp._GP__compute_nlZ(hyp, True, True)
    poisoned = {
        "key": hyp[: cov_N + noise_N].copy(),
        "sn2": 1.0,
        "L": np.eye(N),
        "sl": 1.0,
        "sn2_mult": 1,
        "L_chol": True,
        "pL": np.eye(N),
        "logdet": 0.0,
    }
    nlZ, dnlZ = gp._GP__compute_nlZ(hyp, True, True, poisoned)
    assert nlZ == reference[0] and np.array_equal(dnlZ, reference[1])
    # ... while the no-gradient objective does take a valid hit
    cache = {}
    v0 = gp._GP__compute_nlZ(hyp, False, True, cache)
    assert "key" in cache
    v1 = gp._GP__compute_nlZ(hyp, False, True, cache)
    assert v1 == v0 == gp._GP__compute_nlZ(hyp, False, True)
    hyp2 = hyp.copy()
    hyp2[cov_N + noise_N] += 0.3  # a mean hyperparameter: a hit
    assert gp._GP__compute_nlZ(
        hyp2, False, True, cache
    ) == gp._GP__compute_nlZ(hyp2, False, True)
    hyp3 = hyp.copy()
    hyp3[0] += 0.3  # a covariance hyperparameter: a miss, cache refreshed
    assert gp._GP__compute_nlZ(
        hyp3, False, True, cache
    ) == gp._GP__compute_nlZ(hyp3, False, True)
    assert np.array_equal(cache["key"], hyp3[: cov_N + noise_N])


def test_fit_and_random_function_with_generator():
    """``fit(rng=)`` and ``random_function(rng=)`` draw from the given
    generator: two fits seeded alike agree bit for bit whatever the global
    legacy state does, while ``rng=None`` still follows ``np.random.seed``
    as before generators were supported."""
    state = np.random.get_state()
    options = {
        "n_samples": 4,
        "thin": 2,
        "burn": 4,
        "init_N": 16,
        "opts_N": 1,
        "init_method": "rand",
    }
    try:
        results = []
        for global_seed in (3, 4):
            gp, _ = _small_gp_with_priors(seed=21)
            np.random.seed(global_seed)
            hyp, _, res = gp.fit(options=options, rng=np.random.default_rng(5))
            f = gp.random_function(
                gp.X[:3], add_noise=True, rng=np.random.default_rng(6)
            )
            results.append((hyp, res["samples"], f))
        for a, b in zip(results[0], results[1]):
            assert np.array_equal(a, b)
        legacy = []
        for _ in range(2):
            gp, _ = _small_gp_with_priors(seed=21)
            np.random.seed(8)
            hyp, _, _ = gp.fit(options=options)
            legacy.append((hyp, gp.random_function(gp.X[:3])))
        assert np.array_equal(legacy[0][0], legacy[1][0])
        assert np.array_equal(legacy[0][1], legacy[1][1])
    finally:
        np.random.set_state(state)


@pytest.mark.parametrize("init_method", ["rand", "sobol"])
@pytest.mark.parametrize("seed_kind", ["integer", "seed_sequence"])
def test_fit_seed_continues_design_stream(init_method, seed_kind):
    """Seeding a fit is equivalent to passing the corresponding generator."""
    seed = 5 if seed_kind == "integer" else np.random.SeedSequence(5)
    results = []
    for rng in (seed, np.random.default_rng(5)):
        gp, _ = _small_gp_with_priors(seed=21)
        hyp, _, sampling = gp.fit(
            options={
                "n_samples": 4,
                "thin": 2,
                "burn": 4,
                "init_N": 16,
                "opts_N": 0,
                "init_method": init_method,
            },
            rng=rng,
        )
        results.append((hyp, sampling["samples"], sampling["f_vals"]))
    for seeded, generated in zip(*results):
        assert np.array_equal(seeded, generated)


def test_predict_full_add_noise_per_point():
    """``predict_full(add_noise=True)`` adds the observation noise on the
    diagonal, so the returned matrix stays a covariance also when the noise
    varies from point to point, and its diagonal is ``predict``'s."""
    N = 12
    D = 2
    rng = np.random.default_rng(4)
    X = rng.uniform(-2, 2, size=(N, D))
    y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2])

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=True
        ),
    )
    # [log ell (D), log sf, log sn, m0]
    hyp = np.array([[0.0, 0.0, 0.0, np.log(0.2), 0.0]])
    gp.update(X_new=X, y_new=y, hyp=hyp)

    x_star = rng.uniform(-2, 2, size=(5, D))
    s2_star = np.array([[0.5], [0.1], [0.3], [0.05], [0.4]])

    __, cov = gp.predict_full(x_star, s2_star=s2_star, add_noise=True)
    C = cov[:, :, 0]
    assert np.allclose(C, C.T, rtol=0, atol=1e-14)
    assert np.min(np.linalg.eigvalsh((C + C.T) / 2)) > -1e-10

    __, s2 = gp.predict(x_star, s2_star=s2_star, add_noise=True)
    assert np.allclose(np.diag(C), s2[:, 0], rtol=1e-12, atol=1e-14)

    # The latent covariance differs from the noisy one on the diagonal only.
    __, cov_latent = gp.predict_full(x_star, s2_star=s2_star)
    added = C - cov_latent[:, :, 0]
    assert np.allclose(added, np.diag(np.diag(added)), rtol=0, atol=1e-14)
    assert np.allclose(np.diag(added), np.ravel(s2_star) + 0.04)


def test_failed_factorization_raises_linalgerror():
    """A training covariance that stays singular after every retry of the
    noise inflation reports a ``LinAlgError`` in both noise
    parametrizations, the low-noise one included."""
    D = 2
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(40, D))
    y = np.sum(X, 1).reshape(-1, 1)

    def make_gp():
        return gpr.GP(
            D=D,
            covariance=gpr.covariance_functions.SquaredExponential(),
            mean=gpr.mean_functions.NegativeQuadratic(),
            noise=gpr.noise_functions.GaussianNoise(constant_add=True),
        )

    # [log ell (D), log sf, log sn, m0, mode location (D), log scale (D)]
    hyp = np.zeros(3 * D + 3)
    hyp[0:D] = np.log(1e5)
    hyp[D] = 20.0
    hyp[2 * D + 3 :] = np.log(3.0)

    # min(sn2) < 1e-6: the low-noise parametrization.
    hyp[D + 1] = np.log(1e-7)
    with pytest.raises(scipy.linalg.LinAlgError) as execinfo:
        make_gp().update(X_new=X, y_new=y, hyp=hyp[None, :])
    assert "Singular matrix" in execinfo.value.args[0]

    # And the Cholesky parametrization, for contrast.
    hyp[0:D] = np.log(1e6)
    hyp[D] = 25.0
    hyp[D + 1] = np.log(3e-3)
    with pytest.raises(scipy.linalg.LinAlgError) as execinfo:
        make_gp().update(X_new=X, y_new=y, hyp=hyp[None, :])
    assert "Singular matrix" in execinfo.value.args[0]


def test_robust_cholesky_factors_matrices_cholesky_refuses():
    """The eigenvalue fallback returns a factor of the matrix it was given:
    ``T.T @ T == sigma`` for the semidefinite matrices a direct Cholesky
    decomposition refuses."""
    robust_cholesky = gpr.GP._GP__robust_cholesky
    rng = np.random.default_rng(7)

    cases = {}
    # Rank-deficient but positive semidefinite.
    A = rng.standard_normal((6, 3))
    cases["rank deficient"] = A @ A.T
    # A kernel matrix at duplicated points, exactly singular.
    kernel = gpr.covariance_functions.SquaredExponential()
    X = np.array([[0.0], [0.0], [1.0], [1.0], [-1.5]])
    cases["duplicate points"] = kernel.compute(np.array([0.0, 0.0]), X)
    # A repeated positive eigenvalue, whose eigenspace the general solver
    # need not return an orthogonal basis of.
    Q, __ = np.linalg.qr(rng.standard_normal((5, 5)))
    cases["repeated eigenvalue"] = Q @ np.diag([1.0, 1.0, 1.0, 0.0, 0.0]) @ Q.T

    for name, sigma in cases.items():
        sigma = (sigma + sigma.T) / 2
        with pytest.raises(scipy.linalg.LinAlgError):
            scipy.linalg.cholesky(sigma, check_finite=False)
        T = robust_cholesky(sigma)
        assert np.isrealobj(T), name
        assert np.allclose(
            T.T @ T, sigma, rtol=0, atol=1e-10 * np.max(np.abs(sigma))
        ), name


def test_random_function_on_a_dense_grid():
    """A predictive covariance on a dense one-dimensional grid is
    numerically singular, so the draw goes through the eigenvalue
    fallback. Eigenvalues of rounding size, which such a matrix has of
    both signs, count as zeros, and the draws are draws: they differ
    between generators and carry the predictive covariance."""
    rng_data = np.random.default_rng(77)
    X = rng_data.uniform(-2, 2, size=(40, 1))
    y = np.sin(2 * X)

    gp = gpr.GP(
        D=1,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    # [log ell, log sf, log sn, m0, mode location, log scale]
    hyp = np.array(
        [[np.log(0.7), np.log(1.2), np.log(1e-3), 0.0, 0.0, np.log(1.5)]]
    )
    gp.update(X_new=X, y_new=y, hyp=hyp)

    x_star = np.reshape(np.linspace(-2.5, 2.5, 100), (-1, 1))
    __, cov = gp.predict_full(x_star)
    C = (cov[:, :, 0] + cov[:, :, 0].T) / 2
    with pytest.raises(scipy.linalg.LinAlgError):
        scipy.linalg.cholesky(C, check_finite=False)

    f_1 = gp.random_function(x_star, rng=np.random.default_rng(1))
    f_2 = gp.random_function(x_star, rng=np.random.default_rng(2))
    assert not np.array_equal(f_1, f_2)
    mu, __ = gp.predict(x_star)
    assert not np.allclose(f_1, mu)

    rng = np.random.default_rng(11)
    draws = np.concatenate(
        [gp.random_function(x_star, rng=rng) for __ in range(1000)], axis=1
    )
    empirical = np.cov(draws, ddof=1)
    assert np.linalg.norm(empirical - C) < 0.2 * np.linalg.norm(C)
    assert np.allclose(np.mean(draws, 1), np.ravel(mu), rtol=0, atol=0.05)


def test_robust_cholesky_refuses_an_indefinite_matrix():
    """A negative eigenvalue larger than the rounding tolerance means the
    matrix is no covariance matrix, and no factor of it exists."""
    sigma = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    with pytest.raises(scipy.linalg.LinAlgError) as execinfo:
        gpr.GP._GP__robust_cholesky(sigma)
    assert "not positive semidefinite" in execinfo.value.args[0]


def _low_noise_rank_one_gp(D=2):
    """A GP in the low-noise parametrization of the posterior factor: with
    no constant noise term ``min(sn2)`` is ``eps``, below the 1e-6 the
    Cholesky representation needs."""
    return gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=False),
    )


def test_rank_one_update_low_noise_branch():
    """A well-separated observation appended to a low-noise posterior takes
    the rank-one shortcut and agrees with a full recomputation."""
    D = 2
    rng = np.random.default_rng(3)
    X = rng.uniform(-3, 3, size=(12, D))
    y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2])
    # [log ell (D), log sf, m0]: no noise hyperparameter.
    hyp = np.array([[0.0, 0.0, 0.0, 0.0]])
    x_new = np.array([[2.9, -2.9]])
    y_new = np.sin(x_new[:, 0:1]) + np.cos(x_new[:, 1:2])

    gp = _low_noise_rank_one_gp(D)
    gp.update(X_new=X, y_new=y, hyp=hyp)
    assert not gp.posteriors[0].L_chol
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gp.update(X_new=x_new, y_new=y_new)

    gp_ref = _low_noise_rank_one_gp(D)
    gp_ref.update(
        X_new=np.concatenate((X, x_new)),
        y_new=np.concatenate((y, y_new)),
        hyp=hyp,
    )
    assert not gp_ref.posteriors[0].L_chol
    assert np.allclose(
        gp.posteriors[0].alpha, gp_ref.posteriors[0].alpha, rtol=1e-8
    )

    x_star = rng.uniform(-3, 3, size=(5, D))
    f_mu, f_s2 = gp.predict(x_star)
    f_mu_ref, f_s2_ref = gp_ref.predict(x_star)
    assert np.allclose(f_mu, f_mu_ref, rtol=1e-8, atol=1e-10)
    assert np.allclose(f_s2, f_s2_ref, rtol=1e-8, atol=1e-10)


def test_rank_one_update_low_noise_duplicate_recomputes():
    """An observation at an existing training input leaves the low-noise
    rank-one update dividing by a variance the clamp of ``predict``
    produced, which is no variance: it warns and recomputes in full, as
    the Cholesky branch does."""
    D = 2
    rng = np.random.default_rng(3)
    X = rng.uniform(-3, 3, size=(12, D))
    y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2])
    hyp = np.array([[0.0, 0.0, 0.0, 0.0]])
    x_new = X[0:1].copy()
    y_new = y[0:1].copy()

    gp = _low_noise_rank_one_gp(D)
    gp.update(X_new=X, y_new=y, hyp=hyp)
    assert not gp.posteriors[0].L_chol
    with pytest.warns(UserWarning, match="Reverting to full update"):
        gp.update(X_new=x_new, y_new=y_new)

    gp_ref = _low_noise_rank_one_gp(D)
    gp_ref.update(
        X_new=np.concatenate((X, x_new)),
        y_new=np.concatenate((y, y_new)),
        hyp=hyp,
    )
    assert np.array_equal(gp.posteriors[0].alpha, gp_ref.posteriors[0].alpha)
    assert np.array_equal(gp.posteriors[0].L, gp_ref.posteriors[0].L)

    x_star = rng.uniform(-3, 3, size=(5, D))
    f_mu, f_s2 = gp.predict(x_star)
    f_mu_ref, f_s2_ref = gp_ref.predict(x_star)
    assert np.array_equal(f_mu, f_mu_ref)
    assert np.array_equal(f_s2, f_s2_ref)


@pytest.mark.parametrize("state", ["cleaned", "no_posterior"])
def test_single_point_update_without_posterior_factors(state):
    """The rank-one shortcut extends the stored factors, so a GP that
    carries none -- after ``clean`` or after an update with
    ``compute_posterior=False`` -- recomputes in full instead."""
    N = 10
    D = 2
    rng = np.random.default_rng(9)
    X = rng.uniform(-3, 3, size=(N, D))
    y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2])
    hyp = np.array([[0.0, 0.0, 0.0, np.log(0.1), 0.0]])
    x_new = rng.uniform(-3, 3, size=(1, D))
    y_new = np.sin(x_new[:, 0:1]) + np.cos(x_new[:, 1:2])

    def make_gp():
        return gpr.GP(
            D=D,
            covariance=gpr.covariance_functions.SquaredExponential(),
            mean=gpr.mean_functions.ConstantMean(),
            noise=gpr.noise_functions.GaussianNoise(constant_add=True),
        )

    gp = make_gp()
    if state == "cleaned":
        gp.update(X_new=X, y_new=y, hyp=hyp)
        gp.clean()
    else:
        gp.update(X_new=X, y_new=y, hyp=hyp, compute_posterior=False)
    assert gp.posteriors[0].alpha is None

    gp.update(X_new=x_new, y_new=y_new)
    assert gp.posteriors[0].alpha is not None

    gp_ref = make_gp()
    gp_ref.update(
        X_new=np.concatenate((X, x_new)),
        y_new=np.concatenate((y, y_new)),
        hyp=hyp,
    )
    assert np.array_equal(gp.posteriors[0].alpha, gp_ref.posteriors[0].alpha)
    assert np.array_equal(gp.posteriors[0].L, gp_ref.posteriors[0].L)


def _reference_log_prior(kind, params, x):
    """The hyperprior densities of ``set_priors``, written from their
    definitions: a Gaussian and a Student's t, and their smooth-box
    counterparts, uniform on ``[a, b]`` with tails of scale ``sigma``."""
    if kind == "gaussian":
        mu, sigma = params
        return scipy.stats.norm.logpdf(x, loc=mu, scale=sigma)
    if kind == "student_t":
        mu, sigma, df = params
        return scipy.stats.t.logpdf(x, df, loc=mu, scale=sigma)
    a, b, sigma = params[0], params[1], params[2]
    z = 0.0
    if x < a:
        z = (x - a) / sigma
    elif x > b:
        z = (x - b) / sigma
    if kind == "smoothbox":
        # The plateau has the density of the tails at their peak, so the
        # normalizer is the plateau's length in units of that peak plus one.
        C = 1.0 + (b - a) / (sigma * np.sqrt(2 * np.pi))
        return -np.log(C * sigma * np.sqrt(2 * np.pi)) - 0.5 * z**2
    if kind == "smoothbox_student_t":
        df = params[3]
        peak = np.exp(
            scipy.special.gammaln(0.5 * (df + 1))
            - scipy.special.gammaln(0.5 * df)
        ) / (sigma * np.sqrt(df * np.pi))
        C = 1.0 + (b - a) * peak
        return np.log(peak / C) - 0.5 * (df + 1) * np.log1p(z**2 / df)
    raise ValueError("unknown prior " + kind)


def test_log_prior_matches_the_documented_densities():
    """``log_posterior`` minus ``log_likelihood`` is the log hyperprior:
    the documented Gaussian, Student's t and smooth-box densities, with an
    infinite or missing number of degrees of freedom naming the Gaussian
    families as ``gplite_nlZ.m`` documents."""
    D = 1
    X = np.reshape(np.linspace(-2, 2, 8), (-1, 1))
    y = np.sin(X)

    # In the order of the hyperparameter array: covariance, noise, mean.
    priors = {
        "covariance_log_lengthscale": ("gaussian", (0.3, 1.2)),
        "covariance_log_outputscale": ("student_t", (-0.2, 0.8, 5.0)),
        "noise_log_scale": ("smoothbox", (-1.0, 1.0, 0.7)),
        "mean_const": ("smoothbox_student_t", (-2.0, 0.5, 0.9, 4.0)),
    }

    # The reference densities are normalized, which fixes their constants
    # independently of gpyreg.
    for kind, params in priors.values():
        mass, __ = quad(
            lambda t: np.exp(_reference_log_prior(kind, params, t)),
            -np.inf,
            np.inf,
        )
        assert np.isclose(mass, 1.0, rtol=1e-6), kind

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    # The smooth-box coordinates are the last two: inside their boxes in
    # the first vector, in either tail in the second.
    for hyp in (
        np.array([0.5, -0.4, 0.2, -0.3]),
        np.array([-1.1, 1.3, -1.5, 0.9]),
    ):
        gp.update(X_new=X, y_new=y, hyp=hyp[None, :])
        expected = sum(
            _reference_log_prior(kind, params, x)
            for (kind, params), x in zip(priors.values(), hyp)
        )
        # No bounds are set, so no prior is truncated and the
        # renormalization over the bounds is zero.
        gp.set_priors(priors)
        assert np.all(np.isnan(gp.lower_bounds))
        log_prior = gp.log_posterior(hyp) - gp.log_likelihood(hyp)
        assert np.isclose(log_prior, expected, rtol=1e-12)

        # A Gaussian and a smooth box are also written with an infinite
        # number of degrees of freedom, and `fit` leaves NaN where it has
        # no default to fill: both name the same density.
        for df in (np.inf, np.nan):
            gp.set_priors(priors)
            gp.hyper_priors["df"][0] = df  # the Gaussian
            gp.hyper_priors["df"][2] = df  # the smooth box
            log_prior = gp.log_posterior(hyp) - gp.log_likelihood(hyp)
            assert np.isclose(log_prior, expected, rtol=1e-12)


@pytest.mark.parametrize("family", ["smoothbox", "smoothbox_student_t"])
@pytest.mark.parametrize("D", [2, 3])
def test_smooth_box_prior_over_a_block(family, D):
    """A smooth-box prior set on a block of several hyperparameters has one
    normalization constant per coordinate of the block, whichever side of
    the box each coordinate falls on."""
    params = (
        (-1.0, 1.0, 0.7) if family == "smoothbox" else (-1.0, 1.0, 0.7, 4.0)
    )
    priors = {
        "covariance_log_lengthscale": (family, params),
        "covariance_log_outputscale": None,
        "noise_log_scale": None,
        "mean_const": None,
    }

    X = np.reshape(np.linspace(-2, 2, 6 * D), (-1, D))
    y = np.sum(np.sin(X), 1)

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    # The lengthscales fall below the box, on the plateau and above it.
    lengthscales = np.array([-1.5, 0.2, 1.4])[:D]
    hyp = np.concatenate((lengthscales, [0.0, np.log(0.1), 0.0]))
    gp.update(X_new=X, y_new=y, hyp=hyp[None, :])
    gp.set_priors(priors)

    expected = sum(
        _reference_log_prior(family, params, x) for x in lengthscales
    )
    log_prior = gp.log_posterior(hyp) - gp.log_likelihood(hyp)
    assert np.isclose(log_prior, expected, rtol=1e-12)

    returned = gp.get_priors()["covariance_log_lengthscale"]
    assert returned[0] == family
    for value, expected_value in zip(returned[1], params):
        assert np.all(value == expected_value)


def _gp_1d():
    """A one-dimensional GP with four hyperparameters: two of the kernel,
    one of the noise and one of the mean."""
    return gpr.GP(
        D=1,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )


def _no_priors():
    return {
        "covariance_log_lengthscale": None,
        "covariance_log_outputscale": None,
        "noise_log_scale": None,
        "mean_const": None,
    }


@pytest.mark.parametrize("sigma", [np.inf, -np.inf, np.nan, 0.0, -2.0])
def test_set_priors_refuses_a_scale_that_is_not_positive(sigma):
    """A prior needs a finite, positive scale; no prior is ``None``."""
    priors = _no_priors()
    priors["mean_const"] = ("gaussian", (0.0, sigma))
    with pytest.raises(ValueError) as execinfo:
        _gp_1d().set_priors(priors)
    assert "mean_const" in execinfo.value.args[0]
    assert "None" in execinfo.value.args[0]


def test_set_priors_and_set_bounds_refuse_an_unknown_hyperparameter():
    """A name outside the model is a mistake, and silently setting no
    prior on it is what `set_priors`' own docstring promises against."""
    gp = _gp_1d()
    priors = _no_priors()
    priors["not_a_hyperparameter"] = ("gaussian", (0.0, 1.0))
    with pytest.raises(ValueError) as execinfo:
        gp.set_priors(priors)
    assert "not_a_hyperparameter" in execinfo.value.args[0]

    bounds = {name: None for name in _no_priors()}
    bounds["not_a_hyperparameter"] = (-1.0, 1.0)
    with pytest.raises(ValueError) as execinfo:
        gp.set_bounds(bounds)
    assert "not_a_hyperparameter" in execinfo.value.args[0]


def test_get_recommended_bounds_input_checks():
    """Bounds may be given as any array_like, the message of the upper
    bounds names them, and an inverted pair the caller gave is refused."""
    X = np.reshape(np.linspace(-2, 2, 8), (-1, 1))
    y = np.sin(X)
    gp = _gp_1d()
    gp.update(X_new=X, y_new=y, hyp=np.array([[0.0, 0.0, np.log(0.1), 0.0]]))

    recommended = gp.get_recommended_bounds()
    for given in ([np.nan] * 4, (np.nan,) * 4):
        bounds = gp.get_recommended_bounds(given, given)
        for name, pair in bounds.items():
            assert np.array_equal(pair[0], recommended[name][0])
            assert np.array_equal(pair[1], recommended[name][1])

    with pytest.raises(ValueError) as execinfo:
        gp.get_recommended_bounds(upper_bounds="nonsense")
    assert "`upper_bounds`" in execinfo.value.args[0]

    with pytest.raises(ValueError) as execinfo:
        gp.get_recommended_bounds(np.ones(4), -np.ones(4))
    assert "upper bound" in execinfo.value.args[0]


def test_update_without_hyperparameters_raises():
    """A posterior cannot be computed from hyperparameters that were never
    set: the message names them instead of leaving NaN factors behind."""
    X = np.reshape(np.linspace(-2, 2, 8), (-1, 1))
    y = np.sin(X)
    gp = _gp_1d()
    with pytest.raises(ValueError) as execinfo:
        gp.update(X_new=X, y_new=y)
    message = execinfo.value.args[0]
    for name in _no_priors():
        assert name in message


def test_fit_leaves_the_prior_degrees_of_freedom_alone():
    """``df_base`` fills the degrees of freedom a prior leaves unset for
    the duration of the fit; the GP keeps the priors the caller set, so a
    second fit with another value uses it."""
    X = np.reshape(np.linspace(-2, 2, 12), (-1, 1))
    y = np.sin(X)
    gp = _gp_1d()
    priors = _no_priors()
    priors["covariance_log_outputscale"] = ("student_t", (0.0, 1.0, np.nan))
    gp.set_priors(priors)
    df_before = gp.hyper_priors["df"].copy()
    assert np.all(np.isnan(df_before))

    hyp0 = np.array([[0.0, 0.0, np.log(0.1), 0.0]])
    options = {"init_N": 0, "opts_N": 1, "n_samples": 0}

    results = []
    for df_base in (7, 400):
        __, result, __ = gp.fit(
            X=X, y=y, hyp0=hyp0, options={**options, "df_base": df_base}
        )
        assert np.array_equal(gp.hyper_priors["df"], df_before, equal_nan=True)
        results.append(result.fun)
    assert results[0] != results[1]


@pytest.mark.parametrize("df", [5.0, 400.0])
def test_smooth_box_student_t_prior_with_many_degrees_of_freedom(df):
    """The normalizer of the smooth-box Student's t is a ratio of gamma
    functions, both of which overflow from a few hundred degrees of
    freedom; the log prior stays finite and keeps its value."""
    params = (-1.0, 1.0, 0.7, df)
    priors = _no_priors()
    priors["mean_const"] = ("smoothbox_student_t", params)

    X = np.reshape(np.linspace(-2, 2, 8), (-1, 1))
    y = np.sin(X)
    gp = _gp_1d()
    hyp = np.array([0.0, 0.0, np.log(0.1), 1.4])  # above the box
    gp.update(X_new=X, y_new=y, hyp=hyp[None, :])
    gp.set_priors(priors)

    log_prior = gp.log_posterior(hyp) - gp.log_likelihood(hyp)
    assert np.isfinite(log_prior)
    assert np.isclose(
        log_prior,
        _reference_log_prior("smoothbox_student_t", params, hyp[3]),
        rtol=1e-12,
    )


@pytest.mark.parametrize("key", ["sampler_name", "sampler"])
def test_fit_reads_the_documented_sampler_option(key):
    """`fit` documents the sampler under `sampler_name` and read it under
    `sampler`, so the documented spelling was ignored. Both are read, the
    documented one first."""
    X = np.reshape(np.linspace(-2, 2, 10), (-1, 1))
    y = np.sin(X)
    gp = _gp_1d()
    options = {
        "init_N": 0,
        "opts_N": 1,
        "n_samples": 1,
        "thin": 1,
        "burn": 2,
        key: "does_not_exist",
    }
    with pytest.raises(ValueError) as execinfo:
        gp.fit(
            X=X,
            y=y,
            hyp0=np.array([[0.0, 0.0, np.log(0.1), 0.0]]),
            options=options,
        )
    assert "Unknown sampler!" in execinfo.value.args[0]


def test_log_likelihood_and_posterior_take_a_dictionary():
    """Both methods document a dictionary of hyperparameters, which
    `hyperparameters_from_dict` returns as one row of an array."""
    gp, hyp = _small_gp_with_priors(seed=4)
    hyp = np.ravel(hyp)
    as_dict = gp.hyperparameters_to_dict(hyp)[0]

    for method in (gp.log_likelihood, gp.log_posterior):
        assert np.array_equal(method(as_dict), method(hyp))
        value, gradient = method(as_dict, compute_grad=True)
        value_ref, gradient_ref = method(hyp, compute_grad=True)
        assert np.array_equal(value, value_ref)
        assert np.array_equal(gradient, gradient_ref)


def test_fit_does_not_write_into_the_space_filling_design(monkeypatch):
    """The low-noise starting point is written into the array of starting
    points, which was a view of the design, so the sampler widths are the
    standard deviation of the design as `f_min_fill` returned it."""
    from gpyreg import gaussian_process as gp_module

    records = {}
    real_f_min_fill = gp_module.f_min_fill

    def recording_f_min_fill(*args, **kwargs):
        X0, y0 = real_f_min_fill(*args, **kwargs)
        records["design"] = X0
        records["as_returned"] = X0.copy()
        return X0, y0

    class RecordingSliceSampler(gp_module.SliceSampler):
        def __init__(self, f, x0, widths, *args, **kwargs):
            records["widths"] = np.array(widths, copy=True)
            super().__init__(f, x0, widths, *args, **kwargs)

    monkeypatch.setattr(gp_module, "f_min_fill", recording_f_min_fill)
    monkeypatch.setattr(gp_module, "SliceSampler", RecordingSliceSampler)

    X = np.reshape(np.linspace(-3, 3, 20), (-1, 1))
    y = np.sin(X)
    gp = _gp_1d()
    gp.fit(
        X=X,
        y=y,
        options={
            "opts_N": 3,
            "init_N": 64,
            "n_samples": 2,
            "thin": 1,
            "burn": 2,
        },
        rng=np.random.default_rng(0),
    )
    assert np.array_equal(records["design"], records["as_returned"])
    assert np.allclose(
        records["widths"], np.std(records["as_returned"], axis=0, ddof=1)
    )


def test_noise_gradient_with_a_constant_total_noise():
    """A scale for the user-provided variance gives the noise function two
    hyperparameters and a gradient with one row per training input, while
    the total noise stays a scalar as long as no variance is given."""
    D = 2
    N = 14
    rng = np.random.default_rng(5)
    X = rng.uniform(-2, 2, size=(N, D))
    y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2])

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True,
            user_provided_add=True,
            scale_user_provided=True,
        ),
    )
    assert gp.noise.hyperparameter_count() == 2
    # [log ell (D), log sf, log sn, log s2 multiplier, m0]
    hyp = np.array([0.1, -0.2, 0.3, np.log(0.2), 0.4, 0.5])
    gp.update(X_new=X, y_new=y, hyp=hyp[None, :])
    assert gp.s2 is None

    __, gradient = gp.log_likelihood(hyp, compute_grad=True)
    assert np.all(
        check_grad(
            gp.log_likelihood,
            lambda h: gp.log_likelihood(h, compute_grad=True)[1],
            hyp,
        )
        < 1e-5
    )
    # With no variance given the multiplier does not enter the noise.
    assert gradient[D + 2] == 0.0


def test_quad_input_checks():
    """Bayesian quadrature needs the training data, the posterior factors,
    a mean function whose hyperparameters it can place and measures with
    one column per input dimension."""
    D = 2
    rng = np.random.default_rng(8)
    X = rng.uniform(-2, 2, size=(12, D))
    y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2])
    hyp = np.array([[0.0, 0.0, 0.0, np.log(0.1), 0.0]])

    def make_gp():
        return gpr.GP(
            D=D,
            covariance=gpr.covariance_functions.SquaredExponential(),
            mean=gpr.mean_functions.ConstantMean(),
            noise=gpr.noise_functions.GaussianNoise(constant_add=True),
        )

    gp = make_gp()
    with pytest.raises(ValueError) as execinfo:
        gp.quad(0.0, 1.0)
    assert "training data" in execinfo.value.args[0]

    gp.update(X_new=X, y_new=y, hyp=hyp)

    # A one-dimensional measure is one measure of D dimensions.
    F_flat = gp.quad(np.zeros(D), np.ones(D))
    F_row = gp.quad(np.zeros((1, D)), np.ones((1, D)))
    assert F_flat.shape == (1, 1)
    assert np.array_equal(F_flat, F_row)

    with pytest.raises(ValueError) as execinfo:
        gp.quad(np.zeros((1, D + 1)), 1.0)
    assert "one column per input" in execinfo.value.args[0]

    # A mean function of the caller's own, whose hyperparameters quadrature
    # cannot place.
    other = make_gp()
    other.update(X_new=X, y_new=y, hyp=hyp)
    other.mean = object()
    with pytest.raises(ValueError) as execinfo:
        other.quad(0.0, 1.0)
    assert "mean function" in execinfo.value.args[0]

    gp.clean()
    with pytest.raises(ValueError) as execinfo:
        gp.quad(0.0, 1.0)
    assert "posterior factors" in execinfo.value.args[0]


def test_convert_shapes_input_checks():
    """A variance given as a number of any kind is one variance for every
    input; an array carries one row per input, as `gplite_pred.m:16-23`
    requires."""
    N = 5
    gp = _gp_1d()
    X = np.ones((N, 1))

    for s2 in (3, 3.0, np.int64(3), np.float64(3.0), np.array(3.0)):
        __, __, converted = gp._convert_shapes(X, None, s2)
        assert converted.shape == (N, 1)
        assert np.all(converted == 3.0)

    with pytest.raises(ValueError) as execinfo:
        gp._convert_shapes(X, None, np.ones((1, N)))
    assert "rows" in execinfo.value.args[0]

    with pytest.raises(ValueError):
        gp._convert_shapes(X, None, "nonsense")

    gp3 = gpr.GP(
        D=3,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    with pytest.raises(ValueError) as execinfo:
        gp3._convert_shapes(np.ones((N, 5)), None, None)
    assert "input data 5 doesn't match" in execinfo.value.args[0]


def test_update_checks_the_hyperparameter_width():
    """`update(hyp=...)` stored a row of any width, which `predict` then
    read block by block at the wrong offsets."""
    gp = _gp_1d()  # two kernel, one noise and one mean hyperparameter
    with pytest.raises(ValueError) as execinfo:
        gp.update(hyp=np.zeros((1, 5)))
    assert "4 hyperparameters" in execinfo.value.args[0]

    with pytest.raises(ValueError) as execinfo:
        gp.update(hyp=np.zeros(4))
    assert "one row per hyperparameter sample" in execinfo.value.args[0]

    gp.update(hyp=np.zeros((2, 4)))
    assert np.size(gp.posteriors) == 2
