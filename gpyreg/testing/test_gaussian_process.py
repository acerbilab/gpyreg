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
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gp_gradient_computations(seed):
    rng = np.random.default_rng(seed)
    N = 20
    D = 2
    X = rng.standard_normal(size=(N, D))

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    cov_N = gp.covariance.hyperparameter_count(D)
    mean_N = gp.mean.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()

    N_s = rng.integers(1, 3)
    hyp = rng.standard_normal(size=(N_s, cov_N + noise_N + mean_N))
    hyp[:, D] *= 0.2
    hyp[:, D + 1 : D + 1 + noise_N] *= 0.3

    gp.update(hyp=hyp, compute_posterior=False)
    y = gp.random_function(X, rng=rng)

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
                hyp0 * np.exp(0.1 * rng.uniform(size=hyp0.size)),
            ),
            0.0,
            atol=1e-6,
        )
    )

    # Check GP hyperparameters log prior gradient computation.
    hyp1 = hyp0 * np.exp(0.1 * rng.uniform(size=hyp0.size))
    prior_types = rng.permutation(range(0, 5))
    for i in range(0, cov_N + mean_N + noise_N):
        prior_type = prior_types[i]
        if prior_type == 1:  # 'gaussian'
            gp.hyper_priors["mu"][i] = rng.standard_normal()
            gp.hyper_priors["sigma"][i] = np.exp(rng.standard_normal())
            gp.hyper_priors["df"][i] = 0
        elif prior_type == 2:  #'student_t'
            gp.hyper_priors["mu"][i] = rng.standard_normal()
            gp.hyper_priors["sigma"][i] = rng.standard_normal()
            gp.hyper_priors["df"][i] = np.exp(rng.standard_normal())
        elif prior_type == 3:  # 'smoothbox'
            gp.hyper_priors["a"][i] = -3
            gp.hyper_priors["b"][i] = 3
            gp.hyper_priors["sigma"][i] = rng.standard_normal()
            gp.hyper_priors["df"][i] = 0
        elif prior_type == 4:  # 'smoothbox_student_t'
            gp.hyper_priors["a"][i] = -3
            gp.hyper_priors["b"][i] = 3
            gp.hyper_priors["sigma"][i] = rng.standard_normal()
            gp.hyper_priors["df"][i] = np.exp(rng.standard_normal())
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


def test_quad_without_stored_noise_scale():
    # Posteriors pickled by earlier versions have no ``sl`` attribute; the
    # variance of an integral then recovers the scale of the Cholesky
    # factor from ``sW``.
    N = 12
    rng = np.random.default_rng(15)
    X = np.reshape(np.linspace(-2, 2, N), (-1, 1))
    y = np.sin(X) + 0.1 * rng.standard_normal((N, 1))
    hyp = np.array([[0.0, 0.0, np.log(0.1), 0.0]])
    gp = gpr.GP(
        D=1,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    gp.update(X_new=X, y_new=y, hyp=hyp)
    assert gp.posteriors[0].L_chol
    F_ref, F_var_ref = gp.quad(0.0, 1.0, compute_var=True)

    del gp.posteriors[0].sl
    F, F_var = gp.quad(0.0, 1.0, compute_var=True)

    assert np.array_equal(F, F_ref)
    assert np.allclose(F_var, F_var_ref, rtol=1e-12, atol=0.0)


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


@pytest.mark.parametrize("N_s", [1, 2])
def test_split_update(N_s):
    """Data added in two updates give the posteriors that the same data
    added at once give, for one hyperparameter sample and for several."""
    N = 20
    D = 2
    rng = np.random.default_rng(17)
    X = rng.standard_normal((N, D))
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

    hyp = rng.standard_normal((N_s, cov_N + noise_N + mean_N))
    hyp[:, D] *= 0.2
    hyp[:, D + 1 : D + 1 + noise_N] *= 0.3

    gp.update(hyp=hyp, compute_posterior=False)
    y = gp.random_function(X, rng=rng)

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
    assert np.size(gp.posteriors) == np.size(gp1.posteriors) == N_s

    for post, post1, row in zip(gp.posteriors, gp1.posteriors, hyp):
        assert np.all(post.hyp == row)
        assert np.all(post1.hyp == row)
        # These only approximately the same I think.
        assert np.all(np.isclose(post.alpha, post1.alpha))
        assert np.all(np.isclose(post.sW, post1.sW))
        assert np.all(np.isclose(post.L, post1.L))
        assert np.isclose(post.sn2_mult, post1.sn2_mult)
        assert post.L_chol and post1.L_chol


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
@pytest.mark.parametrize(
    "mean_prior",
    [
        None,
        ("smoothbox", (0.0, 1.0, 0.5)),
        ("smoothbox_student_t", (0.0, 1.0, 0.5, 3.0)),
    ],
)
def test_fitting_with_fixed_bounds(mean_prior):
    """A hyperparameter whose two bounds are equal keeps its value. Its
    entry of the gradient of the log prior is that of its own prior, which
    is zero without a prior and inside a smooth box, so the optimizer takes
    the other hyperparameters to a stationary point."""
    N = 20
    D = 1
    X = np.reshape(np.linspace(-10, 10, N), (-1, 1))
    y = 1 + np.sin(X)

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
        "mean_const": mean_prior,
    }

    def make_gp():
        gp = gpr.GP(
            D=D,
            covariance=gpr.covariance_functions.Matern(3),
            mean=gpr.mean_functions.ConstantMean(),
            noise=gpr.noise_functions.GaussianNoise(constant_add=True),
        )
        gp.set_priors(gp_priors)
        gp.set_bounds(gp_bounds)
        return gp

    gp = make_gp()
    assert gp.get_bounds() == gp_bounds

    hyp, _, _ = gp.fit(X=X, y=y)

    assert np.all(hyp[:, 3] == 0.5)

    # The optimum alone, where the gradient of the free hyperparameters
    # vanishes to the optimizer's tolerance.
    gp = make_gp()
    hyp, _, _ = gp.fit(
        X=X, y=y, options={"n_samples": 0}, rng=np.random.default_rng(0)
    )
    assert hyp[0, 3] == 0.5
    __, gradient = gp.log_posterior(hyp[0], compute_grad=True)
    __, gradient_likelihood = gp.log_likelihood(hyp[0], compute_grad=True)
    assert gradient[3] == gradient_likelihood[3]
    assert np.all(np.abs(gradient[:3]) < 1e-2)

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


def _fit_with_thin(gp, thin, n_samples=2):
    X = np.reshape(np.linspace(-2, 2, 10), (-1, 1))
    hyp, _, _ = gp.fit(
        X,
        np.sin(X),
        options={"thin": thin, "n_samples": n_samples, "init_N": 16},
        rng=0,
    )
    return hyp


@pytest.mark.parametrize("thin", [2.0, np.int64(2), np.float64(2.0)])
def test_fit_takes_a_whole_thin_of_any_type(thin):
    """The thinning factor is a count, which ``fit`` takes as a whole number
    of an integer or a float type, as ``SliceSampler.sample`` takes it. A
    whole float raised ``TypeError`` from the fit's own use of it."""
    hyp = _fit_with_thin(_gp_1d(), thin)
    hyp_int = _fit_with_thin(_gp_1d(), 2)
    assert hyp.shape[0] == 2
    assert np.array_equal(hyp, hyp_int)


@pytest.mark.parametrize("n_samples", [0, 2])
@pytest.mark.parametrize(
    "thin", [2.5, 0, 0.0, -1, -2.0, True, np.inf, np.nan, "2"]
)
def test_fit_refuses_a_thin_that_is_not_a_positive_whole_number(
    thin, n_samples
):
    """A thinning factor that is not a whole number greater than zero is
    refused before the fit changes anything, whether or not the fit draws
    samples. A fraction raised ``TypeError``, zero and negative numbers
    raised another error after the optimization, a bool ran as the
    integer it stands for, and without samples any value passed."""
    gp = _gp_1d()
    with pytest.raises(ValueError) as execinfo:
        _fit_with_thin(gp, thin, n_samples)
    assert "The option thin" in execinfo.value.args[0]
    assert gp.X is None


def _fit_with_counts(gp, **counts):
    X = np.reshape(np.linspace(-2, 2, 10), (-1, 1))
    options = {"n_samples": 2, "opts_N": 2, "init_N": 16}
    options.update(counts)
    hyp, _, _ = gp.fit(X, np.sin(X), options=options, rng=0)
    return hyp


@pytest.mark.parametrize(
    "name, whole",
    [
        ("n_samples", 2.0),
        ("n_samples", np.float64(3.0)),
        ("n_samples", 0.0),
        ("opts_N", 2.0),
        ("opts_N", np.float64(1.0)),
        ("opts_N", 0.0),
        ("init_N", 16.0),
        ("init_N", np.float64(8.0)),
        ("init_N", 0.0),
        ("init_N", np.int64(16)),
    ],
)
def test_fit_takes_whole_counts_of_any_type(name, whole):
    """The options ``n_samples``, ``opts_N`` and ``init_N`` are counts,
    which ``fit`` takes as whole numbers of an integer or a float type, as
    it takes ``thin``; the fit is that of the integer. A whole float raised
    ``TypeError`` from the fit's own use of it (``0.0`` passed for
    ``n_samples`` and ``init_N``, which the fit compares with zero)."""
    hyp = _fit_with_counts(_gp_1d(), **{name: whole})
    hyp_int = _fit_with_counts(_gp_1d(), **{name: int(whole)})
    assert np.array_equal(hyp, hyp_int)


@pytest.mark.parametrize("name", ["n_samples", "opts_N", "init_N"])
@pytest.mark.parametrize("value", [2.5, -1, -1.0, True, np.inf, np.nan, "2"])
def test_fit_refuses_counts_that_are_not_whole_numbers(name, value):
    """A count that is not a whole number of at least zero is refused
    before the fit changes anything. A fraction raised ``TypeError`` or
    another error after the optimization; a negative ``opts_N`` or
    ``init_N``, or a NaN ``init_N``, ran as zero; a bool ran as the
    integer it stands for."""
    gp = _gp_1d()
    with pytest.raises(ValueError) as execinfo:
        _fit_with_counts(gp, **{name: value})
    assert f"The option {name}" in execinfo.value.args[0]
    assert gp.X is None


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
    hyp_N = (
        gp.covariance.hyperparameter_count(D)
        + gp.noise.hyperparameter_count()
        + gp.mean.hyperparameter_count(D)
    )

    # Every combination of the three sizes that can be switched off, run
    # in a row on one GP. `n_samples` decides how many hyperparameter
    # vectors come back, `opts_N` whether an optimization result does, and
    # `n_samples` again whether a sampling result does.
    cases = [
        ({"opts_N": 0}, 10, False, True),
        ({"n_samples": 0}, 1, True, False),
        ({"init_N": 0}, 10, True, True),
        ({"opts_N": 0, "n_samples": 0}, 1, False, False),
        ({"n_samples": 0, "init_N": 0}, 1, True, False),
        ({"opts_N": 0, "init_N": 0}, 10, False, True),
        ({"opts_N": 0, "n_samples": 0, "init_N": 0}, 1, False, False),
        ({"init_N": 1}, 10, True, True),
    ]
    for options, rows, optimized, sampled in cases:
        hyp, optimize_result, sampling_result = gp.fit(
            X=X, y=y, options=options
        )
        assert hyp.shape == (rows, hyp_N), options
        assert np.all(np.isfinite(hyp)), options
        assert (optimize_result is not None) is optimized, options
        assert (sampling_result is not None) is sampled, options
        if sampled:
            assert sampling_result["samples"].shape[1] == hyp_N, options
        assert np.size(gp.posteriors) == rows, options
        assert np.array_equal(
            gp.get_hyperparameters(as_array=True), hyp
        ), options


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
    rng = np.random.default_rng(6)
    hyp = rng.standard_normal((N_s, cov_N + noise_N + mean_N))
    hyp[:, D] *= 0.3
    hyp[:, D + 1 : D + 1 + noise_N] *= 0.3

    gp.update(hyp=hyp, compute_posterior=False)
    y = gp.random_function(X, add_noise=True, rng=rng)
    gp.update(X_new=X, y_new=y, hyp=hyp, compute_posterior=True)

    gp1 = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(5),
        mean=gpr.mean_functions.ZeroMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    gp_train = {"n_samples": 0}
    hyp2, _, _ = gp1.fit(X=X, y=y, options=gp_train, rng=rng)

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
    """A GP in three dimensions with a prior of each family. Each smooth
    box is set on a block of three hyperparameters (the length scales, and
    the log scales of the mean), with a box per coordinate that puts the
    starting values, all within (-1, 1), above the first box, inside the
    second and below the third."""
    rng = np.random.default_rng(seed)
    N, D = 25, 3
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
    a = np.array([-3.0, -1.0, 1.0])
    b = np.array([-1.0, 1.0, 3.0])
    priors = {
        names[0]: ("smoothbox", (a, b, np.array([0.5, 0.7, 0.4]))),
        names[1]: ("gaussian", (np.zeros(1), np.ones(1))),
        names[2]: (
            "student_t",
            (np.zeros(1), np.ones(1), np.full(1, 3.0)),
        ),
        names[3]: None,
        names[4]: ("gaussian", (np.zeros(D), np.full(D, 2.0))),
        names[5]: (
            "smoothbox_student_t",
            (a, b, np.array([0.5, 0.3, 0.6]), np.array([4.0, 3.0, 5.0])),
        ),
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


def _dense_grid_gp(hyp):
    """The GP of the dense-grid draws: 40 points of ``sin(2 x)`` on
    [-2, 2], with the hyperparameters ``hyp``, one row per sample, in the
    order [log ell, log sf, log sn, m0, mode location, log scale]."""
    rng_data = np.random.default_rng(77)
    X = rng_data.uniform(-2, 2, size=(40, 1))
    y = np.sin(2 * X)

    gp = gpr.GP(
        D=1,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    gp.update(X_new=X, y_new=y, hyp=hyp)
    return gp


def _predictive_covariance(gp, x_star, s=0):
    """The predictive covariance of hyperparameter sample ``s`` at
    ``x_star``, solved against a Cholesky factor of the training
    covariance with the noise the posterior was computed with."""
    posterior = gp.posteriors[s]
    cov_N = gp.covariance.hyperparameter_count(gp.D)
    noise_N = gp.noise.hyperparameter_count()
    hyp_cov = posterior.hyp[:cov_N]
    hyp_noise = posterior.hyp[cov_N : cov_N + noise_N]
    K = gp.covariance.compute(hyp_cov, gp.X)
    sn2 = gp.noise.compute(hyp_noise, gp.X, gp.y, gp.s2)
    K_noisy = K + posterior.sn2_mult * sn2 * np.eye(gp.X.shape[0])
    Ks = gp.covariance.compute(hyp_cov, gp.X, X_star=x_star)
    factor = scipy.linalg.cho_factor(K_noisy)
    C = gp.covariance.compute(hyp_cov, x_star) - Ks.T @ scipy.linalg.cho_solve(
        factor, Ks
    )
    return (C + C.T) / 2


@pytest.mark.parametrize(
    "noise_sd, cholesky_factor", [(2e-3, True), (1e-4, False), (1e-6, False)]
)
@pytest.mark.parametrize("grid", [(-2.5, 2.5), (-1.0, 1.0)])
def test_random_function_on_a_dense_grid(grid, noise_sd, cholesky_factor):
    """A predictive covariance on a dense one-dimensional grid is
    numerically singular, so the draw goes through the eigenvalue
    fallback. Eigenvalues of rounding size, which such a matrix has of
    both signs, count as zeros, and the draws are draws: they differ
    between generators and carry the predictive covariance. The
    covariance is the prior covariance minus what the data explain, so
    its rounding is that of the prior variance: on the grid inside the
    data, where the largest eigenvalue is four orders of magnitude or more
    below the prior variance, the negative eigenvalues are still rounding.
    This holds in both representations of the posterior: the Cholesky
    factor of the training covariance, and, below a noise variance of
    1e-6, the negative inverse, from which the predictive covariance would
    carry a rounding error that grows as the noise shrinks. The grid on
    [-2.5, 2.5] reaches past the data on both sides."""
    hyp = np.array(
        [[np.log(0.7), np.log(1.2), np.log(noise_sd), 0.0, 0.0, np.log(1.5)]]
    )
    gp = _dense_grid_gp(hyp)
    if cholesky_factor:
        assert gp.posteriors[0].L_chol
    else:
        assert not gp.posteriors[0].L_chol

    x_star = np.reshape(np.linspace(*grid, 100), (-1, 1))
    C = _predictive_covariance(gp, x_star)
    with pytest.raises(scipy.linalg.LinAlgError):
        scipy.linalg.cholesky(C, check_finite=False)

    f_1 = gp.random_function(x_star, rng=np.random.default_rng(1))
    f_2 = gp.random_function(x_star, rng=np.random.default_rng(2))
    assert not np.array_equal(f_1, f_2)

    rng = np.random.default_rng(11)
    draws = np.concatenate(
        [gp.random_function(x_star, rng=rng) for __ in range(1000)], axis=1
    )
    empirical = np.cov(draws, ddof=1)
    assert np.linalg.norm(empirical - C) < 0.2 * np.linalg.norm(C)
    mu, __ = gp.predict(x_star)
    sd_max = np.sqrt(np.max(np.diag(C)))
    assert np.allclose(
        np.mean(draws, 1), np.ravel(mu), rtol=0, atol=0.15 * sd_max
    )


@pytest.mark.parametrize(
    "noise_sds, cholesky_factor",
    [((1e-2, 1e-1), True), ((1e-4, 1e-6), False)],
)
def test_random_function_draws_from_one_hyperparameter_sample(
    noise_sds, cholesky_factor
):
    """With several hyperparameter samples, each draw comes from the
    posterior of one of them, in either representation of the posterior,
    and ``add_noise`` adds the observation noise of that sample. The same
    generator state draws the same sample and the same function with and
    without the noise, so their difference is the noise alone; the noise
    variances of the two samples are a hundred times apart or more, so the
    noise of a draw tells which sample it came from."""
    hyp = np.array(
        [
            [np.log(ell), np.log(sf), np.log(sd), 0.0, 0.0, np.log(1.5)]
            for ell, sf, sd in zip((0.7, 0.9), (1.2, 1.0), noise_sds)
        ]
    )
    gp = _dense_grid_gp(hyp)
    for posterior in gp.posteriors:
        if cholesky_factor:
            assert posterior.L_chol
        else:
            assert not posterior.L_chol

    x_star = np.reshape(np.linspace(-2.5, 2.5, 30), (-1, 1))
    functions, noise = [], []
    for seed in range(1000):
        f = gp.random_function(x_star, rng=np.random.default_rng(seed))
        y = gp.random_function(
            x_star, add_noise=True, rng=np.random.default_rng(seed)
        )
        functions.append(f)
        noise.append(y - f)
    functions = np.concatenate(functions, axis=1)
    noise = np.concatenate(noise, axis=1)

    noise_variances = np.array(
        [
            np.exp(2 * posterior.hyp[2]) * posterior.sn2_mult
            for posterior in gp.posteriors
        ]
    )
    log_ratios = np.log(np.mean(noise**2, axis=0)[:, None] / noise_variances)
    drawn = np.argmin(np.abs(log_ratios), axis=1)
    assert np.all(np.abs(log_ratios[np.arange(1000), drawn]) < np.log(5))

    mu, __ = gp.predict(x_star, separate_samples=True)
    for s in range(2):
        assert 300 < np.sum(drawn == s) < 700
        assert np.mean(noise[:, drawn == s] ** 2) == pytest.approx(
            noise_variances[s], rel=0.1
        )
        C = _predictive_covariance(gp, x_star, s)
        empirical = np.cov(functions[:, drawn == s], ddof=1)
        assert np.linalg.norm(empirical - C) < 0.2 * np.linalg.norm(C)
        sd_max = np.sqrt(np.max(np.diag(C)))
        assert np.allclose(
            np.mean(functions[:, drawn == s], 1),
            mu[:, s],
            rtol=0,
            atol=0.15 * sd_max,
        )


def test_robust_cholesky_refuses_an_indefinite_matrix():
    """A negative eigenvalue larger than the rounding tolerance means the
    matrix is no covariance matrix, and no factor of it exists. The
    tolerance is measured against the scale of the terms that formed the
    matrix where the caller gives it, and against the largest eigenvalue
    otherwise."""
    robust_cholesky = gpr.GP._GP__robust_cholesky
    sigma = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    for scale in (None, 10.0):
        with pytest.raises(scipy.linalg.LinAlgError) as execinfo:
            robust_cholesky(sigma, scale=scale)
        assert "not positive semidefinite" in execinfo.value.args[0]

    # A matrix formed by cancellation from terms of order one, whose
    # largest eigenvalue is 1e-5: a negative eigenvalue of -1e-15 is
    # rounding of those terms, one of -1e-11 is not.
    Q, __ = np.linalg.qr(np.random.default_rng(5).standard_normal((3, 3)))
    for smallest, refused in ((-1e-15, False), (-1e-11, True)):
        sigma = Q @ np.diag([1e-5, 1e-6, smallest]) @ Q.T
        sigma = (sigma + sigma.T) / 2
        with pytest.raises(scipy.linalg.LinAlgError):
            robust_cholesky(sigma)
        if refused:
            with pytest.raises(scipy.linalg.LinAlgError):
                robust_cholesky(sigma, scale=1.44)
        else:
            T = robust_cholesky(sigma, scale=1.44)
            assert np.allclose(T.T @ T, sigma, rtol=0, atol=1e-14)


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


def test_rank_one_update_low_noise_duplicate_recomputes(monkeypatch):
    """Where rounding drives the latent variance of the new point to zero
    or below, ``predict`` clamps it, and the low-noise rank-one update
    would divide by the noise alone, which is no predictive variance: it
    warns and recomputes in full, as the Cholesky branch does. Rounding
    takes an observation at an existing training input to either side of
    zero depending on the platform, so ``predict`` is made to return the
    clamped value, the noise, whatever the rounding."""
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

    # The noise variance of the new point, computed as `update` computes
    # it: what the clamped predictive variance equals.
    posterior = gp.posteriors[0]
    cov_N = gp.covariance.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()
    sn2 = np.ravel(
        gp.noise.compute(
            posterior.hyp[cov_N : cov_N + noise_N], x_new, y_new, 0
        )
    )[0]
    clamped = sn2 * posterior.sn2_mult
    predict = gp.predict

    def predict_at_the_clamp(*args, **kwargs):
        mu, s2 = predict(*args, **kwargs)
        return mu, np.full_like(s2, clamped)

    with monkeypatch.context() as patch:
        patch.setattr(gp, "predict", predict_at_the_clamp)
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


def _low_noise_gp(X, y, noise_sd=1e-6):
    """A one-dimensional GP with a squared exponential kernel of unit
    length and output scales whose noise variance, below 1e-6, puts its
    posterior in the low-noise representation."""
    gp = gpr.GP(
        D=1,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    gp.update(
        X_new=X, y_new=y, hyp=np.array([[0.0, 0.0, np.log(noise_sd), 0.0]])
    )
    assert not gp.posteriors[0].L_chol
    return gp


def test_low_noise_variance_of_two_points():
    """Two training points at distance ``d`` have, with the kernel
    ``k = exp(-d**2 / 2)`` and the noise variance ``n``, the predictive
    variance ``n * ((1 - k) * (1 + k) + n) / ((1 - k + n) * (1 + k + n))``
    at either point, a form without cancellation. Formed from the explicit
    inverse of the training covariance, which the low-noise representation
    holds, the variance carried the rounding of the inverse, which grows as
    the noise shrinks: here 1e-12 against rounding of order 1e-10. Formed
    from a Cholesky factor, it is within the rounding of a difference of
    terms of the size of the prior variance, one."""
    d = 1e-3
    X = np.array([[0.0], [d]])
    gp = _low_noise_gp(X, np.array([[0.3], [-0.2]]))
    n = np.exp(2 * np.log(1e-6)) * gp.posteriors[0].sn2_mult
    k = np.exp(-(d**2) / 2)
    one_minus_k = -np.expm1(-(d**2) / 2)
    expected = (
        n * (one_minus_k * (1 + k) + n) / ((one_minus_k + n) * (1 + k + n))
    )
    tolerance = 8 * np.finfo(float).eps

    __, s2 = gp.predict(X)
    __, cov = gp.predict_full(X)

    assert np.all(s2 >= 0.0)
    assert np.all(np.abs(s2[:, 0] - expected) <= tolerance)
    assert np.all(np.abs(np.diag(cov[:, :, 0]) - expected) <= tolerance)


def test_low_noise_predictions_at_the_training_inputs():
    """At a noise standard deviation of 1e-6, the predictive variances at
    the training inputs, of order 1e-12, are non-negative and agree with
    ``sn2 * K @ inv(K + sn2 I)`` from the eigendecomposition of ``K``,
    where each term of the diagonal is non-negative, to ``N * eps``; formed
    from the explicit inverse they were off by up to 2e-4. The full
    predictive covariance has no eigenvalue below the rounding of the prior
    covariance it is subtracted from, where it had eigenvalues down to
    -8e-4."""
    N = 30
    eps = np.finfo(float).eps
    rng = np.random.default_rng(0)
    X = np.sort(rng.uniform(-2, 2, N))[:, None]
    gp = _low_noise_gp(X, np.sin(2 * X))
    posterior = gp.posteriors[0]
    sn2 = np.exp(2 * posterior.hyp[2]) * posterior.sn2_mult
    K = gp.covariance.compute(posterior.hyp[:2], X)
    lam, U = np.linalg.eigh(K)
    lam = np.maximum(lam, 0.0)
    reference = (U**2) @ (lam * sn2 / (lam + sn2))

    __, s2 = gp.predict(X)

    assert np.all(s2 >= 0.0)
    assert np.all(np.abs(s2[:, 0] - reference) <= N * eps)

    x_star = np.vstack((X, np.linspace(-2.5, 2.5, 60)[:, None]))
    __, cov = gp.predict_full(x_star)
    K_star = gp.covariance.compute(posterior.hyp[:2], x_star)
    M = x_star.shape[0]
    tolerance = 10 * M * eps * np.linalg.norm(K_star, 2)
    assert np.min(np.linalg.eigvalsh(cov[:, :, 0])) > -tolerance


def test_low_noise_rank_one_updates_match_a_full_recomputation():
    """Ten single-point updates of a posterior in the low-noise
    representation give the predictive mean and the variances at the
    training inputs of a full recomputation. The extension takes
    ``inv(K + sn2 I) k*`` and the Schur complement ``v_star`` from the
    Cholesky factor; with an inaccurate ``v_star``, formed from the explicit
    inverse, the mean was off by 1e-3 and the variances by 2e-4."""
    N = 30
    rng = np.random.default_rng(1)
    X = np.sort(rng.uniform(-2, 2, N))[:, None]
    y = np.sin(2 * X)
    gp = _low_noise_gp(X[:20], y[:20])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for i in range(20, N):
            gp.update(X_new=X[i : i + 1], y_new=y[i : i + 1])
    gp_ref = _low_noise_gp(X, y)

    x_star = np.linspace(-2.5, 2.5, 201)[:, None]
    f_mu, __ = gp.predict(x_star)
    f_mu_ref, __ = gp_ref.predict(x_star)
    __, s2 = gp.predict(X)
    __, s2_ref = gp_ref.predict(X)

    assert np.max(np.abs(f_mu - f_mu_ref)) < 1e-6
    assert np.all(s2 >= 0.0)
    assert np.max(np.abs(s2 - s2_ref)) <= N * np.finfo(float).eps


def test_low_noise_posterior_without_its_factor():
    """A posterior pickled without the Cholesky factor of the low-noise
    representation has it computed again: its predictions and draws are
    those of the posterior that holds it, and a single-point update
    recomputes it in full."""
    rng = np.random.default_rng(2)
    X = np.sort(rng.uniform(-2, 2, 15))[:, None]
    y = np.sin(2 * X)
    gp = _low_noise_gp(X, y)
    stripped = copy.deepcopy(gp)
    del stripped.posteriors[0].L_factor

    x_star = np.linspace(-2.5, 2.5, 9)[:, None]
    for a, b in zip(gp.predict(x_star), stripped.predict(x_star)):
        assert np.array_equal(a, b)
    for a, b in zip(gp.predict_full(x_star), stripped.predict_full(x_star)):
        assert np.array_equal(a, b)
    assert np.array_equal(
        gp.random_function(x_star, rng=np.random.default_rng(3)),
        stripped.random_function(x_star, rng=np.random.default_rng(3)),
    )
    for a, b in zip(
        gp.quad(0.2, 0.5, compute_var=True),
        stripped.quad(0.2, 0.5, compute_var=True),
    ):
        assert np.array_equal(a, b)

    x_new = np.array([[0.123]])
    stripped.update(X_new=x_new, y_new=np.sin(2 * x_new))
    gp_ref = _low_noise_gp(
        np.vstack((X, x_new)), np.vstack((y, np.sin(2 * x_new)))
    )
    for key in ("alpha", "L", "L_factor"):
        assert np.array_equal(
            getattr(stripped.posteriors[0], key),
            getattr(gp_ref.posteriors[0], key),
        )


@pytest.mark.parametrize("d", [1e-3, 1e-2])
def test_low_noise_quadrature_variance_of_two_points(d):
    """Two training points at distance ``d``, with the kernel
    ``c = exp(-d**2 / 2)`` between them and the noise variance ``n``, and a
    Gaussian measure of standard deviation ``sigma`` centred between them:
    the kernel means ``z`` of the two points are equal, ``(1, 1)`` is an
    eigenvector of ``K + n I`` with eigenvalue ``1 + c + n``, and the
    variance of the integral is ``nf - 2 z**2 / (1 + c + n)`` with
    ``nf = 1 / sqrt(1 + 2 sigma**2)``, a form without the ill-conditioning
    of ``K + n I``. Formed from the explicit inverse that the low-noise
    representation holds, the variance carried the rounding of the
    inverse, which grows as the noise shrinks; formed from a Cholesky
    factor, it is within the rounding of a difference of terms of order
    one."""
    X = np.array([[0.0], [d]])
    gp = _low_noise_gp(X, np.array([[0.3], [-0.2]]))
    n = np.exp(2 * np.log(1e-6)) * gp.posteriors[0].sn2_mult
    c = np.exp(-(d**2) / 2)
    sigma = 0.5
    z = np.exp(-((d / 2) ** 2) / (2 * (1 + sigma**2))) / np.sqrt(
        1 + sigma**2
    )
    expected = 1 / np.sqrt(1 + 2 * sigma**2) - 2 * z**2 / (1 + c + n)

    __, F_var = gp.quad(d / 2, sigma, compute_var=True)

    assert np.abs(F_var[0, 0] - expected) <= 8 * np.finfo(float).eps


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
        # or a NaN number of degrees of freedom. Outside `fit`, either
        # value makes a Student's t a Gaussian, as in `gplite_hypprior.m`,
        # and a smooth-box Student's t a smooth box, a reading of gpyreg's
        # own (gplite has no smooth-box priors).
        for df in (np.inf, np.nan):
            gp.set_priors(priors)
            gp.hyper_priors["df"][0] = df  # the Gaussian
            gp.hyper_priors["df"][2] = df  # the smooth box
            log_prior = gp.log_posterior(hyp) - gp.log_likelihood(hyp)
            assert np.isclose(log_prior, expected, rtol=1e-12)


def _reference_mass(kind, params, lower, upper):
    """The mass of a reference density between two bounds, by quadrature
    over the pieces between its kinks, the ends of a smooth box."""
    density = lambda t: np.exp(_reference_log_prior(kind, params, t))
    cuts = [lower, upper]
    if kind.startswith("smoothbox"):
        cuts += [end for end in params[:2] if lower < end < upper]
    cuts = sorted(cuts)
    return sum(
        quad(density, low, high, epsabs=0.0, epsrel=1e-12, limit=200)[0]
        for low, high in zip(cuts[:-1], cuts[1:])
    )


@pytest.mark.parametrize(
    "bounds",
    [
        # Around the centre of every prior.
        [(-1.0, 2.0), (-1.5, 1.0), (-2.0, 0.5), (-3.0, 2.0)],
        # Above the centre: in the upper tail, or from inside a box.
        [(2.5, 5.0), (1.0, 3.0), (0.2, 3.5), (-0.5, 4.0)],
        # Below the centre.
        [(-5.0, -2.0), (-4.0, -1.5), (-3.5, -1.5), (-5.0, -3.0)],
        # One finite bound.
        [(0.0, np.inf), (-np.inf, 0.5), (-np.inf, 2.0), (-1.0, np.inf)],
    ],
    ids=["centre", "upper", "lower", "one_bound"],
)
def test_log_prior_matches_the_truncated_densities(bounds):
    """With bounds, each hyperprior is renormalized over them: the log prior
    is that of the documented density truncated to the bounds, whose mass
    between them is computed here by quadrature of the density."""
    # In the order of the hyperparameter array: covariance, noise, mean.
    priors = {
        "covariance_log_lengthscale": ("gaussian", (0.3, 1.2)),
        "covariance_log_outputscale": ("student_t", (-0.2, 0.8, 5.0)),
        "noise_log_scale": ("smoothbox", (-1.0, 1.0, 0.7)),
        "mean_const": ("smoothbox_student_t", (-2.0, 0.5, 0.9, 4.0)),
    }
    gp = gpr.GP(
        D=1,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    gp.set_priors(priors)
    gp.set_bounds(dict(zip(priors, bounds)))

    # A point between the bounds of each hyperparameter.
    hyp = np.array(
        [
            low + 0.3 * (high - low)
            if np.isfinite(low) and np.isfinite(high)
            else (low + 0.7 if np.isfinite(low) else high - 0.7)
            for low, high in bounds
        ]
    )
    X = np.reshape(np.linspace(-2, 2, 8), (-1, 1))
    gp.update(X_new=X, y_new=np.sin(X), hyp=hyp[None, :])

    expected = sum(
        _reference_log_prior(kind, params, x)
        - np.log(_reference_mass(kind, params, low, high))
        for (kind, params), x, (low, high) in zip(priors.values(), hyp, bounds)
    )
    log_prior = gp.log_posterior(hyp) - gp.log_likelihood(hyp)
    assert np.isclose(log_prior, expected, rtol=1e-10)


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


@pytest.mark.parametrize(
    "sigma, problem",
    [
        (np.inf, "infinite"),
        (-np.inf, "infinite"),
        (np.nan, "NaN"),
        (0.0, "zero or negative"),
        (-2.0, "zero or negative"),
    ],
)
def test_set_priors_refuses_a_scale_that_is_not_positive(sigma, problem):
    """A prior needs a finite, positive scale; no prior is ``None``. The
    message names the hyperparameter and what is wrong with its scale."""
    priors = _no_priors()
    priors["mean_const"] = ("gaussian", (0.0, sigma))
    with pytest.raises(ValueError) as execinfo:
        _gp_1d().set_priors(priors)
    message = execinfo.value.args[0]
    assert "mean_const" in message
    assert problem in message
    assert "None" in message


@pytest.mark.parametrize("had_priors", [False, True])
def test_a_refused_set_priors_leaves_the_gp_as_it_was(had_priors):
    """``set_priors`` changes nothing when it refuses its argument: not the
    priors, and not the flag that says whether the GP has any, which
    decides whether ``fit`` adds the log prior to its objective and what
    ``str`` reports."""
    gp = _gp_1d()
    if had_priors:
        priors = _no_priors()
        priors["mean_const"] = ("gaussian", (0.0, 1.0))
        gp.set_priors(priors)
    hyper_priors = copy.deepcopy(gp.hyper_priors)
    no_prior = gp.no_prior
    text = str(gp)

    refused = _no_priors()
    refused["mean_const"] = ("gaussian", (0.0, -1.0))
    missing = {"mean_const": ("gaussian", (0.0, 1.0))}
    unknown = dict(_no_priors(), not_a_hyperparameter=None)
    for priors in (refused, missing, unknown):
        with pytest.raises(ValueError):
            gp.set_priors(priors)
        assert gp.no_prior is no_prior
        assert str(gp) == text
        for key, value in hyper_priors.items():
            assert np.array_equal(gp.hyper_priors[key], value, equal_nan=True)


def _gp_2d():
    """A two-dimensional GP with five hyperparameters: the two length
    scales and the output scale of the kernel, one of the noise and one of
    the mean."""
    return gpr.GP(
        D=2,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )


@pytest.mark.parametrize(
    "family, params",
    [
        ("gaussian", (0.3, 1.2)),
        ("student_t", (0.3, 1.2, 5.0)),
        ("smoothbox", (-1.0, 1.0, 0.7)),
        ("smoothbox_student_t", (-1.0, 1.0, 0.7, 4.0)),
    ],
)
def test_set_priors_takes_a_coordinate_without_a_prior(family, params):
    """A coordinate of a block whose location and ``sigma`` are both NaN
    has no prior: the location is ``mu`` for the Gaussian and Student's t
    families and the box ``[a, b]`` for the smooth-box ones. The block's
    log prior is that of its other coordinates alone. PyVBMC writes such
    a block for the rectified output-dependent noise."""
    # The first length scale has no prior; its degrees of freedom, where
    # the family has them, are NaN as well.
    block = tuple(np.array([np.nan, value]) for value in params)
    priors = _no_priors()
    priors["covariance_log_lengthscale"] = (family, block)

    X = np.reshape(np.linspace(-2, 2, 12), (-1, 2))
    y = np.sum(np.sin(X), 1)
    gp = _gp_2d()
    hyp = np.array([0.4, -1.4, 0.0, np.log(0.1), 0.0])
    gp.update(X_new=X, y_new=y, hyp=hyp[None, :])
    gp.set_priors(priors)

    expected = _reference_log_prior(family, params, hyp[1])
    for moved in (0.4, 3.0):
        hyp[0] = moved
        log_prior = gp.log_posterior(hyp) - gp.log_likelihood(hyp)
        assert np.isclose(log_prior, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "family, params",
    [
        # A NaN number of degrees of freedom, on the whole block.
        ("student_t", (0.3, 1.2, np.nan)),
        ("smoothbox_student_t", (-1.0, 1.0, 0.7, np.nan)),
        # A block whose first coordinate has no prior.
        ("gaussian", ([np.nan, 0.3], [np.nan, 1.2])),
        ("student_t", ([np.nan, 0.3], [np.nan, 1.2], [np.nan, 3.0])),
        ("smoothbox", ([np.nan, -1.0], [np.nan, 1.0], [np.nan, 0.7])),
        (
            "smoothbox_student_t",
            ([np.nan, -1.0], [np.nan, 1.0], [np.nan, 0.7], [np.nan, 4.0]),
        ),
        # A prior of each family on the whole block.
        ("gaussian", (0.3, 1.2)),
        ("student_t", (0.3, 1.2, 3.0)),
        ("smoothbox", (-1.0, 1.0, 0.7)),
        ("smoothbox_student_t", (-1.0, 1.0, 0.7, 4.0)),
        # Smooth boxes of zero width, on the whole block and on one
        # coordinate.
        ("smoothbox", (0.5, 0.5, 0.7)),
        ("smoothbox_student_t", ([-1.0, 0.5], [1.0, 0.5], 0.7, 4.0)),
    ],
)
def test_get_priors_returns_what_set_priors_reads_back(family, params):
    """``get_priors`` returns a prior whose degrees of freedom are NaN, and
    a block with a coordinate that has no prior, in a form ``set_priors``
    reads back unchanged, before and after a ``fit``. Outside ``fit`` a
    NaN ``df`` names the Gaussian family, as in ``gplite_hypprior.m``, and
    the smooth box for the smooth-box Student's t, gpyreg's own family;
    ``fit`` fills it with ``df_base`` for its own duration alone."""
    X = np.reshape(np.linspace(-2, 2, 12), (-1, 2))
    y = np.sum(np.sin(X), 1)
    hyp0 = np.array([[0.4, -0.4, 0.0, np.log(0.1), 0.0]])
    gp = _gp_2d()
    priors = _no_priors()
    priors["covariance_log_lengthscale"] = (family, params)
    gp.set_priors(priors)
    hyper_priors = copy.deepcopy(gp.hyper_priors)

    for fitted in (False, True):
        if fitted:
            gp.fit(
                X=X,
                y=y,
                hyp0=hyp0,
                options={"init_N": 0, "opts_N": 1, "n_samples": 0},
            )
        returned = gp.get_priors()["covariance_log_lengthscale"]
        assert returned is not None
        assert returned[0] == family
        for value, given in zip(returned[1], params):
            assert np.array_equal(
                value, np.broadcast_to(given, (2,)), equal_nan=True
            )

        other = _gp_2d()
        other.set_priors(gp.get_priors())
        for key, value in hyper_priors.items():
            assert np.array_equal(gp.hyper_priors[key], value, equal_nan=True)
            assert np.array_equal(
                other.hyper_priors[key], value, equal_nan=True
            )


@pytest.mark.parametrize(
    "family, params",
    [
        # Degrees of freedom that mix zero, NaN, infinity and a number,
        # which `set_priors` writes as they are given.
        ("student_t", ([0.3, 0.1], [1.2, 0.5], [0.0, np.nan])),
        ("student_t", ([0.3, 0.1], [1.2, 0.5], [0.0, 3.0])),
        ("student_t", ([0.3, 0.1], [1.2, 0.5], [np.inf, 0.0])),
        (
            "smoothbox_student_t",
            ([-1.0, 0.0], [1.0, 0.5], [0.7, 0.2], [0.0, np.nan]),
        ),
        (
            "smoothbox_student_t",
            ([-1.0, 0.0], [1.0, 0.5], [0.7, 0.2], [0.0, 4.0]),
        ),
        # Degrees of freedom that name a Gaussian family throughout.
        ("student_t", (0.3, 1.2, np.inf)),
        ("student_t", (0.3, 1.2, 0.0)),
        ("smoothbox_student_t", (-1.0, 1.0, 0.7, np.inf)),
        ("smoothbox_student_t", (-1.0, 1.0, 0.7, 0.0)),
        # A family set on a block without a prior in any coordinate.
        ("gaussian", (np.nan, np.nan)),
        ("student_t", (np.nan, np.nan, 3.0)),
        ("smoothbox", (np.nan, np.nan, np.nan)),
    ],
)
def test_set_priors_of_get_priors_changes_nothing(family, params):
    """``set_priors(get_priors())`` writes back every array of the priors
    as the GP holds them, and the flag that says whether it has any, for
    any prior that ``set_priors`` takes. A Student's t block whose degrees
    of freedom mix zero with NaN or a number came back as ``None``, which
    dropped it; one whose degrees of freedom are infinite throughout came
    back as a Gaussian family with zero; and a family set on a block
    without a prior came back as ``None``, with NaN degrees of freedom."""
    gp = _gp_2d()
    priors = _no_priors()
    priors["covariance_log_lengthscale"] = (family, params)
    priors["mean_const"] = ("student_t", (0.0, 2.0, 5.0))
    gp.set_priors(priors)

    other = _gp_2d()
    other.set_priors(gp.get_priors())

    for key, value in gp.hyper_priors.items():
        assert np.array_equal(other.hyper_priors[key], value, equal_nan=True)
    assert other.no_prior is gp.no_prior


@pytest.mark.parametrize(
    "written, problem",
    [
        # A smooth-box coordinate with finite ends and a NaN sigma.
        (
            {"a": [-1.0, 0.0], "b": [1.0, 0.5], "sigma": [0.7, np.nan]},
            "a NaN sigma",
        ),
        # A Gaussian coordinate with a mu and a NaN sigma.
        ({"mu": [0.3, 0.1], "sigma": [1.2, np.nan]}, "a NaN sigma"),
        ({"mu": [0.3, 0.1], "sigma": [1.2, -0.5]}, "zero or negative"),
        (
            {"a": [1.0, 0.0], "b": [-1.0, 0.5], "sigma": [0.7, 0.2]},
            "above its upper end",
        ),
        # A mu beside the ends of a smooth box.
        (
            {
                "mu": [0.3, np.nan],
                "a": [np.nan, 0.0],
                "b": [np.nan, 0.5],
                "sigma": [1.2, 0.2],
            },
            "both a `mu` and the ends of a smooth box",
        ),
    ],
)
def test_get_priors_refuses_priors_that_set_priors_refuses(written, problem):
    """``get_priors`` returns the priors in the form ``set_priors`` takes.
    Priors written into ``hyper_priors`` directly that ``set_priors``
    refuses have no such form, and ``get_priors`` says what is wrong with
    them, naming the hyperparameter, where it returned a block that
    ``set_priors`` then refused, or ``None``, which dropped the prior."""
    gp = _gp_2d()
    gp.hyper_priors["df"][0:2] = 0.0
    for key, value in written.items():
        gp.hyper_priors[key][0:2] = value

    with pytest.raises(ValueError) as execinfo:
        gp.get_priors()

    message = execinfo.value.args[0]
    assert "covariance_log_lengthscale" in message
    assert problem in message


@pytest.mark.parametrize(
    "family, params",
    [
        ("gaussian", ([0.0, 0.3], [np.nan, 1.2])),
        ("student_t", ([0.0, 0.3], [np.nan, 1.2], 5.0)),
        ("smoothbox", ([-1.0, -1.0], [1.0, 1.0], [np.nan, 0.7])),
        (
            "smoothbox_student_t",
            ([-1.0, -1.0], [1.0, 1.0], [np.nan, 0.7], 4.0),
        ),
    ],
)
def test_set_priors_refuses_a_nan_scale_beside_a_location(family, params):
    """A coordinate whose location is set needs a finite, positive
    ``sigma``: a NaN one beside it is refused, with a message that says
    the scale is NaN."""
    priors = _no_priors()
    priors["covariance_log_lengthscale"] = (family, params)
    with pytest.raises(ValueError) as execinfo:
        _gp_2d().set_priors(priors)
    message = execinfo.value.args[0]
    assert "covariance_log_lengthscale" in message
    assert "NaN" in message
    assert "infinite" not in message


@pytest.mark.parametrize(
    "family, params, problem",
    [
        ("gaussian", (np.inf, 1.0), "infinite mu"),
        ("gaussian", (-np.inf, 1.0), "infinite mu"),
        ("gaussian", (np.nan, 1.0), "NaN mu"),
        ("student_t", (np.inf, 1.0, 3.0), "infinite mu"),
        ("student_t", (np.nan, 1.0, 3.0), "NaN mu"),
        ("smoothbox", (0.0, np.inf, 1.0), "infinite end"),
        ("smoothbox", (-np.inf, 0.0, 1.0), "infinite end"),
        ("smoothbox", (np.nan, 0.0, 1.0), "NaN end"),
        ("smoothbox", (np.nan, np.nan, 1.0), "NaN end"),
        ("smoothbox_student_t", (0.0, np.inf, 1.0, 3.0), "infinite end"),
        ("smoothbox_student_t", (np.nan, 1.0, 1.0, 3.0), "NaN end"),
    ],
)
def test_set_priors_refuses_a_location_that_is_not_finite(
    family, params, problem
):
    """A coordinate that has a prior needs a finite location (``mu``, or
    both ends of a smooth box) beside its finite ``sigma``, as it needs a
    finite, positive ``sigma``: the log posterior of such a prior is NaN
    or infinite. The message names the hyperparameter and what is wrong
    with its location."""
    priors = _no_priors()
    priors["mean_const"] = (family, params)
    with pytest.raises(ValueError) as execinfo:
        _gp_1d().set_priors(priors)
    message = execinfo.value.args[0]
    assert "mean_const" in message
    assert problem in message
    assert "None" in message


def test_set_priors_refuses_a_location_that_is_not_finite_in_a_block():
    """In a block, a coordinate whose location and ``sigma`` are both NaN
    has no prior, and another coordinate with an infinite location is
    refused."""
    priors = _no_priors()
    priors["covariance_log_lengthscale"] = (
        "gaussian",
        (np.array([np.nan, np.inf]), np.array([np.nan, 1.0])),
    )
    with pytest.raises(ValueError) as execinfo:
        _gp_2d().set_priors(priors)
    message = execinfo.value.args[0]
    assert "covariance_log_lengthscale" in message
    assert "infinite mu" in message


@pytest.mark.parametrize(
    "family, params",
    [
        ("smoothbox", (1.0, -1.0, 0.7)),
        ("smoothbox_student_t", (3.0, -3.0, 0.7, 4.0)),
        # One coordinate of a block, beside a coordinate without a prior.
        ("smoothbox", ([np.nan, 1.0], [np.nan, 0.9], [np.nan, 0.7])),
    ],
)
def test_set_priors_refuses_an_inverted_smooth_box(family, params):
    """A smooth box needs its lower end ``a`` at or below its upper end
    ``b``: an inverted box has a normalizer below one, or negative, and a
    log prior that is wrong or NaN. The message names the hyperparameter
    and says what is wrong with its box."""
    priors = _no_priors()
    if np.ndim(params[0]) == 0:
        priors["mean_const"] = (family, params)
        name, gp = "mean_const", _gp_1d()
    else:
        priors["covariance_log_lengthscale"] = (family, params)
        name, gp = "covariance_log_lengthscale", _gp_2d()
    with pytest.raises(ValueError) as execinfo:
        gp.set_priors(priors)
    message = execinfo.value.args[0]
    assert name in message
    assert "above its upper end" in message
    assert "None" in message


@pytest.mark.parametrize("family", ["smoothbox", "smoothbox_student_t"])
def test_smooth_box_of_zero_width(family):
    """A smooth box whose two ends are equal has no plateau and is the
    Gaussian, or the Student's t, centred at that point with the box's
    scale; ``set_priors`` takes it, and the log prior is that density."""
    if family == "smoothbox":
        params = (0.5, 0.5, 0.7)
        density = scipy.stats.norm(loc=0.5, scale=0.7)
    else:
        params = (0.5, 0.5, 0.7, 4.0)
        density = scipy.stats.t(4.0, loc=0.5, scale=0.7)
    priors = _no_priors()
    priors["mean_const"] = (family, params)

    X = np.reshape(np.linspace(-2, 2, 8), (-1, 1))
    gp = _gp_1d()
    hyp = np.array([0.0, 0.0, np.log(0.1), 0.0])
    gp.update(X_new=X, y_new=np.sin(X), hyp=hyp[None, :])
    gp.set_priors(priors)
    for mean_const in (-0.4, 0.5, 1.9):
        hyp[3] = mean_const
        log_prior = gp.log_posterior(hyp) - gp.log_likelihood(hyp)
        assert np.isclose(log_prior, density.logpdf(mean_const), rtol=1e-12)


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


def test_set_bounds_refuses_an_inverted_pair():
    """``set_bounds`` refuses a lower bound above its upper bound, naming
    the hyperparameter, as ``get_recommended_bounds`` and ``fit`` refuse
    it, and leaves the bounds as they were; equal bounds, which fix a
    hyperparameter, are taken. It stored the inverted pair, which the
    next ``fit`` refused."""
    gp = _gp_2d()
    bounds = {name: None for name in _no_priors()}
    bounds["mean_const"] = (-1.0, 1.0)
    gp.set_bounds(bounds)
    lower_bounds = gp.lower_bounds.copy()
    upper_bounds = gp.upper_bounds.copy()

    inverted = dict(bounds)
    inverted["covariance_log_lengthscale"] = ([0.0, 2.0], [1.0, 1.0])
    with pytest.raises(ValueError) as execinfo:
        gp.set_bounds(inverted)

    message = execinfo.value.args[0]
    assert "Lower bound above upper bound" in message
    assert "covariance_log_lengthscale" in message
    assert "mean_const" not in message
    assert np.array_equal(gp.lower_bounds, lower_bounds, equal_nan=True)
    assert np.array_equal(gp.upper_bounds, upper_bounds, equal_nan=True)

    fixed = dict(bounds)
    fixed["covariance_log_lengthscale"] = (1.0, 1.0)
    gp.set_bounds(fixed)
    assert np.all(gp.lower_bounds[:2] == 1.0)
    assert np.all(gp.upper_bounds[:2] == 1.0)


def test_fit_raises_what_it_documents():
    """``fit`` passes on the ``ValueError`` of ``get_recommended_bounds``,
    through which it fills its bounds, and that of ``update``, which
    computes the posterior of the fitted hyperparameters, as its
    ``Raises`` section says."""
    X = np.reshape(np.linspace(-2, 2, 8), (-1, 1))
    y = np.sin(X)
    options = {"n_samples": 0, "init_N": 8}

    with pytest.raises(ValueError, match="`lower_bounds` should be"):
        _gp_1d().fit(X, y, options=dict(options, lower_bounds="nonsense"))
    with pytest.raises(ValueError, match="Lower bound above upper bound"):
        _gp_1d().fit(
            X,
            y,
            options=dict(
                options, lower_bounds=np.ones(4), upper_bounds=-np.ones(4)
            ),
        )
    # Without a space-filling design or an optimization, the fit keeps
    # the starting point it is given, NaN included.
    with pytest.raises(ValueError, match="are NaN"):
        _gp_1d().fit(
            X,
            y,
            hyp0=np.full((1, 4), np.nan),
            options={"n_samples": 0, "init_N": 0, "opts_N": 0},
        )


@pytest.mark.parametrize("given", ["neither", "X", "y"])
def test_fit_without_training_data_raises(given):
    """A fit needs training inputs and targets, given to it or held by the
    GP. Without them it says so, naming what is missing, before it changes
    anything. It raised ``AttributeError`` from a component or from the
    check of the shapes, or, given ``X`` alone, the ``ValueError`` of
    ``get_recommended_bounds`` after storing ``X``."""
    X = np.reshape(np.linspace(-2, 2, 8), (-1, 1))
    y = np.sin(X)
    data = {"X": {"X": X}, "y": {"y": y}, "neither": {}}[given]
    gp = _gp_1d()

    with pytest.raises(ValueError) as execinfo:
        gp.fit(**data, options={"n_samples": 0, "init_N": 8})

    message = execinfo.value.args[0]
    assert "no training data" in message
    missing = {"X": "y", "y": "X", "neither": "X and y"}[given]
    assert f"missing {missing}." in message
    assert gp.X is None and gp.y is None
    assert np.all(np.isnan(gp.lower_bounds))


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


@pytest.mark.parametrize(
    "family, params, bounds",
    [
        ("gaussian", (0.0, 1.0), (9.0, 10.0)),
        ("student_t", (0.0, 1.0, 3.0), (1e6, 2e6)),
        ("smoothbox", (-0.5, 0.5, 1.0), (9.5, 10.5)),
        ("smoothbox_student_t", (-0.5, 0.5, 1.0, 3.0), (1e6, 2e6)),
    ],
)
def test_prior_mass_in_the_upper_tail(family, params, bounds):
    """The mass of a prior between two bounds far in its upper tail, by
    which the log prior is renormalized, is that between the mirrored
    bounds in its lower tail, since each prior here is symmetric about
    zero. The cumulative distribution function rounds to one at both
    bounds of the upper tail, so a mass taken as the difference of its two
    values is zero and the log posterior infinite."""
    X = np.reshape(np.linspace(-2, 2, 8), (-1, 1))
    priors = _no_priors()
    priors["mean_const"] = (family, params)

    masses = []
    log_priors = []
    for lower, upper in (bounds, (-bounds[1], -bounds[0])):
        # The targets follow the constant mean into the tail, so that the
        # log likelihood stays of the size of the log prior.
        m0 = 0.5 * (lower + upper)
        hyp = np.array([0.0, 0.0, np.log(0.1), m0])
        gp = _gp_1d()
        gp.update(X_new=X, y_new=np.sin(X) + m0, hyp=hyp[None, :])
        gp_bounds = {name: (-np.inf, np.inf) for name in priors}
        gp_bounds["mean_const"] = (lower, upper)
        gp.set_bounds(gp_bounds)
        gp.set_priors(priors)
        masses.append(gp.normalization_constants[3])
        log_priors.append(gp.log_posterior(hyp) - gp.log_likelihood(hyp))

    assert masses[0] > 0.0
    assert np.isclose(masses[0], masses[1], rtol=1e-12)
    assert np.isfinite(log_priors[0])
    assert np.isclose(log_priors[0], log_priors[1], rtol=1e-12)


@pytest.mark.parametrize(
    "family", ["gaussian", "student_t", "smoothbox", "smoothbox_student_t"]
)
def test_prior_mass_from_the_centre_down(family):
    """Where the lower bound is not above the centre of the prior (``mu``,
    or the middle of the box), as for every prior that PyVBMC sets, the
    mass inside the bounds is the difference of the cumulative
    distribution function at the two bounds, bit for bit. The lower bound
    here is at the centre and below it."""
    from gpyreg.f_min_fill import smoothbox_cdf, smoothbox_student_t_cdf

    mu, sigma, df, a, b = 0.3, 1.2, 3.0, -1.0, 1.6
    centre = mu
    if family == "gaussian":
        params = (mu, sigma)
        cdf = lambda x: scipy.stats.norm.cdf(x, loc=mu, scale=sigma)
    elif family == "student_t":
        params = (mu, sigma, df)
        cdf = lambda x: scipy.stats.t.cdf(x, df, loc=mu, scale=sigma)
    elif family == "smoothbox":
        params = (a, b, sigma)
        cdf = lambda x: smoothbox_cdf(x, sigma, a, b)
        centre = 0.5 * (a + b)
    else:
        params = (a, b, sigma, df)
        cdf = lambda x: smoothbox_student_t_cdf(x, df, sigma, a, b)
        centre = 0.5 * (a + b)
    priors = _no_priors()
    priors["mean_const"] = (family, params)

    for lower, upper in (
        (centre, 4.0),
        (centre - 1.3, 4.0),
        (-5.0, -1.0),
        (centre, np.inf),
    ):
        gp = _gp_1d()
        gp_bounds = {name: (-np.inf, np.inf) for name in priors}
        gp_bounds["mean_const"] = (lower, upper)
        gp.set_bounds(gp_bounds)
        gp.set_priors(priors)
        assert gp.normalization_constants[3] == cdf(upper) - cdf(lower)


@pytest.mark.parametrize(
    "mu, df, a, b, bounds",
    [
        (0.0, 0.0, np.nan, np.nan, (9.0, 10.0)),
        (0.0, 3.0, np.nan, np.nan, (1e6, 2e6)),
        (np.nan, 0.0, -0.5, 0.5, (9.5, 10.5)),
        (np.nan, 3.0, -0.5, 0.5, (1e6, 2e6)),
    ],
)
def test_space_filling_design_in_the_upper_tail(mu, df, a, b, bounds):
    """The space-filling design maps its unit-cube draws through the
    prior truncated to the bounds. With both bounds far in the upper tail
    of the prior, the design is the mirror image of the design between the
    mirrored bounds in the lower tail, since each prior here (a Gaussian, a
    Student's t and their smooth boxes) is symmetric about zero and the
    draws of one coordinate of the unscrambled Sobol sequence are
    symmetric about one half. The cumulative distribution function rounds
    to one at both bounds of the upper tail, and a design mapped through it
    lies at infinity."""
    from gpyreg.f_min_fill import f_min_fill

    hprior = {
        "mu": np.array([mu]),
        "sigma": np.array([1.0]),
        "df": np.array([df]),
        "a": np.array([a]),
        "b": np.array([b]),
    }
    designs = []
    for lower, upper in (bounds, (-bounds[1], -bounds[0])):
        LB, UB = np.array([lower]), np.array([upper])
        X, __ = f_min_fill(
            lambda x: 0.0,
            np.array([[0.5 * (lower + upper)]]),
            LB,
            UB,
            LB,
            UB,
            hprior,
            64,
            rng=np.random.default_rng(0),
        )
        designs.append(np.sort(X[:, 0]))

    assert np.all((designs[0] >= bounds[0]) & (designs[0] <= bounds[1]))
    assert np.allclose(designs[0], -designs[1][::-1], rtol=1e-10, atol=0.0)


def test_fit_with_samples_in_the_upper_tail_of_a_prior(monkeypatch):
    """A fit with hyperparameter samples, where both bounds of the constant
    mean lie far in the upper tail of its Gaussian prior, completes as the
    mirrored fit in the lower tail does: its space-filling design is the
    mirror image of the other's, and its samples lie inside the bounds. A
    design at infinity gave the slice sampler a width of NaN."""
    from gpyreg import gaussian_process as gp_module

    designs = []
    real_f_min_fill = gp_module.f_min_fill

    def recording_f_min_fill(*args, **kwargs):
        X0, y0 = real_f_min_fill(*args, **kwargs)
        designs.append(np.sort(X0[:, 3]))
        return X0, y0

    monkeypatch.setattr(gp_module, "f_min_fill", recording_f_min_fill)

    X = np.reshape(np.linspace(-1, 1, 12), (-1, 1))
    priors = _no_priors()
    priors["mean_const"] = ("gaussian", (0.0, 1.0))
    for sign in (1.0, -1.0):
        # The targets follow the constant mean into the tail, and change
        # sign with it, so that the two posteriors mirror each other.
        y = sign * (9.5 + np.sin(2 * X))
        gp = _gp_1d()
        gp.X, gp.y, gp.s2 = gp._convert_shapes(X, y, None)
        gp_bounds = gp.get_recommended_bounds()
        gp_bounds["mean_const"] = (
            np.array([min(9.0 * sign, 10.0 * sign)]),
            np.array([max(9.0 * sign, 10.0 * sign)]),
        )
        gp.set_bounds(gp_bounds)
        gp.set_priors(priors)
        hyp, __, __ = gp.fit(
            options={"n_samples": 3, "init_N": 64, "opts_N": 2},
            rng=np.random.default_rng(0),
        )
        assert np.all(np.isfinite(hyp))
        assert np.all((9.0 <= sign * hyp[:, 3]) & (sign * hyp[:, 3] <= 10.0))

    assert np.allclose(designs[0], -designs[1][::-1], rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("below_centre", [0.0, 1.8])
@pytest.mark.parametrize(
    "family", ["gaussian", "student_t", "smoothbox", "smoothbox_student_t"]
)
def test_space_filling_design_from_the_centre_down(family, below_centre):
    """Where the lower bound is not above the centre of the prior (``mu``,
    or the middle of the box), as for every prior that PyVBMC sets (the
    lower bound of its noise prior is the prior's centre), the
    space-filling design maps its unit-cube draws through the cumulative
    distribution function of the prior at the two bounds and its percent
    point function, bit for bit. The lower bound here is at the centre and
    below it."""
    from gpyreg.f_min_fill import (
        f_min_fill,
        smoothbox_cdf,
        smoothbox_ppf,
        smoothbox_student_t_cdf,
        smoothbox_student_t_ppf,
    )

    mu, sigma, a, b = 0.3, 1.2, -1.0, 1.6
    df = 3.0 if family.endswith("student_t") else 0.0
    smooth_box = family.startswith("smoothbox")
    centre = 0.5 * (a + b) if smooth_box else mu
    hprior = {
        "mu": np.array([np.nan if smooth_box else mu]),
        "sigma": np.array([sigma]),
        "df": np.array([df]),
        "a": np.array([a if smooth_box else np.nan]),
        "b": np.array([b if smooth_box else np.nan]),
    }
    LB, UB = np.array([centre - below_centre]), np.array([4.0])
    N = 257
    x0 = np.array([[1.0]])
    X, __ = f_min_fill(
        lambda x: x[0],
        x0,
        LB,
        UB,
        LB,
        UB,
        hprior,
        N,
        rng=np.random.default_rng(0),
    )

    # The draws of `f_min_fill`: the unscrambled Sobol sequence without its
    # first point (a single column, which the shuffle leaves alone).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = scipy.stats.qmc.Sobol(d=1, scramble=False).random(n=N)[1:, 0]
    if family == "gaussian":
        cdf_lb = scipy.stats.norm.cdf((LB[0] - mu) / sigma)
        cdf_ub = scipy.stats.norm.cdf((UB[0] - mu) / sigma)
        expected = (
            scipy.stats.norm.ppf(cdf_lb + (cdf_ub - cdf_lb) * S) * sigma + mu
        )
    elif family == "student_t":
        cdf_lb = scipy.stats.t.cdf((LB[0] - mu) / sigma, df)
        cdf_ub = scipy.stats.t.cdf((UB[0] - mu) / sigma, df)
        expected = (
            scipy.stats.t.ppf(cdf_lb + (cdf_ub - cdf_lb) * S, df) * sigma + mu
        )
    elif family == "smoothbox":
        cdf_lb = smoothbox_cdf(LB[0], sigma, a, b)
        cdf_ub = smoothbox_cdf(UB[0], sigma, a, b)
        expected = np.array(
            [
                smoothbox_ppf(q, sigma, a, b)
                for q in cdf_lb + (cdf_ub - cdf_lb) * S
            ]
        )
    else:
        cdf_lb = smoothbox_student_t_cdf(LB[0], df, sigma, a, b)
        cdf_ub = smoothbox_student_t_cdf(UB[0], df, sigma, a, b)
        expected = np.array(
            [
                smoothbox_student_t_ppf(q, df, sigma, a, b)
                for q in cdf_lb + (cdf_ub - cdf_lb) * S
            ]
        )

    # `f_min_fill` returns its points in the order of the objective, which
    # is the value itself here.
    expected = np.sort(np.concatenate((x0[:, 0], expected)))
    assert np.array_equal(X[:, 0], expected)


@pytest.mark.parametrize(
    "prior",
    [
        # Priors of the form PyVBMC gives its noise, centred above the
        # fixed value and below it.
        ("student_t", (np.log(0.2), 0.5, 3.0)),
        ("student_t", (-8.0, 0.5, 3.0)),
        ("gaussian", (np.log(0.2), 0.5)),
        # A smooth box around the fixed value, and one below it.
        ("smoothbox", (-7.0, -4.0, 0.5)),
        ("smoothbox_student_t", (-9.0, -8.0, 0.5, 3.0)),
    ],
)
def test_space_filling_design_of_a_fixed_coordinate_with_a_prior(
    monkeypatch, prior
):
    """A hyperparameter whose two bounds are equal takes their value at
    every point of the space-filling design, whatever its prior. Mapped
    through the prior's quantile function, the value comes back an ulp or
    two off, where the log prior is ``-inf``. The noise here is fixed at
    ``log(sqrt(1e-5))``, as a PyVBMC-built GP fixes it for targets of a
    small range."""
    from gpyreg import gaussian_process as gp_module

    received = {}
    real_f_min_fill = gp_module.f_min_fill

    def recording_f_min_fill(*args, **kwargs):
        X0, y0 = real_f_min_fill(*args, **kwargs)
        received.update(X0=X0.copy(), y0=y0.copy())
        return X0, y0

    monkeypatch.setattr(gp_module, "f_min_fill", recording_f_min_fill)

    fixed = np.log(np.sqrt(1e-5))
    X = np.reshape(np.linspace(-2, 2, 12), (-1, 1))
    gp = _gp_1d()
    bounds = {name: None for name in _no_priors()}
    bounds["noise_log_scale"] = (fixed, fixed)
    gp.set_bounds(bounds)
    priors = _no_priors()
    priors["noise_log_scale"] = prior
    gp.set_priors(priors)
    gp.fit(
        X=X,
        y=np.sin(X),
        options={"n_samples": 0, "init_N": 64, "opts_N": 1},
        rng=np.random.default_rng(0),
    )

    assert np.all(received["X0"][:, 2] == fixed)
    assert np.all(np.isfinite(received["y0"]))


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


@pytest.mark.parametrize(
    "features",
    [
        {"user_provided_add": True},
        {"user_provided_add": True, "scale_user_provided": True},
        {"rectified_linear_output_dependent_add": True},
        {
            "user_provided_add": True,
            "scale_user_provided": True,
            "rectified_linear_output_dependent_add": True,
        },
    ],
)
def test_likelihood_gradient_with_a_noise_per_point(features):
    """Where the noise varies from point to point, the gradient of the log
    marginal likelihood with respect to the noise hyperparameters is a sum
    over the training points of the noise function's gradient, weighted by
    the diagonal of ``inv(C) - alpha alpha^T``. It agrees with finite
    differences for a user-provided variance, with and without a scale of
    its own, and for the rectified output-dependent noise, whose threshold
    lies inside the range of the targets, so that it adds noise at some
    points and not at others. One input dimension, with repeated inputs."""
    rng = np.random.default_rng(12)
    X = rng.uniform(-1, 1, size=(10, 1))
    X = np.concatenate((X, X[:3], X[:1]))
    N = X.shape[0]
    y = -2 * X**2 + 0.1 * rng.standard_normal((N, 1))
    s2 = 0.01 / rng.integers(1, 4, size=(N, 1))

    gp = gpr.GP(
        D=1,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True, **features),
    )
    # [log ell, log sf], the noise, [m0, xm, log omega]
    noise_hyp = [np.log(0.05)]
    if features.get("scale_user_provided"):
        noise_hyp.append(np.log(1.5))
    if features.get("rectified_linear_output_dependent_add"):
        # The threshold in the middle of the widest gap between the
        # targets, away from the kinks of the noise at each target.
        y_sorted = np.sort(y[:, 0])
        k = 3 + np.argmax(np.diff(y_sorted[3:-3]))
        noise_hyp += [0.5 * (y_sorted[k] + y_sorted[k + 1]), np.log(0.5)]
    hyp = np.concatenate(
        ([np.log(0.4), 0.0], noise_hyp, [0.2, 0.1, np.log(0.8)])
    )
    given_s2 = s2 if features.get("user_provided_add") else None
    gp.update(X_new=X, y_new=y, s2_new=given_s2, hyp=hyp[None, :])

    __, gradient = gp.log_likelihood(hyp, compute_grad=True)
    error = check_grad(
        gp.log_likelihood,
        lambda h: gp.log_likelihood(h, compute_grad=True)[1],
        hyp,
    )
    assert np.all(error < 1e-6 * np.max(np.abs(gradient)))


@pytest.mark.parametrize("sn2, low_noise", [(0.9e-6, True), (1.1e-6, False)])
@pytest.mark.parametrize("mean_name", ["constant", "negative_quadratic"])
def test_likelihood_in_the_low_noise_representation(sn2, low_noise, mean_name):
    """Below a smallest noise variance of 1e-6 the posterior holds the
    inverse of the training covariance instead of its Cholesky factor.
    Just below the switch, on a set where the covariance is well
    conditioned, the negative log marginal likelihood equals a dense
    evaluation and its gradient agrees with finite differences, as they do
    just above it."""
    X = np.reshape(np.linspace(-1, 1, 8), (-1, 1))
    y = np.sin(3 * X)
    N = X.shape[0]
    if mean_name == "constant":
        mean, mean_hyp = gpr.mean_functions.ConstantMean(), [0.1]
    else:
        mean = gpr.mean_functions.NegativeQuadratic()
        mean_hyp = [0.5, 0.1, np.log(0.8)]
    gp = gpr.GP(
        D=1,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=mean,
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    # A short length scale keeps the covariance close to its diagonal.
    hyp = np.concatenate(([np.log(0.08), 0.0, 0.5 * np.log(sn2)], mean_hyp))
    gp.update(X_new=X, y_new=y, hyp=hyp[None, :])
    assert gp.posteriors[0].L_chol == (not low_noise)

    C = np.exp(-0.5 * ((X - X.T) / 0.08) ** 2) + sn2 * np.eye(N)
    r = y - mean.compute(np.array(mean_hyp), X).reshape(-1, 1)
    dense_nlZ = (
        0.5 * (r.T @ np.linalg.solve(C, r))[0, 0]
        + 0.5 * np.linalg.slogdet(C)[1]
        + 0.5 * N * np.log(2 * np.pi)
    )
    assert np.isclose(-gp.log_likelihood(hyp), dense_nlZ, rtol=1e-12)

    __, gradient = gp.log_likelihood(hyp, compute_grad=True)
    error = check_grad(
        gp.log_likelihood,
        lambda h: gp.log_likelihood(h, compute_grad=True)[1],
        hyp,
    )
    assert np.all(error < 1e-8 * np.max(np.abs(gradient)))


def test_quad_takes_one_width_per_measure():
    """A ``sigma`` of one column holds one standard deviation per measure,
    the same in every dimension, as ``gplite_quad.m`` broadcasts it and as
    release 1.2.1 took it: the integrals and their variances are those of
    the same widths written out per dimension. A ``sigma`` of another width
    than one or ``D`` is refused."""
    D = 3
    rng = np.random.default_rng(9)
    X = rng.uniform(-2, 2, size=(15, D))
    y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2]) * X[:, 2:3]
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    hyp = np.concatenate(
        [np.zeros(D), [0.0], [np.log(0.1)], [0.5], np.zeros(D), np.zeros(D)]
    )
    gp.update(X_new=X, y_new=y, hyp=hyp[None, :])
    mu = rng.uniform(-1, 1, size=(4, D))
    widths = np.array([[0.3], [0.7], [1.1], [2.0]])

    F_one, V_one = gp.quad(mu, widths, compute_var=True)
    F_all, V_all = gp.quad(mu, np.tile(widths, (1, D)), compute_var=True)

    assert np.array_equal(F_one, F_all)
    assert np.array_equal(V_one, V_all)
    with pytest.raises(ValueError) as execinfo:
        gp.quad(mu, np.ones((4, 2)))
    assert "one column per input" in execinfo.value.args[0]


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
    requires. A variance of another type is a ``TypeError``, a wrong row
    count a ``ValueError``."""
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

    for s2 in ("nonsense", [1.0] * N):
        with pytest.raises(TypeError):
            gp._convert_shapes(X, None, s2)

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


def test_fit_with_targets_of_a_tiny_range(monkeypatch):
    """Targets whose standard deviation is below 1e-3, the noise's
    plausible lower bound, give the noise an inverted recommended
    plausible pair, ``[0.5 * log(tol), log(std(y))]``, as gplite's noise
    does. The clips into the hard box keep it inverted while the range of
    the targets is above 1e-6 (below that the hard pair collapses first),
    and the space-filling design needs ``PLB <= PUB``. These targets have
    a range of 8.2e-5 and a standard deviation of 2.6e-5."""
    from gpyreg import gaussian_process as gp_module

    received = {}
    real_f_min_fill = gp_module.f_min_fill

    def recording_f_min_fill(fun, x0, LB, UB, PLB, PUB, *args, **kwargs):
        received.update(LB=LB.copy(), UB=UB.copy())
        received.update(PLB=PLB.copy(), PUB=PUB.copy())
        return real_f_min_fill(fun, x0, LB, UB, PLB, PUB, *args, **kwargs)

    monkeypatch.setattr(gp_module, "f_min_fill", recording_f_min_fill)

    rng = np.random.default_rng(2)
    X = rng.uniform(-2, 2, size=(20, 1))
    y = 1.0 + 1e-4 * rng.random((20, 1))

    gp = _gp_1d()
    hyp, __, __ = gp.fit(
        X=X,
        y=y,
        options={"n_samples": 0, "opts_N": 1, "init_N": 16},
        rng=np.random.default_rng(3),
    )
    assert np.all(np.isfinite(hyp))
    # The design receives an ordered plausible pair inside the hard box.
    assert np.all(received["PLB"] <= received["PUB"])
    assert np.all(received["LB"] <= received["PLB"])
    assert np.all(received["PUB"] <= received["UB"])
