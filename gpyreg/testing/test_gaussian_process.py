import copy

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats
from scipy.integrate import quad
from scipy.misc import derivative

import gpyreg as gpr


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

    gp.update(compute_posterior=True)

    for i in range(0, 10):
        assert np.all(posteriors[i].hyp == gp.posteriors[i].hyp)
        assert np.all(posteriors[i].alpha == gp.posteriors[i].alpha)
        assert np.all(posteriors[i].sW == gp.posteriors[i].sW)
        assert np.all(posteriors[i].L == gp.posteriors[i].L)
        assert posteriors[i].L_chol == gp.posteriors[i].L_chol
        assert posteriors[i].sn2_mult == posteriors[i].sn2_mult

    # gp.plot()


def partial(f, x0_orig, x0_i, i):
    x0 = x0_orig.copy()
    x0[i] = x0_i
    return f(x0)


def compute_gradient(f, x0):
    num_grad = np.zeros(x0.shape)

    for i in range(0, np.size(x0)):
        f_i = lambda x0_i: partial(f, x0, x0_i, i)
        tmp = scipy.misc.derivative(
            f_i, x0[i], dx=np.finfo(float).eps ** (1 / 5.0), order=5
        )
        num_grad[i] = tmp

    return num_grad


def check_grad(f, grad, x0):
    analytical_grad = grad(x0)
    numerical_grad = compute_gradient(f, x0)
    return np.abs(analytical_grad - numerical_grad)


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


@pytest.mark.filterwarnings(
    """ignore:Matplotlib is currently using agg:UserWarning"""
)
def test_quadrature_with_noise():
    N = 500
    D = 1
    s2_constant = 0.01
    X = np.reshape(np.linspace(-15, 15, N), (-1, 1))
    s2 = np.full(X.shape, s2_constant)

    mu_N = 1000
    x_star = np.reshape(np.linspace(-15, 15, mu_N), (-1, 1))

    y = np.sin(X) + np.sqrt(s2) * scipy.stats.norm.ppf(
        np.random.random_sample(X.shape)
    )
    y[y < 0] = -np.abs(3 * y[y < 0]) ** 2

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
    gp.fit(X=X, y=y, s2=s2, options=gp_train)

    f_mu, f_cov = gp.predict_full(x_star, s2_star=s2_constant, add_noise=True)
    F_predict = 0
    for i in range(0, mu_N):
        F_predict += f_mu[i, 0] * scipy.stats.norm.pdf(x_star[i], scale=0.11)
    F_predict *= 30 / mu_N

    pdf_tmp = np.reshape(scipy.stats.norm.pdf(x_star, scale=0.1), (-1, 1))
    tmp = np.dot(pdf_tmp, pdf_tmp.T)
    F_predict_var = np.sum(np.sum(f_cov[:, :, 0] * tmp)) * (30 / mu_N) ** 2

    F_bayes, F_bayes_var = gp.quad(0, 0.1, compute_var=True)

    assert np.abs(F_bayes_var - F_predict_var) < 0.01
    assert np.abs(F_bayes - F_predict) < 0.01

    def f(x):
        y = np.sin(x)
        if y < 0:
            return -np.abs(3 * y) ** 2
        return y

    f_p = lambda x: f(x) * scipy.stats.norm.pdf(x, scale=0.1)

    F_true = scipy.integrate.quad(f_p, -np.inf, np.inf)[0]

    assert np.abs(F_true - F_bayes) < 0.1

    # gp.plot()


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
    y = 1 + np.sin(X)

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
    N = 1000
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
    hyp = np.array(
        [
            [
                # Covariance
                0.0,
                0.0,
                0.0,  # log ell
                1.0,  # log sf2
                # Noise
                np.log(np.pi),  # log std. dev. of noise
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
                0.0,
                0.0,
                0.0,  # log ell
                1.0,  # log sf2
                # Noise
                np.log(np.pi),  # log std. dev. of noise
                # Mean
                -(D / 2) * np.log(2 * np.pi),  # MVN mode
                0.0,
                0.0,
                0.0,  # Mode location
                0.0,
                0.0,
                0.0,  # log scale
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
    s2_star = np.arange(-3, 3).reshape((-1, 1))
    s2_star = np.zeros((6, 1))
    f_mu, f_s2, lpd = gp.predict(
        X_star, y_star, s2_star=s2_star, return_lpd=True
    )
    assert np.allclose(
        lpd,
        scipy.stats.norm.logpdf(
            y_star, loc=f_mu, scale=np.sqrt(np.pi * s2_star + f_s2)
        ),
    )
    __, __, lpd2 = gp.predict(
        X_star, y_star, s2_star=s2_star, return_lpd=True, add_noise=True
    )
    assert np.all(lpd2 == lpd)
    __, __, lpd3 = gp.predict(
        X_star,
        y_star,
        s2_star=s2_star,
        return_lpd=True,
        add_noise=True,
        separate_samples=True,
    )
    assert np.all(lpd3[:, 0:1] == lpd)
    assert np.all(lpd3[:, 1:2] == lpd)


def test__str__and__repr__():
    # 1-D:
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
    y = 1 + np.sin(X)

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
