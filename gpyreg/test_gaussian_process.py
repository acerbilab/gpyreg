import numpy as np
import scipy as sp

import gpyreg as gpr


def partial(f, x0_orig, x0_i, i):
    x0 = x0_orig.copy()
    x0[i] = x0_i
    return f(x0)


def compute_gradient(f, x0):
    num_grad = np.zeros(x0.shape)

    for i in range(0, np.size(x0)):
        f_i = lambda x0_i: partial(f, x0, x0_i, i)
        tmp = sp.misc.derivative(
            f_i, x0[i], dx=np.finfo(float).eps ** (1 / 5.0), order=5
        )
        num_grad[i] = tmp

    return num_grad


def check_grad(f, grad, x0):
    analytical_grad = grad(x0)
    numerical_grad = compute_gradient(f, x0)
    return np.abs(analytical_grad - numerical_grad)


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

    gp.update(X_new=X, hyp=hyp, compute_posterior=False)
    y = gp.random_function(X)

    gp.update(y_new=y, hyp=hyp)

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
    gp.set_priors({})

    hyp1 = hyp0 * np.exp(0.1 * np.random.uniform(size=hyp0.size))
    for i in range(0, cov_N + mean_N + noise_N):
        prior_type = np.random.randint(1, 6)
        if prior_type == 1:  # 'gaussian'
            gp.hyper_priors["mu"][i] = np.random.standard_normal()
            gp.hyper_priors["sigma"][i] = np.exp(np.random.standard_normal())
            gp.hyper_priors["df"][i] = 0
        elif prior_type == 2:  #'student_t'
            gp.hyper_priors["mu"][i] = np.random.standard_normal()
            gp.hyper_priors["sigma"][i] = np.random.standard_normal()
            gp.hyper_priors["df"][i] = np.exp(np.random.standard_normal())
        elif prior_type == 3:  # 'smoothbox'
            gp.hyper_priors["a"][i] = -10
            gp.hyper_priors["b"][i] = 10
            gp.hyper_priors["sigma"][i] = np.random.standard_normal()
            gp.hyper_priors["df"][i] = 0
        elif prior_type == 4:  # 'smoothbox_student_t'
            gp.hyper_priors["a"][i] = -10
            gp.hyper_priors["b"][i] = 10
            gp.hyper_priors["sigma"][i] = np.random.standard_normal()
            gp.hyper_priors["df"][i] = np.exp(np.random.standard_normal())
        else:  # None
            pass

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
        gp1.update(X_new=X[i, :], y_new=y[i])

    assert np.all(np.isclose(gp.X, gp1.X))
    assert np.all(np.isclose(gp.y, gp1.y))

    assert np.all(np.isclose(gp.posteriors[0].hyp, gp1.posteriors[0].hyp))
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

    # Test plotting
    gp.plot()


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

    bounds = gp.get_bounds()
    assert np.all(bounds["covariance_log_lengthscale"][0] == -np.inf)
    assert np.all(bounds["covariance_log_lengthscale"][1] == np.inf)
    assert np.all(bounds["covariance_log_outputscale"][0] == -np.inf)
    assert np.all(bounds["covariance_log_outputscale"][1] == np.inf)
    assert np.all(bounds["noise_log_scale"][0] == -np.inf)
    assert np.all(bounds["noise_log_scale"][1] == np.inf)
    assert np.all(bounds["mean_const"][0] == -np.inf)
    assert np.all(bounds["mean_const"][1] == np.inf)

    hyp_dict_list = gp.get_hyperparameters()
    assert len(hyp_dict_list) == 1

    hyp_dict = hyp_dict_list[0]
    assert np.all(np.isnan(hyp_dict["covariance_log_lengthscale"]))
    assert np.all(np.isnan(hyp_dict["covariance_log_outputscale"]))
    assert np.all(np.isnan(hyp_dict["noise_log_scale"]))
    assert np.all(np.isnan(hyp_dict["mean_const"]))

    assert np.all(np.isnan(gp.get_hyperparameters(as_array=True)))

    prior = gp.get_priors()
    assert prior["covariance_log_lengthscale"] is None
    assert prior["covariance_log_outputscale"] is None
    assert prior["noise_log_scale"] is None
    assert prior["mean_const"] is None

    gp_priors_mistaken = {
        "covariance_log_outputscal": ("student_t", (0, np.log(10), 3)),
        "covariance_log_lengthscale": (
            "gaussian",
            (np.log(np.std(X, ddof=1)), np.log(10)),
        ),
        "noise_log_scale": ("gaussian", (np.log(1e-3), 1.0)),
        "mean_const": ("smoothbox", (np.min(y), np.max(y), 1.0)),
    }

    mistaken = gp.set_priors(gp_priors_mistaken)
    assert len(mistaken) == 1 and mistaken[0] == "covariance_log_outputscale"

    gp_priors = {
        "covariance_log_outputscale": ("student_t", (0, np.log(10), 3)),
        "covariance_log_lengthscale": (
            "gaussian",
            (np.log(np.std(X, ddof=1)), np.log(10)),
        ),
        "noise_log_scale": ("gaussian", (np.log(1e-3), 1.0)),
        "mean_const": ("smoothbox", (np.min(y), np.max(y), 1.0)),
    }

    gp.set_priors(gp_priors)
    prior = gp.get_priors()
    assert gp_priors == prior

    hyp_arr = np.array(
        [[-0.4630094, -0.78566179, -0.2209450, -7.2947503, 0.03713608]]
    )
    hyp = gp.hyperparameters_to_dict(hyp_arr)
    gp.set_hyperparameters(hyp)
    assert np.all(gp.get_hyperparameters(as_array=True) == hyp_arr)

    gp.set_hyperparameters(hyp_arr)
    assert np.all(gp.get_hyperparameters(as_array=True) == hyp_arr)

    gp_train = {"n_samples": 10}
    hyp, _ = gp.fit(X=X, y=y, options=gp_train)

    assert np.all(gp.get_hyperparameters(as_array=True) == hyp)

    hyp_dict_list = gp.get_hyperparameters()
    for i, hyp_dict in enumerate(hyp_dict_list):
        assert np.all(hyp_dict["covariance_log_lengthscale"] == hyp[i, 0:2])
        assert np.all(hyp_dict["covariance_log_outputscale"] == hyp[i, 2])
        assert np.all(hyp_dict["noise_log_scale"] == hyp[i, 3])
        assert np.all(hyp_dict["mean_const"] == hyp[i, 4])

    prior = gp.get_priors()
    assert gp_priors == prior

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


def incomplete_test_fitting():
    np.random.seed(123456)
    N = 500
    D = 1
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

    N_s = 1
    hyp = np.random.standard_normal(size=(N_s, cov_N + noise_N + mean_N))
    hyp[:, D] *= 0.2
    hyp[:, D + 1 : D + 1 + noise_N] *= 0.3
    print(hyp)

    gp.update(hyp=hyp, compute_posterior=False)
    y = gp.random_function(X, add_noise=True)
    gp.update(X_new=X, y_new=y, hyp=hyp, compute_posterior=True)
    gp.plot()

    aaa

    gp1 = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )

    gp_train = {"n_samples": 0}
    gp_priors = {
        # "noise_log_scale": ("student_t", (np.log(1e-3), 1.0, 7)),
    }

    gp1.set_priors(gp_priors)
    gp1.fit(X=X, y=y, options=gp_train)
    hyp2 = gp1.get_hyperparameters(as_array=True)
    print(hyp2)
    print(gp1._GP__compute_nlZ(hyp[0, :], False, False))
    print(gp1._GP__compute_nlZ(hyp2[0, :], False, False))
    gp1.plot()
