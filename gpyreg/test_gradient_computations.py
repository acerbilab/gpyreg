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
    hyp = np.random.standard_normal(size=(cov_N + noise_N + mean_N, N_s))
    hyp[D, :] *= 0.2
    hyp[D + 1 : D + 1 + noise_N, :] *= 0.3

    gp.update(X_new=X, hyp=hyp, compute_posterior=False)
    y = gp.random_function(X)

    gp.update(y_new=y, hyp=hyp)

    hyp0 = hyp[:, 0]

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
    for i in range(0, cov_N + mean_N + noise_N):
        prior_type = np.random.randint(0, 4)
        if prior_type == 0:  # 'gaussian'
            gp.hyper_priors.mu[i] = np.random.standard_normal()
            gp.hyper_priors.sigma[i] = np.exp(np.random.standard_normal())
            gp.hyper_priors.df[i] = 0
        elif prior_type == 1:  #'student_t'
            gp.hyper_priors.mu[i] = np.random.standard_normal()
            gp.hyper_priors.sigma[i] = np.random.standard_normal()
            gp.hyper_priors.df[i] = np.exp(np.random.standard_normal())
        elif prior_type == 2:  # 'smoothbox'
            gp.hyper_priors.a[i] = -10
            gp.hyper_priors.b[i] = 10
            gp.hyper_priors.sigma[i] = np.random.standard_normal()
            gp.hyper_priors.df[i] = 0
        elif prior_type == 3:  # 'smoothbox_student_t'
            gp.hyper_priors.a[i] = -10
            gp.hyper_priors.b[i] = 10
            gp.hyper_priors.sigma[i] = np.random.standard_normal()
            gp.hyper_priors.df[i] = np.exp(np.random.standard_normal())
    f = lambda hyp_: gp._GP__compute_log_priors(hyp_, False)
    f_grad = lambda hyp_: gp._GP__compute_log_priors(hyp_, True)[1]
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

    # Test plotting
    gp.plot()
