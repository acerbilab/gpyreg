import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats
from scipy.integrate import quad

from gpyreg.f_min_fill import smoothbox_student_t_cdf, smoothbox_student_t_ppf


def pdf(x, sigma, df, a, b):
    # Normalization constant so that integral over pdf is 1.
    c = scipy.special.gamma(0.5 * (df + 1)) / (
        scipy.special.gamma(0.5 * df) * sigma * np.sqrt(df * np.pi)
    )
    C = 1.0 + (b - a) * c

    if x < a:
        return scipy.stats.t.pdf(x, df, loc=a, scale=sigma) / C

    if x <= b:
        return c / C

    return scipy.stats.t.pdf(x, df, loc=b, scale=sigma) / C


@pytest.mark.filterwarnings(
    """ignore:Matplotlib is currently using agg:UserWarning"""
)
def test_pdf():
    sigma = 3
    df = 3
    a = -3
    b = 3

    assert np.isclose(
        1.0, quad(pdf, -np.inf, np.inf, args=(df, sigma, a, b))[0]
    )

    N = 10000
    xx = np.linspace(-15, 15, N)
    yy = np.zeros((N,))
    for i in range(0, N):
        yy[i] = pdf(xx[i], df, sigma, a, b)
    # plt.plot(xx, yy)
    # plt.show()

    assert np.isclose(
        pdf(a - np.spacing(a), df, sigma, a, b),
        pdf(a + np.spacing(a), df, sigma, a, b),
    )
    assert np.isclose(
        pdf(b - np.spacing(b), df, sigma, a, b),
        pdf(b + np.spacing(b), df, sigma, a, b),
    )


def test_cdf_limits():
    sigma = 5
    df = 3
    a = -5
    b = 10

    assert np.isclose(smoothbox_student_t_cdf(-np.inf, df, sigma, a, b), 0.0)
    assert np.isclose(smoothbox_student_t_cdf(np.inf, df, sigma, a, b), 1.0)


def test_ppf_limits():
    sigma = 5
    df = 3
    a = -5
    b = 10

    assert np.isclose(smoothbox_student_t_ppf(0.0, df, sigma, a, b), -np.inf)
    assert np.isclose(smoothbox_student_t_ppf(1.0, df, sigma, a, b), np.inf)


def test_cdf_ppf():
    xx = np.linspace(-5, 5, 200)
    df = 7
    sigma = 3
    a = -2
    b = 3

    for x in xx:
        assert np.isclose(
            smoothbox_student_t_ppf(
                smoothbox_student_t_cdf(x, df, sigma, a, b), df, sigma, a, b
            ),
            x,
        )


def test_ppf_cdf():
    qq = np.linspace(0, 1, 200)
    df = 3
    sigma = 0.1
    a = -4
    b = 0

    for q in qq:
        assert np.isclose(
            smoothbox_student_t_cdf(
                smoothbox_student_t_ppf(q, df, sigma, a, b), df, sigma, a, b
            ),
            q,
        )
