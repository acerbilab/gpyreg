import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats
from scipy.integrate import quad

from gpyreg.f_min_fill import smoothbox_cdf, smoothbox_ppf, uuinv


def pdf(x, sigma, a, b):
    # Normalization constant so that integral over pdf is 1.
    C = 1.0 + (b - a) / (sigma * np.sqrt(2 * np.pi))

    if x < a:
        return scipy.stats.norm.pdf(x, loc=a, scale=sigma) / C

    if x <= b:
        return 1 / (sigma * np.sqrt(2 * np.pi)) / C

    return scipy.stats.norm.pdf(x, loc=b, scale=sigma) / C


@pytest.mark.filterwarnings(
    """ignore:Matplotlib is currently using agg:UserWarning"""
)
def test_pdf():
    sigma = 3
    a = -2
    b = 3

    assert np.isclose(1.0, quad(pdf, -np.inf, np.inf, args=(sigma, a, b))[0])

    N = 10000
    xx = np.linspace(-15, 15, N)
    yy = np.zeros((N,))
    for i in range(0, N):
        yy[i] = pdf(xx[i], sigma, a, b)
    # plt.plot(xx, yy)
    # plt.show()

    assert np.isclose(
        pdf(a - np.spacing(a), sigma, a, b),
        pdf(a + np.spacing(a), sigma, a, b),
    )
    assert np.isclose(
        pdf(b - np.spacing(b), sigma, a, b),
        pdf(b + np.spacing(b), sigma, a, b),
    )


def test_cdf_limits():
    sigma = 5
    a = -5
    b = 10

    assert np.isclose(smoothbox_cdf(-np.inf, sigma, a, b), 0.0)
    assert np.isclose(smoothbox_cdf(np.inf, sigma, a, b), 1.0)


def test_ppf_limits():
    sigma = 5
    a = -5
    b = 10

    assert np.isclose(smoothbox_ppf(0.0, sigma, a, b), -np.inf)
    assert np.isclose(smoothbox_ppf(1.0, sigma, a, b), np.inf)


def test_cdf_ppf():
    xx = np.linspace(-5, 5, 200)
    sigma = 3
    a = -2
    b = 3

    for x in xx:
        assert np.isclose(
            smoothbox_ppf(smoothbox_cdf(x, sigma, a, b), sigma, a, b), x
        )


def test_ppf_cdf():
    qq = np.linspace(0, 1, 200)
    sigma = 0.1
    a = -4
    b = 0

    for q in qq:
        assert np.isclose(
            smoothbox_cdf(smoothbox_ppf(q, sigma, a, b), sigma, a, b), q
        )


def test_uuinv():
    p = np.linspace(0, 1, 1000)
    Bs = [
        [0, 0, 30, 66],
        [0, 10, 30, 66],
        [0, 10, 66, 66],
        [0, 10, 10, 66],
    ]
    ws = [0, 0.6, 1]

    for B, w in itertools.product(Bs, ws):
        x = uuinv(p, B, w)
        assert np.isclose(
            sum((x <= B[2]) & (B[1] <= x)) / np.size(x), w, atol=1e-3
        )
        assert np.isclose(
            sum((x < B[1]) & (B[0] <= x)) / np.size(x),
            (1 - w) * (B[1] - B[0]) / (B[1] - B[0] + B[3] - B[2]),
            atol=1e-3,
        )
        assert np.isclose(
            sum((x <= B[3]) & (B[2] < x)) / np.size(x),
            (1 - w) * (B[3] - B[2]) / (B[1] - B[0] + B[3] - B[2]),
            atol=1e-3,
        )

    # Degenerate to mixture of delta and uniform distributions
    Bs = [[0, 0, 66, 66]]
    ws = [0, 0.6, 1]
    for B, w in itertools.product(Bs, ws):
        x = uuinv(p, B, w)
        assert np.isclose(
            sum((x < B[2]) & (B[1] < x)) / np.size(x), w, atol=2e-3
        )
        assert np.isclose(sum(x == B[0]) / np.size(x), (1 - w) / 2, atol=2e-3)
        assert np.isclose(sum(x == B[3]) / np.size(x), (1 - w) / 2, atol=2e-3)
