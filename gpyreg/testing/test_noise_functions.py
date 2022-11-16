import numpy as np
import pytest

from gpyreg.noise_functions import GaussianNoise


def test_gaussian_noise_compute_sanity_checks():
    gaussiannoise = GaussianNoise(True, True, True, True)
    D = 3
    N = 20
    X = np.ones((N, D))
    y = np.ones((N, 1))

    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones((4 + 1))
        gaussiannoise.compute(hyp, X, y)
    assert (
        "Expected 4 noise function hyperparameters" in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones((4, 1))
        gaussiannoise.compute(hyp, X, y)
    assert (
        "Noise function output is available only for" in execinfo.value.args[0]
    )
