import numpy as np
import pytest

from gpyreg.mean_functions import ConstantMean, NegativeQuadratic, ZeroMean


def test_constant_mean_compute_sanity_checks():
    constantmean = ConstantMean()
    D = 3
    N = 20
    X = np.ones((N, D))

    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(D + 2)
        constantmean.compute(hyp, X)
    assert "Expected 1 mean function hyperparameters" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones((1, 1))
        constantmean.compute(hyp, X)
    assert (
        "Mean function output is available only for" in execinfo.value.args[0]
    )


def test_negative_quadratic_compute_sanity_checks():
    negative_quadratic = NegativeQuadratic()
    D = 3
    N = 20
    X = np.ones((N, D))

    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(D + 2)
        negative_quadratic.compute(hyp, X)
    assert "Expected 7 mean function hyperparameters" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones((7, 1))
        negative_quadratic.compute(hyp, X)
    assert (
        "Mean function output is available only for" in execinfo.value.args[0]
    )


def test_zero_mean_compute_sanity_checks():
    zeromean = ZeroMean()
    D = 3
    N = 20
    X = np.ones((N, D))

    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones(D + 2)
        zeromean.compute(hyp, X)
    assert "Expected 0 mean function hyperparameters" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        hyp = np.ones((0, 0))
        zeromean.compute(hyp, X)
    assert (
        "Mean function output is available only for" in execinfo.value.args[0]
    )


@pytest.mark.parametrize("D", [1, 3, 8, 12])
@pytest.mark.parametrize(
    "mean_cls", [ZeroMean, ConstantMean, NegativeQuadratic]
)
def test_compute_batched_matches_compute(mean_cls, D):
    """Column ``s`` of ``compute_batched`` is bit-identical to
    ``compute(hyp[s], X)`` (the batched form reduces the same contiguous
    last axis; ``predict`` relies on this)."""
    rng = np.random.default_rng(D)
    mean = mean_cls()
    mean_N = mean.hyperparameter_count(D)
    N, Ns = 13, 5
    X = rng.standard_normal((N, D))
    hyp = rng.standard_normal((Ns, mean_N))
    batched = mean.compute_batched(hyp, X)
    assert batched.shape == (N, Ns)
    for s in range(Ns):
        assert np.array_equal(batched[:, s], mean.compute(hyp[s], X))
    with pytest.raises(ValueError):
        mean.compute_batched(hyp[0], X)
    with pytest.raises(ValueError):
        mean.compute_batched(np.zeros((Ns, mean_N + 1)), X)
