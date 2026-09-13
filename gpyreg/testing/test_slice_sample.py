import copy
import logging
import math
import pickle

import numpy as np
import pytest
from scipy.stats import (
    beta,
    expon,
    multivariate_normal,
    multivariate_t,
    norm,
    uniform,
)

from gpyreg.slice_sample import SliceSampler

options = {"display": "off", "diagnostics": True}
threshold = 0.1


def _normal_metropolis_proposal():
    return np.random.normal(size=1)


def test_multiple_runs():
    state = np.random.get_state()

    np.random.seed(1234)
    slicer1 = SliceSampler(norm.logpdf, np.array([0.5]), options=options)
    res1 = slicer1.sample(300)

    np.random.seed(1234)
    slicer2 = SliceSampler(norm.logpdf, np.array([0.5]), options=options)
    res2 = slicer2.sample(100, burn=100)
    res3 = slicer2.sample(100)
    res4 = slicer2.sample(100)

    np.random.set_state(state)

    assert np.all(
        res1["samples"]
        == np.concatenate((res2["samples"], res3["samples"], res4["samples"]))
    )


@pytest.mark.parametrize("step_out", [False, True])
def test_evaluations_stay_on_coordinate_line(step_out):
    # Coordinate-wise slice sampling evaluates the density along one axis at
    # a time, so consecutive evaluations differ in at most one coordinate.
    # With step_out=True the bracket ends x_l and x_r are evaluated as full
    # vectors, and their entries for earlier dimensions must hold the
    # accepted coordinates rather than those dimensions' shrunk bracket
    # edges (GitHub issue #44).
    D = 3
    rho = 0.9
    cov = rho * np.ones((D, D)) + (1 - rho) * np.eye(D)
    precision = np.linalg.inv(cov)
    evaluated = []

    def logpdf(x):
        evaluated.append(np.array(x, copy=True))
        return -0.5 * (x @ precision @ x)

    N = 200
    slicer = SliceSampler(
        logpdf,
        np.zeros(D),
        widths=0.5,
        options={"display": "off", "step_out": step_out},
        rng=np.random.default_rng(7),
    )
    samples = slicer.sample(N, burn=0)["samples"]

    evaluated = np.array(evaluated)
    n_changed = np.count_nonzero(np.diff(evaluated, axis=0), axis=1)
    assert np.all(n_changed <= 1)
    # The checks above are not vacuous: every coordinate moved, and with
    # step-out both bracket ends were evaluated for every coordinate.
    assert np.all(np.ptp(samples, axis=0) > 0)
    if step_out:
        assert len(evaluated) >= 2 * D * N


def _split(trace):
    """Split a trace into the two half-chains the diagnostics compare."""
    n = math.floor(trace.shape[0] / 2)
    return np.array([trace[0:n, :], trace[n : 2 * n, :]])


def _diagnostics_sampler():
    """A sampler whose private diagnostics helpers can be called directly."""
    return SliceSampler(norm.logpdf, np.array([0.0]), options=options)


def test_short_chain_diagnostics_do_not_claim_success():
    """Four draws are too few to diagnose anything, and the effective
    sample size of such a chain is still positive and finite."""
    sampler = SliceSampler(
        norm.logpdf,
        np.array([0.0]),
        widths=1.0,
        options=options,
        rng=np.random.default_rng(0),
    )
    res = sampler.sample(4, burn=0)

    assert res["exit_flag"] == -3
    assert np.all(np.isnan(res["R"]))
    assert np.all(np.isnan(res["eff_N"]))

    eff_N = sampler._SliceSampler__effective_n(_split(res["samples"]))
    assert np.all(eff_N > 0)
    assert np.all(np.isfinite(eff_N))


def test_constant_trace_in_a_free_parameter_fails_the_diagnostics(caplog):
    """A parameter that is free to move but whose chain stayed put has
    undefined diagnostics, which must not pass as convergence."""
    sampler = _diagnostics_sampler()
    samples = np.zeros((20, 1))

    with caplog.at_level(logging.INFO, logger="SliceSampler"):
        exit_flag, R, eff_N = sampler._SliceSampler__diagnose(samples)

    assert exit_flag == -3
    assert np.all(np.isnan(R))
    assert np.all(np.isnan(eff_N))
    assert "did not move" in caplog.text


def test_fixed_parameter_is_left_out_of_the_diagnostics():
    """A parameter fixed by LB == UB has no diagnostics to report, and the
    checks look only at the parameter that is actually sampled."""
    rv = multivariate_normal(np.zeros(2), np.eye(2))
    sampler = SliceSampler(
        rv.logpdf,
        np.array([0.0, 1.0]),
        LB=np.array([-np.inf, 1.0]),
        UB=np.array([np.inf, 1.0]),
        options=options,
        rng=np.random.default_rng(2),
    )
    res = sampler.sample(200)

    assert np.all(res["samples"][:, 1] == 1.0)
    assert np.isnan(res["R"][1])
    assert np.isnan(res["eff_N"][1])
    assert np.isfinite(res["R"][0])
    assert res["eff_N"][0] > 0
    assert res["exit_flag"] == 1


def test_anticorrelated_trace_has_a_large_but_bounded_effective_n():
    """Anticorrelated draws carry more information than independent ones,
    so the effective sample size exceeds the number of draws, up to the
    cap that keeps the estimate finite."""
    sampler = _diagnostics_sampler()
    trace = np.tile([-1.0, 1.0], 10)[:, None]
    split = _split(trace)
    m, n = split.shape[0], split.shape[1]

    eff_N = sampler._SliceSampler__effective_n(split)

    cap = m * n * np.log10(m * n)
    assert np.all(np.isfinite(eff_N))
    assert np.all(eff_N > trace.shape[0])
    assert np.all(eff_N <= cap * (1 + 1e-12))


def test_positively_correlated_trace_has_a_small_effective_n():
    """An AR(1) chain with a high coefficient mixes slowly, so its
    effective sample size is well below the number of draws."""
    sampler = _diagnostics_sampler()
    rng = np.random.default_rng(5)
    N = 400
    phi = 0.9
    innovations = rng.standard_normal(N)
    trace = np.zeros(N)
    for i in range(1, N):
        trace[i] = phi * trace[i - 1] + innovations[i]

    eff_N = sampler._SliceSampler__effective_n(_split(trace[:, None]))

    assert np.all(eff_N > 0)
    assert np.all(eff_N < N / 4)


def _geyer_effective_n_reference(split):
    """Effective sample size of two half-chains, written out directly.

    Estimates the autocorrelations from the split-chain variance and the
    variogram, sums them in consecutive pairs while the pair sum is
    positive, and floors the integrated autocorrelation time at
    ``1 / log10(m * n)``.
    """
    m, n = split.shape
    chain_means = split.mean(axis=1)
    B_over_n = np.sum((chain_means - split.mean()) ** 2) / (m - 1)
    W = np.sum((split - chain_means[:, None]) ** 2) / (m * (n - 1))
    s2 = W * (n - 1) / n + B_over_n
    rho = np.ones(n)
    for t in range(1, n):
        variogram = np.sum((split[:, t:] - split[:, :-t]) ** 2)
        variogram /= m * (n - t)
        rho[t] = 1.0 - variogram / (2.0 * s2)
    tau = -1.0
    for t in range(0, n - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        tau += 2 * pair
    tau = max(tau, 1.0 / np.log10(m * n))
    return m * n / tau


def test_healthy_chain_effective_n_matches_reference():
    """A mixing chain whose autocorrelation dies out ends the pair sum on
    a non-positive pair. The estimator then equals Geyer's initial positive
    sequence written out directly, and the diagnostics report success."""
    rng = np.random.default_rng(8)
    N = 300
    phi = 0.5
    innovations = rng.standard_normal(N)
    trace = np.zeros(N)
    for i in range(1, N):
        trace[i] = phi * trace[i - 1] + innovations[i]
    split = _split(trace[:, None])

    sampler = _diagnostics_sampler()
    eff_N = sampler._SliceSampler__effective_n(split)
    expected = _geyer_effective_n_reference(split[:, :, 0])
    assert 0 < expected < N
    np.testing.assert_allclose(eff_N, expected, rtol=1e-12)

    rv = multivariate_normal(np.zeros(2), np.eye(2))
    slicer = SliceSampler(
        rv.logpdf,
        np.zeros(2),
        options=options,
        rng=np.random.default_rng(9),
    )
    res = slicer.sample(400)
    assert res["exit_flag"] == 1
    assert np.all(np.isfinite(res["R"]))
    assert np.all(np.isfinite(res["eff_N"]))
    assert np.all(res["eff_N"] >= 400 / 10)


# The following tests can fail with some small probability.


def test_normal():
    slicer = SliceSampler(norm.logpdf, np.array([0.5]), options=options)
    samples = slicer.sample(20000)["samples"]

    assert np.abs(norm.mean() - np.mean(samples)) < threshold
    assert np.abs(norm.var() - np.var(samples)) < threshold


def test_normal_step_out():
    new_options = options = {
        "display": "off",
        "diagnostics": True,
        "step_out": True,
    }
    slicer = SliceSampler(norm.logpdf, np.array([0.5]), options=new_options)
    samples = slicer.sample(20000)["samples"]

    assert np.abs(norm.mean() - np.mean(samples)) < threshold
    assert np.abs(norm.var() - np.var(samples)) < threshold


def test_correlated_normal_step_out():
    # Strongly correlated target, where step-out decisions taken off the
    # current coordinate line would distort the brackets.
    rho = 0.9
    cov = np.array([[1.0, rho], [rho, 1.0]])
    precision = np.linalg.inv(cov)
    logpdf = lambda x: -0.5 * (x @ precision @ x)
    new_options = {"display": "off", "diagnostics": True, "step_out": True}
    slicer = SliceSampler(
        logpdf,
        np.array([0.5, -0.5]),
        options=new_options,
        rng=np.random.default_rng(1234),
    )
    samples = slicer.sample(20000)["samples"]

    assert np.all(np.abs(np.mean(samples, axis=0)) < threshold)
    assert np.all(np.abs(cov - np.cov(samples.T)) < threshold)


def test_normal_mixture():
    p = 0.7
    rv1 = norm(0, 1)
    rv2 = norm(6, 2)
    pdf = lambda x: p * rv1.pdf(x) + (1 - p) * rv2.pdf(x)
    logpdf = lambda x: np.log(pdf(x))  # if pdf(x) > np.spacing(0) else -np.inf
    slicer = SliceSampler(logpdf, np.array([0.5]), options=options)
    samples = slicer.sample(20000)["samples"]

    assert np.abs((1 - p) * 6 - np.mean(samples)) < threshold
    # plt.scatter(samples , pdf(samples))
    # plt.show()


def test_exponential():
    slicer = SliceSampler(
        expon.logpdf, np.array([0.5]), LB=0.0, options=options
    )
    samples = slicer.sample(20000)["samples"]

    # plt.scatter(samples, expon.pdf(samples))
    # plt.show()
    assert np.abs(expon.mean() - np.mean(samples)) < threshold
    assert np.abs(expon.var() - np.var(samples)) < threshold


def test_uniform():
    slicer = SliceSampler(
        uniform.logpdf, np.array([0.5]), LB=0.0, UB=1.0, options=options
    )
    samples = slicer.sample(20000)["samples"]

    assert np.abs(uniform.mean() - np.mean(samples)) < threshold
    assert np.abs(uniform.var() - np.var(samples)) < threshold


def test_beta():
    a, b = 2.31, 0.627
    rv = beta(a, b)
    slicer = SliceSampler(
        rv.logpdf, np.array([0.5]), LB=0.0, UB=1.0, options=options
    )
    samples = slicer.sample(20000)["samples"]

    assert np.abs(rv.mean() - np.mean(samples)) < threshold
    assert np.abs(rv.var() - np.var(samples)) < threshold


def test_multivariate_normal():
    mean = np.array([0.68, 0.6, 0.4])
    cov = np.array(
        [[1.58, 0.96, -1.2], [0.96, 2.17, -1.725], [-1.2, -1.725, 1.85]]
    )
    rv = multivariate_normal(mean, cov)
    slicer = SliceSampler(
        rv.logpdf, np.array([0.5, -0.5, 1.0]), options=options
    )
    samples = slicer.sample(20000)["samples"]

    assert np.all(np.abs(mean - np.mean(samples, axis=0)) < threshold)
    # assert np.all(np.abs(cov - np.cov(samples.T)) < threshold)


def test_multivariate_t():
    x = [1.0, -0.5]
    loc = [[2.1, 0.3], [0.3, 1.5]]
    rv = multivariate_t(x, loc, df=3)
    slicer = SliceSampler(rv.logpdf, np.array([0.5, 0.5]), options=options)
    samples = slicer.sample(20000)["samples"]

    assert np.all(np.abs(x - np.mean(samples, axis=0)) < threshold)


def test_init_sanity_checks():
    """
    Just some basic tests to check for incorrect input for __init__.
    """
    x = [1.0, -0.5]
    loc = [[2.1, 0.3], [0.3, 1.5]]
    rv = multivariate_t(x, loc, df=3)
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2, 2)))
    assert "initial point x0 needs to be a scalar" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2)), LB=np.zeros((2, 2)))
    assert "LB and UB need to be None, scalars" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(
            rv.logpdf, np.zeros((2)), LB=np.zeros((2, 2)), UB=np.ones((2, 2))
        )
    assert "LB and UB need to be None, scalars" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2)), UB=np.zeros((2, 2)))
    assert "LB and UB need to be None, scalars" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2)), LB=1, UB=0)
    assert "UB need to be equal or greater than" in execinfo.value.args[0]
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2)), widths=-1, UB=0)
    assert (
        "The widths vector needs to be all positive real numbers"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2)), widths=1 + 2j, UB=0)
    assert (
        "The widths vector needs to be all positive real numbers"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2)), LB=1, UB=2)
    assert (
        "The initial starting point X0 is outside the bounds"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        SliceSampler(rv.logpdf, np.zeros((2)), LB=-2, UB=-1)
    assert (
        "The initial starting point X0 is outside the bounds"
        in execinfo.value.args[0]
    )


def test_init_logger():
    mean = np.ones(3)
    cov = np.eye(3)
    rv = multivariate_normal(mean, cov)
    options = {"display": "off"}
    slicer = SliceSampler(rv.logpdf, np.ones(3), options=options)
    assert slicer.logger.getEffectiveLevel() == 30  # WARNING
    options = {"display": "summary"}
    slicer = SliceSampler(rv.logpdf, np.ones(3), options=options)
    assert slicer.logger.getEffectiveLevel() == 20  # INFO
    options = {"display": "full"}
    slicer = SliceSampler(rv.logpdf, np.ones(3), options=options)
    assert slicer.logger.getEffectiveLevel() == 10  # DEBUG


def test_sample_sanity_checks():
    """
    Just some basic tests to check for incorrect input for sample.
    """
    mean = np.ones(3)
    cov = np.eye(3)
    rv = multivariate_normal(mean, cov)
    slicer = SliceSampler(rv.logpdf, np.ones(3))
    with pytest.raises(ValueError) as execinfo:
        slicer.sample(3, thin=-1)
    assert (
        "The thinning factor option needs to be a positive integer"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        slicer.sample(3, thin=np.ones((3, 3)))
    assert (
        "The thinning factor option needs to be a positive integer"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        slicer.sample(3, burn=-1)
    assert (
        "burn-in samples option needs to be a non-negative"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        slicer.sample(3, burn=np.ones((3, 3)))
    assert (
        "burn-in samples option needs to be a non-negative"
        in execinfo.value.args[0]
    )
    slicer.x0 = slicer.x0 * np.nan
    with pytest.raises(ValueError) as execinfo:
        slicer.sample(3)
    assert (
        "The initial starting point X0 needs to evaluate to a"
        in execinfo.value.args[0]
    )


def test_generator_runs_are_reproducible_and_independent_of_global_state():
    """With ``rng`` a ``Generator``, two samplers seeded alike give the same
    chain whatever the global legacy state does, one generator shared across
    ``sample`` calls continues its stream, and ``rng=None`` still follows
    ``np.random.seed`` exactly as before."""
    state = np.random.get_state()
    try:
        np.random.seed(99)
        slicer1 = SliceSampler(
            norm.logpdf,
            np.array([0.5]),
            options=options,
            rng=np.random.default_rng(7),
        )
        res1 = slicer1.sample(300)
        np.random.seed(1)  # a different global state must not matter
        rng = np.random.default_rng(7)
        slicer2 = SliceSampler(
            norm.logpdf, np.array([0.5]), options=options, rng=rng
        )
        res2 = slicer2.sample(100, burn=100)
        res3 = slicer2.sample(100)
        res4 = slicer2.sample(100)
        assert np.all(
            res1["samples"]
            == np.concatenate(
                (res2["samples"], res3["samples"], res4["samples"])
            )
        )
        # the legacy path: rng=None draws from the global stream as always
        np.random.seed(1234)
        legacy = SliceSampler(norm.logpdf, np.array([0.5]), options=options)
        a = legacy.sample(50)["samples"]
        np.random.seed(1234)
        explicit = SliceSampler(
            norm.logpdf, np.array([0.5]), options=options, rng=None
        )
        assert np.array_equal(a, explicit.sample(50)["samples"])
    finally:
        np.random.set_state(state)


@pytest.mark.parametrize("serialization", ["pickle", "deepcopy"])
@pytest.mark.parametrize("rng_kind", ["legacy", "generator", "old_pickle"])
@pytest.mark.parametrize("metropolis", [False, True])
def test_serialized_sampler_continues_stream(
    serialization, rng_kind, metropolis
):
    """Copied samplers resume with generator state or the current global stream.

    Old pickle state has no rng attribute, including when Metropolis steps
    are enabled. Serializing a legacy sampler must not capture global state.
    """
    state = np.random.get_state()
    try:
        np.random.seed(718)
        sampler = SliceSampler(
            norm.logpdf,
            np.array([0.5]),
            widths=1.0,
            options={"display": "off", "diagnostics": False},
            rng=7 if rng_kind == "generator" else None,
        )
        if metropolis:
            sampler.metropolis_pdf = norm.pdf
            sampler.metropolis_rnd = _normal_metropolis_proposal
            sampler.metropolis_flag = True
        sampler.sample(5, burn=5)
        if rng_kind == "old_pickle":
            del sampler.rng
        if serialization == "pickle":
            restored = pickle.loads(pickle.dumps(sampler))
        else:
            restored = copy.deepcopy(sampler)
        if rng_kind == "old_pickle":
            from gpyreg.rng import resolve_rng

            sampler.rng = resolve_rng()
        # Reseeding after serialization proves the legacy stream stays live.
        np.random.seed(919)
        expected = sampler.sample(12, burn=0)["samples"]
        expected_state = np.random.get_state()
        np.random.seed(919)
        actual = restored.sample(12, burn=0)["samples"]
        assert np.array_equal(actual, expected)
        actual_state = np.random.get_state()
        assert all(
            np.array_equal(a, b) for a, b in zip(actual_state, expected_state)
        )
    finally:
        np.random.set_state(state)
