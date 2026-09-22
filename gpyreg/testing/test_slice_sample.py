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


@pytest.mark.parametrize("burn", [2, 3])
def test_short_burn_in_keeps_the_chain_moving(burn):
    """The width adaptation at the end of the burn-in needs a window of at
    least two iterations. A window of one iteration has no variance to
    estimate, and a width of zero brackets nothing, so the coordinate would
    stop and the chain return its starting point once per requested sample."""
    rv = multivariate_normal(np.zeros(2), np.eye(2))
    sampler = SliceSampler(
        rv.logpdf,
        np.zeros(2),
        widths=1.0,
        options={"display": "off", "diagnostics": False},
        rng=np.random.default_rng(3),
    )
    res = sampler.sample(6, burn=burn)

    assert np.all(sampler.widths > 0)
    assert np.unique(res["samples"], axis=0).shape[0] == 6


def test_infinite_width_does_not_come_back_after_the_burn_in():
    """An infinite width is legal for an unbounded coordinate and is
    replaced by 10 at construction. The base widths that the geometric-mean
    recombination at the end of the burn-in uses are the replaced ones: an
    infinite base width would come back there and fill the chain with NaN."""
    rv = multivariate_normal(np.zeros(2), np.eye(2))
    sampler = SliceSampler(
        rv.logpdf,
        np.zeros(2),
        widths=[np.inf, 1.0],
        LB=[-np.inf, -5.0],
        UB=[np.inf, 5.0],
        options={"display": "off", "diagnostics": False},
        rng=np.random.default_rng(6),
    )
    res = sampler.sample(20, burn=10)

    assert np.all(np.isfinite(sampler.base_widths))
    assert np.all(np.isfinite(sampler.widths))
    assert np.all(np.isfinite(res["samples"]))


def test_burn_in_statistics_window_is_the_second_half():
    """The adapted widths come from the last ``floor(burn / 2)`` burn-in
    iterations, one term per iteration and the same number in the divisor.

    On a uniform target whose widths already span the whole box, every
    proposal is accepted at the first try and the within-burn-in adaptation
    leaves the widths untouched, so a second sampler with the same seed and
    no adaptation walks the identical chain and hands over its iterates.
    """
    burn = 7  # odd, where MATLAB accumulates one term more than it divides by
    LB = np.array([-2.0, -1.0])
    UB = np.array([3.0, 4.0])
    widths = UB - LB
    uniform_logpdf = lambda x: 0.0

    def sampler(adaptive):
        return SliceSampler(
            uniform_logpdf,
            np.array([0.0, 0.5]),
            widths=widths,
            LB=LB,
            UB=UB,
            options={
                "display": "off",
                "diagnostics": False,
                "adaptive": adaptive,
            },
            rng=np.random.default_rng(11),
        )

    adapted = sampler(True)
    adapted.sample(1, burn=burn)
    # The same chain without adaptation, recorded iteration by iteration.
    trace = sampler(False).sample(burn + 1, burn=0)["samples"]

    def widths_from(window):
        n = window.shape[0]
        var = (window**2).sum(0) / n - (window.sum(0) / n) ** 2
        new_widths = np.fmin(
            5 * np.sqrt(np.maximum(var, 0)), adapted.UB_out - adapted.LB_out
        )
        return np.maximum(new_widths, np.sqrt(new_widths * widths))

    expected = widths_from(trace[math.ceil(burn / 2) : burn])
    np.testing.assert_allclose(adapted.widths, expected, rtol=1e-12)
    # Not vacuous: MATLAB's window, one iteration longer, gives other widths.
    assert trace[math.ceil(burn / 2) : burn].shape[0] == burn // 2
    assert np.any(widths_from(trace[burn // 2 : burn]) != expected)


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


@pytest.mark.parametrize("constant", [0.0, 1.0, 0.3, 1e-3, -7.0, np.log(10)])
def test_constant_trace_in_a_free_parameter_fails_the_diagnostics(
    caplog, constant
):
    """A parameter that is free to move but whose chain stayed put has
    undefined diagnostics, which must not pass as convergence.

    Whether the estimates come out non-finite depends on the rounding of
    the constant's own mean, so the chain is recognized by its range.
    """
    sampler = _diagnostics_sampler()
    samples = np.full((20, 1), constant)

    with caplog.at_level(logging.INFO, logger="SliceSampler"):
        exit_flag, R, eff_N = sampler._SliceSampler__diagnose(samples)

    assert exit_flag == -3
    assert np.all(np.isnan(R))
    assert np.all(np.isnan(eff_N))
    assert "did not move" in caplog.text


@pytest.mark.parametrize("constant", [0.0, 1.0, 0.3])
def test_fixed_parameter_is_left_out_of_the_diagnostics(constant):
    """A parameter fixed by LB == UB has no diagnostics to report, and the
    checks look only at the parameter that is actually sampled."""
    rv = multivariate_normal(np.zeros(2), np.eye(2))
    sampler = SliceSampler(
        rv.logpdf,
        np.array([0.0, constant]),
        LB=np.array([-np.inf, constant]),
        UB=np.array([np.inf, constant]),
        options=options,
        rng=np.random.default_rng(2),
    )
    res = sampler.sample(200)

    assert np.all(res["samples"][:, 1] == constant)
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
    ``1 / log10(m * n)``. The pairs start at lag 0, the convention of the
    estimator under test, so this is a second reading of the estimator's
    definition and not an external check of the convention itself.
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


def test_healthy_chain_effective_n_matches_reference(caplog):
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
    with caplog.at_level(logging.INFO, logger="SliceSampler"):
        res = slicer.sample(400)
    assert res["exit_flag"] == 1
    assert np.all(np.isfinite(res["R"]))
    assert np.all(np.isfinite(res["eff_N"]))
    assert np.all(res["eff_N"] >= 400 / 10)
    assert "No violations of convergence" in caplog.text


def test_list_bounds_detect_fixed_parameter():
    """Bounds and widths given as lists behave like arrays: a coordinate
    with equal bounds is fixed, its width is irrelevant, and it is left
    out of the diagnostics."""
    rv = multivariate_normal(np.zeros(2), np.eye(2))
    sampler = SliceSampler(
        rv.logpdf,
        np.array([0.0, 1.0]),
        widths=[1.0, 3.0],
        LB=[-np.inf, 1.0],
        UB=[np.inf, 1.0],
        options=options,
        rng=np.random.default_rng(4),
    )
    for bounds in (sampler.LB, sampler.UB, sampler.widths):
        assert isinstance(bounds, np.ndarray)
        assert bounds.dtype == float
    assert sampler.widths[1] == 1.0

    res = sampler.sample(100)
    assert np.all(res["samples"][:, 1] == 1.0)
    assert res["exit_flag"] == 1


# The following tests can fail with some small probability.


def test_normal():
    slicer = SliceSampler(
        norm.logpdf,
        np.array([0.5]),
        options=options,
        rng=np.random.default_rng(0),
    )
    samples = slicer.sample(20000)["samples"]

    assert np.abs(norm.mean() - np.mean(samples)) < threshold
    assert np.abs(norm.var() - np.var(samples)) < threshold


def test_normal_step_out():
    new_options = options = {
        "display": "off",
        "diagnostics": True,
        "step_out": True,
    }
    slicer = SliceSampler(
        norm.logpdf,
        np.array([0.5]),
        options=new_options,
        rng=np.random.default_rng(1),
    )
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
    slicer = SliceSampler(
        logpdf,
        np.array([0.5]),
        options=options,
        rng=np.random.default_rng(2),
    )
    samples = slicer.sample(20000)["samples"]

    assert np.abs((1 - p) * 6 - np.mean(samples)) < threshold
    # plt.scatter(samples , pdf(samples))
    # plt.show()


def test_exponential():
    slicer = SliceSampler(
        expon.logpdf,
        np.array([0.5]),
        LB=0.0,
        options=options,
        rng=np.random.default_rng(3),
    )
    samples = slicer.sample(20000)["samples"]

    # plt.scatter(samples, expon.pdf(samples))
    # plt.show()
    assert np.abs(expon.mean() - np.mean(samples)) < threshold
    assert np.abs(expon.var() - np.var(samples)) < threshold


def test_uniform():
    slicer = SliceSampler(
        uniform.logpdf,
        np.array([0.5]),
        LB=0.0,
        UB=1.0,
        options=options,
        rng=np.random.default_rng(4),
    )
    samples = slicer.sample(20000)["samples"]

    assert np.abs(uniform.mean() - np.mean(samples)) < threshold
    assert np.abs(uniform.var() - np.var(samples)) < threshold


def test_beta():
    a, b = 2.31, 0.627
    rv = beta(a, b)
    slicer = SliceSampler(
        rv.logpdf,
        np.array([0.5]),
        LB=0.0,
        UB=1.0,
        options=options,
        rng=np.random.default_rng(5),
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
        rv.logpdf,
        np.array([0.5, -0.5, 1.0]),
        options=options,
        rng=np.random.default_rng(6),
    )
    samples = slicer.sample(20000)["samples"]

    assert np.all(np.abs(mean - np.mean(samples, axis=0)) < threshold)
    # assert np.all(np.abs(cov - np.cov(samples.T)) < threshold)


def test_multivariate_t():
    x = [1.0, -0.5]
    loc = [[2.1, 0.3], [0.3, 1.5]]
    rv = multivariate_t(x, loc, df=3)
    slicer = SliceSampler(
        rv.logpdf,
        np.array([0.5, 0.5]),
        options=options,
        rng=np.random.default_rng(7),
    )
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
    with pytest.raises(ValueError) as execinfo:
        slicer.sample(3, thin=1.5)
    assert (
        "The thinning factor option needs to be a positive integer"
        in execinfo.value.args[0]
    )
    with pytest.raises(ValueError) as execinfo:
        slicer.sample(3, burn=2.5)
    assert (
        "burn-in samples option needs to be a non-negative"
        in execinfo.value.args[0]
    )
    # A whole number of another type is still a whole number.
    slicer.sample(3, thin=2.0, burn=1.0)

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
def test_serialized_sampler_continues_stream(serialization, rng_kind):
    """Copied samplers resume with generator state or the current global stream.

    Old pickle state has no rng attribute. Serializing a legacy sampler
    must not capture global state.
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
