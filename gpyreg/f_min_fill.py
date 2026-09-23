"""Module for helper functions for GP training."""

import operator
import re
import warnings

import numpy as np
import scipy as sp

from gpyreg.rng import resolve_rng


def f_min_fill(
    f,
    x0,
    LB: np.ndarray,
    UB: np.ndarray,
    PLB: np.ndarray,
    PUB: np.ndarray,
    hprior: dict,
    N: int,
    design: str = None,
    rng=None,
):
    """
    Create a space-filling design, evaluates the function ``f``
    on the points of the design and sorts the points from smallest
    value of ``f`` to largest.

    Parameters
    ==========
    f : callable
        The function to evaluate on the design points.
    x0 : ndarray, shape (M, hyp_N)
        A 2D array of points to include in the design, with each row
        containing a design point.
    LB : ndarray, shape (hyp_N,)
        The lower bounds.
    UB : ndarray, shape (hyp_N,)
        The upper bounds.
    PLB : ndarray, shape (hyp_N,)
        The plausible lower bounds.
    PUB : ndarray, shape (hyp_N,)
        The plausible upper bounds.
    hprior : dict
        Hyperparameter prior dictionary.
    N : int
        Design size to use.
    init_method : {'sobol', 'rand'}, defaults to 'sobol'
        Specify what kind of method to use to construct the space-filling
        design.
    rng : None, numpy.random.Generator or seed, optional
        Where the random draws come from (the permutation of the Sobol
        columns, or the uniform design). ``None`` keeps NumPy's global
        legacy stream, as before generators were supported; see
        :func:`gpyreg.rng.resolve_rng`.

    Returns
    =======
    X : ndarray, shape (N, hyp_N)
        An array of the design points sorted according to the value
        ``f`` has at those points.
    y : ndarray, shape (N,)
        An array of the sorted values of ``f`` at the design points.
    """
    if design is None:
        design = "sobol"
    rng = resolve_rng(rng)

    # Helper for comparing version numbers.
    def ge_versions(version1, version2):
        def normalize(v):
            return [int(x) for x in re.sub(r"(\.0+)*$", "", v).split(".")]

        return operator.ge(normalize(version1), normalize(version2))

    # Check version number to make sure qmc exists.
    # Remove in the future when Anaconda has SciPy 1.7.0
    if design == "sobol" and not ge_versions(sp.__version__, "1.7.0"):
        design = "rand"

    N0 = x0.shape[0]
    n_vars = np.max(
        [x0.shape[1], np.size(LB), np.size(UB), np.size(PLB), np.size(PUB)]
    )

    # Force provided points to be inside bounds
    x0 = np.minimum(np.maximum(x0, LB), UB)

    sX = None

    if N > N0:
        # First test hyperparameters on a space-filling initial design
        if design == "sobol":
            sampler = sp.stats.qmc.Sobol(d=n_vars, scramble=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Get rid of first zero.
                S = sampler.random(n=N - N0 + 1)[1:, :]
            # Randomly permute columns
            rng.shuffle(S.T)
        elif design == "rand":
            S = rng.uniform(size=(N - N0, n_vars))
        else:
            raise ValueError(
                "Unknown design: got "
                + design
                + ' and expected either "sobol" or "rand"'
            )
        sX = np.zeros((N - N0, n_vars))

        # If a prior is specified use that: the draws are mapped through
        # the prior truncated to the bounds. Where the lower bound lies
        # above the centre of the prior (its median), the cumulative
        # distribution function is close to one at both bounds, and far in
        # the upper tail it rounds to one at both, so that a draw mapped
        # through it lands at infinity; there the draws go through the
        # survival function and its inverse, which give the same points in
        # exact arithmetic. `GP` takes the prior's mass inside the bounds
        # on the same switch.
        for i in range(0, n_vars):
            mu = hprior["mu"][i]
            sigma = hprior["sigma"][i]
            a = hprior["a"][i]
            b = hprior["b"][i]

            if not np.isfinite(mu) and not np.isfinite(
                sigma
            ):  # Uniform distribution?
                if np.isfinite(LB[i]) and np.isfinite(UB[i]):
                    # Fixed dimension
                    if LB[i] == UB[i]:
                        sX[:, i] = LB[i]
                    else:
                        # Mixture of uniforms
                        # (full bounds and plausible bounds)

                        # Each coordinate that takes this branch puts the
                        # weight w on its plausible interval, so the
                        # design puts the weight w ** m = 0.5 ** (m /
                        # n_vars) on the plausible intervals of the m
                        # coordinates that take it: one half where all
                        # n_vars do (no coordinate has a prior or an
                        # infinite bound, and none is fixed by LB == UB).
                        # A fixed coordinate takes its value, one with an
                        # infinite bound its plausible interval, and one
                        # with a prior the branches of the priors below.
                        w = 0.5 ** (1 / n_vars)

                        sX[:, i] = uuinv(
                            S[:, i], [LB[i], PLB[i], PUB[i], UB[i]], w
                        )
                else:
                    # All starting points from inside the plausible box
                    sX[:, i] = S[:, i] * (PUB[i] - PLB[i]) + PLB[i]
            elif np.isfinite(a) and np.isfinite(
                b
            ):  # Smooth box student's t prior
                df = hprior["df"][i]
                # Force fat tails
                if not np.isfinite(df):
                    df = 3
                df = np.minimum(df, 3)
                upper_half = LB[i] > 0.5 * (a + b)
                if df == 0 and upper_half:
                    sf_lb = smoothbox_sf(LB[i], sigma, a, b)
                    sf_ub = smoothbox_sf(UB[i], sigma, a, b)
                    S_scaled = sf_lb - (sf_lb - sf_ub) * S[:, i]
                    for j in range(0, (N - N0)):
                        sX[j, i] = smoothbox_isf(S_scaled[j], sigma, a, b)
                elif df == 0:
                    cdf_lb = smoothbox_cdf(LB[i], sigma, a, b)
                    cdf_ub = smoothbox_cdf(UB[i], sigma, a, b)
                    S_scaled = cdf_lb + (cdf_ub - cdf_lb) * S[:, i]
                    for j in range(0, (N - N0)):
                        sX[j, i] = smoothbox_ppf(S_scaled[j], sigma, a, b)
                elif upper_half:
                    tsf_lb = smoothbox_student_t_sf(LB[i], df, sigma, a, b)
                    tsf_ub = smoothbox_student_t_sf(UB[i], df, sigma, a, b)
                    S_scaled = tsf_lb - (tsf_lb - tsf_ub) * S[:, i]
                    for j in range(0, (N - N0)):
                        sX[j, i] = smoothbox_student_t_isf(
                            S_scaled[j], df, sigma, a, b
                        )
                else:
                    tcdf_lb = smoothbox_student_t_cdf(LB[i], df, sigma, a, b)
                    tcdf_ub = smoothbox_student_t_cdf(UB[i], df, sigma, a, b)
                    S_scaled = tcdf_lb + (tcdf_ub - tcdf_lb) * S[:, i]
                    for j in range(0, (N - N0)):
                        sX[j, i] = smoothbox_student_t_ppf(
                            S_scaled[j], df, sigma, a, b
                        )
            else:  # Student's t prior
                df = hprior["df"][i]
                # Force fat tails
                if not np.isfinite(df):
                    df = 3
                df = np.minimum(df, 3)
                upper_half = LB[i] > mu
                if df == 0 and upper_half:
                    sf_lb = sp.stats.norm.sf((LB[i] - mu) / sigma)
                    sf_ub = sp.stats.norm.sf((UB[i] - mu) / sigma)
                    S_scaled = sf_lb - (sf_lb - sf_ub) * S[:, i]
                    sX[:, i] = sp.stats.norm.isf(S_scaled) * sigma + mu
                elif df == 0:
                    cdf_lb = sp.stats.norm.cdf((LB[i] - mu) / sigma)
                    cdf_ub = sp.stats.norm.cdf((UB[i] - mu) / sigma)
                    S_scaled = cdf_lb + (cdf_ub - cdf_lb) * S[:, i]
                    sX[:, i] = sp.stats.norm.ppf(S_scaled) * sigma + mu
                elif upper_half:
                    tsf_lb = sp.stats.t.sf((LB[i] - mu) / sigma, df)
                    tsf_ub = sp.stats.t.sf((UB[i] - mu) / sigma, df)
                    S_scaled = tsf_lb - (tsf_lb - tsf_ub) * S[:, i]
                    sX[:, i] = sp.stats.t.isf(S_scaled, df) * sigma + mu
                else:
                    tcdf_lb = sp.stats.t.cdf((LB[i] - mu) / sigma, df)
                    tcdf_ub = sp.stats.t.cdf((UB[i] - mu) / sigma, df)
                    S_scaled = tcdf_lb + (tcdf_ub - tcdf_lb) * S[:, i]
                    sX[:, i] = sp.stats.t.ppf(S_scaled, df) * sigma + mu

    if sX is None:
        X = x0
    else:
        X = np.concatenate([x0, sX])
    y = np.full((N,), np.inf)
    for i in range(0, N):
        y[i] = f(X[i, :])

    order = np.argsort(y)

    return X[order, :], y[order]


def uuinv(p, B, w):
    r"""
    Inverse of cumulative distribution function of mixture of uniform
    distributions. The mixture puts the weight ``w`` on the plausible box
    and spreads the remaining weight over the union of the two tails in
    proportion to their lengths:

    .. math::
        w \, \text{Uniform}(B[1], B[2]) +
        (1 - w) \frac{B[1] - B[0]}{L} \text{Uniform}(B[0], B[1]) +
        (1 - w) \frac{B[3] - B[2]}{L} \text{Uniform}(B[2], B[3]),

    with :math:`L = (B[1] - B[0]) + (B[3] - B[2])`. The two tails
    therefore carry the same density as each other, and a tail of length
    zero carries no weight. Where both tails have length zero the mixture
    degenerates to the plausible box with a point mass of
    :math:`(1 - w) / 2` at each of ``B[0]`` and ``B[3]``.

    Parameters
    ----------
    p : ndarray
        1D array of cumulative function values.
    B : ndarray, list
        1D array or list containing [LB, PLB, PUB, UB].
    w : float
        The coefficient for mixture of uniform distributions.
        :math:`0 \leq w \leq 1`.

    Returns
    -------
    x : ndarray
        1D array of samples corresponding to `p`. Entries of `p` outside
        :math:`[0, 1]` are returned as NaN.
    """
    assert B[0] <= B[1] <= B[2] <= B[3]
    assert 0 <= w <= 1
    x = np.zeros(p.shape)
    L = B[3] - B[0] + B[1] - B[2]

    if w == 1:
        x = p * (B[2] - B[1]) + B[1]
    elif L == 0:
        # Degenerate to mixture of delta and uniform distributions
        i1 = p <= (1 - w) / 2
        x[i1] = B[0]

        if w != 0:
            i2 = (p <= (1 - w) / 2 + w) & ~i1
            x[i2] = (p[i2] - (1 - w) / 2) * (B[2] - B[1]) / w + B[1]

        i3 = p > (1 - w) / 2 + w
        x[i3] = B[3]
    else:
        # First step
        i1 = p <= (1 - w) * (B[1] - B[0]) / L
        x[i1] = B[0] + p[i1] * L / (1 - w)

        # Second step
        i2 = (p <= (1 - w) * (B[1] - B[0]) / L + w) & ~i1
        if w != 0:
            x[i2] = (p[i2] - (1 - w) * (B[1] - B[0]) / L) * (
                B[2] - B[1]
            ) / w + B[1]

        # Third step
        i3 = p > (1 - w) * (B[1] - B[0]) / L + w
        x[i3] = (p[i3] - w - (1 - w) * (B[1] - B[0]) / L) * L / (1 - w) + B[2]

    # Outside the unit interval p is not a quantile, in all three cases.
    x[p < 0] = np.nan
    x[p > 1] = np.nan

    return x


def smoothbox_cdf(x: float, sigma: float, a: float, b: float):
    """
    Compute the value of the cumulative distribution function
    for the smooth box distribution.

    Parameters
    ==========
    x : float
        The point where we want the value of the cdf.
    sigma : float
        Value of sigma for the smooth box distribution.
    a : float
        Value of a for the smooth box distribution.
    b : float
        Value of b for the smooth box distribution.
    """
    # Normalization constant so that integral over pdf is 1.
    C = 1.0 + (b - a) / (sigma * np.sqrt(2 * np.pi))

    if x < a:
        return sp.stats.norm.cdf(x, loc=a, scale=sigma) / C

    if x <= b:
        return (0.5 + (x - a) / (sigma * np.sqrt(2 * np.pi))) / C

    return (C - 1.0 + sp.stats.norm.cdf(x, loc=b, scale=sigma)) / C


def smoothbox_sf(x: float, sigma: float, a: float, b: float):
    """
    Compute the value of the survival function (one minus the cumulative
    distribution function) for the smooth box distribution. Above the box
    it keeps the probability of the upper tail, which one minus
    :func:`smoothbox_cdf` loses where the cumulative distribution function
    rounds to one.

    Parameters
    ==========
    x : float
        The point where we want the value of the survival function.
    sigma : float
        Value of sigma for the smooth box distribution.
    a : float
        Value of a for the smooth box distribution.
    b : float
        Value of b for the smooth box distribution.
    """
    # Normalization constant so that integral over pdf is 1.
    C = 1.0 + (b - a) / (sigma * np.sqrt(2 * np.pi))

    if x > b:
        return sp.stats.norm.sf(x, loc=b, scale=sigma) / C

    if x >= a:
        return (0.5 + (b - x) / (sigma * np.sqrt(2 * np.pi))) / C

    return (C - 1.0 + sp.stats.norm.sf(x, loc=a, scale=sigma)) / C


def smoothbox_student_t_cdf(
    x: float, df: float, sigma: float, a: float, b: float
):
    """
    Compute the value of the cumulative distribution function
    for the smooth box student t distribution.

    Parameters
    ==========
    x : float
        The point where we want the value of the cdf.
    df : float
        The degrees of freedom of the distribution.
    sigma : float
        Value of sigma for the distribution.
    a : float
        Value of a for the distribution.
    b : float
        Value of b for the distribution.
    """
    # Normalization constant so that integral over pdf is 1. The ratio of
    # the two gamma functions is of order sqrt(df) but each of them
    # overflows from a df of about 340, so take it through their logs.
    c = np.exp(
        sp.special.gammaln(0.5 * (df + 1)) - sp.special.gammaln(0.5 * df)
    ) / (sigma * np.sqrt(df * np.pi))
    C = 1.0 + (b - a) * c

    if x < a:
        return sp.stats.t.cdf(x, df, loc=a, scale=sigma) / C

    if x <= b:
        return (0.5 + (x - a) * c) / C

    return (C - 1.0 + sp.stats.t.cdf(x, df, loc=b, scale=sigma)) / C


def smoothbox_student_t_sf(
    x: float, df: float, sigma: float, a: float, b: float
):
    """
    Compute the value of the survival function (one minus the cumulative
    distribution function) for the smooth box student t distribution.
    Above the box it keeps the probability of the upper tail, which one
    minus :func:`smoothbox_student_t_cdf` loses where the cumulative
    distribution function rounds to one.

    Parameters
    ==========
    x : float
        The point where we want the value of the survival function.
    df : float
        The degrees of freedom of the distribution.
    sigma : float
        Value of sigma for the distribution.
    a : float
        Value of a for the distribution.
    b : float
        Value of b for the distribution.
    """
    # Normalization constant so that integral over pdf is 1, as in
    # `smoothbox_student_t_cdf`.
    c = np.exp(
        sp.special.gammaln(0.5 * (df + 1)) - sp.special.gammaln(0.5 * df)
    ) / (sigma * np.sqrt(df * np.pi))
    C = 1.0 + (b - a) * c

    if x > b:
        return sp.stats.t.sf(x, df, loc=b, scale=sigma) / C

    if x >= a:
        return (0.5 + (b - x) * c) / C

    return (C - 1.0 + sp.stats.t.sf(x, df, loc=a, scale=sigma)) / C


def smoothbox_ppf(q: float, sigma: float, a: float, b: float):
    """
    Compute the value of the percent point function for
    the smooth box distribution.

    Parameters
    ==========
    q : float
        The quantile where we want the value of the ppf.
    sigma : float
        Value of sigma for the smooth box distribution.
    a : float
        Value of a for the smooth box distribution.
    b : float
        Value of b for the smooth box distribution.
    """
    # Normalization constant so that integral over pdf is 1.
    C = 1.0 + (b - a) / (sigma * np.sqrt(2 * np.pi))

    if q < 0.5 / C:
        return sp.stats.norm.ppf(C * q, loc=a, scale=sigma)

    if q <= (C - 0.5) / C:
        return (q * C - 0.5) * sigma * np.sqrt(2 * np.pi) + a

    return sp.stats.norm.ppf(C * q - (C - 1), loc=b, scale=sigma)


def smoothbox_isf(p: float, sigma: float, a: float, b: float):
    """
    Compute the value of the inverse survival function for the smooth box
    distribution: the point above which the distribution puts the
    probability ``p``. It is the percent point function of ``1 - p``,
    without the loss of precision of that difference for a small ``p``.

    Parameters
    ==========
    p : float
        The probability of the upper tail.
    sigma : float
        Value of sigma for the smooth box distribution.
    a : float
        Value of a for the smooth box distribution.
    b : float
        Value of b for the smooth box distribution.
    """
    # Normalization constant so that integral over pdf is 1.
    C = 1.0 + (b - a) / (sigma * np.sqrt(2 * np.pi))

    if p < 0.5 / C:
        return sp.stats.norm.isf(C * p, loc=b, scale=sigma)

    if p <= (C - 0.5) / C:
        return b - (p * C - 0.5) * sigma * np.sqrt(2 * np.pi)

    return sp.stats.norm.isf(C * p - (C - 1), loc=a, scale=sigma)


def smoothbox_student_t_ppf(
    q: float, df: float, sigma: float, a: float, b: float
):
    """
    Compute the value of the percent point function for
    the smooth box student t distribution.

    Parameters
    ==========
    q : float
        The quantile where we want the value of the ppf.
    df : float
        The degrees of freedom of the distribution.
    sigma : float
        Value of sigma for the distribution.
    a : float
        Value of a for the distribution.
    b : float
        Value of b for the distribution.
    """
    # Normalization constant so that integral over pdf is 1. The ratio of
    # the two gamma functions is of order sqrt(df) but each of them
    # overflows from a df of about 340, so take it through their logs.
    c = np.exp(
        sp.special.gammaln(0.5 * (df + 1)) - sp.special.gammaln(0.5 * df)
    ) / (sigma * np.sqrt(df * np.pi))
    C = 1.0 + (b - a) * c

    if q < 0.5 / C:
        return sp.stats.t.ppf(C * q, df, loc=a, scale=sigma)

    if q <= (C - 0.5) / C:
        return (q * C - 0.5) / c + a

    return sp.stats.t.ppf(C * q - (C - 1), df, loc=b, scale=sigma)


def smoothbox_student_t_isf(
    p: float, df: float, sigma: float, a: float, b: float
):
    """
    Compute the value of the inverse survival function for the smooth box
    student t distribution: the point above which the distribution puts
    the probability ``p``. It is the percent point function of ``1 - p``,
    without the loss of precision of that difference for a small ``p``.

    Parameters
    ==========
    p : float
        The probability of the upper tail.
    df : float
        The degrees of freedom of the distribution.
    sigma : float
        Value of sigma for the distribution.
    a : float
        Value of a for the distribution.
    b : float
        Value of b for the distribution.
    """
    # Normalization constant so that integral over pdf is 1, as in
    # `smoothbox_student_t_ppf`.
    c = np.exp(
        sp.special.gammaln(0.5 * (df + 1)) - sp.special.gammaln(0.5 * df)
    ) / (sigma * np.sqrt(df * np.pi))
    C = 1.0 + (b - a) * c

    if p < 0.5 / C:
        return sp.stats.t.isf(C * p, df, loc=b, scale=sigma)

    if p <= (C - 0.5) / C:
        return b - (p * C - 0.5) / c

    return sp.stats.t.isf(C * p - (C - 1), df, loc=a, scale=sigma)
