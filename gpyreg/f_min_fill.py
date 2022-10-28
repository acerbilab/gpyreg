"""Module for helper functions for GP training."""
import operator
import re
import warnings

import numpy as np
import scipy as sp


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
            np.random.shuffle(S.T)
        elif design == "rand":
            S = np.random.uniform(size=(N - N0, n_vars))
        else:
            raise ValueError(
                "Unknown design: got "
                + design
                + ' and expected either "sobol" or "rand"'
            )
        sX = np.zeros((N - N0, n_vars))

        # If a prior is specified use that.
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

                        # Half of all starting points from inside the
                        # plausible box
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
                if df == 0:
                    cdf_lb = smoothbox_cdf(LB[i], sigma, a, b)
                    cdf_ub = smoothbox_cdf(UB[i], sigma, a, b)
                    S_scaled = cdf_lb + (cdf_ub - cdf_lb) * S[:, i]
                    for j in range(0, (N - N0)):
                        sX[j, i] = smoothbox_ppf(S_scaled[j], sigma, a, b)
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
                if df == 0:
                    cdf_lb = sp.stats.norm.cdf((LB[i] - mu) / sigma)
                    cdf_ub = sp.stats.norm.cdf((UB[i] - mu) / sigma)
                    S_scaled = cdf_lb + (cdf_ub - cdf_lb) * S[:, i]
                    sX[:, i] = sp.stats.norm.ppf(S_scaled) * sigma + mu
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
    """
    Inverse of cumulative distribution function of mixture of uniform
    distributions. The mixture is:
    .. math::
    w \text{Uniform}(B[1], B[2]) +
    \frac{1 - w}{2} (\text{Uniform}(B[0], B[1]) + \text{Uniform}(B[2], B[3]))

    Parameters
    ----------
    p : ndarray
        1D array of cumulative function values.
    B : ndarray, list
        1D array or list containing [LB, PLB, PUB, UB].
    w : float
        The coefficient for mixture of uniform distributions.
        :math: `0 \leq w \leq 1`.

    Returns
    -------
    x : ndarray
        1D array of samples corresponding to `p`.
    """
    assert B[0] <= B[1] <= B[2] <= B[3]
    assert 0 <= w <= 1
    x = np.zeros(p.shape)
    L = B[3] - B[0] + B[1] - B[2]

    if w == 1:
        x = p * (B[2] - B[1]) + B[1]
        return x

    if L == 0:
        # Degenerate to mixture of delta and uniform distributions
        i1 = p <= (1 - w) / 2
        x[i1] = B[0]

        if w != 0:
            i2 = (p <= (1 - w) / 2 + w) & ~i1
            x[i2] = (p[i2] - (1 - w) / 2) * (B[2] - B[1]) / w + B[1]

        i3 = p > (1 - w) / 2 + w
        x[i3] = B[3]
        return x

    # First step
    i1 = p <= (1 - w) * (B[1] - B[0]) / L
    x[i1] = B[0] + p[i1] * L / (1 - w)

    # Second step
    i2 = (p <= (1 - w) * (B[1] - B[0]) / L + w) & ~i1
    if w != 0:
        x[i2] = (p[i2] - (1 - w) * (B[1] - B[0]) / L) * (B[2] - B[1]) / w + B[
            1
        ]

    # Third step
    i3 = p > (1 - w) * (B[1] - B[0]) / L + w
    x[i3] = (p[i3] - w - (1 - w) * (B[1] - B[0]) / L) * L / (1 - w) + B[2]

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
    # Normalization constant so that integral over pdf is 1.
    c = sp.special.gamma(0.5 * (df + 1)) / (
        sp.special.gamma(0.5 * df) * sigma * np.sqrt(df * np.pi)
    )
    C = 1.0 + (b - a) * c

    if x < a:
        return sp.stats.t.cdf(x, df, loc=a, scale=sigma) / C

    if x <= b:
        return (0.5 + (x - a) * c) / C

    return (C - 1.0 + sp.stats.t.cdf(x, df, loc=b, scale=sigma)) / C


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
    # Normalization constant so that integral over pdf is 1.
    c = sp.special.gamma(0.5 * (df + 1)) / (
        sp.special.gamma(0.5 * df) * sigma * np.sqrt(df * np.pi)
    )
    C = 1.0 + (b - a) * c

    if q < 0.5 / C:
        return sp.stats.t.ppf(C * q, df, loc=a, scale=sigma)

    if q <= (C - 0.5) / C:
        return (q * C - 0.5) / c + a

    return sp.stats.t.ppf(C * q - (C - 1), df, loc=b, scale=sigma)
