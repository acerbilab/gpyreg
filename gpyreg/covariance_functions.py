"""Module for different covariance functions used by Gaussian Processes."""

import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


def _target_spread(y: np.ndarray):
    """Return the range and the standard deviation of the training targets.

    The recommended bounds of every component take the scale of the
    targets from these two numbers. Targets that are all equal have
    neither: the range is zero, its logarithm is ``-inf``, and the bounds
    built from it are unusable (the output scale of a kernel gets the pair
    ``(-inf, -inf)``, which L-BFGS-B refuses). Such a training set is
    given the scale of a unit range instead, with a warning, as both
    gpyreg and gplite already do for a single target. The other statistics
    of the targets, their location among them, are left alone.

    Parameters
    ----------
    y : ndarray
        The training targets.

    Returns
    -------
    height : float
        The range of the targets, or one where they are all equal and
        finite. It is NaN where a target is NaN or where every target is
        the same infinity, and is returned as it is.
    y_std : float
        Their standard deviation, or that of a unit range where they are
        all equal.
    """
    height = np.max(y) - np.min(y)
    if height != 0:
        return height, np.std(y, ddof=1)

    warnings.warn(
        "The training targets are all equal, so they have no scale for "
        "the recommended bounds to take: a range of one is assumed "
        "instead."
    )
    unit_range = np.array([0.0, 1.0])
    return (
        np.max(unit_range) - np.min(unit_range),
        np.std(unit_range, ddof=1),
    )


def _input_spread(X: np.ndarray):
    """Return the width and the standard deviation of each column of the
    training inputs.

    The recommended bounds of a length scale, and of the scale of
    :class:`gpyreg.mean_functions.NegativeQuadratic`, are built from the
    logarithm of the width of a column, and their starting value from the
    logarithm of its standard deviation. A column without spread, as every
    column of a single input is, has a width of zero, whose logarithm is
    ``-inf``: the bounds built from it are ``-inf``, which
    :meth:`gpyreg.GP.get_recommended_bounds` refuses unless the caller
    gives a finite lower bound, so the callers take these logarithms
    without NumPy's warning on a logarithm of zero. The standard deviation
    is the sample one (``ddof=1``): zero for such a column of two inputs
    or more, and NaN for a single input. The NaN is returned here without
    NumPy's warnings on a sample of one, and the starting value built from
    it falls back to the middle of the plausible bounds.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        The training inputs.

    Returns
    -------
    width : ndarray, shape (D,)
        The range of each column.
    x_std : ndarray, shape (D,)
        The sample standard deviation of each column, NaN in every column
        when ``N`` is one.
    """
    width = np.max(X, axis=0) - np.min(X, axis=0)
    if X.shape[0] > 1:
        return width, np.std(X, axis=0, ddof=1)
    return width, np.full(X.shape[1], np.nan)


def _validate_periods(periods):
    """Return the periods given to a kernel as a float array, or ``None``.

    Parameters
    ----------
    periods : array_like or None
        One period per input dimension: a positive number, or ``np.inf``
        for a dimension that is not periodic.

    Returns
    -------
    periods : ndarray or None
        A float copy of the periods, or ``None`` where none was given or
        every period is infinite, so that no dimension is periodic.

    Raises
    ------
    ValueError
        Raised when the periods are not a one-dimensional array with at
        least one entry, or when one of them is NaN, zero or negative.
    """
    if periods is None:
        return None
    periods = np.array(periods, dtype=float)
    if periods.ndim != 1 or periods.size == 0:
        raise ValueError(
            "The periods must be a one-dimensional array with one entry "
            "per input dimension."
        )
    if np.any(np.isnan(periods)) or np.any(periods <= 0):
        raise ValueError(
            "Each period must be a positive number, or np.inf for a "
            "dimension that is not periodic."
        )
    if np.all(np.isinf(periods)):
        return None
    return periods


def _check_periods_match(periods, D):
    """Raise ``ValueError`` when a kernel's periods are not one per input
    dimension."""
    if periods is not None and periods.size != D:
        raise ValueError(
            f"The covariance function has {periods.size} periods, one per "
            f"input dimension, but the inputs have {D} dimensions."
        )


def _on_circle(x, period):
    """Return the coordinates along a periodic dimension mapped onto a
    circle of circumference ``period``.

    The squared Euclidean distance between two mapped points is the squared
    chord ``(p / pi)**2 * sin(pi * delta / p)**2`` of the difference
    ``delta`` of their coordinates, for a period ``p``. Each coordinate is
    first wrapped, exactly, into ``(-p / 2, p / 2]``: ``fmod`` is exact,
    and so is the shift by one period that follows it, of a value at least
    half a period from zero (Sterbenz's lemma). Coordinates a whole number
    of periods apart are therefore mapped to the same point, whose squared
    distance is exactly zero. Where the half angle ``phi = pi * x / p`` is
    small, the second coordinate is ``x`` to second order, with its
    relative precision, and the first one is of order ``x**2 / p``, so
    that a period far beyond the spread of the inputs leaves their
    differences as they are.

    Parameters
    ----------
    x : ndarray, shape (N,)
        The coordinates along the dimension.
    period : float
        The finite period of the dimension.

    Returns
    -------
    circle : ndarray, shape (N, 2)
        ``R * (1 - cos(2 * phi), sin(2 * phi))`` with ``R = p / (2 * pi)``,
        computed as ``(p / pi) * (sin(phi)**2, sin(phi) * cos(phi))``: a
        point of the circle of radius ``R`` through the origin, at the
        origin for the angle zero.
    """
    half = period / 2
    r = np.fmod(x, period)
    r = np.where(r > half, r - period, r)
    r = np.where(r <= -half, r + period, r)
    phi = np.pi / period * r
    s = np.sin(phi)
    # The radius multiplies the unit-circle terms here, and the caller's
    # scale multiplies the result: p / pi stays finite for any finite
    # period, where the product of p and a scale could overflow and give
    # NaN against a zero sine.
    radius = period / np.pi
    return np.stack((radius * (s * s), radius * (s * np.cos(phi))), axis=1)


def _scaled_coordinates(X, scale, periods):
    """Return coordinates whose squared Euclidean distances are the scaled
    squared distances of an ARD kernel with periods.

    A dimension without a period gives its scaled coordinate, a periodic
    one the two scaled coordinates of :func:`_on_circle`.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        The points.
    scale : ndarray, shape (D,)
        The scale of each dimension.
    periods : ndarray, shape (D,)
        The period of each dimension, or ``np.inf``.

    Returns
    -------
    coordinates : ndarray, shape (N, D + P)
        The coordinates of the dimensions without a period, in their order,
        followed by those of the ``P`` periodic ones.
    """
    periodic = np.isfinite(periods)
    other = np.logical_not(periodic)
    columns = [X[:, other] * scale[other]]
    for d in np.flatnonzero(periodic):
        columns.append(scale[d] * _on_circle(X[:, d], periods[d]))
    return np.hstack(columns)


def _scaled_sq_diff(x, scale, period):
    """Return the scaled squared differences along one input dimension.

    The term of one dimension in the squared distance of an ARD kernel is
    ``scale**2 * delta**2``, where ``delta`` is the difference of the two
    coordinates. Along a dimension with a finite period ``p``, ``delta**2``
    is replaced by the squared chord ``(p / pi)**2 * sin(pi * delta / p)**2``,
    the squared distance between the two points mapped onto a circle of
    circumference ``p`` by :func:`_on_circle`, zero at every multiple of the
    period.

    Parameters
    ----------
    x : ndarray, shape (N,)
        The coordinates of the points.
    scale : float
        The scale of the dimension, the inverse of its length scale times
        any factor of the kernel.
    period : float
        The period of the dimension, or ``np.inf``.

    Returns
    -------
    sq_diff : ndarray, shape (N, N)
        The scaled squared differences, exactly symmetric with zeros on
        the diagonal.
    """
    if not np.isfinite(period):
        a = np.reshape(scale * x, (-1, 1))
    else:
        a = scale * _on_circle(x, period)
    return cdist(a, a, "sqeuclidean")


def _scaled_sq_dist(X, X_star, scale, periods):
    """Return the scaled squared distances of an ARD kernel with periods.

    The distance is the sum over the input dimensions of
    ``scale[d]**2 * delta[d]**2``, with the squared chord of
    :func:`_scaled_sq_diff` in place of ``delta[d]**2`` along a dimension
    with a finite period. It is computed as one squared Euclidean distance
    between the points' :func:`_scaled_coordinates`, whose trigonometric
    functions take one evaluation per point and periodic dimension.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        The first set of points.
    X_star : ndarray, shape (M, D), optional
        The second set of points; ``X`` again when ``None``.
    scale : ndarray, shape (D,)
        The scale of each dimension.
    periods : ndarray, shape (D,)
        The period of each dimension, or ``np.inf``.

    Returns
    -------
    sq_dist : ndarray, shape (N, M)
        The scaled squared distances, exactly symmetric with zeros on the
        diagonal where ``X_star`` is ``None``.
    """
    a = _scaled_coordinates(X, scale, periods)
    b = a if X_star is None else _scaled_coordinates(X_star, scale, periods)
    return cdist(a, b, "sqeuclidean")


def _pairwise(a, metric):
    """Return ``squareform(pdist(a, metric))``, bit for bit: the distances
    among the scaled training inputs of an ARD kernel without periods.

    For float64 coordinates it is ``cdist(a, a, metric)``, which sums the
    same differences in the same order and skips ``pdist``'s wrapper and the
    ``squareform`` pass, with its diagonal set to zero, as ``squareform``
    sets it (``cdist``'s is NaN at an infinite coordinate). Coordinates of
    another type take ``pdist`` itself, whose arithmetic for them depends on
    the SciPy version.
    """
    if a.dtype != np.float64:
        return squareform(pdist(a, metric))
    d = cdist(a, a, metric)
    np.fill_diagonal(d, 0.0)
    return d


def _sq_diff(x):
    """Return ``squareform(pdist(x[:, None], "sqeuclidean"))``: the squared
    differences of the coordinates ``x`` of one dimension."""
    return squareform(pdist(np.reshape(x, (-1, 1)), "sqeuclidean"))


def _sq_diffs(rows, out):
    """Write :func:`_sq_diff` of each dimension into ``out``, bit for bit,
    for the gradients of an ARD kernel without periods, where it can.

    The squared Euclidean distance of one coordinate is the square of its
    one difference, and a difference and its negative have the same square:
    for float64 coordinates, one broadcast computes every dimension, without
    the calls and the copies of ``pdist`` and ``squareform`` for each, and
    the diagonal is set to zero, as ``squareform`` sets it (the difference
    is NaN at an infinite coordinate). Coordinates of another type are left
    to :func:`_sq_diff`, whose arithmetic for them depends on the SciPy
    version.

    Parameters
    ----------
    rows : list of ndarray, shape (N,)
        The scaled coordinates of each dimension, as the kernel's formula
        computes them.
    out : ndarray, shape (D, N, N)
        The float64 array that receives the squared differences.

    Returns
    -------
    written : bool
        Whether ``out`` holds the squared differences, with
        ``out[d, i, j] = (rows[d][i] - rows[d][j])**2``: ``False``, and
        ``out`` untouched, for coordinates other than float64.
    """
    if rows[0].dtype != np.float64:
        return False
    a = np.stack(rows)
    # pdist, which computes in C, warns of no overflow.
    with np.errstate(all="ignore"):
        np.subtract(a[:, :, None], a[:, None, :], out=out)
        np.multiply(out, out, out=out)
    diagonal = np.arange(a.shape[1])
    out[:, diagonal, diagonal] = 0.0
    return True


class AbstractKernel(ABC):
    """Abstract base class for covariance kernels."""

    #: The period of each input dimension, ``np.inf`` for a dimension that
    #: is not periodic, or ``None`` for a kernel that is not periodic along
    #: any dimension.
    periods = None

    @abstractmethod
    def compute(
        self,
        hyp: np.ndarray,
        X: np.ndarray,
        X_star: np.ndarray = None,
        compute_diag: bool = False,
        compute_grad: bool = False,
    ):
        """
        Compute the covariance matrix for given training points
        and test points.

        Parameters
        ----------
        hyp : ndarray, shape (cov_N,)
            A 1D array of hyperparameters, where ``cov_N`` is
            the number of hyperparameters.
        X : ndarray, shape (N, D)
            A 2D array where each row is a training point.
        X_star : ndarray, shape (M, D), optional
            A 2D array where each row is a test point. If this is not
            given, the self-covariance matrix is being computed.
        compute_diag : bool, defaults to False
            Whether to only compute the diagonal of the self-covariance
            matrix.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to the
            hyperparameters.

        Returns
        -------
        K : ndarray
            The covariance matrix which is by default of shape ``(N, N)``. If
            ``compute_diag = True`` the shape is ``(N, 1)``.
        dK : ndarray, shape (N, N, cov_N), optional
            The gradient of the covariance matrix with respect to the
            hyperparameters.

        Raises
        ------
        ValueError
            Raised when `hyp` has not the expected number of hyperparameters.
        ValueError
            Raised when `hyp` is not an 1D array but of higher dimension.
        ValueError
            Raised when `compute_diag` and `compute_grad` are both True:
            the gradient is available for the full covariance matrix only.
        ValueError
            Raised when the kernel has periods and their number differs
            from the number of columns of `X`.
        """

    def hyperparameter_count(self, D: int):
        """
        Return the number of hyperparameters this covariance function has.

        Parameters
        ----------
        D : int
            The dimensionality of the kernel.

        Returns
        -------
        count : int
            The number of hyperparameters.
        """
        return D + 1

    def hyperparameter_info(self, D: int):
        """
        Return information on the names of hyperparameters for setting
        them in other parts of the program.

        Parameters
        ----------
        D : int
            The dimensionality of the kernel.

        Returns
        -------
        hyper_info : array_like
            A list of tuples of hyperparameter names and their number,
            in the order they are in the hyperparameter array.
        """
        return [
            ("covariance_log_lengthscale", D),
            ("covariance_log_outputscale", 1),
        ]

    def get_bounds_info(self, X: np.ndarray, y: np.ndarray):
        """
        Return information on the lower, upper, plausible lower
        and plausible upper bounds of the hyperparameters of this
        covariance function.

        Parameters
        ----------
        X : ndarray, shape (N, D)
            A 2D array where each row is a test point.
        y : ndarray, shape (N, 1)
            A 2D array where each row is a test target.

        Returns
        -------
        cov_bound_info: dict
            A dictionary containing the bound info with the following elements:

            **LB** : np.ndarray, shape (cov_N, 1)
                    The lower bounds of the hyperparameters.
            **UB** : np.ndarray, shape (cov_N, 1)
                    The upper bounds of the hyperparameters.
            **PLB** : np.ndarray, shape (cov_N, 1)
                    The plausible lower bounds of the hyperparameters.
            **PUB** : np.ndarray, shape (cov_N, 1)
                    The plausible upper bounds of the hyperparameters.
            **x0** : np.ndarray, shape (cov_N, 1)
                    The plausible starting point.

            where ``cov_N`` is the number of hyperparameters.
        """
        cov_N = self.hyperparameter_count(X.shape[1])
        return _bounds_info_helper(cov_N, X, y)


class SquaredExponential(AbstractKernel):
    """
    Squared exponential kernel.

    Parameters
    ----------
    periods : array_like, shape (D,), optional
        The period of each input dimension along which the kernel is
        periodic, and ``np.inf`` for each dimension along which it is not.
        Along a dimension of period ``p``, the squared difference
        ``delta**2`` of two coordinates is replaced by the squared chord
        ``(p / pi)**2 * sin(pi * delta / p)**2``, the squared distance
        between the two points mapped onto a circle of circumference
        ``p``. It is zero at every multiple of the period and equals
        ``delta**2`` to second order in ``delta``, so that the length scale
        keeps the units of the input. The periods are fixed constants, not
        hyperparameters. ``None``, the default, or periods that are all
        infinite give the non-periodic kernel. A period that is NaN, zero
        or negative, or periods that are not a one-dimensional array, raise
        ``ValueError``.
    """

    def __init__(self, periods: np.ndarray = None):
        self.periods = _validate_periods(periods)

    # Overriding abstract method
    def compute(
        self,
        hyp: np.ndarray,
        X: np.ndarray,
        X_star: np.ndarray = None,
        compute_diag: bool = False,
        compute_grad: bool = False,
    ):

        N, D = X.shape
        cov_N = self.hyperparameter_count(D)

        if hyp.size != cov_N:
            raise ValueError(
                f"Expected {cov_N} covariance function hyperparameters, "
                f"{hyp.size} passed instead."
            )
        if hyp.ndim != 1:
            raise ValueError(
                "Covariance function output is available only for "
                "one-sample hyperparameter inputs."
            )
        if compute_diag and compute_grad:
            raise ValueError(
                "compute_diag and compute_grad cannot both be True: the "
                "gradient is available for the full covariance matrix "
                "only."
            )

        _check_periods_match(self.periods, D)

        ell = np.exp(hyp[0:D])
        sf2 = np.exp(2 * hyp[D])

        if X_star is None and compute_diag:
            # The diagonal is sf2 * exp(-0 / 2) = sf2 exactly.
            return np.full((N, 1), sf2)

        if X_star is None:
            if compute_diag:
                tmp = np.zeros((N, 1))
            elif self.periods is not None:
                tmp = _scaled_sq_dist(X, None, 1.0 / ell, self.periods)
            else:
                # cdist(Xs, Xs) equals squareform(pdist(Xs)) bit for bit
                # (the same per-dimension differences summed in the same
                # order, exactly symmetric, exact zeros on the diagonal)
                # and skips pdist's wrapper and the squareform pass.
                Xs = X / ell
                tmp = cdist(Xs, Xs, "sqeuclidean")
        elif self.periods is not None:
            tmp = _scaled_sq_dist(X, X_star, 1.0 / ell, self.periods)
        else:
            tmp = cdist(X / ell, X_star / ell, "sqeuclidean")

        # K = sf2 * exp(-tmp / 2), computed in tmp, a fresh array: the
        # operations of the expression, in its order, without its
        # temporaries.
        K = np.negative(tmp, out=tmp)
        K /= 2
        np.exp(K, out=K)
        K *= sf2

        if compute_grad:
            if X_star is not None:
                raise ValueError(
                    "X_star should be None when compute_grad is True."
                )
            # Every entry of dK is written below.
            dK = np.empty((cov_N, N, N))
            # Gradient of cov length scales
            if self.periods is not None:
                for i in range(0, D):
                    dK[i, :, :] = K * _scaled_sq_diff(
                        X[:, i], 1.0 / ell[i], self.periods[i]
                    )
            else:
                rows = [X[:, i] / ell[i] for i in range(0, D)]
                if _sq_diffs(rows, out=dK[0:D]):
                    np.multiply(K, dK[0:D], out=dK[0:D])
                else:
                    for i in range(0, D):
                        dK[i, :, :] = K * _sq_diff(rows[i])
            # Gradient of cov output scale.
            np.multiply(2, K, out=dK[D])
            return K, dK.transpose(1, 2, 0)

        return K


class Matern(AbstractKernel):
    """
    Matern kernel.

    Parameters
    ----------
    degree : {1, 3, 5}
        The degree of the Matern kernel.

        Currently the only supported degrees are 1, 3, 5, and if
        some other degree is provided a ``ValueError`` exception is raised.
    periods : array_like, shape (D,), optional
        The period of each input dimension along which the kernel is
        periodic, and ``np.inf`` for each dimension along which it is not.
        Along a dimension of period ``p``, the squared difference
        ``delta**2`` of two coordinates is replaced by the squared chord
        ``(p / pi)**2 * sin(pi * delta / p)**2``, the squared distance
        between the two points mapped onto a circle of circumference
        ``p``. It is zero at every multiple of the period and equals
        ``delta**2`` to second order in ``delta``, so that the length scale
        keeps the units of the input. The periods are fixed constants, not
        hyperparameters. ``None``, the default, or periods that are all
        infinite give the non-periodic kernel. A period that is NaN, zero
        or negative, or periods that are not a one-dimensional array, raise
        ``ValueError``.
    """

    def __init__(self, degree: int, periods: np.ndarray = None):
        if degree not in (1, 3, 5):
            raise ValueError(
                "Only degrees 1, 3 and 5 are supported for the "
                "Matern covariance function."
            )

        self.degree = degree
        if self.degree == 1:
            self.f = lambda t: 1
            self.df = lambda t: 1 / t
        elif self.degree == 3:
            self.f = lambda t: 1 + t
            self.df = lambda t: 1
        else:
            self.f = lambda t: 1 + t * (1 + t / 3)
            self.df = lambda t: (1 + t) / 3
        self.periods = _validate_periods(periods)

    # Overriding abstract method
    def compute(
        self,
        hyp: np.ndarray,
        X: np.ndarray,
        X_star: np.ndarray = None,
        compute_diag: bool = False,
        compute_grad: bool = False,
    ):

        N, D = X.shape
        cov_N = self.hyperparameter_count(D)

        if hyp.size != cov_N:
            raise ValueError(
                f"Expected {cov_N} covariance function hyperparameters, "
                f"{hyp.size} passed instead."
            )
        if hyp.ndim != 1:
            raise ValueError(
                "Covariance function output is available only for "
                "one-sample hyperparameter inputs."
            )
        if compute_diag and compute_grad:
            raise ValueError(
                "compute_diag and compute_grad cannot both be True: the "
                "gradient is available for the full covariance matrix "
                "only."
            )

        _check_periods_match(self.periods, D)

        ell = np.exp(hyp[0:D])
        sf2 = np.exp(2 * hyp[D])

        if X_star is None:
            if compute_diag:
                tmp = np.zeros((N, 1))
            elif self.periods is not None:
                tmp = np.sqrt(
                    _scaled_sq_dist(
                        X, None, np.sqrt(self.degree) / ell, self.periods
                    )
                )
            else:
                tmp = _pairwise(
                    X @ np.diag(np.sqrt(self.degree) / ell), "euclidean"
                )
        elif self.periods is not None:
            tmp = np.sqrt(
                _scaled_sq_dist(
                    X, X_star, np.sqrt(self.degree) / ell, self.periods
                )
            )
        else:
            a = X @ np.diag(np.sqrt(self.degree) / ell)
            b = X_star @ np.diag(np.sqrt(self.degree) / ell)
            tmp = cdist(a, b)

        K = sf2 * self.f(tmp) * np.exp(-tmp)

        if compute_grad:
            if X_star is not None:
                raise ValueError(
                    "X_star should be None when compute_grad is True."
                )
            # Every entry of dK is written below.
            dK = np.empty((cov_N, N, N))
            # The factor of the length scales' gradients that is the same
            # for every dimension. At d=1 it is infinite where two inputs
            # coincide, the diagonal among them, since df(0) = 1 / 0.
            with np.errstate(all="ignore"):
                dK_factor = sf2 * (self.df(tmp) * np.exp(-tmp))
            # Where two inputs share the i-th coordinate, or lie a multiple
            # of its period apart, the kernel does not depend on that
            # length scale, so the derivative is zero. Where they coincide,
            # the d=1 factor is infinite and inf * 0 = NaN, which would
            # poison the gradient of the marginal likelihood through the
            # whole diagonal, so the product is taken as the zero it is.
            if self.periods is not None:
                for i in range(0, D):
                    Ki = _scaled_sq_diff(
                        X[:, i],
                        np.sqrt(self.degree) / ell[i],
                        self.periods[i],
                    )
                    with np.errstate(all="ignore"):
                        dK[i, :, :] = np.where(Ki > 0, dK_factor * Ki, 0.0)
            else:
                rows = [
                    np.sqrt(self.degree) / ell[i] * X[:, i]
                    for i in range(0, D)
                ]
                Ks = dK[0:D]
                if _sq_diffs(rows, out=Ks):
                    # np.where(Ki > 0, dK_factor * Ki, 0.0) for every
                    # dimension at once, in place.
                    positive = Ks > 0
                    with np.errstate(all="ignore"):
                        np.multiply(dK_factor, Ks, out=Ks, where=positive)
                    np.logical_not(positive, out=positive)
                    Ks[positive] = 0.0
                else:
                    for i in range(0, D):
                        Ki = _sq_diff(rows[i])
                        with np.errstate(all="ignore"):
                            dK[i, :, :] = np.where(Ki > 0, dK_factor * Ki, 0.0)
            # Gradient of cov output scale
            np.multiply(2, K, out=dK[D])
            return K, dK.transpose(1, 2, 0)

        return K


class RationalQuadraticARD(AbstractKernel):
    """
    Rational Quadratic ARD kernel.

    Parameters
    ----------
    periods : array_like, shape (D,), optional
        The period of each input dimension along which the kernel is
        periodic, and ``np.inf`` for each dimension along which it is not.
        Along a dimension of period ``p``, the squared difference
        ``delta**2`` of two coordinates is replaced by the squared chord
        ``(p / pi)**2 * sin(pi * delta / p)**2``, the squared distance
        between the two points mapped onto a circle of circumference
        ``p``. It is zero at every multiple of the period and equals
        ``delta**2`` to second order in ``delta``, so that the length scale
        keeps the units of the input. The periods are fixed constants, not
        hyperparameters. ``None``, the default, or periods that are all
        infinite give the non-periodic kernel. A period that is NaN, zero
        or negative, or periods that are not a one-dimensional array, raise
        ``ValueError``.
    """

    def __init__(self, periods: np.ndarray = None):
        self.periods = _validate_periods(periods)

    def hyperparameter_count(self, D: int):
        return D + 2

    def hyperparameter_info(self, D: int):
        return [
            ("covariance_log_lengthscale", D),
            ("covariance_log_outputscale", 1),
            ("covariance_log_shape", 1),
        ]

    def compute(
        self,
        hyp: np.ndarray,
        X: np.ndarray,
        X_star: np.ndarray = None,
        compute_diag: bool = False,
        compute_grad: bool = False,
    ):

        N, D = X.shape
        cov_N = self.hyperparameter_count(D)

        if hyp.size != cov_N:
            raise ValueError(
                f"Expected {cov_N} covariance function hyperparameters, "
                f"{hyp.size} passed instead."
            )
        if hyp.ndim != 1:
            raise ValueError(
                "Covariance function output is available only for "
                "one-sample hyperparameter inputs."
            )
        if compute_diag and compute_grad:
            raise ValueError(
                "compute_diag and compute_grad cannot both be True: the "
                "gradient is available for the full covariance matrix "
                "only."
            )

        _check_periods_match(self.periods, D)

        ell = np.exp(hyp[0:D])
        sf2 = np.exp(2 * hyp[D])
        alpha = np.exp(hyp[D + 1])

        if X_star is None:
            if compute_diag:
                tmp = np.zeros((N, 1))
            elif self.periods is not None:
                tmp = _scaled_sq_dist(X, None, 1.0 / ell, self.periods)
            else:
                tmp = _pairwise(X @ np.diag(1.0 / ell), "sqeuclidean")
        elif self.periods is not None:
            tmp = _scaled_sq_dist(X, X_star, 1.0 / ell, self.periods)
        else:
            a = X @ np.diag(1.0 / ell)
            b = X_star @ np.diag(1.0 / ell)
            tmp = cdist(a, b, "sqeuclidean")

        # The kernel and its gradient are computed in tmp, a fresh array,
        # and in arrays of their own: the operations of the expressions in
        # the comments, in their order, without their temporaries. The
        # powers take the operator, as the expressions do, since NumPy
        # takes an exponent of -1 (alpha = 1) as a reciprocal.
        if not compute_grad:
            # K = sf2 * (1 + 0.5 * tmp / alpha) ** (-alpha)
            tmp *= 0.5
            tmp /= alpha
            tmp += 1
            tmp **= -alpha
            tmp *= sf2
            return tmp

        if X_star is not None:
            raise ValueError(
                "X_star should be None when compute_grad is True."
            )

        # M = 1 + 0.5 * tmp / alpha, K = sf2 * M ** (-alpha); tmp is kept
        # for the gradient of the shape.
        M = np.multiply(0.5, tmp)
        M /= alpha
        M += 1
        K = M ** (-alpha)
        K *= sf2

        # Every entry of dK is written below.
        dK = np.empty((cov_N, N, N))

        # Gradient with respect to the length scales, whose factor
        # sf2 * M ** (-alpha - 1) is the same for every dimension.
        with np.errstate(all="ignore"):
            dK_factor = M ** (-alpha - 1)
            dK_factor *= sf2
        if self.periods is not None:
            for i in range(0, D):
                Ki = _scaled_sq_diff(X[:, i], 1.0 / ell[i], self.periods[i])
                with np.errstate(all="ignore"):
                    dK[i, :, :] = dK_factor * Ki
        else:
            rows = [1.0 / ell[i] * X[:, i] for i in range(0, D)]
            if _sq_diffs(rows, out=dK[0:D]):
                with np.errstate(all="ignore"):
                    np.multiply(dK_factor, dK[0:D], out=dK[0:D])
            else:
                for i in range(0, D):
                    Ki = _sq_diff(rows[i])
                    with np.errstate(all="ignore"):
                        dK[i, :, :] = dK_factor * Ki

        # Gradient of cov output scale: 2 * K.
        np.multiply(2, K, out=dK[D])

        # Gradient respect of alpha: K * (0.5 * tmp / M - alpha * log(M)).
        tmp *= 0.5
        tmp /= M
        log_M = np.log(M, out=dK_factor)
        log_M *= alpha
        tmp -= log_M
        np.multiply(K, tmp, out=dK[D + 1])

        return K, dK.transpose(1, 2, 0)

    def get_bounds_info(self, X: np.ndarray, y: np.ndarray):
        # The length scales and the output scale have the bounds of the
        # other kernels. The shape hyperparameter, last in the vector, has
        # the bounds and the starting value of BADS; a better
        # initialization should be considered for future releases.
        bounds_info = _bounds_info_helper(X.shape[1] + 1, X, y)
        shape = {"LB": -5.0, "UB": 5.0, "PLB": -5.0, "PUB": 5.0, "x0": 1.0}
        for key, value in shape.items():
            bounds_info[key] = np.append(bounds_info[key], value)
        return bounds_info


def _bounds_info_helper(cov_N, X, y):
    _, D = X.shape
    tol = 1e-6
    lower_bounds = np.full((cov_N,), -np.inf)
    upper_bounds = np.full((cov_N,), np.inf)
    plausible_lower_bounds = np.full((cov_N,), -np.inf)
    plausible_upper_bounds = np.full((cov_N,), np.inf)
    plausible_x0 = np.full((cov_N,), np.nan)

    width, x_std = _input_spread(X)
    if np.size(y) <= 1:
        y = np.array([0, 1])
    height, y_std = _target_spread(y)

    # A column without spread gives -inf (see _input_spread).
    with np.errstate(divide="ignore"):
        lower_bounds[0:D] = np.log(width) + np.log(tol)
        upper_bounds[0:D] = np.log(width * 10)
        plausible_lower_bounds[0:D] = np.log(width) + 0.5 * np.log(tol)
        plausible_upper_bounds[0:D] = np.log(width)
        plausible_x0[0:D] = np.log(x_std)

    lower_bounds[D] = np.log(height) + np.log(tol)
    upper_bounds[D] = np.log(height * 10)
    plausible_lower_bounds[D] = np.log(height) + 0.5 * np.log(tol)
    plausible_upper_bounds[D] = np.log(height)
    plausible_x0[D] = np.log(y_std)

    # Plausible starting point
    i_nan = np.isnan(plausible_x0)
    plausible_x0[i_nan] = 0.5 * (
        plausible_lower_bounds[i_nan] + plausible_upper_bounds[i_nan]
    )

    bounds_info = {
        "LB": lower_bounds,
        "UB": upper_bounds,
        "PLB": plausible_lower_bounds,
        "PUB": plausible_upper_bounds,
        "x0": plausible_x0,
    }
    return bounds_info
