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


class AbstractKernel(ABC):
    """Abstract base class for covariance kernels."""

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
    """Squared exponential kernel."""

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

        ell = np.exp(hyp[0:D])
        sf2 = np.exp(2 * hyp[D])

        if X_star is None and compute_diag:
            # The diagonal is sf2 * exp(-0 / 2) = sf2 exactly.
            return np.full((N, 1), sf2)

        if X_star is None:
            if compute_diag:
                tmp = np.zeros((N, 1))
            else:
                # cdist(Xs, Xs) equals squareform(pdist(Xs)) bit for bit
                # (the same per-dimension differences summed in the same
                # order, exactly symmetric, exact zeros on the diagonal)
                # and skips pdist's wrapper and the squareform pass.
                Xs = X / ell
                tmp = cdist(Xs, Xs, "sqeuclidean")
        else:
            tmp = cdist(X / ell, X_star / ell, "sqeuclidean")

        K = sf2 * np.exp(-tmp / 2)

        if compute_grad:
            if X_star is not None:
                raise ValueError(
                    "X_star should be None when compute_grad is True."
                )
            dK = np.zeros((cov_N, N, N))
            for i in range(0, D):
                # Gradient of cov length scales
                dK[i, :, :] = K * squareform(
                    pdist(np.reshape(X[:, i] / ell[i], (-1, 1)), "sqeuclidean")
                )
            # Gradient of cov output scale.
            dK[D, :, :] = 2 * K
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
    """

    def __init__(self, degree: int):
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

        ell = np.exp(hyp[0:D])
        sf2 = np.exp(2 * hyp[D])

        if X_star is None:
            if compute_diag:
                tmp = np.zeros((N, 1))
            else:
                tmp = squareform(
                    pdist(X @ np.diag(np.sqrt(self.degree) / ell))
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
            dK = np.zeros((cov_N, N, N))
            for i in range(0, D):
                Ki = squareform(
                    pdist(
                        np.reshape(
                            np.sqrt(self.degree) / ell[i] * X[:, i], (-1, 1)
                        ),
                        "sqeuclidean",
                    )
                )
                # Where two inputs share the i-th coordinate the kernel
                # does not depend on that length scale, so the derivative
                # is zero. The d=1 kernel divides by zero there and gives
                # inf * 0 = NaN, which would poison the gradient of the
                # marginal likelihood through the whole diagonal, so the
                # product is taken as the zero it is.
                with np.errstate(all="ignore"):
                    dK[i, :, :] = np.where(
                        Ki > 0,
                        sf2 * (self.df(tmp) * np.exp(-tmp)) * Ki,
                        0.0,
                    )
            # Gradient of cov output scale
            dK[D, :, :] = 2 * K
            return K, dK.transpose(1, 2, 0)

        return K


class RationalQuadraticARD(AbstractKernel):
    """Rational Quadratic ARD kernel"""

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

        ell = np.exp(hyp[0:D])
        sf2 = np.exp(2 * hyp[D])
        alpha = np.exp(hyp[D + 1])

        if X_star is None:
            if compute_diag:
                tmp = np.zeros((N, 1))
            else:
                tmp = squareform(pdist(X @ np.diag(1.0 / ell), "sqeuclidean"))
        else:
            a = X @ np.diag(1.0 / ell)
            b = X_star @ np.diag(1.0 / ell)
            tmp = cdist(a, b, "sqeuclidean")

        M = 1 + 0.5 * tmp / alpha
        K = sf2 * M ** (-alpha)

        if compute_grad:
            if X_star is not None:
                raise ValueError(
                    "X_star should be None when compute_grad is True."
                )
            dK = np.zeros((cov_N, N, N))

            # Gradient respect of lenght scale.
            for i in range(0, D):
                Ki = squareform(
                    pdist(
                        np.reshape(1.0 / ell[i] * X[:, i], (-1, 1)),
                        "sqeuclidean",
                    )
                )
                with np.errstate(all="ignore"):
                    dK[i, :, :] = sf2 * M ** (-alpha - 1) * Ki

            # Gradient of cov output scale.
            dK[D, :, :] = 2 * K

            # Gradient respect of alpha.
            dK[D + 1, :, :] = K * (0.5 * tmp / M - alpha * np.log(M))

            return K, dK.transpose(1, 2, 0)

        return K

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

    width = np.max(X, axis=0) - np.min(X, axis=0)
    if np.size(y) <= 1:
        y = np.array([0, 1])
    height, y_std = _target_spread(y)

    lower_bounds[0:D] = np.log(width) + np.log(tol)
    upper_bounds[0:D] = np.log(width * 10)
    plausible_lower_bounds[0:D] = np.log(width) + 0.5 * np.log(tol)
    plausible_upper_bounds[0:D] = np.log(width)
    plausible_x0[0:D] = np.log(np.std(X, axis=0, ddof=1))

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
