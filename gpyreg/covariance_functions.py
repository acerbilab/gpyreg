"""Module for different covariance functions used by Gaussian Processes."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


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
            ``compute_diag = True`` the shape is ``(N,)``.
        dK : ndarray, shape (N, N, cov_N), optional
            The gradient of the covariance matrix with respect to the
            hyperparameters.

        Raises
        ------
        ValueError
            Raised when `hyp` has not the expected number of hyperparameters.
        ValueError
            Raised when `hyp` is not an 1D array but of higher dimension.
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

        ell = np.exp(hyp[0:D])
        sf2 = np.exp(2 * hyp[D])

        if X_star is None:
            if compute_diag:
                tmp = np.zeros((N, 1))
            else:
                tmp = squareform(pdist(X / ell, "sqeuclidean"))
        else:
            tmp = cdist(X / ell, X_star / ell, "sqeuclidean")

        K = sf2 * np.exp(-tmp / 2)

        if compute_grad:
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
                # With d=1 kernel there will be issues caused by zero
                # divisions. This is OK, the kernel is just not
                # differentiable there.
                with np.errstate(all="ignore"):
                    dK[i, :, :] = sf2 * (self.df(tmp) * np.exp(-tmp)) * Ki
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
        # Override the default get_bounds_info function. The function has the
        # same behavior of the `_bounds_info_helper` But it also initializes
        # the covariance_log_shape hyperparameter as in BADS. A better
        # initialization should be considered for future releases.
        cov_N = self.hyperparameter_count(X.shape[1])
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
        height = np.max(y) - np.min(y)

        lower_bounds[0:D] = np.log(width) + np.log(tol)
        upper_bounds[0:D] = np.log(width * 10)
        plausible_lower_bounds[0:D] = np.log(width) + 0.5 * np.log(tol)
        plausible_upper_bounds[0:D] = np.log(width)
        plausible_x0[0:D] = np.log(np.std(X, ddof=1))

        lower_bounds[D] = np.log(height) + np.log(tol)
        upper_bounds[D] = np.log(height * 10)
        plausible_lower_bounds[D] = np.log(height) + 0.5 * np.log(tol)
        plausible_upper_bounds[D] = np.log(height)
        plausible_x0[D] = np.log(np.std(y, ddof=1))

        # Initialization of the covariance_log_shape hyperparameter (like in
        # BADS)
        lower_bounds[-1] = -5.0
        upper_bounds[-1] = 5
        plausible_lower_bounds[-1] = -5.0
        plausible_upper_bounds[D] = 5.0
        plausible_x0[-1] = 1.0

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
    height = np.max(y) - np.min(y)

    lower_bounds[0:D] = np.log(width) + np.log(tol)
    upper_bounds[0:D] = np.log(width * 10)
    plausible_lower_bounds[0:D] = np.log(width) + 0.5 * np.log(tol)
    plausible_upper_bounds[0:D] = np.log(width)
    plausible_x0[0:D] = np.log(np.std(X, ddof=1))

    lower_bounds[D] = np.log(height) + np.log(tol)
    upper_bounds[D] = np.log(height * 10)
    plausible_lower_bounds[D] = np.log(height) + 0.5 * np.log(tol)
    plausible_upper_bounds[D] = np.log(height)
    plausible_x0[D] = np.log(np.std(y, ddof=1))

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
