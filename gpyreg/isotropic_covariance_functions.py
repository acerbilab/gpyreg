from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from .covariance_functions import AbstractKernel, Matern, SquaredExponential

class AbstractIsotropicKernel(AbstractKernel):
    """Abstract base class for isotropic kernel functions.

    The two default hyperparameters are the log-lengthscale and
    log-outputscale.
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
        return 2

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
            ("covariance_log_lengthscale", 1),
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
        return _isotropic_bounds_info_helper(cov_N, X, y)

class MaternIsotropic(AbstractIsotropicKernel, Matern):
    """
    Isotropic Matern kernel.

    Overrides `compute`. Inherits `hyperparameter_count` and
    `hyperparameter_info` from :class:`AbstractIsotropicKernel`. Inherits other
    methods from :class:`Matern`.

    Parameters
    ----------
    degree : {1, 3, 5}
        The degree of the isotropic Matern kernel.

        Currently the only supported degrees are 1, 3, 5, and if
        some other degree is provided a ``ValueError`` exception is raised.
    """

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

        ell = np.exp(hyp[0])
        sf2 = np.exp(2 * hyp[1])

        if X_star is None:
            if compute_diag:
                tmp = np.zeros((N, 1))
            else:
                tmp = squareform(
                    pdist(X * np.sqrt(self.degree) / ell)
                )
        else:
            a = X * np.sqrt(self.degree) / ell
            b = X_star * np.sqrt(self.degree) / ell
            tmp = cdist(a, b)

        K = sf2 * self.f(tmp) * np.exp(-tmp)

        if compute_grad:
            if X_star is not None:
                raise ValueError("X_star should be None when compute_grad is True.")
            dK = np.zeros((cov_N, N, N))
            # Gradient of cov length scale
            K_ls = squareform(pdist(np.sqrt(self.degree) / ell * X, "sqeuclidean"))
            # With d=1 kernel there will be issues caused by zero
            # divisions. This is OK, the kernel is just not
            # differentiable there.
            with np.errstate(all="ignore"):
                dK[0, :, :] = sf2 * (self.df(tmp) * np.exp(-tmp)) * K_ls
            # Gradient of cov output scale
            dK[1, :, :] = 2 * K
            return K, dK.transpose(1, 2, 0)

        return K

class SquaredExponentialIsotropic(AbstractIsotropicKernel, SquaredExponential):
    """Isotropic squared exponential kernel.

    Overrides `compute`. Inherits `hyperparameter_count` and
    `hyperparameter_info` from :class:`AbstractIsotropicKernel`. Inherits other
    methods from :class:`SquaredExponential`.
    """

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

        ell = np.exp(hyp[0])
        sf2 = np.exp(2 * hyp[1])

        if X_star is None:
            if compute_diag:
                tmp = np.zeros((N, 1))
            else:
                tmp = squareform(pdist(X / ell, "sqeuclidean"))
        else:
            tmp = cdist(X / ell, X_star / ell, "sqeuclidean")

        K = sf2 * np.exp(-tmp / 2)

        if compute_grad:
            if X_star is not None:
                raise ValueError("X_star should be None when compute_grad is True.")
            dK = np.zeros((cov_N, N, N))
            # Gradient of cov length scale
            dK[0, :, :] = K * squareform(pdist(X / ell, "sqeuclidean"))
            # Gradient of cov output scale.
            dK[1, :, :] = 2 * K
            return K, dK.transpose(1, 2, 0)

        return K

def _isotropic_bounds_info_helper(cov_N, X, y):
    _, D = X.shape
    tol = 1e-6
    lower_bounds = np.full((cov_N,), -np.inf)
    upper_bounds = np.full((cov_N,), np.inf)
    plausible_lower_bounds = np.full((cov_N,), -np.inf)
    plausible_upper_bounds = np.full((cov_N,), np.inf)
    plausible_x0 = np.full((cov_N,), np.nan)

    width = np.mean(np.max(X, axis=0) - np.min(X, axis=0))
    if np.size(y) <= 1:
        y = np.array([0, 1])
    height = np.max(y) - np.min(y)

    lower_bounds[0:cov_N - 1] = np.log(width) + np.log(tol)
    upper_bounds[0:cov_N - 1] = np.log(width * 10)
    plausible_lower_bounds[0:cov_N - 1] = np.log(width) + 0.5 * np.log(tol)
    plausible_upper_bounds[0:cov_N - 1] = np.log(width)
    plausible_x0[0:cov_N - 1] = np.log(np.std(X, ddof=1))

    lower_bounds[cov_N - 1] = np.log(height) + np.log(tol)
    upper_bounds[cov_N - 1] = np.log(height * 10)
    plausible_lower_bounds[cov_N - 1] = np.log(height) + 0.5 * np.log(tol)
    plausible_upper_bounds[cov_N - 1] = np.log(height)
    plausible_x0[cov_N - 1] = np.log(np.std(y, ddof=1))

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
