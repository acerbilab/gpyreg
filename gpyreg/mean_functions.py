"""Module for different mean functions used by a Gaussian process."""

import numpy as np


class ZeroMean:
    """Zero mean function."""

    def __init__(self):
        pass

    @staticmethod
    def hyperparameter_count(_):
        """
        Return the number of hyperparameters of this mean function.

        Parameters
        ----------
        D : int
            The dimensionality of the mean function (unused).

        Returns
        -------
        count : int
            The number of hyperparameters.
        """
        return 0

    @staticmethod
    def hyperparameter_info(_):
        """
        Return information on the names of hyperparameters for setting
        them in other parts of the program.

        Parameters
        ----------
        D : int
            The dimensionality of the mean function (unused).

        Returns
        -------
        hyper_info : array_like
            A list of tuples of hyperparameter names and their number,
            in the order they are in the hyperparameter array.
        """
        return []

    def get_bounds_info(self, X: np.ndarray, y: np.ndarray):
        """
        Return information on the lower, upper, plausible lower
        and plausible upper bounds of the hyperparameters of this
        mean function.

        Parameters
        ----------
        X : ndarray, shape (N, D)
            A 2D array where each row is a test point.
        y : ndarray, shape (N, 1)
            A 2D array where each row is a test target.

        Returns
        -------
        mean_bound_info: dict
            A dictionary containing the bound info with the following elements:

            **LB** : np.ndarray, shape (mean_N, 1)
                    The lower bounds of the hyperparameters.
            **UB** : np.ndarray, shape (mean_N, 1)
                    The upper bounds of the hyperparameters.
            **PLB** : np.ndarray, shape (mean_N, 1)
                    The plausible lower bounds of the hyperparameters.
            **PUB** : np.ndarray, shape (mean_N, 1)
                    The plausible upper bounds of the hyperparameters.
            **x0** : np.ndarray, shape (mean_N, 1)
                    The plausible starting point.

            where ``mean_N`` is the number of hyperparameters.
        """
        mean_N = self.hyperparameter_count(X.shape[1])
        return _bounds_info_helper(mean_N, X, y, 0)

    def compute(
        self, hyp: np.ndarray, X: np.ndarray, compute_grad: bool = False
    ):
        """
        Compute the mean function at given test points.

        Parameters
        ----------
        hyp : ndarray, shape (mean_N,)
            A 1D array of hyperparameters, where ``mean_N`` is
            the number returned by the function ``hyperparameter_count``.
        X : ndarray, shape (N, D)
            A 2D array where each row is a test point.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to hyperparameters.

        Returns
        -------
        m : ndarray, shape (N,)
            The mean values.
        dm : ndarray, shape (N, mean_N), optional
            The gradient with respect to hyperparameters.

        Raises
        ------
        ValueError
            Raised when `hyp` has not the expected number of hyperparameters.
        ValueError
            Raised when `hyp` is not an 1D array but of higher dimension.
        """
        N, D = X.shape
        mean_N = self.hyperparameter_count(D)

        if hyp.size != mean_N:
            raise ValueError(
                f"Expected {mean_N} mean function hyperparameters, "
                f"{hyp.size} passed instead."
            )
        if hyp.ndim != 1:
            raise ValueError(
                "Mean function output is available only for "
                "one-sample hyperparameter inputs."
            )

        m = np.zeros((N,))

        if compute_grad:
            return m, []

        return m


class ConstantMean:
    """Constant mean function."""

    def __init__(self):
        pass

    @staticmethod
    def hyperparameter_count(_):
        """
        Return the number of hyperparameters of this mean function.

        Parameters
        ----------
        D : int
            The dimensionality of the mean function (unused).

        Returns
        -------
        count : int
            The number of hyperparameters.
        """
        return 1

    @staticmethod
    def hyperparameter_info(_):
        """
        Return information on the names of hyperparameters for setting
        them in other parts of the program.

        Parameters
        ----------
        D : int
            The dimensionality of the mean function (unused).

        Returns
        -------
        hyper_info : array_like
            A list of tuples of hyperparameter names and their number,
            in the order they are in the hyperparameter array.
        """
        return [("mean_const", 1)]

    def get_bounds_info(self, X: np.ndarray, y: np.ndarray):
        """
        Return information on the lower, upper, plausible lower
        and plausible upper bounds of the hyperparameters of this
        mean function.

        Parameters
        ----------
        X : ndarray, shape (N, D)
            A 2D array where each row is a test point.
        y : ndarray, shape (N, 1)
            A 2D array where each row is a test target.

        Returns
        -------
        mean_bound_info: dict
            A dictionary containing the bound info with the following elements:

            **LB** : np.ndarray, shape (mean_N, 1)
                    The lower bounds of the hyperparameters.
            **UB** : np.ndarray, shape (mean_N, 1)
                    The upper bounds of the hyperparameters.
            **PLB** : np.ndarray, shape (mean_N, 1)
                    The plausible lower bounds of the hyperparameters.
            **PUB** : np.ndarray, shape (mean_N, 1)
                    The plausible upper bounds of the hyperparameters.
            **x0** : np.ndarray, shape (mean_N, 1)
                    The plausible starting point.

            where ``mean_N`` is the number of hyperparameters.
        """
        mean_N = self.hyperparameter_count(X.shape[1])
        return _bounds_info_helper(mean_N, X, y, 1)

    def compute(
        self, hyp: np.ndarray, X: np.ndarray, compute_grad: bool = False
    ):
        """
        Compute the mean function at given test points.

        Parameters
        ----------
        hyp : ndarray, shape (mean_N,)
            A 1D array of hyperparameters, where ``mean_N`` is
            the number returned by the function ``hyperparameter_count``.
        X : ndarray, shape (N, D)
            A 2D array where each row is a test point.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to hyperparameters.

        Returns
        -------
        m : ndarray, shape (N,)
            The mean values.
        dm : ndarray, shape (N, mean_N), optional
            The gradient with respect to hyperparameters.

        Raises
        ------
        ValueError
            Raised when `hyp` has not the expected number of hyperparameters.
        ValueError
            Raised when `hyp` is not an 1D array but of higher dimension.
        """
        N, D = X.shape
        mean_N = self.hyperparameter_count(D)

        if hyp.size != mean_N:
            raise ValueError(
                f"Expected {mean_N} mean function hyperparameters, "
                f"{hyp.size} passed instead."
            )
        if hyp.ndim != 1:
            raise ValueError(
                "Mean function output is available only for "
                "one-sample hyperparameter inputs."
            )

        m0 = hyp[0]
        m = m0 * np.ones((N,))

        if compute_grad:
            return m, np.ones((N, 1))

        return m


class NegativeQuadratic:
    """Centered negative quadratic mean functions."""

    def __init__(self):
        pass

    @staticmethod
    def hyperparameter_count(D: int):
        """
        Return the number of hyperparameters this mean function has.

        Parameters
        ----------
        D : int
            The dimensionality of the mean function.

        Returns
        -------
        count : int
            The number of hyperparameters.
        """
        return 1 + 2 * D

    @staticmethod
    def hyperparameter_info(D: int):
        """
        Return information on the names of hyperparameters for setting
        them in other parts of the program.

        Parameters
        ----------
        D : int
            The dimensionality of the mean function.

        Returns
        -------
        hyper_info : array_like
            A list of tuples of hyperparameter names and their number,
            in the order they are in the hyperparameter array.
        """
        return [("mean_const", 1), ("mean_location", D), ("mean_log_scale", D)]

    def get_bounds_info(self, X: np.ndarray, y: np.ndarray):
        """
        Return information on the lower, upper, plausible lower
        and plausible upper bounds of the hyperparameters of this
        mean function.

        Parameters
        ----------
        X : ndarray, shape (N, D)
            A 2D array where each row is a test point.
        y : ndarray, shape (N, 1)
            A 2D array where each row is a test target.

        Returns
        -------
        mean_bound_info: dict
            A dictionary containing the bound info with the following elements:

            **LB** : np.ndarray, shape (mean_N, 1)
                    The lower bounds of the hyperparameters.
            **UB** : np.ndarray, shape (mean_N, 1)
                    The upper bounds of the hyperparameters.
            **PLB** : np.ndarray, shape (mean_N, 1)
                    The plausible lower bounds of the hyperparameters.
            **PUB** : np.ndarray, shape (mean_N, 1)
                    The plausible upper bounds of the hyperparameters.
            **x0** : np.ndarray, shape (mean_N, 1)
                    The plausible starting point.

            where ``mean_N`` is the number of hyperparameters.

        """
        mean_N = self.hyperparameter_count(X.shape[1])
        return _bounds_info_helper(mean_N, X, y, 2)

    def compute(
        self, hyp: np.ndarray, X: np.ndarray, compute_grad: bool = False
    ):
        """
        Compute the mean function at given test points.

        Parameters
        ----------
        hyp : ndarray, shape (mean_N,)
            A 1D array of hyperparameters, where ``mean_N`` is
            the number returned by the function ``hyperparameter_count``.
        X : ndarray, shape (N, D)
            A 2D array where each row is a test point.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to hyperparameters.

        Returns
        -------
        m : ndarray, shape (N,)
            The mean values.
        dm : ndarray, shape (N, mean_N), optional
            The gradient with respect to hyperparameters.

        Raises
        ------
        ValueError
            Raised when `hyp` has not the expected number of hyperparameters.
        ValueError
            Raised when `hyp` is not an 1D array but of higher dimension.
        """
        N, D = X.shape
        mean_N = self.hyperparameter_count(D)

        if hyp.size != mean_N:
            raise ValueError(
                f"Expected {mean_N} mean function hyperparameters, "
                f"{hyp.size} passed instead."
            )
        if hyp.ndim != 1:
            raise ValueError(
                "Mean function output is available only for one-sample "
                "hyperparameter inputs."
            )

        m_0 = hyp[0]
        x_m = hyp[1 : (1 + D)]
        omega = np.exp(hyp[(1 + D) : (1 + 2 * D)])
        z_2 = ((X - x_m) / omega) ** 2
        m = m_0 - 0.5 * np.sum(z_2, 1)

        if compute_grad:
            dm = np.zeros((N, mean_N))
            dm[:, 0] = np.ones((N,))
            dm[:, 1 : D + 1] = (X - x_m) / omega ** 2
            dm[:, D + 1 :] = z_2
            return m, dm

        return m


def _bounds_info_helper(mean_N, X, y, idx):
    _, D = X.shape
    tol = 1e-6
    big = np.exp(3)
    LB = np.full((mean_N,), -np.inf)
    UB = np.full((mean_N,), np.inf)
    PLB = np.full((mean_N,), -np.inf)
    PUB = np.full((mean_N,), np.inf)
    x0 = np.full((mean_N,), np.nan)

    w = np.max(X) - np.min(X)
    if np.size(y) <= 1:
        y = np.array([0, 1])
    h = np.max(y) - np.min(y)

    if idx == 0:
        pass
    elif idx == 1:
        LB[0] = np.min(y) - 0.5 * h
        UB[0] = np.max(y) + 0.5 * h
        # For future reference note that quantile behaviour in
        # MATLAB and NumPy is slightly different.
        PLB[0] = np.quantile(y, 0.1)
        PUB[0] = np.quantile(y, 0.9)
        x0[0] = np.median(y)
    else:
        LB[0] = np.min(y)
        UB[0] = np.max(y) + h
        PLB[0] = np.median(y)
        PUB[0] = np.max(y)
        x0[0] = np.quantile(y, 0.9)

        # xm
        LB[1 : 1 + D] = np.min(X) - 0.5 * w
        UB[1 : 1 + D] = np.max(X) + 0.5 * w
        PLB[1 : 1 + D] = np.min(X)
        PUB[1 : 1 + D] = np.max(X)
        x0[1 : 1 + D] = np.median(X)

        # omega
        LB[1 + D : mean_N] = np.log(w) + np.log(tol)
        UB[1 + D : mean_N] = np.log(w) + np.log(big)
        PLB[1 + D : mean_N] = np.log(w) + 0.5 * np.log(tol)
        PUB[1 + D : mean_N] = np.log(w)
        # For future reference note that std behaviour in
        # MATLAB and NumPy is slightly different.
        x0[1 + D : mean_N] = np.log(np.std(X, ddof=1))

    # Plausible starting point
    i_nan = np.isnan(x0)
    x0[i_nan] = 0.5 * (PLB[i_nan] + PUB[i_nan])

    bounds_info = {
        "LB": LB,
        "PLB": PLB,
        "UB": UB,
        "PUB": PUB,
        "x0": x0,
    }
    return bounds_info
