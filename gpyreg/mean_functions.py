"""Module for different mean functions used by a Gaussian process."""

import numpy as np


class ZeroMean:
    """Zero mean function."""

    def __init__(self):
        pass

    @staticmethod
    def hyperparameter_count(_):
        """Gives the number of hyperparameters this mean function has.

        Parameters
        ----------
        d : int
            The degree of the mean function.

        Returns
        -------
        count : int
            The amount of hyperparameters.
        """
        return 0

    @staticmethod
    def hyperparameter_info(_):
        """Gives information on the names of hyperparameters for setting
        them in other parts of the program.

        Parameters
        ----------
        d : int
            The degree of the kernel.

        Returns
        -------
        hyper_info : array_like
            A list of tuples of hyperparameter names and their number,
            in the order they are used in computations.
        """
        return []

    def get_bounds_info(self, X, y):
        """Gives information on the lower, upper, plausible lower
        and plausible upper bounds of the hyperparameters of this
        mean function.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            A 2D array where each row is a test point.
        y : ndarray, shape (n, 1)
            A 2D array where each row is a test target.

        Returns
        -------
        mean_bound_info: dict
            A dictionary containing the bound info.
        """
        mean_N = self.hyperparameter_count(X.shape[1])
        return _bounds_info_helper(mean_N, X, y, 0)

    def compute(self, hyp, X, compute_grad=False):
        """Computes the mean function at given test points.

        Parameters
        ----------
        hyp : ndarray, shape (mean_n,)
            A 1D array of hyperparameters, where ``mean_n`` is
            the number returned by the function ``hyperparameter_count``.
        X : ndarray, shape (n, d)
            A 2D array where each row is a test point.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to hyperparameters.

        Returns
        -------
        m : ndarray, shape (n,)
            The mean values.
        dm : ndarray, shape (n, mean_n), optional
            The gradient with respect to hyperparameters.
        """
        N, D = X.shape
        mean_N = self.hyperparameter_count(D)

        if hyp.size != mean_N:
            raise ValueError(
                "Expected %d mean function hyperparameters, %d "
                "passed instead." % (mean_N, hyp.size)
            )
        if hyp.ndim != 1:
            raise ValueError(
                "Mean function output is available only for "
                "one-sample hyperparameter inputs."
            )

        m = np.zeros((N, 1))

        if compute_grad:
            return m, []

        return m


class ConstantMean:
    """Constant mean function."""

    def __init__(self):
        pass

    @staticmethod
    def hyperparameter_count(_):
        """Gives the number of hyperparameters this mean function has.

        Parameters
        ----------
        d : int
            The degree of the mean function.

        Returns
        -------
        count : int
            The amount of hyperparameters.
        """
        return 1

    @staticmethod
    def hyperparameter_info(_):
        """Gives information on the names of hyperparameters for setting
        them in other parts of the program.

        Parameters
        ----------
        d : int
            The degree of the kernel.

        Returns
        -------
        hyper_info : array_like
            A list of tuples of hyperparameter names and their number,
            in the order they are used in computations.
        """
        return [("mean_const", 1)]

    def get_bounds_info(self, X, y):
        """Gives information on the lower, upper, plausible lower
        and plausible upper bounds of the hyperparameters of this
        mean function.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            A 2D array where each row is a test point.
        y : ndarray, shape (n, 1)
            A 2D array where each row is a test target.

        Returns
        -------
        mean_bound_info: dict
            A dictionary containing the bound info.
        """
        mean_N = self.hyperparameter_count(X.shape[1])
        return _bounds_info_helper(mean_N, X, y, 1)

    def compute(self, hyp, X, compute_grad=False):
        """Computes the mean function at given test points.

        Parameters
        ----------
        hyp : ndarray, shape (mean_n,)
            A 1D array of hyperparameters, where ``mean_n`` is
            the number returned by the function ``hyperparameter_count``.
        X : ndarray, shape (n, d)
            A 2D array where each row is a test point.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to hyperparameters.

        Returns
        -------
        m : ndarray, shape (n,)
            The mean values.
        dm : ndarray, shape (n, mean_n), optional
            The gradient with respect to hyperparameters.
        """
        N, D = X.shape
        mean_N = self.hyperparameter_count(D)

        if hyp.size != mean_N:
            raise ValueError(
                "Expected %d mean function hyperparameters, %d "
                "passed instead." % (mean_N, hyp.size)
            )
        if hyp.ndim != 1:
            raise ValueError(
                "Mean function output is available only for "
                "one-sample hyperparameter inputs."
            )

        m0 = hyp[0]
        m = m0 * np.ones((N, 1))

        if compute_grad:
            return m, np.ones((N, 1))

        return m


class NegativeQuadratic:
    """Centered negative quadratic mean functions."""

    def __init__(self):
        pass

    @staticmethod
    def hyperparameter_count(d):
        """Gives the number of hyperparameters this mean function has.

        Parameters
        ----------
        d : int
            The degree of the mean function.

        Returns
        -------
        count : int
            The amount of hyperparameters.
        """
        return 1 + 2 * d

    @staticmethod
    def hyperparameter_info(d):
        """Gives information on the names of hyperparameters for setting
        them in other parts of the program.

        Parameters
        ----------
        d : int
            The degree of the kernel.

        Returns
        -------
        hyper_info : array_like
            A list of tuples of hyperparameter names and their number,
            in the order they are used in computations.
        """
        return [("mean_const", 1), ("mean_location", d), ("mean_log_scale", d)]

    def get_bounds_info(self, X, y):
        """Gives information on the lower, upper, plausible lower
        and plausible upper bounds of the hyperparameters of this
        mean function.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            A 2D array where each row is a test point.
        y : ndarray, shape (n, 1)
            A 2D array where each row is a test target.

        Returns
        -------
        mean_bound_info: dict
            A dictionary containing the bound info.
        """
        mean_N = self.hyperparameter_count(X.shape[1])
        return _bounds_info_helper(mean_N, X, y, 2)

    def compute(self, hyp, X, compute_grad=False):
        """Computes the mean function at given test points.

        Parameters
        ----------
        hyp : ndarray, shape (mean_n,)
            A 1D array of hyperparameters, where ``mean_n`` is
            the number returned by the function ``hyperparameter_count``.
        X : ndarray, shape (n, d)
            A 2D array where each row is a test point.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to hyperparameters.

        Returns
        -------
        m : ndarray, shape (n,)
            The mean values.
        dm : ndarray, shape (n, mean_n), optional
            The gradient with respect to hyperparameters.
        """
        N, D = X.shape
        mean_N = self.hyperparameter_count(D)

        if hyp.size != mean_N:
            raise ValueError(
                "Expected %d mean function hyperparameters, %d "
                "passed instead." % (mean_N, hyp.size)
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
