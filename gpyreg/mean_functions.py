"""Module for different mean functions used by a Gaussian process."""

import numpy as np


class ZeroMean:
    """Zero mean function."""

    def __init__(self):
        pass

    @staticmethod
    def hyperparameter_count(_):
        """Counts the number of hyperparameters this mean function has.

        Parameters
        ----------
        d : int
            The degree we are interested in.

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
            The degree we are interested in.

        Returns
        -------
        hyper_info : array_like
            A list of tuples containing hyperparameter names along with
            how many parameters with such a name there are, in the order
            they are used in computations.
        """
        return []

    def get_info(self, X, y):
        """Gives additional information on the hyperparameters.

        Parameters
        ----------
        X : array_like
            Matrix of training inputss.
        y : array_like
            Vector of training targets.

        Returns
        -------
        mean_info : MeanInfo
            The additional info represented as a ``MeanInfo`` object.
        """
        mean_N = self.hyperparameter_count(X.shape[1])
        return MeanInfo(mean_N, X, y, 0)

    def compute(self, hyp, X, compute_grad=False):
        """Computes the mean function at test points.

        Parameters
        ----------
        hyp : array_like
            Vector of hyperparameters.
        X : array_like
            Matrix of test points.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to hyperparameters.

        Returns
        -------
        m : array_like
            The mean values.
        dm : array_like, optional
            The gradient.
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
        """Counts the numver of hyperparameters this mean function has.

        Parameters
        ----------
        d : int
            The degree we are interested in.

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
            The degree we are interested in.

        Returns
        -------
        hyper_info : array_like
            A list of tuples containing hyperparameter names along with
            how many parameters with such a name there are, in the order
            they are used in computations.
        """
        return [("mean_const", 1)]

    def get_info(self, X, y):
        """Gives additional information on the hyperparameters.

        Parameters
        ----------
        X : array_like
            Matrix of training inputs.
        y : array_like
            Vector of training targets.

        Returns
        -------
        mean_info : CovarianceInfo
            The additional info represented as a ``MeanInfo`` object.
        """
        mean_N = self.hyperparameter_count(X.shape[1])
        return MeanInfo(mean_N, X, y, 1)

    def compute(self, hyp, X, compute_grad=False):
        """Computes the mean function at test points.

        Parameters
        ----------
        hyp : array_like
            Vector of hyperparameters.
        X : array_like
            Matrix of test points.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to hyperparameters.

        Returns
        -------
        m : array_like
            The mean values.
        dm : array_like, optional
            The gradient.
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
        """Counts the numver of hyperparameters this mean function has.

        Parameters
        ----------
        d : int
            The degree we are interested in.

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
            The degree we are interested in.

        Returns
        -------

        hyper_info : array_like
            A list of tuples containing hyperparameter names along with
            how many parameters with such a name there are, in the order
            they are used in computations.
        """
        return [("mean_const", 1), ("mean_location", d), ("mean_log_scale", d)]

    def get_info(self, X, y):
        """Gives additional information on the hyperparameters.

        Parameters
        ----------
        X : array_like
            Matrix of training inputs.
        y : array_like
            Vector of training targets.

        Returns
        -------
        cov_info : MeanInfo
            The additional info represented as a ``MeanInfo`` object.
        """
        mean_N = self.hyperparameter_count(X.shape[1])
        return MeanInfo(mean_N, X, y, 2)

    def compute(self, hyp, X, compute_grad=False):
        """Computes the mean function at test points.

        Parameters
        ----------
        hyp : array_like
            Vector of hyperparameters.
        X : array_like
            Matrix of test points.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to hyperparameters.

        Returns
        -------
        m : array_like
            The mean values.
        dm : array_like, optional
            The gradient.
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


class MeanInfo:
    def __init__(self, mean_N, X, y, idx):
        _, D = X.shape
        tol = 1e-6
        big = np.exp(3)
        self.LB = np.full((mean_N,), -np.inf)
        self.UB = np.full((mean_N,), np.inf)
        self.PLB = np.full((mean_N,), -np.inf)
        self.PUB = np.full((mean_N,), np.inf)
        self.x0 = np.full((mean_N,), np.nan)

        w = np.max(X) - np.min(X)
        if np.size(y) <= 1:
            y = np.array([0, 1])
        h = np.max(y) - np.min(y)

        if idx == 0:
            pass
        elif idx == 1:
            self.LB[0] = np.min(y) - 0.5 * h
            self.UB[0] = np.max(y) + 0.5 * h
            # For future reference note that quantile behaviour in
            # MATLAB and NumPy is slightly different.
            self.PLB[0] = np.quantile(y, 0.1)
            self.PUB[0] = np.quantile(y, 0.9)
            self.x0[0] = np.median(y)
        else:
            self.LB[0] = np.min(y)
            self.UB[0] = np.max(y) + h
            self.PLB[0] = np.median(y)
            self.PUB[0] = np.max(y)
            self.x0[0] = np.quantile(y, 0.9)

            # xm
            self.LB[1 : 1 + D] = np.min(X) - 0.5 * w
            self.UB[1 : 1 + D] = np.max(X) + 0.5 * w
            self.PLB[1 : 1 + D] = np.min(X)
            self.PUB[1 : 1 + D] = np.max(X)
            self.x0[1 : 1 + D] = np.median(X)

            # omega
            self.LB[1 + D : mean_N] = np.log(w) + np.log(tol)
            self.UB[1 + D : mean_N] = np.log(w) + np.log(big)
            self.PLB[1 + D : mean_N] = np.log(w) + 0.5 * np.log(tol)
            self.PUB[1 + D : mean_N] = np.log(w)
            # For future reference note that std behaviour in
            # MATLAB and NumPy is slightly different.
            self.x0[1 + D : mean_N] = np.log(np.std(X, ddof=1))
        # Plausible starting point
        i_nan = np.isnan(self.x0)
        self.x0[i_nan] = 0.5 * (self.PLB[i_nan] + self.PUB[i_nan])
