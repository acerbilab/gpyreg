"""Module for different covariance functions used by a Gaussian process."""

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform


class SquaredExponential:
    """Squared exponential kernel."""

    def __init__(self):
        pass

    @staticmethod
    def hyperparameter_count(d):
        """Counts the number of hyperparameters this covariance function has.

        Parameters
        ----------
        d : int
            The degree we are interested in.

        Returns
        -------

        count : int
            The amount of hyperparameters.
        """
        return d + 1

    @staticmethod
    def hyperparameter_info(d):
        """Gives information on the names of hyperparameters for setting them
        in other parts of the program.

        Parameters
        ----------
        d : int
            The degree we are interested in.

        Returns
        -------
        hyper_info : array_like
            A list of tuples containing hyperparameter names along with how
            many parameters with such a name there are, in the order they
            are used in computations.
        """
        return [
            ("covariance_log_lengthscale", d),
            ("covariance_log_outputscale", 1),
        ]

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
        cov_info : CovarianceInfo
            The additional info represented as a ``CovarianceInfo`` object.
        """
        cov_N = self.hyperparameter_count(X.shape[1])
        return CovarianceInfo(cov_N, X, y)

    def compute(self, hyp, X, X_star=None, compute_grad=False):
        """Computes the self-covariance matrix of the training points.

        Parameters
        ----------
        X : array_like
            Matrix of training inputs.
        X_star : array_like, optional
            Matrix of additional training inputs.
        compute_grad : bool, defaults to False
            Flag for computing the gradient.

        Returns
        -------
        K : array_like
            The self-covariance matrix.
        dK : array_like, optional
            The gradient of the self-covariance matrix.
        """
        N, D = X.shape
        cov_N = self.hyperparameter_count(D)

        if hyp.size != cov_N:
            raise ValueError(
                "Expected %d covariance function hyperparameters, %d "
                "passed instead." % (cov_N, hyp.size)
            )
        if hyp.ndim != 1:
            raise ValueError(
                "Covariance function output is available only for "
                "one-sample hyperparameter inputs."
            )

        ell = np.exp(hyp[0:D])
        sf2 = np.exp(2 * hyp[D])

        if X_star is None:
            tmp = squareform(pdist(X / ell, "sqeuclidean"))
        elif isinstance(X_star, str):
            tmp = np.zeros((N, 1))
        else:
            tmp = cdist(X / ell, X_star / ell, "sqeuclidean")

        K = sf2 * np.exp(-tmp / 2)

        if compute_grad:
            dK = np.zeros((N, N, cov_N))
            for i in range(0, D):
                # Gradient of cov length scales
                dK[:, :, i] = K * squareform(
                    pdist(np.reshape(X[:, i] / ell[i], (-1, 1)), "sqeuclidean")
                )
            # Gradient of cov output scale.
            dK[:, :, D] = 2 * K
            return K, dK

        return K


class Matern:
    """Matern kernel.

    Parameters
    ----------
    degree : {1, 3, 5}
        The degree of the Matern kernel.

        Currently the only supported degrees are 1, 3, 5.

    """

    def __init__(self, degree):
        self.degree = degree

    @staticmethod
    def hyperparameter_count(d):
        """Counts the number of hyperparameters this covariance function has.

        Parameters
        ----------
        d : int
            The degree we are interested in.

        Returns
        -------

        count : int
            The amount of hyperparameters.
        """
        return d + 1

    @staticmethod
    def hyperparameter_info(d):
        """Gives information on the names of hyperparameters for setting them
        in other parts of the program.

        Parameters
        ----------
        d : int
            The degree we are interested in.

        Returns
        -------
        hyper_info : array_like
            A list of tuples containing hyperparameter names along with how
            many parameters with such a name there are, in the order they
            are used in computations.
        """
        return [
            ("covariance_log_lengthscale", d),
            ("covariance_log_outputscale", 1),
        ]

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
        cov_info : CovarianceInfo
            The additional info represented as a ``CovarianceInfo`` object.
        """
        cov_N = self.hyperparameter_count(X.shape[1])
        return CovarianceInfo(cov_N, X, y)

    def compute(self, hyp, X, X_star=None, compute_grad=False):
        """Computes the self-covariance matrix of the training points.

        Parameters
        ----------
        X : array_like
            Matrix of training inputs.
        X_star : array_like, optional
            Matrix of additional training inputs.
        compute_grad : bool, defaults to False
            Flag for computing the gradient.

        Returns
        -------
        K : array_like
            The self-covariance matrix.
        dK : array_like, optional
            The gradient of the self-covariance matrix.
        """
        N, D = X.shape
        cov_N = self.hyperparameter_count(D)

        if hyp.size != cov_N:
            raise ValueError(
                "Expected %d covariance function hyperparameters "
                ", %d passed instead." % (cov_N, hyp.size)
            )
        if hyp.ndim != 1:
            raise ValueError(
                "Covariance function output is available only for "
                "one-sample hyperparameter inputs."
            )

        ell = np.exp(hyp[0:D])
        sf2 = np.exp(2 * hyp[D])

        f = df = None
        if self.degree == 1:
            f = lambda t: 1
            df = lambda t: 1 / t
        elif self.degree == 3:
            f = lambda t: 1 + t
            df = lambda t: 1
        elif self.degree == 5:
            f = lambda t: 1 + t * (1 + t / 3)
            df = lambda t: (1 + t) / 3
        else:
            raise Exception(
                "Only degrees 1, 3 and 5 are supported for the "
                "Matern covariance function."
            )

        if X_star is None:
            tmp = squareform(pdist(X @ np.diag(np.sqrt(self.degree) / ell)))
        elif isinstance(X_star, str):
            tmp = np.zeros((X.shape[0], 1))
        else:
            a = X @ np.diag(np.sqrt(self.degree) / ell)
            b = X_star @ np.diag(np.sqrt(self.degree) / ell)
            tmp = cdist(a, b)

        K = sf2 * f(tmp) * np.exp(-tmp)

        if compute_grad:
            dK = np.zeros((N, N, cov_N))
            for i in range(0, D):
                Ki = squareform(
                    pdist(
                        np.reshape(
                            np.sqrt(self.degree) / ell[i] * X[:, i], (-1, 1)
                        ),
                        "sqeuclidean",
                    )
                )
                dK[:, :, i] = sf2 * (df(tmp) * np.exp(-tmp)) * Ki
            # Gradient of cov output scale
            dK[:, :, D] = 2 * K
            return K, dK

        return K


class CovarianceInfo:
    def __init__(self, cov_N, X, y):
        _, D = X.shape
        tol = 1e-6
        self.LB = np.full((cov_N,), -np.inf)
        self.UB = np.full((cov_N,), np.inf)
        self.PLB = np.full((cov_N,), -np.inf)
        self.PUB = np.full((cov_N,), np.inf)
        self.x0 = np.full((cov_N,), np.nan)

        width = np.max(X, axis=0) - np.min(X, axis=0)
        if np.size(y) <= 1:
            y = np.array([0, 1])
        height = np.max(y) - np.min(y)

        self.LB[0:D] = np.log(width) + np.log(tol)
        self.UB[0:D] = np.log(width * 10)
        self.PLB[0:D] = np.log(width) + 0.5 * np.log(tol)
        self.PUB[0:D] = np.log(width)
        self.x0[0:D] = np.log(np.std(X, ddof=1))

        self.LB[D] = np.log(height) + np.log(tol)
        self.UB[D] = np.log(height * 10)
        self.PLB[D] = np.log(height) + 0.5 * np.log(tol)
        self.PUB[D] = np.log(height)
        self.x0[D] = np.log(np.std(y, ddof=1))

        # Plausible starting point
        i_nan = np.isnan(self.x0)
        self.x0[i_nan] = 0.5 * (self.PLB[i_nan] + self.PUB[i_nan])
