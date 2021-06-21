"""Module for different noise functions used by a Gaussian process."""

import numpy as np


class GaussianNoise:
    """Gaussian noise

    Parameters
    ==========
    constant_add : bool, defaults to False
        Whether to add constant noise.
    user_provided_add : bool, defaults to False
        Whether to add user provided noise.
    scale_user_provided : bool, defaults to False
        Whether to scale uncertainty in provided noise.
    rectified_linear_output_dependent_add : bool, defaults to False
        Whether to add rectified linear output-dependent noise.
    """

    def __init__(
        self,
        constant_add=False,
        user_provided_add=False,
        scale_user_provided=False,
        rectified_linear_output_dependent_add=False,
    ):
        self.parameters = np.zeros((3,))
        if constant_add:
            self.parameters[0] = 1
        if user_provided_add:
            self.parameters[1] = 1
            if scale_user_provided:
                self.parameters[1] += 1
        if rectified_linear_output_dependent_add:
            self.parameters[2] = 1

    def hyperparameter_count(self):
        """Counts the number of hyperparameters this noise function has.

        Returns
        -------

        count : int
            The amount of hyperparameters.
        """
        noise_N = 0
        if self.parameters[0] == 1:
            noise_N += 1
        if self.parameters[1] == 2:
            noise_N += 1
        if self.parameters[2] == 1:
            noise_N += 2
        return noise_N

    def hyperparameter_info(self):
        """Gives information on the names of hyperparameters for setting them in other parts of the program.

        Returns
        -------
        hyper_info : array_like
            A list of tuples containing hyperparameter names along with how many parameters with such a name there are, in the order they are used in computations.
        """
        hyper_info = []
        if self.parameters[0] == 1:
            hyper_info.append(("noise_log_scale", 1))
        if self.parameters[1] == 2:
            hyper_info.append(("noise_provided_log_multiplier", 1))
        if self.parameters[2] == 1:
            hyper_info.append(("noise_rectified_log_multiplier", 2))

        return hyper_info

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
        noise_info : NoiseInfo
            The additional info represented as a ``NoiseInfo`` object.
        """
        _, D = X.shape
        noise_N = self.hyperparameter_count()
        tol = 1e-6
        info = NoiseInfo(
            np.full((noise_N,), -np.inf),
            np.full((noise_N,), np.inf),
            np.full((noise_N,), -np.inf),
            np.full((noise_N,), np.inf),
            np.full((noise_N,), np.nan),
        )

        if np.size(y) <= 1:
            y = np.array([0, 1])
        height = np.max(y) - np.min(y)

        i = 0
        # Base constant noise
        if self.parameters[0] == 1:
            # Constant noise (log standard deviation)
            info.LB[i] = np.log(tol)
            info.UB[i] = np.log(height)
            info.PLB[i] = 0.5 * np.log(tol)
            info.PUB[i] = np.log(np.std(y, ddof=1))
            info.x0[i] = np.log(1e-3)
            i += 1

        # User provided noise.
        if self.parameters[1] == 2:
            info.LB[i] = np.log(1e-3)
            info.UB[i] = np.log(1e3)
            info.PLB[i] = np.log(0.5)
            info.PUB[i] = np.log(2)
            info.x0[i] = np.log(1)
            i += 1

        # Output dependent noise
        if self.parameters[2] == 1:
            min_y = np.min(y)
            max_y = np.max(y)
            info.LB[i] = min_y
            info.UB[i] = max_y
            info.PLB[i] = min_y
            info.PUB[i] = np.max(max_y - 5 * D, min_y)
            info.x0[i] = np.max(max_y - 10 * D, min_y)
            i += 1

            info.LB[i] = np.log(1e-3)
            info.UB[i] = np.log(0.1)
            info.PLB[i] = np.log(0.01)
            info.PUB[i] = np.log(0.1)
            info.x0[i] = np.log(0.1)
            i += 1

        # Plausible starting point
        i_nan = np.isnan(info.x0)
        info.x0[i_nan] = 0.5 * (info.PLB[i_nan] + info.PUB[i_nan])

        return info

    def compute(self, hyp, X, y, s2, compute_grad=False):
        """Computes the noise function at test points.

        Parameters
        ----------
        hyp : array_like
            Vector of hyperparameters.
        X : array_like
            Matrix of test points.
        y : array_like
            Vector of test targets.
        s2 : array_like
            Estimated noise variance associated with each training input vector.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to the hyperparameters.

        Returns
        -------
        sn2 : array_like
            The variance of observation noise evaluated at test points.
        dsn2 : array_like, optional
            The gradient.
        """
        N, _ = X.shape
        noise_N = self.hyperparameter_count()

        if hyp.size != noise_N:
            raise ValueError(
                "Expected %d noise function hyperparameters, %d passed instead."
                % (noise_N, hyp.size)
            )
        if hyp.ndim != 1:
            raise ValueError(
                "Noise function output is available only for one-sample hyperparameter inputs."
            )

        dsn2 = None
        if compute_grad:
            if any(x > 0 for x in self.parameters[1:]):
                dsn2 = np.zeros((N, noise_N))
            else:
                dsn2 = np.zeros((1, noise_N))

        i = 0
        sn2 = 0
        if self.parameters[0] == 0:
            sn2 = np.spacing(1.0)
        else:
            sn2 = np.exp(2 * hyp[i])
            if compute_grad:
                dsn2[:, i] = 2 * sn2
            i += 1

        if self.parameters[1] == 1:
            sn2 += s2
        elif self.parameters[1] == 2:
            sn2 += np.exp(hyp[i]) * s2
            if compute_grad:
                dsn2[:, i] = np.exp(hyp[i]) * s2
            i += 1

        if self.parameters[2] == 1:
            if y is not None:
                y_tresh = hyp[i]
                w2 = np.exp(2 * hyp[i + 1])
                zz = np.maximum(0, y_tresh - y)

                sn2 += w2 @ zz ** 2
                if compute_grad:
                    dsn2[:, i] = 2 * w2 * (y_tresh - y) * (zz > 0)
                    dsn2[:, i + 1] = 2 * w2 * zz ** 2
            i += 2

        if compute_grad:
            return sn2, dsn2

        return sn2


class NoiseInfo:
    def __init__(self, LB, UB, PLB, PUB, x0):
        self.LB = LB
        self.UB = UB
        self.PLB = PLB
        self.PUB = PUB
        self.x0 = x0
