"""Module for different noise functions used by Gaussian processes."""

import numpy as np


class GaussianNoise:
    """
    Gaussian noise function.

    Total noise variance is is obtained by summing the independent
    contribution of each noise feature.

    Parameters
    ==========
    constant_add : bool, defaults to False
        Whether to add constant noise.
    user_provided_add : bool, defaults to False
        Whether to add user provided (input) noise.
    scale_user_provided : bool, defaults to False
        Whether to scale uncertainty in provided noise. If
        ``user_provided_add = False`` then this does nothing.
    rectified_linear_output_dependent_add : bool, defaults to False
        Whether to add rectified linear output-dependent noise.
    """

    def __init__(
        self,
        constant_add: bool = False,
        user_provided_add: bool = False,
        scale_user_provided: bool = False,
        rectified_linear_output_dependent_add: bool = False,
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
        """
        Returns the number of hyperparameters this noise function has.

        Returns
        -------
        count : int
            The number of hyperparameters.
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
        """
        Returns information on the names of hyperparameters for setting
        them in other parts of the program.

        Returns
        -------
        hyper_info : array_like
            A list of tuples of hyperparameter names and their number,
            in the order they are in the hyperparameter array.
        """
        hyper_info = []
        if self.parameters[0] == 1:
            hyper_info.append(("noise_log_scale", 1))
        if self.parameters[1] == 2:
            hyper_info.append(("noise_provided_log_multiplier", 1))
        if self.parameters[2] == 1:
            hyper_info.append(("noise_rectified_log_multiplier", 2))

        return hyper_info

    def get_bounds_info(self, X: np.ndarray, y: np.ndarray):
        """
        Returns information on the lower, upper, plausible lower
        and plausible upper bounds of the hyperparameters of this
        noise function.

        Parameters
        ----------
        X : ndarray, shape (N, D)
            A 2D array where each row is a test point.
        y : ndarray, shape (N, 1)
            A 2D array where each row is a test target.

        Returns
        -------
        noise_bound_info: dict
            A dictionary containing the bound info with the following elements:

            **LB** : np.ndarray, shape (noise_N, 1)
                    The lower bounds of the hyperparameters.
            **UB** : np.ndarray, shape (noise_N, 1)
                    The upper bounds of the hyperparameters.
            **PLB** : np.ndarray, shape (noise_N, 1)
                    The plausible lower bounds of the hyperparameters.
            **PUB** : np.ndarray, shape (noise_N, 1)
                    The plausible upper bounds of the hyperparameters.
            **x0** : np.ndarray, shape (noise_N, 1)
                    The plausible starting point.

            where ``noise_N`` is the number of hyperparameters.
        """
        _, D = X.shape
        noise_N = self.hyperparameter_count()
        tol = 1e-6
        lower_bounds = np.full((noise_N,), -np.inf)
        upper_bounds = np.full((noise_N,), np.inf)
        plausible_lower_bounds = np.full((noise_N,), -np.inf)
        plausible_upper_bounds = np.full((noise_N,), np.inf)
        plausible_x0 = np.full((noise_N,), np.nan)

        if np.size(y) <= 1:
            y = np.array([0, 1])
        height = np.max(y) - np.min(y)

        i = 0
        # Base constant noise
        if self.parameters[0] == 1:
            # Constant noise (log standard deviation)
            lower_bounds[i] = np.log(tol)
            upper_bounds[i] = np.log(height)
            plausible_lower_bounds[i] = 0.5 * np.log(tol)
            plausible_upper_bounds[i] = np.log(np.std(y, ddof=1))
            plausible_x0[i] = np.log(1e-3)
            i += 1

        # User provided noise.
        if self.parameters[1] == 2:
            lower_bounds[i] = np.log(1e-3)
            upper_bounds[i] = np.log(1e3)
            plausible_lower_bounds[i] = np.log(0.5)
            plausible_upper_bounds[i] = np.log(2)
            plausible_x0[i] = np.log(1)
            i += 1

        # Output dependent noise
        if self.parameters[2] == 1:
            min_y = np.min(y)
            max_y = np.max(y)
            lower_bounds[i] = min_y
            upper_bounds[i] = max_y
            plausible_lower_bounds[i] = min_y
            plausible_upper_bounds[i] = np.maximum(max_y - 5 * D, min_y)
            plausible_x0[i] = np.maximum(max_y - 10 * D, min_y)
            i += 1

            lower_bounds[i] = np.log(1e-3)
            upper_bounds[i] = np.log(0.1)
            plausible_lower_bounds[i] = np.log(0.01)
            plausible_upper_bounds[i] = np.log(0.1)
            plausible_x0[i] = np.log(0.1)
            i += 1

        # Plausible starting point
        i_nan = np.isnan(plausible_x0)
        plausible_x0[i_nan] = 0.5 * (
            plausible_lower_bounds[i_nan] + plausible_upper_bounds[i_nan]
        )

        noise_bound_info = {
            "LB": lower_bounds,
            "PLB": plausible_lower_bounds,
            "PUB": plausible_upper_bounds,
            "UB": upper_bounds,
            "x0": plausible_x0,
        }
        return noise_bound_info

    def compute(
        self,
        hyp: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        s2: np.ndarray = None,
        compute_grad: bool = False,
    ):
        """
        Compute the noise function at test points, that is the variance
        of observation noise evaluated at the test points.

        Parameters
        ----------
        hyp : ndarray, shape (noise_N,)
            A 1D array of hyperparameters, where ``noise_N`` is
            the number returned by the function ``hyperparameter_count``.
        X : ndarray, shape (N, D)
            A 2D array where each row is a test point.
        y : ndarray, shape (N, 1)
            A 2D array where each row is a test target.
        s2 : ndarray, shape (N, 1), optional
            A 2D array of estimated noise variance associated
            with each test point. Only required if
            ``user_provided_add = True``.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to the
            hyperparameters.

        Returns
        -------
        sn2 : ndarray
            The variance of observation noise evaluated at test points.
            If there is no input or output dependent noise, ``sn2`` is
            a scalar since it does not change, while otherwise it
            is of the same shape as ``s2``.
        dsn2 : ndarray, optional
            The gradient with respect to hyperparameters. If there is no
            input or output dependent noise, ``dsn2`` is of shape
            ``(1, noise_N)`` while otherwise it is of shape ``(N, noise_N)``.

        Raises
        ------
        ValueError
            Raised when `hyp` has not the expected number of hyperparameters.
        ValueError
            Raised when `hyp` is not an 1D array but of higher dimension.
        """
        N, _ = X.shape
        noise_N = self.hyperparameter_count()

        if hyp.size != noise_N:
            raise ValueError(
                f"Expected {noise_N} noise function hyperparameters, "
                f"{hyp.size} passed instead."
            )
        if hyp.ndim != 1:
            raise ValueError(
                "Noise function output is available only for "
                "one-sample hyperparameter inputs."
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
                dsn2[:, i : i + 1] = np.exp(hyp[i]) * s2
            i += 1

        if self.parameters[2] == 1:
            if y is not None:
                y_tresh = hyp[i]
                w2 = np.exp(2 * hyp[i + 1])
                zz = np.maximum(0, y_tresh - y)

                sn2 += w2 * zz ** 2
                if compute_grad:
                    dsn2[:, i : i + 1] = 2 * w2 * (y_tresh - y) * (zz > 0)
                    dsn2[:, i + 1 : i + 2] = 2 * w2 * zz ** 2
            i += 2

        if compute_grad:
            return sn2, dsn2

        return sn2
