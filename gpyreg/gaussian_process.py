"""Module for Gaussian processes."""

import math
import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import gpyreg.covariance_functions
import gpyreg.mean_functions

from gpyreg.f_min_fill import f_min_fill
from gpyreg.slice_sample import SliceSampler


class GP:
    """A single gaussian process.

    Parameters
    ==========
    D : int
        The dimension for the gaussian process.
    covariance : object
        The covariance function to use.
    mean : object
        The mean function to use.
    noise : object
        The noise function to use.
    s2 : array_like, optional
        User-provided noise.
    """

    def __init__(self, D, covariance, mean, noise, s2=None):
        self.D = D
        self.covariance = covariance
        self.mean = mean
        self.noise = noise
        self.s2 = None
        if s2 is not None:
            self.s2 = s2.copy()
        self.X = None
        self.y = None
        self.posteriors = None
        self.set_bounds()
        self.set_priors()

    def __repr__(self):
        dimension = "Dimension: " + str(self.D) + "\n"

        cov_N = self.covariance.hyperparameter_count(self.D)
        cov = "Covariance function: " + self.covariance.__class__.__name__
        if self.covariance.__class__.__name__ == "Matern":
            cov += "(degree=" + str(self.covariance.degree) + ")\n"
        if cov_N == 1:
            cov += ", " + str(cov_N) + " parameter\n"
        else:
            cov += ", " + str(cov_N) + " parameters\n"

        mean_N = self.mean.hyperparameter_count(self.D)
        mean = "Mean function: " + self.mean.__class__.__name__
        if mean_N == 1:
            mean += ", " + str(mean_N) + " parameter\n"
        else:
            mean += ", " + str(mean_N) + " parameters\n"

        noise_N = self.noise.hyperparameter_count()
        noise = "Noise function: " + self.noise.__class__.__name__
        if np.any(self.noise.parameters):
            noise += "("
            add_flag = False
            if self.noise.parameters[0] == 1:
                noise += "constant_add=True"
                add_flag = True

            if self.noise.parameters[1] == 1:
                if add_flag:
                    noise += ", "
                noise += "user_provided_add=True"

            if self.noise.parameters[1] == 2:
                if add_flag:
                    noise += ", "
                noise += "scale_user_provided=True"

            if self.noise.parameters[2] == 1:
                if add_flag:
                    noise += ", "
                noise += "rectified_linear_output_dependent_add=True"

            noise += ")"

        if noise_N == 1:
            noise += ", " + str(noise_N) + " parameter\n"
        else:
            noise += ", " + str(noise_N) + " parameters\n"

        priors = "Hyperparameter priors: "
        if self.no_prior:
            priors += "none\n"
        else:
            priors += "present\n"
        samples = "Hyperparameter samples: "
        if self.posteriors is None:
            samples += "0"
        else:
            samples += str(np.size(self.posteriors))

        total = dimension + cov + mean + noise + priors + samples
        return total

    def set_bounds(self, bounds=None):
        """Sets the hyperparameter priors.

        Parameters
        ==========
        bounds : dict, optional
            A dictionary of hyperparameter names and tuples of their lower and
            upper bounds. If None is given, the the lower bounds will be set
            to -inf and upper bounds to +inf.

        Returns
        =======
        missing : list
            A list of missing hyperparameter bounds that should have been
            given.
        """

        cov_N = self.covariance.hyperparameter_count(self.D)
        cov_hyper_info = self.covariance.hyperparameter_info(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        mean_hyper_info = self.mean.hyperparameter_info(self.D)
        noise_N = self.noise.hyperparameter_count()
        noise_hyper_info = self.noise.hyperparameter_info()
        hyper_info = cov_hyper_info + noise_hyper_info + mean_hyper_info

        hyp_N = cov_N + mean_N + noise_N
        self.lower_bounds = np.full((hyp_N,), -np.inf)
        self.upper_bounds = np.full((hyp_N,), np.inf)

        if bounds is None:
            return []

        lower = 0
        missing = []

        for info in hyper_info:
            try:
                vals = bounds[info[0]]
            except KeyError:
                missing.append(info[0])
                vals = None

            # None indicates no bounds.
            if vals is not None:
                upper = lower + info[1]
                lb, ub = vals
                i = range(lower, upper)
                self.lower_bounds[i] = lb
                self.upper_bounds[i] = ub

            lower += info[1]

        return missing

    def get_bounds(self):
        """Gets the current hyperparameter lower and upper bounds.

        Returns
        =======
        bounds_dict : dict
            A dictionary of the current hyperparameters and their bounds.
        """
        return self.bounds_to_dict(self.lower_bounds, self.upper_bounds)

    def bounds_to_dict(self, lower_bounds, upper_bounds):
        """Converts the given hyperparemeter lower and upper bounds to
        a dictionary.

        Parameters
        ==========
        lower_bounds : array_like
            An 1D array of lower bounds. Must be the same size as
            the array for the upper bounds.
        upper_bounds : array_like
            An 1D array of upper bounds. Must be the same size as
            the array for the lower bounds.
        Returns
        =======
        bounds_dict : dict
            A dictionary of the current hyperparemters and the given
            bounds.
        """

        cov_hyper_info = self.covariance.hyperparameter_info(self.D)
        mean_hyper_info = self.mean.hyperparameter_info(self.D)
        noise_hyper_info = self.noise.hyperparameter_info()
        hyper_info = cov_hyper_info + noise_hyper_info + mean_hyper_info

        bounds_dict = {}
        lower = 0

        for info in hyper_info:
            upper = lower + info[1]
            i = range(lower, upper)

            # Check if we can compress the bounds dictionary.
            lower_same = np.all(lower_bounds[i] == lower_bounds[lower])
            upper_same = np.all(upper_bounds[i] == upper_bounds[lower])
            if lower_same and upper_same:
                i = lower

            bounds_dict[info[0]] = (lower_bounds[i], upper_bounds[i])
            lower += info[1]

        return bounds_dict

    def get_recommended_bounds(self):
        """Gets the recommended hyperparameter lower and upper bounds.

        Returns
        =======
        bounds_dict : dict
            A dictionary of the hyperparameters and their recommended bounds.
        """
        if self.X or self.y is None:
            raise Exception("GP does not have X or y set!")

        cov_N = self.covariance.hyperparameter_count(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        noise_N = self.noise.hyperparameter_count()

        cov_info = self.covariance.get_info(self.X, self.y)
        mean_info = self.mean.get_info(self.X, self.y)
        noise_info = self.noise.get_info(self.X, self.y)

        lb = self.lower_bounds.copy()
        ub = self.upper_bounds.copy()

        lb_cov = lb[0:cov_N]
        lb_noise = lb[cov_N : cov_N + noise_N]
        lb_mean = lb[cov_N + noise_N : cov_N + noise_N + mean_N]

        lb_cov[np.isinf(lb_cov)] = cov_info.LB[np.isinf(lb_cov)]
        lb_noise[np.isinf(lb_noise)] = noise_info.LB[np.isinf(lb_noise)]
        lb_mean[np.isinf(lb_mean)] = mean_info.LB[np.isinf(lb_mean)]

        ub_cov = ub[0:cov_N]
        ub_noise = ub[cov_N : cov_N + noise_N]
        ub_mean = ub[cov_N + noise_N : cov_N + noise_N + mean_N]

        ub_cov[np.isinf(ub_cov)] = cov_info.UB[np.isinf(ub_cov)]
        ub_noise[np.isinf(ub_noise)] = noise_info.UB[np.isinf(ub_noise)]
        ub_mean[np.isinf(ub_mean)] = mean_info.UB[np.isinf(ub_mean)]

        lb = np.concatenate([lb_cov, lb_noise, lb_mean])
        ub = np.concatenate([ub_cov, ub_noise, ub_mean])
        ub = np.maximum(lb, ub)

        return self.bounds_to_dict(lb, ub)

    def get_priors(self):
        """Gets the current hyperparameter priors.

        Returns
        =======
        hyper_priors : dict
            A dictionary of the current hyperparameter priors.
        """

        cov_hyper_info = self.covariance.hyperparameter_info(self.D)
        mean_hyper_info = self.mean.hyperparameter_info(self.D)
        noise_hyper_info = self.noise.hyperparameter_info()
        hyper_info = cov_hyper_info + noise_hyper_info + mean_hyper_info

        hyper_priors = {}
        lower = 0

        mu = self.hyper_priors["mu"].copy()
        sigma = self.hyper_priors["sigma"].copy()
        df = self.hyper_priors["df"].copy()
        a = self.hyper_priors["a"].copy()
        b = self.hyper_priors["b"].copy()

        for info in hyper_info:
            upper = lower + info[1]
            i = range(lower, upper)

            # Check if all the priors of this hyperparameter are equal
            # If they are, we can compress the prior dictionary.
            mu_same = np.all((mu[i] == mu[lower])) or np.all(np.isnan(mu[i]))
            sigma_same = np.all((sigma[i] == sigma[lower])) or np.all(
                np.isnan(sigma[i])
            )
            df_same = np.all((df[i] == df[lower])) or np.all(np.isnan(df[i]))
            a_same = np.all((a[i] == a[lower])) or np.all(np.isnan(a[i]))
            b_same = np.all((b[i] == b[lower])) or np.all(np.isnan(b[i]))
            if mu_same and sigma_same and df_same and a_same and b_same:
                i = lower

            prior_type = prior_params = None
            if (
                np.isfinite(a[i])
                and np.isfinite(b[i])
                and np.isfinite(sigma[i])
            ):
                if df[i] == 0 or df[i] == np.inf:
                    prior_type = "smoothbox"
                    prior_params = (a[i], b[i], sigma[i])
                elif df[i] > 0:
                    prior_type = "smoothbox_student_t"
                    prior_params = (a[i], b[i], sigma[i], df[i])
            elif np.isfinite(mu[i]) and np.isfinite(sigma[i]):
                if df[i] == 0 or df[i] == np.inf:
                    prior_type = "gaussian"
                    prior_params = (mu[i], sigma[i])
                elif df[i] > 0:
                    prior_type = "student_t"
                    prior_params = (mu[i], sigma[i], df[i])

            if prior_type is not None and prior_params is not None:
                hyper_priors[info[0]] = (prior_type, prior_params)
            else:
                hyper_priors[info[0]] = None

            lower += info[1]

        return hyper_priors

    def set_priors(self, priors=None):
        """Sets the hyperparameter priors.

        Parameters
        ==========
        priors : dict, optional
            A dictionary of hyperparameter names and tuples of their values.
            If None is given, there will be no prior.

        Returns
        =======
        missing : list
            A list of missing hyperparameter priors that should have been
            given. If not empty, the elements have been set to no prior.
        """

        self.no_prior = False
        if priors is None:
            self.no_prior = True

        cov_N = self.covariance.hyperparameter_count(self.D)
        cov_hyper_info = self.covariance.hyperparameter_info(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        mean_hyper_info = self.mean.hyperparameter_info(self.D)
        noise_N = self.noise.hyperparameter_count()
        noise_hyper_info = self.noise.hyperparameter_info()
        hyper_info = cov_hyper_info + noise_hyper_info + mean_hyper_info

        hyp_N = cov_N + mean_N + noise_N
        # Set up a hyperprior dictionary with default values which can
        # be updated individually later.
        self.hyper_priors = {
            "mu": np.full((hyp_N,), np.nan),
            "sigma": np.full((hyp_N,), np.nan),
            "df": np.full((hyp_N,), np.nan),
            "a": np.full((hyp_N,), np.nan),
            "b": np.full((hyp_N,), np.nan),
        }

        non_trivial_flag = False
        lower = 0
        missing = []

        for info in hyper_info:
            if self.no_prior:
                vals = None
            else:
                try:
                    vals = priors[info[0]]
                except KeyError:
                    missing.append(info[0])
                    vals = None

            # None indicates no prior
            if vals is not None:
                non_trivial_flag = True
                upper = lower + info[1]
                prior_type, prior_params = vals
                i = range(lower, upper)

                if prior_type == "gaussian":
                    mu, sigma = prior_params
                    self.hyper_priors["mu"][i] = mu
                    self.hyper_priors["sigma"][i] = sigma
                    # Implicit flag for gaussian, is set to inf later.
                    self.hyper_priors["df"][i] = 0
                elif prior_type == "student_t":
                    mu, sigma, df = prior_params
                    self.hyper_priors["mu"][i] = mu
                    self.hyper_priors["sigma"][i] = sigma
                    self.hyper_priors["df"][i] = df
                elif prior_type == "smoothbox":
                    a, b, sigma = prior_params
                    self.hyper_priors["a"][i] = a
                    self.hyper_priors["b"][i] = b
                    self.hyper_priors["sigma"][i] = sigma
                    # Implicit flag for gaussian, is set to inf later.
                    self.hyper_priors["df"][i] = 0
                elif prior_type == "smoothbox_student_t":
                    a, b, sigma, df = prior_params
                    self.hyper_priors["a"][i] = a
                    self.hyper_priors["b"][i] = b
                    self.hyper_priors["sigma"][i] = sigma
                    # Implicit flag for gaussian, is set to inf later.
                    self.hyper_priors["df"][i] = df
                else:
                    raise ValueError("Unknown hyperprior type " + prior_type)

            lower += info[1]

        self.no_prior = non_trivial_flag is not True
        return missing

    def get_hyperparameters(self, as_array=False):
        """Gets the current hyperparameters for the Gaussian process.

        Parameters
        ==========
        as_array : bool, defaults to False
            Whether to return the hyperparameters as an array, or a list
            of dictionaries for each sample.

        Returns
        =======
        hyp : object
            The hyperparameteres in the form specified by as_array.
        """
        # If no hyperparameters have been set return an array/dict with NaN.
        if self.posteriors is None:
            cov_N = self.covariance.hyperparameter_count(self.D)
            mean_N = self.mean.hyperparameter_count(self.D)
            noise_N = self.noise.hyperparameter_count()
            hyp = np.full((1, cov_N + mean_N + noise_N), np.nan)
        else:
            hyp = np.zeros(
                (np.size(self.posteriors), np.size(self.posteriors[0].hyp))
            )
            for i in range(0, np.size(self.posteriors)):
                # Copy for avoiding reference issues.
                hyp[i, :] = self.posteriors[i].hyp.copy()

        if as_array:
            return hyp

        return self.hyperparameters_to_dict(hyp)

    def set_hyperparameters(self, hyp_new, compute_posterior=True):
        """Sets new hyperparameters for the Gaussian process.

        Parameters
        ==========
        hyp_new : object
            The new hyperparameters. This can be an array of size
            (N0, hyp_n) where hyp_N is the number of hyperparametes, and
            N0 is the amount of samples, a single dictionary with
            hyperparameter names and values, or a list of dictionaries.
            The behaviour of passing a single dictionary and a list
            with one dictionary is equivalent.
        compute_posterior : bool, defaults to True
            Whether to compute the posterior for the new hyperparameters.
        """
        if isinstance(hyp_new, np.ndarray):
            cov_N = self.covariance.hyperparameter_count(self.D)
            mean_N = self.mean.hyperparameter_count(self.D)
            noise_N = self.noise.hyperparameter_count()

            if hyp_new.ndim == 1:
                hyp_new = np.reshape(hyp_new, (1, -1))

            if hyp_new.shape[1] != cov_N + mean_N + noise_N:
                raise ValueError(
                    "Input hyperparameter array is the wrong shape!"
                )
            self.update(hyp=hyp_new, compute_posterior=compute_posterior)
        else:
            hyp_new_arr = self.hyperparameters_from_dict(hyp_new)
            self.update(hyp=hyp_new_arr, compute_posterior=compute_posterior)

    def hyperparameters_to_dict(self, hyp_arr):
        """Converts a hyperparameter array to a list of dictionaries
        for each sample with hyperparameter names and values.

        Parameters
        ==========
        hyp_arr : array_like
            An 1D or 2D array containing hyperparameters.

        Returns
        =======
        hyp_dict : object
            The list of dictonaries for each sample with hyperparameter
            names and values.
        """
        hyp = []
        cov_N = self.covariance.hyperparameter_count(self.D)
        cov_hyper_info = self.covariance.hyperparameter_info(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        mean_hyper_info = self.mean.hyperparameter_info(self.D)
        noise_N = self.noise.hyperparameter_count()
        noise_hyper_info = self.noise.hyperparameter_info()

        hyper_info = cov_hyper_info + noise_hyper_info + mean_hyper_info

        if hyp_arr.ndim == 1:
            hyp_arr = np.reshape(hyp_arr, (1, -1))

        if hyp_arr.shape[1] != cov_N + mean_N + noise_N:
            raise ValueError("Input hyperparameter array is the wrong shape!")

        for i in range(0, hyp_arr.shape[0]):
            # Make sure there are no accidents with references etc.
            hyp_tmp = hyp_arr[i, :].copy()
            hyp_dict = {}
            i = 0

            for info in hyper_info:
                hyp_dict[info[0]] = hyp_tmp[i : i + info[1]]
                i += info[1]

            hyp.append(hyp_dict)

        return hyp

    def hyperparameters_from_dict(self, hyp_dict):
        """Converts a list of hyperparameter dictionaries to a hyperparameter
        array.

        Parameters
        ==========
        hyp_dict : object
            A list of hyperparameter dictionaries with hyperparameter names
            and values. One can also pass just one dictionary instead of a
            list with one element.

        Returns
        =======
        hyp_arr : array_like
            The hyperparameter array.
        """
        if isinstance(hyp_dict, dict):
            hyp_dict = [hyp_dict]

        cov_N = self.covariance.hyperparameter_count(self.D)
        cov_hyper_info = self.covariance.hyperparameter_info(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        mean_hyper_info = self.mean.hyperparameter_info(self.D)
        noise_N = self.noise.hyperparameter_count()
        noise_hyper_info = self.noise.hyperparameter_info()

        hyper_info = cov_hyper_info + noise_hyper_info + mean_hyper_info

        hyp_N = cov_N + mean_N + noise_N
        hyp_new_arr = np.zeros((len(hyp_dict), hyp_N))

        for i, hyp_tmp in enumerate(hyp_dict):
            j = 0

            for info in hyper_info:
                hyp_new_arr[i, j : j + info[1]] = hyp_tmp[info[0]]
                j += info[1]

        return hyp_new_arr

    def update(
        self,
        X_new=None,
        y_new=None,
        s2_new=None,
        hyp=None,
        compute_posterior=True,
    ):
        """Adds new data to the gaussian process.

        Parameters
        ==========
        X_new : array_like, optional
            New training inputs, that will be concatenated with the old ones.
        y_new : array_like, optional
            New training targets, that will be concatenated with the old ones.
        s2_new : array_like, optional
            New input-dependent noise that will be concatenated with the
            old ones.
        hyp : array_like, optional
            New hyperparameters that will replace the old ones.
        compute_posterior : bool, defaults to True
            Whether to compute the new posterior or not.
        """

        # Create local copies so we won't get trouble
        # with references later.
        if X_new is not None:
            X_new = X_new.copy()
            if X_new.ndim == 1:
                X_new = np.reshape(X_new, (1, -1))
        if y_new is not None:
            y_new = y_new.copy()
            if y_new.ndim == 1:
                y_new = np.reshape(y_new, (-1, 1))
        if s2_new is not None:
            s2_new = s2_new.copy()
        if hyp is not None:
            hyp = hyp.copy()

        # Check whether to do a rank-1 update.
        rank_one_update = False
        if X_new is not None and y_new is not None:
            if (
                self.X is not None
                and self.y is not None
                and X_new.shape[0] == 1
                and y_new.shape[0] == 1
                and s2_new is None
            ):
                rank_one_update = True

        if rank_one_update:
            cov_N = self.covariance.hyperparameter_count(self.D)
            # mean_N = self.mean.hyperparameter_count(self.D)
            noise_N = self.noise.hyperparameter_count()

            # Compute prediction for all samples.
            m_star, v_star = self.predict(
                X_new, y_new, add_noise=True, separate_samples=True
            )
            s_N = np.size(self.posteriors)

            # Loop over hyperparameter samples.
            for s in range(0, s_N):
                hyp_s = self.posteriors[s].hyp

                hyp_noise = hyp_s[cov_N : cov_N + noise_N]
                sn2 = self.noise.compute(hyp_noise, X_new, y_new, 0)
                sn2_eff = sn2 * self.posteriors[s].sn2_mult

                # Compute covariance and cross-covariance.
                hyp_cov = hyp_s[0:cov_N]
                K = self.covariance.compute(hyp_cov, X_new)
                Ks = self.covariance.compute(hyp_cov, self.X, X_new)

                L = self.posteriors[s].L
                L_chol = self.posteriors[s].L_chol

                if L_chol:  # High-noise parametrization
                    alpha_update = (
                        sp.linalg.solve_triangular(
                            L,
                            sp.linalg.solve_triangular(L, Ks, trans=1),
                            trans=0,
                        )
                        / sn2_eff
                    )
                    new_L_column = (
                        sp.linalg.solve_triangular(L, Ks, trans=1) / sn2_eff
                    )
                    self.posteriors[s].L = np.block(
                        [
                            [L, new_L_column],
                            [
                                np.zeros((1, L.shape[0])),
                                np.sqrt(
                                    1
                                    + K / sn2_eff
                                    - np.dot(new_L_column.T, new_L_column)
                                ),
                            ],
                        ]
                    )
                else:  # Low-noies parametrization
                    alpha_update = np.dot(-L, Ks)
                    v = -alpha_update / v_star[:, s]
                    self.posteriors[s].L = np.block(
                        [
                            [L + np.dot(v, alpha_update.T), -v],
                            [-v.T, -1 / v_star[:, s]],
                        ]
                    )

                self.posteriors[s].sW = np.concatenate(
                    (self.posteriors[s].sW, np.array([[1 / np.sqrt(sn2_eff)]]))
                )

                # alpha_update now contains (K + \sigma^2 I) \ k*
                self.posteriors[s].alpha = np.concatenate(
                    (self.posteriors[s].alpha, np.array([[0]]))
                ) + (m_star[:, s] - y_new) / v_star[:, s] * np.concatenate(
                    (alpha_update, np.array([[-1]]))
                )

        if X_new is not None:
            if self.X is None:
                self.X = X_new
            else:
                self.X = np.concatenate((self.X, X_new))

        if y_new is not None:
            if self.y is None:
                self.y = y_new
            else:
                self.y = np.concatenate((self.y, y_new))

        if s2_new is not None:
            if self.s2 is None:
                self.s2 = s2_new
            else:
                self.s2 = np.concatenate((self.s2, s2_new))

        if not rank_one_update and hyp is not None:
            s_N, _ = hyp.shape
            self.posteriors = np.empty((s_N,), dtype=Posterior)

            if compute_posterior and self.X is not None and self.y is not None:
                for i in range(0, s_N):
                    self.posteriors[i] = self.__core_computation(
                        hyp[i, :], 0, 0
                    )
            else:
                for i in range(0, s_N):
                    self.posteriors[i] = Posterior(
                        hyp[i, :], None, None, None, None, None
                    )

    def fit(self, X=None, y=None, s2=None, hyp0=None, options=None):
        """Trains gaussian process hyperparameters.

        Parameters
        ==========

        X : array_like, optional
            Inputs to train on.
        y : array_like, optional
            Targets to train on.
        options : dict, optional
            A dictionary of options for training. The possible options are:

                **opts_N** : int, defaults to 3
                    Number of hyperparameter optimization runs.
                **init_N** : int, defaults to 1024
                    Initial design size for hyperparameter optimization.
                **df_base** : int, defaults to 7
                    Default degrees of freedom for student's t prior.
                **n_samples** : int, defaults to 10
                    Number of hyperparameters to sample.
                **thin** : int, defaults to 5
                    Thinning parameter for slice sampling.
                **burn** : int, defaults to ``thin * n_samples``
                    Burn parameter for slice sampling.
                **use_recommended_bounds** : bool, defaults to True
                    Whether to only use user provided lower and upper bounds
                    or to try to choose them intelligently.

        Returns
        =======
        hyp : array_like
            The fitted hyperparameters.
        sampling_result : dict
            If sampling was performed this is a dictionary with info on the
            sampling run, and None otherwise.
        """
        ## Default options
        if options is None:
            options = {}
        opts_N = options.get("opts_N", 3)
        init_N = options.get("init_N", 2 ** 10)
        thin = options.get("thin", 5)
        df_base = options.get("df_base", 7)
        s_N = options.get("n_samples", 10)
        burn_in = options.get("burn", thin * s_N)
        use_recommended_bounds = options.get("use_recommended_bounds", True)

        # Initialize GP if requested.
        if X is not None:
            self.X = X
        else:
            X = self.X

        if y is not None:
            if y.ndim == 1:
                y = np.reshape(y, (-1, 1))
            self.y = y
        else:
            y = self.y

        if s2 is not None:
            self.s2 = s2
        else:
            s2 = self.s2

        cov_N = self.covariance.hyperparameter_count(self.D)
        # mean_N = self.mean.hyperparameter_count(self.D)
        noise_N = self.noise.hyperparameter_count()

        ## Initialize inference of GP hyperparameters (bounds, priors, etc.)

        cov_info = self.covariance.get_info(X, y)
        mean_info = self.mean.get_info(X, y)
        noise_info = self.noise.get_info(X, y)

        self.hyper_priors["df"][np.isnan(self.hyper_priors["df"])] = df_base
        if use_recommended_bounds:
            self.set_bounds(self.get_recommended_bounds())
        LB = self.lower_bounds
        UB = self.upper_bounds

        # Plausible bounds for generation of starting points
        PLB = np.concatenate([cov_info.PLB, noise_info.PLB, mean_info.PLB])
        PUB = np.concatenate([cov_info.PUB, noise_info.PUB, mean_info.PUB])
        PLB = np.minimum(np.maximum(PLB, LB), UB)
        PUB = np.maximum(np.minimum(PUB, UB), LB)

        # If we are not provided with an initial hyperparameter guess then
        # either use the current hyperparameters if they exist, or use
        # plausible lower and upper bounds to guess.
        if hyp0 is None:
            if self.posteriors is not None:
                hyp0 = self.get_hyperparameters(as_array=True)
            else:
                hyp0 = np.reshape(
                    np.minimum(np.maximum((PLB + PUB) / 2, LB), UB), (1, -1)
                )
        elif isinstance(hyp0, dict):
            hyp0 = self.hyperparameters_from_dict(hyp0)

        ## Hyperparameter optimization
        objective_f_1 = lambda hyp_: self.__gp_obj_fun(hyp_, False, False)

        # First evaluate GP log posterior on an informed space-filling design.
        t1_s = time.time()
        X0, y0 = f_min_fill(
            objective_f_1, hyp0, LB, UB, PLB, PUB, self.hyper_priors, init_N
        )
        hyp = X0[0:opts_N, :]
        widths_default = np.std(X0, axis=0, ddof=1)

        # Extract a good low-noise starting point for the 2nd optimization.
        if noise_N > 0 and 1 < opts_N < init_N:
            xx = X0[opts_N:, :]
            noise_y = y0[opts_N:]
            noise_params = xx[:, cov_N]

            # Order by noise parameter magnitude.
            order = np.argsort(noise_params)
            xx = xx[order, :]
            noise_y = noise_y[order]
            # Take the best amongst bottom 20% vectors.
            idx_best = np.argmin(
                noise_y[0 : math.ceil(0.2 * np.size(noise_y))]
            )
            hyp[1, :] = xx[idx_best, :]

        # Fix zero widths.
        idx0 = widths_default == 0
        if np.any(idx0):
            if np.shape(hyp)[1] > 1:
                std_hyp = np.std(hyp, axis=1, ddof=1)
                widths_default[idx0] = std_hyp[idx0]
                idx0 = widths_default == 0

            if np.any(idx0):
                widths_default[idx0] = np.minimum(1, UB[idx0] - LB[idx0])

        t1 = time.time() - t1_s

        # Check that hyperparameters are within bounds.
        # Note that with infinite upper and lower bounds we have to be careful
        # with spacing since it returns NaN.
        eps_LB = np.reshape(LB, (1, -1))
        eps_LB[np.isfinite(eps_LB)] += np.spacing(eps_LB[np.isfinite(eps_LB)])
        eps_UB = np.reshape(UB, (1, -1))
        eps_UB[np.isfinite(eps_UB)] -= np.spacing(eps_UB[np.isfinite(eps_UB)])
        hyp = np.minimum(eps_UB, np.maximum(eps_LB, hyp))

        # Perform optimization from most promising NOPTS hyperparameter
        # vectors.
        objective_f_2 = lambda hyp_: self.__gp_obj_fun(hyp_, True, False)
        nll = np.full((opts_N,), np.inf)

        t2_s = time.time()
        for i in range(0, opts_N):
            res = sp.optimize.minimize(
                fun=objective_f_2,
                x0=hyp[i, :],
                jac=True,
                bounds=list(zip(LB, UB)),
            )
            hyp[i, :] = res.x
            nll[i] = res.fun

        # Take the best hyperparameter vector.
        hyp_start = hyp[np.argmin(nll), :]
        t2 = time.time() - t2_s

        # In case n_samples is 0, just return the optimized hyperparameter
        # result.
        if s_N == 0:
            hyp_start = np.reshape(hyp_start, (1, -1))
            self.update(hyp=hyp_start)
            return hyp_start, None

        ## Sample from best hyperparameter vector using slice sampling

        t3_s = time.time()
        # Effective number of samples (thin after)
        eff_s_N = s_N * thin

        sample_f = lambda hyp_: self.__gp_obj_fun(hyp_, False, True)
        options = {"display": "off", "diagnostics": False}
        slicer = SliceSampler(
            sample_f, hyp_start, widths_default, LB, UB, options
        )
        sampling_result = slicer.sample(eff_s_N, burn=burn_in)

        # Thin samples
        hyp_pre_thin = sampling_result["samples"]
        hyp = hyp_pre_thin[thin - 1 :: thin, :]

        t3 = time.time() - t3_s
        print(hyp)
        print(t1, t2, t3)

        # Recompute GP with finalized hyperparameters.
        self.update(hyp=hyp)
        return hyp, sampling_result

    def __compute_log_priors(self, hyp, compute_grad):
        lp = 0
        dlp = None
        if compute_grad:
            dlp = np.zeros(hyp.shape)

        mu = self.hyper_priors["mu"]
        sigma = np.abs(self.hyper_priors["sigma"])
        df = self.hyper_priors["df"]
        a = self.hyper_priors["a"]
        b = self.hyper_priors["b"]
        lb = self.lower_bounds
        ub = self.upper_bounds

        f_idx = lb == ub
        sb_idx = (
            np.isfinite(a)
            & np.isfinite(b)
            & (df == 0 | ~np.isfinite(df))
            & ~np.isfinite(mu)
            & np.isfinite(sigma)
        )
        sb_t_idx = (
            np.isfinite(a)
            & np.isfinite(b)
            & (df > 0)
            & ~np.isfinite(mu)
            & np.isfinite(sigma)
            & np.isfinite(df)
        )
        u_idx = ~np.isfinite(mu) & ~np.isfinite(sigma)
        g_idx = (
            ~u_idx
            & ~sb_idx
            & (df == 0 | ~np.isfinite(df))
            & np.isfinite(sigma)
        )
        t_idx = ~u_idx & ~sb_t_idx & (df > 0) & np.isfinite(df)

        # Quadratic form
        z2 = np.zeros(hyp.shape)
        z2[g_idx | t_idx] = (
            (hyp[g_idx | t_idx] - mu[g_idx | t_idx]) / sigma[g_idx | t_idx]
        ) ** 2

        # Fixed prior
        if np.any(f_idx):
            if np.any(hyp[f_idx] != lb[f_idx]):
                lp = -np.inf
            if compute_grad:
                dlp[f_idx] = np.nan

        # Smooth box prior
        if np.any(sb_idx):
            # Normalization constant so that integral over pdf is 1.
            C = 1.0 + (b[sb_idx] - a[sb_idx]) / (
                sigma[sb_idx] * np.sqrt(2 * np.pi)
            )

            sb_idx_b = (hyp < a) & sb_idx
            sb_idx_a = (hyp > b) & sb_idx
            sb_idx_btw = (hyp >= a) & (hyp <= b) & sb_idx

            z2_tmp = np.zeros(hyp.shape)
            z2_tmp[sb_idx_b] = (
                (hyp[sb_idx_b] - a[sb_idx_b]) / sigma[sb_idx_b]
            ) ** 2
            z2_tmp[sb_idx_a] = (
                (hyp[sb_idx_a] - b[sb_idx_a]) / sigma[sb_idx_a]
            ) ** 2

            if np.any(sb_idx_b | sb_idx_a):
                lp -= 0.5 * np.sum(
                    np.log(
                        C ** 2 * 2 * np.pi * sigma[sb_idx_b | sb_idx_a] ** 2
                    )
                    + z2_tmp[sb_idx_b | sb_idx_a]
                )
            if np.any(sb_idx_btw):
                lp -= np.sum(
                    np.log(C * sigma[sb_idx_btw]) + np.log(np.sqrt(2 * np.pi))
                )

            if compute_grad:
                if np.any(sb_idx_b):
                    dlp[sb_idx_b] = (
                        -(hyp[sb_idx_b] - a[sb_idx_b]) / sigma[sb_idx_b] ** 2
                    )
                if np.any(sb_idx_a):
                    dlp[sb_idx_a] = (
                        -(hyp[sb_idx_a] - b[sb_idx_a]) / sigma[sb_idx_a] ** 2
                    )

        # Smooth box Student's t prior
        if np.any(sb_t_idx):
            # Normalization constant so that integral over pdf is 1.
            C = 1.0 + (b[sb_t_idx] - a[sb_t_idx]) * sp.special.gamma(
                0.5 * (df[sb_t_idx] + 1)
            ) / (
                sp.special.gamma(0.5 * df[sb_t_idx])
                * sigma[sb_t_idx]
                * np.sqrt(df[sb_t_idx] * np.pi)
            )

            sb_t_idx_b = (hyp < a) & sb_t_idx
            sb_t_idx_a = (hyp > b) & sb_t_idx
            sb_t_idx_btw = (hyp >= a) & (hyp <= b) & sb_t_idx

            z2_tmp = np.zeros(hyp.shape)
            z2_tmp[sb_t_idx_b] = (
                (hyp[sb_t_idx_b] - a[sb_t_idx_b]) / sigma[sb_t_idx_b]
            ) ** 2
            z2_tmp[sb_t_idx_a] = (
                (hyp[sb_t_idx_a] - b[sb_t_idx_a]) / sigma[sb_t_idx_a]
            ) ** 2

            if np.any(sb_t_idx_b | sb_t_idx_a):
                tmp_idx = sb_t_idx_b | sb_t_idx_a
                lp += np.sum(
                    sp.special.gammaln(0.5 * (df[tmp_idx] + 1))
                    - sp.special.gammaln(0.5 * df[tmp_idx])
                )
                lp += np.sum(
                    -0.5 * np.log(np.pi * df[tmp_idx])
                    - np.log(C * sigma[tmp_idx])
                    - 0.5
                    * (df[tmp_idx] + 1)
                    * np.log1p(z2_tmp[tmp_idx] / df[tmp_idx])
                )
            if np.any(sb_t_idx_btw):
                tmp_idx = sb_t_idx_btw
                lp += np.sum(
                    sp.special.gammaln(0.5 * (df[tmp_idx] + 1))
                    - sp.special.gammaln(0.5 * df[tmp_idx])
                )
                lp += np.sum(
                    -0.5 * np.log(np.pi * df[tmp_idx])
                    - np.log(C * sigma[tmp_idx])
                )

            if compute_grad:
                if np.any(sb_t_idx_b):
                    dlp[sb_t_idx_b] = (
                        -(df[sb_t_idx_b] + 1)
                        / df[sb_t_idx_b]
                        / (1 + z2_tmp[sb_t_idx_b] / df[sb_t_idx_b])
                        * (hyp[sb_t_idx_b] - a[sb_t_idx_b])
                        / sigma[sb_t_idx_b] ** 2
                    )
                if np.any(sb_t_idx_a):
                    dlp[sb_t_idx_a] = (
                        -(df[sb_t_idx_a] + 1)
                        / df[sb_t_idx_a]
                        / (1 + z2_tmp[sb_t_idx_a] / df[sb_t_idx_a])
                        * (hyp[sb_t_idx_a] - b[sb_t_idx_a])
                        / sigma[sb_t_idx_a] ** 2
                    )

        # Gaussian prior
        if np.any(g_idx):
            lp -= 0.5 * np.sum(
                np.log(2 * np.pi * sigma[g_idx] ** 2) + z2[g_idx]
            )
            if compute_grad:
                dlp[g_idx] = -(hyp[g_idx] - mu[g_idx]) / sigma[g_idx] ** 2

        # Student's t prior
        if np.any(t_idx):
            lp += np.sum(
                sp.special.gammaln(0.5 * (df[t_idx] + 1))
                - sp.special.gammaln(0.5 * df[t_idx])
            )
            lp += np.sum(
                -0.5 * np.log(np.pi * df[t_idx])
                - np.log(sigma[t_idx])
                - 0.5 * (df[t_idx] + 1) * np.log1p(z2[t_idx] / df[t_idx])
            )
            if compute_grad:
                dlp[t_idx] = (
                    -(df[t_idx] + 1)
                    / df[t_idx]
                    / (1 + z2[t_idx] / df[t_idx])
                    * (hyp[t_idx] - mu[t_idx])
                    / sigma[t_idx] ** 2
                )

        if compute_grad:
            return lp, dlp

        return lp

    def log_likelihood(self, hyp, compute_grad=False):
        """Computes (positive) log marginal likelihood of the GP for given
        hyperparameters.

        Parameters
        ==========
        hyp : object
            Either an 1D array or a dictionary of hyperparameters.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to hyperparameters.

        Returns
        =======
        lZ : float
            The positive log marginal likelihood.
        dlZ : array_like, optional
            The gradient.
        """
        if isinstance(hyp, dict):
            hyp = self.hyperparameters_from_dict(hyp)
        return -self.__compute_nlZ(hyp, compute_grad, False)

    def log_posterior(self, hyp, compute_grad=False):
        """Computes (positive) log marginal likelihood of the GP with added
        log prior for given hyperparameters (that is, the unnormalized log
        posterior).

        Parameters
        ==========
        hyp : object
            Either an 1D array or a dictionary of hyperparameters.
        compute_grad : bool, defaults to False
            Whether to compute the gradient with respect to hyperparameters.

        Returns
        =======
        lZ_plus_posterior : float
            The positive log marginal likelihood with added log prior.
        dlZ_plus_d_posterior : array_like, optional
            The gradient.
        """
        if isinstance(hyp, dict):
            hyp = self.hyperparameters_from_dict(hyp)

        return -self.__compute_nlZ(hyp, compute_grad, True)

    def __compute_nlZ(self, hyp, compute_grad, compute_prior):
        if compute_grad:
            nlZ, dnlZ = self.__core_computation(hyp, 1, compute_grad)
        else:
            nlZ = self.__core_computation(hyp, 1, compute_grad)

        if compute_prior:
            if compute_grad:
                P, dP = self.__compute_log_priors(hyp, compute_grad)
                nlZ -= P
                dnlZ -= dP
            else:
                P = self.__compute_log_priors(hyp, compute_grad)
                nlZ -= P

        if compute_grad:
            return nlZ, dnlZ

        return nlZ

    def __gp_obj_fun(self, hyp, compute_grad, swap_sign):
        if compute_grad:
            nlZ, dnlZ = self.__compute_nlZ(
                hyp, compute_grad, self.no_prior is not True
            )
        else:
            nlZ = self.__compute_nlZ(
                hyp, compute_grad, self.no_prior is not True
            )

        # Swap sign of negative log marginal likelihood (e.g. for sampling)
        if swap_sign:
            nlZ *= -1
            if compute_grad:
                dnlZ *= -1

        if compute_grad:
            return nlZ, dnlZ

        return nlZ

    def predict(
        self,
        x_star,
        y_star=None,
        s2_star=0,
        add_noise=False,
        separate_samples=False,
    ):
        """Predict the values of the gaussian process at given points.

        Parameters
        ==========
        x_star : array_like
            The points we want to predict the values at.
        y_star : array_like, optional
            True values at the points.
        s2_star : array_like, optional
            Noise at the points.
        add_noise : bool, defaults to True
            Whether to add noise to the prediction results.
        separate_samples : bool, defaults to False
            Whether to return the results separately for each hyperparameter
            sample or averaged.

        Returns
        =======
        mu : array_like
            Value of mu at the requested points.
        s2 : array_like
            Value noise the at the requested points.
        """
        if x_star.ndim == 1:
            x_star = np.reshape(x_star, (-1, 1))
        N, D = self.X.shape
        s_N = self.posteriors.size
        N_star = x_star.shape[0]

        # Preallocate space
        fmu = np.zeros((N_star, s_N))
        ymu = np.zeros((N_star, s_N))
        fs2 = np.zeros((N_star, s_N))
        ys2 = np.zeros((N_star, s_N))

        for s in range(0, s_N):
            hyp = self.posteriors[s].hyp
            alpha = self.posteriors[s].alpha
            L = self.posteriors[s].L
            L_chol = self.posteriors[s].L_chol
            sW = self.posteriors[s].sW
            sn2_mult = self.posteriors[s].sn2_mult

            cov_N = self.covariance.hyperparameter_count(D)
            mean_N = self.mean.hyperparameter_count(D)
            noise_N = self.noise.hyperparameter_count()
            sn2_star = self.noise.compute(
                hyp[cov_N : cov_N + noise_N], x_star, y_star, s2_star
            )
            m_star = np.reshape(
                self.mean.compute(
                    hyp[cov_N + noise_N : cov_N + noise_N + mean_N], x_star
                ),
                (-1, 1),
            )
            Ks = self.covariance.compute(hyp[0:cov_N], self.X, x_star)
            kss = self.covariance.compute(hyp[0:cov_N], x_star, "diag")

            if N > 0:
                fmu[:, s : s + 1] = m_star + np.dot(
                    Ks.T, alpha
                )  # Conditional mean
            else:
                fmu[:, s : s + 1] = m_star

            ymu[:, s] = fmu[:, s]
            if N > 0:
                if L_chol:
                    V = sp.linalg.solve_triangular(
                        L, np.tile(sW, (1, N_star)) * Ks, trans=1
                    )
                    fs2[:, s : s + 1] = kss - np.reshape(
                        np.sum(V * V, 0), (-1, 1)
                    )  # predictive variance
                else:
                    fs2[:, s : s + 1] = kss + np.reshape(
                        np.sum(Ks * np.dot(L, Ks), 0), (-1, 1)
                    )
            else:
                fs2[:, s : s + 1] = kss

            fs2[:, s] = np.maximum(
                fs2[:, s], 0
            )  # remove numerical noise, i.e. negative variances
            ys2[:, s : s + 1] = fs2[:, s : s + 1] + sn2_star * sn2_mult

        # Unless predictions for samples are requested separately
        # average over samples.
        if s_N > 1 and not separate_samples:
            fbar = np.reshape(np.sum(fmu, 1), (-1, 1)) / s_N
            ybar = np.reshape(np.sum(ymu, 1), (-1, 1)) / s_N
            vf = np.sum((fmu - fbar) ** 2, 1) / (s_N - 1)
            fs2 = np.reshape(np.sum(fs2, 1) / s_N + vf, (-1, 1))
            vy = np.sum((ymu - ybar) ** 2, 1) / (s_N - 1)
            ys2 = np.reshape(np.sum(ys2, 1) / s_N + vy, (-1, 1))

            fmu = fbar
            ymu = ybar

        if add_noise:
            return ymu, ys2
        return fmu, fs2

    def quad(self, mu, sigma, compute_var=False, separate_samples=False):
        """Bayesian quadrature for a gaussian process.

        Parameters
        ==========
        mu : array_like
        sigma : array_like
        compute_var : bool, defaults to False
            Whether to compute variance.
        separate_samples : bool, defaults to False
            Whether to return the results separately for each hyperparameter
            sample or averaged.
        """

        if not isinstance(
            self.covariance, gpyreg.covariance_functions.SquaredExponential
        ):
            raise Exception(
                "Bayesian quadrature only supports the squared exponential "
                "kernel."
            )

        # Number of training points and dimension
        N, D = self.X.shape
        # Number of hyperparameter samples.
        N_s = np.size(self.posteriors)

        # Number of GP hyperparameters.
        cov_N = self.covariance.hyperparameter_count(self.D)
        # mean_N = self.mean.hyperparameter_count(self.D)
        noise_N = self.noise.hyperparameter_count()

        N_star = mu.shape[0]
        if np.size(sigma) == 1:
            sigma = np.tile(sigma, (N_star, 1))

        quadratic_mean_fun = isinstance(
            self.mean, gpyreg.mean_functions.NegativeQuadratic
        )

        F = np.zeros((N_star, N_s))
        if compute_var:
            F_var = np.zeros((N_star, N_s))

        # Loop over hyperparameter samples.
        for s in range(0, N_s):
            hyp = self.posteriors[s].hyp

            # Extract GP hyperparameters
            ell = np.exp(hyp[0:D])
            ln_sf2 = 2 * hyp[D]
            sum_lnell = np.sum(hyp[0:D])

            # GP mean function hyperparameters
            if isinstance(self.mean, gpyreg.mean_functions.ZeroMean):
                m0 = 0
            else:
                m0 = hyp[cov_N + noise_N]

            if quadratic_mean_fun:
                xm = hyp[cov_N + noise_N + 1 : cov_N + noise_N + D + 1]
                omega = np.exp(hyp[cov_N + noise_N + D + 1 :])

            # GP posterior parameters
            alpha = self.posteriors[s].alpha
            L = self.posteriors[s].L
            L_chol = self.posteriors[s].L_chol

            sn2 = np.exp(2 * hyp[cov_N])
            sn2_eff = sn2 * self.posteriors[s].sn2_mult

            # Compute posterior mean of the integral
            tau = np.sqrt(sigma ** 2 + ell ** 2)
            lnnf = (
                ln_sf2 + sum_lnell - np.sum(np.log(tau), 1)
            )  # Covariance normalization factor
            sum_delta2 = np.zeros((N_star, N))

            for i in range(0, D):
                sum_delta2 += (
                    (mu[:, i] - np.reshape(self.X[:, i], (-1, 1))).T
                    / tau[:, i : i + 1]
                ) ** 2
            z = np.exp(np.reshape(lnnf, (-1, 1)) - 0.5 * sum_delta2)
            F[:, s : s + 1] = np.dot(z, alpha) + m0

            if quadratic_mean_fun:
                nu_k = -0.5 * np.sum(
                    1
                    / omega ** 2
                    * (mu ** 2 + sigma ** 2 - 2 * mu * xm + xm ** 2),
                    1,
                )
                F[:, s] += nu_k

            # Compute posterior variance of the integral
            if compute_var:
                tau_kk = np.sqrt(2 * sigma ** 2 + ell ** 2)
                nf_kk = np.exp(ln_sf2 + sum_lnell - np.sum(np.log(tau_kk), 1))
                if L_chol:
                    invKzk = (
                        sp.linalg.solve(
                            L, sp.linalg.solve(L, z.T, trans=1), trans=0
                        )
                        / sn2_eff
                    )
                else:
                    invKzk = np.dot(-L, z.T)
                J_kk = nf_kk - np.sum(z * invKzk.T, 1)
                F_var[:, s] = np.maximum(
                    np.spacing(1), J_kk
                )  # Correct for numerical error

        # Unless predictions for samples are requested separately
        # average over samples
        if N_s > 1 and not separate_samples:
            F_bar = np.reshape(np.sum(F, 1), (-1, 1)) / N_s
            if compute_var:
                Fss_var = np.sum((F - F_bar) ** 2, 1) / (N_s - 1)
                F_var = np.reshape(np.sum(F_var, 1) / N_s + Fss_var, (-1, 1))
            F = F_bar

        if compute_var:
            return F, F_var

        return F

    # sigma doesn't work, requires gplite_quad implementation
    # quantile doesn't work, requires gplite_qpred implementation
    def plot(self, x0=None, lb=None, ub=None, max_min_flag=None):
        """Plot the gaussian process profile centered around a given point.

        The plot is a D-by-D panel matrix, in which panels on the diagonal
        show the profile of the gaussian process prediction (mean and +/- 1 SD)
        by varying one dimension at a time, whereas off-diagonal panels show
        2-D contour plots of the GP mean and standard deviation (respectively,
        above and below diagonal). In each panel, black lines indicate the
        location of the reference point.

        Parameters
        ==========
        x0 : array_like, optional
            The reference point.
        lb : array_like, optional
            Lower bounds for the plotting.
        ub : array_like, optional
            Upper bounds for the plotting.
        max_min_flag : bool, optional
            If False then the minimum, and if True the maximum, of the
            GP training input is used as the reference point.
        """
        delta_y = None
        if np.isscalar(lb) and ub is None:
            delta_y = lb
            lb = None

        _, D = self.X.shape  # Number of training points and dimension
        s_N = self.posteriors.size  # Hyperparameter samples
        x_N = 100  # Grid points per visualization

        # Loop over hyperparameter samples.
        ell = np.zeros((D, s_N))
        for s in range(0, s_N):
            ell[:, s] = np.exp(
                self.posteriors[s].hyp[0:D]
            )  # Extract length scale from HYP
        ellbar = np.sqrt(np.mean(ell ** 2, 1)).T

        if lb is None:
            lb = np.min(self.X, axis=0) - ellbar
        if ub is None:
            ub = np.max(self.X, axis=0) + ellbar

        gutter = [0.05, 0.05]
        margins = [0.1, 0.01, 0.12, 0.01]
        linewidth = 1

        if x0 is None:
            max_min_flag = True
        if max_min_flag is not None:
            if max_min_flag:
                i = np.argmax(self.y)
                x0 = self.X[i, :]
            else:
                i = np.argmin(self.y)
                x0 = self.X[i, :]

        _, ax = plt.subplots(D, D, squeeze=False)

        flo = fhi = None
        for i in range(0, D):
            ax[i, i].set_position(
                self.__tight_subplot(D, D, i, i, gutter, margins)
            )

            xx = None
            xx_vec = np.reshape(
                np.linspace(lb[i], ub[i], np.ceil(x_N ** 1.5).astype(int)),
                (-1, 1),
            )
            if D > 1:
                xx = np.tile(x0, (np.size(xx_vec), 1))
                xx[:, i : i + 1] = xx_vec
            else:
                xx = xx_vec

            # do we need to add quantile prediction stuff etc here?
            fmu, fs2 = self.predict(xx, add_noise=False)
            flo = fmu - 1.96 * np.sqrt(fs2)
            fhi = fmu + 1.96 * np.sqrt(fs2)

            if delta_y is not None:
                # Probably doesn't work
                fmu0, _ = self.predict(x0, add_noise=False)
                dx = xx_vec[1] - xx_vec[0]
                region = np.abs(fmu - fmu0) < delta_y
                if np.any(region):
                    idx1 = np.argmax(region)
                    idx2 = np.size(region) - np.argmax(region[::-1]) - 1
                    lb[i] = x0[idx1] - 0.5 * dx
                    ub[i] = x0[idx2] + 0.5 * dx
                else:
                    lb[i] = x0[i] - 0.5 * dx
                    ub[i] = x0[i] + 0.5 * dx

                xx_vec = np.reshape(
                    np.linspace(lb[i], ub[i], np.ceil(x_N ** 1.5).astype(int)),
                    (-1, 1),
                )
                if D > 1:
                    xx = np.tile(x0, (np.size(xx_vec), 1))
                    xx[:, i : i + 1] = xx_vec
                else:
                    xx = xx_vec

                # do we need to add quantile prediction stuff etc here?
                fmu, fs2 = self.predict(xx, add_noise=False)
                flo = fmu - 1.96 * np.sqrt(fs2)
                fhi = fmu + 1.96 * np.sqrt(fs2)

            ax[i, i].plot(xx_vec, fmu, "-k", linewidth=linewidth)
            ax[i, i].plot(
                xx_vec, fhi, "-", color=(0.8, 0.8, 0.8), linewidth=linewidth
            )
            ax[i, i].plot(
                xx_vec, flo, "-", color=(0.8, 0.8, 0.8), linewidth=linewidth
            )
            ax[i, i].set_xlim(lb[i], ub[i])
            ax[i, i].set_ylim(ax[i, i].get_ylim())

            # ax[i, i].tick_params(direction='out')
            ax[i, i].spines["top"].set_visible(False)
            ax[i, i].spines["right"].set_visible(False)

            if D == 1:
                ax[i, i].set_xlabel("x")
                ax[i, i].set_ylabel("y")
                ax[i, i].scatter(self.X, self.y, color="blue")
            else:
                if i == 0:
                    ax[i, i].set_ylabel(r"$x_" + str(i + 1) + r"$")
                if i == D - 1:
                    ax[i, i].set_xlabel(r"$x_" + str(i + 1) + r"$")
            ax[i, i].vlines(
                x0[i],
                ax[i, i].get_ylim()[0],
                ax[i, i].get_ylim()[1],
                colors="k",
                linewidth=linewidth,
            )

        for i in range(0, D):
            for j in range(0, i):
                xx1_vec = np.reshape(np.linspace(lb[i], ub[i], x_N), (-1, 1)).T
                xx2_vec = np.reshape(np.linspace(lb[j], ub[j], x_N), (-1, 1)).T
                xx_vec = np.array(np.meshgrid(xx1_vec, xx2_vec)).T.reshape(
                    -1, 2
                )

                xx = np.tile(x0, (x_N * x_N, 1))
                xx[:, i] = xx_vec[:, 0]
                xx[:, j] = xx_vec[:, 1]

                fmu, fs2 = self.predict(xx, add_noise=False)

                for k in range(0, 2):
                    if k == 1:
                        i1 = j
                        i2 = i
                        mat = np.reshape(fmu, (x_N, x_N)).T
                    else:
                        i1 = 1
                        i2 = j
                        mat = np.reshape(np.sqrt(fs2), (x_N, x_N))
                    ax[i1, i2].set_position(
                        self.__tight_subplot(D, D, i1, i2, gutter, margins)
                    )
                    ax[i1, i2].spines["top"].set_visible(False)
                    ax[i1, i2].spines["right"].set_visible(False)

                    if k == 1:
                        Xt, Yt = np.meshgrid(xx1_vec, xx2_vec)
                        ax[i1, i2].contour(Xt, Yt, mat)
                    else:
                        Xt, Yt = np.meshgrid(xx2_vec, xx1_vec)
                        ax[i1, i2].contour(Xt, Yt, mat)
                    ax[i1, i2].set_xlim(lb[i2], ub[i2])
                    ax[i1, i2].set_ylim(lb[i1], ub[i1])
                    ax[i1, i2].scatter(
                        self.X[:, i2], self.X[:, i1], color="blue", s=10
                    )

                    ax[i1, i2].hlines(
                        x0[i1],
                        ax[i1, i2].get_xlim()[0],
                        ax[i1, i2].get_xlim()[1],
                        colors="k",
                        linewidth=linewidth,
                    )
                    ax[i1, i2].vlines(
                        x0[i2],
                        ax[i1, i2].get_ylim()[0],
                        ax[i1, i2].get_ylim()[1],
                        colors="k",
                        linewidth=linewidth,
                    )

                if j == 0:
                    ax[i, j].set_ylabel(r"$x_" + str(i + 1) + r"$")
                if i == D - 1:
                    ax[i, j].set_xlabel(r"$x_" + str(j + 1) + r"$")

        plt.show()

    @staticmethod
    def __tight_subplot(m, n, row, col, gutter=None, margins=None):
        if gutter is None:
            gutter = [0.002, 0.002]
        if margins is None:
            margins = [0.06, 0.01, 0.04, 0.04]
        Lmargin = margins[0]
        Rmargin = margins[1]
        Bmargin = margins[2]
        Tmargin = margins[3]

        unit_height = (1 - Bmargin - Tmargin - (m - 1) * gutter[1]) / m
        height = np.size(row) * unit_height + (np.size(row) - 1) * gutter[1]

        unit_width = (1 - Lmargin - Rmargin - (n - 1) * gutter[0]) / n
        width = np.size(col) * unit_width + (np.size(col) - 1) * gutter[0]

        bottom = (m - np.max(row) - 1) * (unit_height + gutter[1]) + Bmargin
        left = np.min(col) * (unit_width + gutter[0]) + Lmargin

        pos_vec = [left, bottom, width, height]

        return pos_vec

    def random_function(self, X_star, add_noise=False):
        """Draws a random function from the gaussian process.

        Parameters
        ==========
        X_star : array_like
            The points at which to evaluate the drawn function.
        add_noise : bool, defaults to False
            Whether to add noise to the values of the drawn function.

        Returns
        =======

        f_star : array_like,
            The values of the drawn function at the requested points.
        """
        N_star = X_star.shape[0]
        N_s = np.size(self.posteriors)

        cov_N = self.covariance.hyperparameter_count(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        noise_N = self.noise.hyperparameter_count()

        # Draw from hyperparameter samples.
        s = np.random.randint(0, N_s)

        hyp = self.posteriors[s].hyp
        alpha = self.posteriors[s].hyp
        L = self.posteriors[s].L
        L_chol = self.posteriors[s].L_chol
        sW = self.posteriors[s].sW

        # Compute GP mean function at test points
        m_star = np.reshape(
            self.mean.compute(
                hyp[cov_N + noise_N : cov_N + noise_N + mean_N], X_star
            ),
            (-1, 1),
        )

        # Compute kernel matrix
        K_star = self.covariance.compute(hyp[0:cov_N], X_star)

        if self.y is None:
            # No data, draw from prior
            f_mu = m_star
            C = K_star + np.spacing(500) * np.eye(N_star)
        else:
            # Compute cross-kernel matrix Ks
            Ks = self.covariance.compute(hyp[0:cov_N], self.X, X_star=X_star)

            # Conditional mean
            f_mu = m_star + np.dot(Ks.T, alpha)

            if L_chol:
                V = np.linalg.solve(L.T, np.tile(sW, (1, N_star)) * Ks)
                C = K_star - np.dot(V.T, V)  # Predictive variances
            else:
                LKs = np.dot(L, Ks)
                C = K_star + np.dot(Ks.T, LKs)

        # Enforce symmetry if lost due to numerical errors.
        C = (C + C.T) / 2

        # Draw random function
        T = sp.linalg.cholesky(C)
        f_star = np.dot(T.T, np.random.standard_normal((T.shape[0], 1))) + f_mu

        # Add observation noise.
        if add_noise:
            # Get observation noise hyperparameters and evaluate noise
            # at test points.
            sn2 = self.noise.compute(
                hyp[cov_N : cov_N + noise_N], X_star, None, None
            )
            sn2_mult = self.posteriors[s].sn2_mult
            if sn2_mult is None:
                sn2_mult = 1
            y_star = f_star + np.sqrt(
                sn2 * sn2_mult
            ) * np.random.standard_normal(size=f_mu.shape)
            return y_star

        return f_star

    def __core_computation(self, hyp, compute_nlZ, compute_nlZ_grad):
        N, d = self.X.shape
        cov_N = self.covariance.hyperparameter_count(d)
        mean_N = self.mean.hyperparameter_count(d)
        noise_N = self.noise.hyperparameter_count()

        if compute_nlZ_grad:
            sn2, dsn2 = self.noise.compute(
                hyp[cov_N : cov_N + noise_N],
                self.X,
                self.y,
                self.s2,
                compute_grad=True,
            )
            m, dm = self.mean.compute(
                hyp[cov_N + noise_N : cov_N + noise_N + mean_N],
                self.X,
                compute_grad=True,
            )

            # This line is actually important due to behaviour of above
            # Maybe change that in the future.
            m = m.reshape((-1, 1))
            K, dK = self.covariance.compute(
                hyp[0:cov_N], self.X, compute_grad=True
            )
        else:
            sn2 = self.noise.compute(
                hyp[cov_N : cov_N + noise_N], self.X, self.y, self.s2
            )
            m = np.reshape(
                self.mean.compute(
                    hyp[cov_N + noise_N : cov_N + noise_N + mean_N], self.X
                ),
                (-1, 1),
            )
            K = self.covariance.compute(hyp[0:cov_N], self.X)
        sn2_mult = 1  # Effective noise variance multiplier

        L_chol = np.min(sn2) >= 1e-6
        if L_chol:
            if np.isscalar(sn2):
                sn2_div = sn2
                sn2_mat = np.eye(N)
            else:
                sn2_div = np.min(sn2)
                sn2_mat = np.diag(sn2.ravel() / sn2_div)
            for i in range(0, 10):
                try:
                    L = sp.linalg.cholesky(K / (sn2_div * sn2_mult) + sn2_mat)
                except sp.linalg.LinAlgError:
                    sn2_mult *= 10
                    continue
                break
            sl = sn2_div * sn2_mult
            pL = L
        else:
            if np.isscalar(sn2):
                sn2_mat = sn2 * np.eye(N)
            else:
                sn2_mat = np.diag(sn2.ravel())

            for i in range(0, 10):
                try:
                    L = sp.linalg.cholesky(K + sn2_mult * sn2_mat)
                except sp.linalg.LinAlgError:
                    sn2_mult *= 10
                    continue
                break
            sl = 1
            if not compute_nlZ:
                pL = sp.linalg.solve_triangular(
                    -L,
                    sp.linalg.solve_triangular(L, np.eye(N), trans=1),
                    trans=0,
                )

        alpha = (
            sp.linalg.solve_triangular(
                L, sp.linalg.solve_triangular(L, self.y - m, trans=1), trans=0
            )
            / sl
        )

        # Negative log marginal likelihood computation
        if compute_nlZ:
            nlZ = (
                np.dot((self.y - m).T, alpha / 2)
                + np.sum(np.log(np.diag(L)))
                + N * np.log(2 * np.pi * sl) / 2
            )

            if compute_nlZ_grad:
                dnlZ = np.zeros(hyp.shape)
                Q = (
                    sp.linalg.solve_triangular(
                        L,
                        sp.linalg.solve_triangular(L, np.eye(N), trans=1),
                        trans=0,
                    )
                    / sl
                    - np.dot(alpha, alpha.T)
                )

                # Gradient of covariance hyperparameters.
                for i in range(0, cov_N):
                    dnlZ[i] = np.sum(np.sum(Q * dK[:, :, i])) / 2

                # Gradient of GP likelihood
                if np.isscalar(sn2):
                    tr_Q = np.trace(Q)
                    for i in range(0, noise_N):
                        dnlZ[cov_N + i] = (
                            0.5 * sn2_mult * np.dot(dsn2[i], tr_Q)
                        )
                else:
                    dg_Q = np.diag(Q)
                    for i in range(0, noise_N):
                        dnlZ[cov_N + i] = (
                            0.5 * sn2_mult * np.sum(dsn2[:, i] * dg_Q)
                        )

                # Gradient of mean function.
                if mean_N > 0:
                    dnlZ[cov_N + noise_N :] = np.dot(-dm.T, alpha)[:, 0]

                return nlZ[0, 0], dnlZ

            return nlZ[0, 0]

        return Posterior(
            hyp,
            alpha,
            np.ones((N, 1)) / np.sqrt(np.min(sn2) * sn2_mult),
            pL,
            sn2_mult,
            L_chol,
        )


class Posterior:
    def __init__(self, hyp, alpha, sW, L, sn2_mult, Lchol):
        self.hyp = hyp
        self.alpha = alpha
        self.sW = sW
        self.L = L
        self.sn2_mult = sn2_mult
        self.L_chol = Lchol
