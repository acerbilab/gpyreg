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
        self.s2 = s2
        self.X = None
        self.y = None
        self.hyper_priors = None
        self.post = None

    def set_priors(self, priors):
        """Sets the hyperparameter priors.

        Parameters
        ==========
        priors : dict
            A dictionary of hyperparameter names and tuples of their values.
        """
        cov_N = self.covariance.hyperparameter_count(self.D)
        cov_hyper_info = self.covariance.hyperparameter_info(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        mean_hyper_info = self.mean.hyperparameter_info(self.D)
        noise_N = self.noise.hyperparameter_count()
        noise_hyper_info = self.noise.hyperparameter_info()

        hyp_N = cov_N + mean_N + noise_N
        # Set up a hyperprior dictionary with default values which can be updated individually later.
        self.hyper_priors = {
            "mu": np.full((hyp_N,), np.nan),
            "sigma": np.full((hyp_N,), np.nan),
            "df": np.full((hyp_N,), np.nan),
            "a": np.full((hyp_N,), np.nan),
            "b": np.full((hyp_N,), np.nan),
            "LB": np.full((hyp_N,), np.nan),
            "UB": np.full((hyp_N,), np.nan),
        }

        for prior in priors:
            # Indices for the lower and upper range of this particular hyperprior.
            lower = 0
            upper = None

            # Go through all possible hyperparameter names and find the one we are interested in.
            # Also update lower and upper as we go.

            for info in cov_hyper_info:
                if prior == info[0]:
                    upper = lower + info[1]
                    break
                lower += info[1]

            if upper is None:
                for info in noise_hyper_info:
                    if prior == info[0]:
                        upper = lower + info[1]
                        break
                    lower += info[1]

            if upper is None:
                for info in mean_hyper_info:
                    if prior == info[0]:
                        upper = lower + info[1]
                        break
                    lower += info[1]

            # If we found something update the hyperprior.
            if upper is not None:
                prior_type, prior_params = priors[prior]
                i = range(lower, upper)
                if prior_type == "fixed":
                    val = prior_params
                    self.hyper_priors["LB"][i] = val
                    self.hyper_priors["UB"][i] = val
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

    def set_hyperparameters(self, hyp_new):
        pass

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
            New input-dependent noise that will be concatenated with the old ones.
        hyp : array_like, optional
            New hyperparameters that will replace the old ones.
        compute_posterior : bool, defaults to True
            Whether to compute the new posterior or not.
        """
        rank_one_update = False
        if X_new is not None and y_new is not None:
            if X_new.ndim == 1:
                X_new = np.reshape(X_new, (1, -1))
            if y_new.ndim == 1:
                y_new = np.reshape(y_new, (-1, 1))

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
            s_N = np.size(self.post)

            # Loop over hyperparameter samples.
            for s in range(0, s_N):
                hyp_s = self.post[s].hyp

                hyp_noise = hyp_s[cov_N : cov_N + noise_N]
                sn2 = self.noise.compute(hyp_noise, X_new, y_new, 0)
                sn2_eff = sn2 * self.post[s].sn2_mult

                # Compute covariance and cross-covariance.
                hyp_cov = hyp_s[0:cov_N]
                K = self.covariance.compute(hyp_cov, X_new)
                Ks = self.covariance.compute(hyp_cov, self.X, X_new)

                L = self.post[s].L
                L_chol = self.post[s].L_chol

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
                    self.post[s].L = np.block(
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
                    self.post[s].L = np.block(
                        [
                            [L + np.dot(v, alpha_update.T), -v],
                            [-v.T, -1 / v_star[:, s]],
                        ]
                    )

                self.post[s].sW = np.concatenate(
                    (self.post[s].sW, np.array([[1 / np.sqrt(sn2_eff)]]))
                )

                # alpha_update now contains (K + \sigma^2 I) \ k*
                self.post[s].alpha = np.concatenate(
                    (self.post[s].alpha, np.array([[0]]))
                ) + (m_star[:, s] - y_new) / v_star[:, s] * np.concatenate(
                    (alpha_update, np.array([[-1]]))
                )

        if X_new is not None:
            if self.X is None:
                self.X = X_new
            else:
                self.X = np.concatenate((self.X, X_new))

        if y_new is not None:
            # Change from 1D to 2D internally.
            if y_new is not None and y_new.ndim == 1:
                y_new = np.reshape(y_new, (-1, 1))

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
            _, s_N = hyp.shape
            self.post = np.empty((s_N,), dtype=Posterior)

            if compute_posterior:
                for i in range(0, s_N):
                    self.post[i] = self.__core_computation(hyp[:, i], 0, 0)
            else:
                for i in range(0, s_N):
                    self.post[i] = Posterior(
                        hyp[:, i], None, None, None, None, None
                    )

    def fit(self, X=None, y=None, s2=None, options=None):
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

        Returns
        =======

        hyp : object
            In case ``n_samples`` is 0, we return the best result of optimization without sampling,
            and if not then we return the ``SamplingResult``object from sampling.
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
        mean_N = self.mean.hyperparameter_count(self.D)
        noise_N = self.noise.hyperparameter_count()
        hyp_N = cov_N + mean_N + noise_N
        hyp0 = np.zeros((hyp_N,))

        LB = self.hyper_priors["LB"]
        UB = self.hyper_priors["UB"]

        ## Initialize inference of GP hyperparameters (bounds, priors, etc.)

        cov_info = self.covariance.get_info(X, y)
        mean_info = self.mean.get_info(X, y)
        noise_info = self.noise.get_info(X, y)

        self.hyper_priors["df"][np.isnan(self.hyper_priors["df"])] = df_base

        # Set covariance/noise/mean function hyperparameter lower bounds.
        LB_cov = LB[0:cov_N]
        LB_noise = LB[cov_N : cov_N + noise_N]
        LB_mean = LB[cov_N + noise_N : cov_N + noise_N + mean_N]
        LB_cov[np.isnan(LB_cov)] = cov_info.LB[np.isnan(LB_cov)]
        LB_noise[np.isnan(LB_noise)] = noise_info.LB[np.isnan(LB_noise)]
        LB_mean[np.isnan(LB_mean)] = mean_info.LB[np.isnan(LB_mean)]

        # Set covariance/noise/mean function hyperparameter upper bounds.
        UB_cov = UB[0:cov_N]
        UB_noise = UB[cov_N : cov_N + noise_N]
        UB_mean = UB[cov_N + noise_N : cov_N + noise_N + mean_N]
        UB_cov[np.isnan(UB_cov)] = cov_info.UB[np.isnan(UB_cov)]
        UB_noise[np.isnan(UB_noise)] = noise_info.UB[np.isnan(UB_noise)]
        UB_mean[np.isnan(UB_mean)] = mean_info.UB[np.isnan(UB_mean)]

        # Create lower and upper bounds
        LB = np.concatenate([LB_cov, LB_noise, LB_mean])
        UB = np.concatenate([UB_cov, UB_noise, UB_mean])
        UB = np.maximum(LB, UB)

        # Plausible bounds for generation of starting points
        PLB = np.concatenate([cov_info.PLB, noise_info.PLB, mean_info.PLB])
        PUB = np.concatenate([cov_info.PUB, noise_info.PUB, mean_info.PUB])
        PLB = np.minimum(np.maximum(PLB, LB), UB)
        PUB = np.maximum(np.minimum(PUB, UB), LB)

        ## Hyperparameter optimization
        objective_f_1 = lambda hyp_: self.__gp_obj_fun(hyp_, False, False)

        # First evaluate GP log posterior on an informed space-filling design.
        t1_s = time.time()
        X0, y0 = f_min_fill(
            objective_f_1, hyp0, LB, UB, PLB, PUB, self.hyper_priors, init_N
        )
        hyp = X0[0:opts_N, :].T
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
            hyp[:, 1] = xx[idx_best, :]

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
        eps_LB = np.reshape(LB, (-1, 1)) + np.spacing(np.reshape(LB, (-1, 1)))
        eps_UB = np.reshape(UB, (-1, 1)) - np.spacing(np.reshape(UB, (-1, 1)))
        hyp = np.minimum(eps_UB, np.maximum(eps_LB, hyp))

        # Perform optimization from most promising NOPTS hyperparameter vectors.
        objective_f_2 = lambda hyp_: self.__gp_obj_fun(hyp_, True, False)
        nll = np.full((opts_N,), np.inf)

        t2_s = time.time()
        for i in range(0, opts_N):
            # res = sp.optimize.minimize(fun=objective_f_1, x0=hyp[:, i], bounds=list(zip(LB, UB)))
            res = sp.optimize.minimize(
                fun=objective_f_2,
                x0=hyp[:, i],
                jac=True,
                bounds=list(zip(LB, UB)),
            )
            hyp[:, i] = res.x
            nll[i] = res.fun

        # Take the best hyperparameter vector.
        hyp_start = hyp[:, np.argmin(nll)]
        t2 = time.time() - t2_s

        # In case n_samples is 0, just return the optimized hyperparameter result.
        if s_N == 0:
            self.update(hyp=hyp_start)
            return hyp_start

        ## Sample from best hyperparameter vector using slice sampling

        t3_s = time.time()
        # Effective number of samples (thin after)
        eff_s_N = s_N * thin

        sample_f = lambda hyp_: self.__gp_obj_fun(hyp_, False, True)
        options = {"display": "off", "diagnostics": False}
        slicer = SliceSampler(
            sample_f, hyp_start, widths_default, LB, UB, options
        )
        res = slicer.sample(eff_s_N, burn=burn_in)

        # Thin samples
        hyp_pre_thin = res.samples.T
        hyp = hyp_pre_thin[:, thin - 1 :: thin]

        t3 = time.time() - t3_s
        # print(hyp)
        print(t1, t2, t3)

        # Recompute GP with finalized hyperparameters.
        self.update(hyp=hyp)
        return res

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
        lb = self.hyper_priors["LB"]
        ub = self.hyper_priors["UB"]

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
                hyp, compute_grad, self.hyper_priors is not None
            )
        else:
            nlZ = self.__compute_nlZ(
                hyp, compute_grad, self.hyper_priors is not None
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
            Whether to return the results separately for each hyperparameter sample or averaged.

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
        s_N = self.post.size
        N_star = x_star.shape[0]

        # Preallocate space
        fmu = np.zeros((N_star, s_N))
        ymu = np.zeros((N_star, s_N))
        fs2 = np.zeros((N_star, s_N))
        ys2 = np.zeros((N_star, s_N))

        for s in range(0, s_N):
            hyp = self.post[s].hyp
            alpha = self.post[s].alpha
            L = self.post[s].L
            L_chol = self.post[s].L_chol
            sW = self.post[s].sW
            sn2_mult = self.post[s].sn2_mult

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

        # Unless predictions for samples are requested separately, average over samples.
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
            Whether to return the results separately for each hyperparameter sample or averaged.
        """

        if not isinstance(
            self.covariance, gpyreg.covariance_functions.SquaredExponential
        ):
            raise Exception(
                "Bayesian quadrature only supports the squared exponential kernel."
            )

        # Number of training points and dimension
        N, D = self.X.shape
        # Number of hyperparameter samples.
        N_s = np.size(self.post)

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
            hyp = self.post[s].hyp

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
            alpha = self.post[s].alpha
            L = self.post[s].L
            L_chol = self.post[s].L_chol

            sn2 = np.exp(2 * hyp[cov_N])
            sn2_eff = sn2 * self.post[s].sn2_mult

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

        # Unless predictions for samples are requested separately, average over samples
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
        s_N = self.post.size  # Hyperparameter samples
        x_N = 100  # Grid points per visualization

        # Loop over hyperparameter samples.
        ell = np.zeros((D, s_N))
        for s in range(0, s_N):
            ell[:, s] = np.exp(
                self.post[s].hyp[0:D]
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
        N_s = np.size(self.post)

        cov_N = self.covariance.hyperparameter_count(self.D)
        mean_N = self.mean.hyperparameter_count(self.D)
        noise_N = self.noise.hyperparameter_count()

        # Draw from hyperparameter samples.
        s = np.random.randint(0, N_s)

        hyp = self.post[s].hyp
        alpha = self.post[s].hyp
        L = self.post[s].L
        L_chol = self.post[s].L_chol
        sW = self.post[s].sW

        # Compute GP mena function at test points
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
            C = K_star + np.spacing(1) * np.eye(N_star)
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
            # Get observation noise hyperparameters and evaluate noise at test points.
            sn2 = self.noise.compute(
                hyp[cov_N : cov_N + noise_N], X_star, None, None
            )
            sn2_mult = self.post[s].sn2_mult
            y_star = f_star + np.sqrt(
                sn2 * sn2_mult
            ) * np.random.standard_normal(size=f_mu.size)
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
            # This line is actually important due to behaviour of above, maybe change that in the future.
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
