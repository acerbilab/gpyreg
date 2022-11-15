"""Module for slice sampling."""

import logging
import math

import numpy as np


class SliceSampler:
    """Class for drawing random samples from a target distribution with a
    given log probability density function using the coordinate-wise
    slice sampling method.

    Parameters
    ----------
    log_f : callable
        The log pdf of the target distribution. It takes one argument as
        input that has the same type and size as ``x0`` and returns the
        target log density function (minus a constant; that is, the
        normalization constant of the pdf need not be known).

        Note that ``log_f`` can return either a scalar
        (the value of the log probability density at ``x``) or a row vector
        (the value of the log probability density at ``x`` for each data
        point; each column corresponds to a different data point). In the
        latter case, the total log pdf is obtained by summing the log pdf
        per each individual data point. Also, ``f_vals`` in the object
        returned by ``sample`` is a matrix (each row corresponds to a
        sampled point, each column to a different data point). Knowing the
        log pdf of the sampled points per each data point can be useful to
        compute estimates of predictive error such as the widely applicable
        information criterion (WAIC); see [3]_.
    x0 : ndarray, shape (D,)
        Initial value of the random sample sequence. It must be within the
        domain of the target distribution. The number of independent
        variables is ``D``.
    widths : array_like, optional
        A parameter for typical widths. Either a scalar or a 1D array.
        If it is a scalar, all dimensions are assumed to have the same
        typical widths. If it is a 1D array, each element of the array
        is the typical width of the marginal target distribution in that
        dimension. The default value of ``widths[i]`` is ``(UB[i]-LB[i])/2``
        if the ``i``-th bounds are finite, or 10 otherwise. By default an
        an adaptive widths method during the burn-in period is being used, so
        the choice of typical widths is not crucial.
    LB : array_like, optional
        An array of lower bounds on the domain of the target density
        function, which is assumed to be zero outside the range
        ``LB <= x <= UB``. If not given no lower bounds are assumed.
        Set ``LB[i] = -inf`` if ``x[i]`` is unbounded below. If
        ``LB[i] == UB[i]``, the variable is assumed to be fixed on
        that dimension.
    UB : array_like, optional
        An array of upper bounds on the domain of the target density
        function, which is assumed to be zero outside the range
        ``LB <= x <= UB``. If not given no upper bounds are assumed.
        Set ``UB[i] = inf`` if ``x[i]`` is unbounded above. If
        ``LB[i] == UB[i]``, the variable is assumed to be fixed on that
        dimension.
    options : dict, optional
        A dictionary of sampler options. The possible options are:

            **step_out** : bool, defaults to False
                If set to true, performs the stepping-out action when
                the current window does not bracket the probability density.
                For details see [1]_.
            **display** : {'off', 'summary', 'full'}, defaults to 'full'
                Defines the level of display.
            **log_prior** : callable, optional
                Allows the user to specify a prior over ``x``. The function
                ``log_prior`` takes one argument as input that has the same
                type and size as ``x0`` and returns the log prior density
                function at X. The generated samples will be then drawn
                from the log density ``log_f + log_prior``.
            **adaptive** : bool, defaults to True
                Specifies whether to adapt ``widths`` at the end of the
                burn-in period based on the samples obtained so far.
                Disabling this works best if we already have good estimates.
            **diagnostics** : bool, defaults to True
                Specifies whether convergence diagnostics are performed at
                the end of the run. The diagnostic tests are from [4]_.

    Raises
    ------
    ValueError
        Raised when `x0` is not  a scalar or a 1D array.
    ValueError
        Raised when `LB` or `UB` are not None, scalars, or 1D arrays of the
        same size as `x0`.
    ValueError
        Raised when `LB` > `UB`.
    ValueError
        Raised when `widths` does not only contain positive real numbers.
    ValueError
        Raised when the initial starting point `x0` is outside the bounds (`LB`
        and `UB`).

    Notes
    -----

    Inspired by a MATLAB implementation of slice sampling by Iain Murray.
    See pseudo-code in [2]_.

    References
    ----------
    .. [1] R. Neal (2003), Slice Sampling, Annals of Statistics,
       31(3), p705-67.
    .. [2] D. J. MacKay (2003), Information theory, inference and learning
       algorithms, Cambridge university press, p374-7.
    .. [3] S. Watanabe (2010), Asymptotic equivalence of Bayes cross
       validation and widely applicable information criterion in singular
       learning theory, The Journal of Machine Learning Research,
       11, p3571-94.
    .. [4] A. Gelman, et al (2013), Bayesian data analysis. Vol. 2.
       Boca Raton, FL, USA: Chapman & Hall/CRC.

    """

    def __init__(
        self,
        log_f,
        x0: np.ndarray,
        widths=None,
        LB=None,
        UB=None,
        options: dict = None,
    ):
        D = x0.size
        self.log_f = log_f
        self.x0 = x0.copy()

        if LB is None:
            self.LB = np.tile(-np.inf, D)
            self.LB_out = np.tile(-np.inf, D)
        else:
            if np.size(LB) == 1:
                self.LB = np.tile(LB, D)
            else:
                self.LB = LB.copy()
        # np.spacing could return negative numbers so use nextafter
        self.LB_out = np.nextafter(self.LB, np.inf)

        if UB is None:
            self.UB = np.tile(np.inf, D)
            self.UB_out = np.tile(np.inf, D)
        else:
            if np.size(UB) == 1:
                self.UB = np.tile(UB, D)
            else:
                self.UB = UB.copy()
        # np.spacing could return negative numbers so use nextafter
        self.UB_out = np.nextafter(self.UB, -np.inf)

        if widths is None:
            self.widths = ((self.UB - self.LB) / 2).copy()
            self.base_widths = None
        else:
            if np.size(widths) == 1:
                self.widths = np.tile(widths, D)
            else:
                self.widths = widths.copy()
            self.base_widths = self.widths.copy()

        self.widths[np.isinf(self.widths)] = 10
        self.widths[
            self.LB == self.UB
        ] = 1  # Widths is irrelevant when LB == UB, set to 1

        # Sanity checks
        if np.ndim(self.x0) > 1:
            raise ValueError(
                "The initial point x0 needs to be a scalar or a 1D array"
            )

        if np.shape(self.LB) != np.shape(self.x0) or np.shape(
            self.UB
        ) != np.shape(self.x0):
            raise ValueError(
                "LB and UB need to be None, scalars, or 1D arrays of "
                "the same size as X0."
            )

        if not np.all(self.UB >= self.LB):
            raise ValueError(
                "All upper bounds UB need to be equal or greater than "
                "lower bounds LB."
            )

        if (
            np.any(self.widths <= 0)
            or np.any(~np.isfinite(self.widths))
            or np.any(~np.isreal(self.widths))
        ):
            raise ValueError(
                "The widths vector needs to be all positive real numbers."
            )

        if np.any(self.x0 < self.LB) or np.any(self.x0 > self.UB):
            raise ValueError(
                "The initial starting point X0 is outside the bounds."
            )

        self.func_count = 0

        # Default options
        if options is None:
            options = {}
        self.step_out = options.get("step_out", False)
        self.display = options.get("display", "full")
        self.adaptive = options.get("adaptive", True)
        self.log_prior = options.get("log_prior", None)
        self.diagnostics = options.get("diagnostics", True)
        self.metropolis_pdf = options.get("metropolis_pdf", None)
        self.metropolis_rnd = options.get("metopolis_rnd", None)
        self.metropolis_flag = (
            self.metropolis_pdf is not None and self.metropolis_rnd is not None
        )

        # Logging
        self.logger = logging.getLogger("SliceSampler")
        # Remember to only add the handler once.
        if len(self.logger.handlers) == 0:
            self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)
        if self.display == "off":
            self.logger.setLevel(logging.WARN)
        elif self.display == "summary":
            self.logger.setLevel(logging.INFO)
        elif self.display == "full":
            self.logger.setLevel(logging.DEBUG)

    def sample(self, N: int, thin: int = 1, burn: int = None):
        """Samples an arbitrary number of points from the distribution.

        Parameters
        ----------
        N : int
            The number of samples to return.
        thin : int, optional
            The thinning parameter will omit ``thin-1`` out of ``thin`` values
            in the generated sequence (after burn-in).
        burn : int, optional
            The burn parameter omits the first ``burn`` points before starting
            recording points for the generated sequence.
            In case this is the first time sampling, the default value of burn
            is ``round(N/3)`` (that is, one third of the number of recorded
            samples), while otherwise it is 0.

        Returns
        -------
        res : dict
          The sampling result represented as a dictionary with attributes

            **samples** : array_like
                The actual sampled points.
            **f_vals** : array_like
                The sequence of values of the target log pdf at the sampled
                points. If a prior is specified in ``log_prior``, then
                ``f_vals`` does NOT include the contribution of the prior.
            **exit_flag** : { 1, 0, -1, -2, -3 }
                Possible values and the corresponding exit conditions are

                    1, Target number of recorded samples reached,
                      with no explicit violation of convergence
                      (this does not ensure convergence).

                    0, Target number of recorded samples reached,
                      convergence status is unknown (no diagnostics have
                      been run).

                    -1, No explicit violation of convergence detected, but
                      the number of effective (independent) samples in the
                      sampled sequence is much lower than the number of
                      requested samples N for at least one dimension.

                    -2, Detected probable lack of convergence of the sampling
                      procedure.

                    -3, Detected lack of convergence of the sampling
                      procedure.
            **log_priors** : array_like
                The sequence of the values of the log prior at the sampled
                points.
            **R** : array_like
                Estimate of the potential scale reduction factor in each
                dimension.
            **eff_N** : array_like
                Estimate of the effective number of samples in each
                dimension.

        Raises
        ------
        ValueError
            Raised when `thin` is not a positive integer.
        ValueError
            Raised when `burn` is not a integer >= 0.
        ValueError
            Raised when the initial starting point X0 does not evaluate to a
            real number (e.g. Inf or NaN).
        """

        # Reference to x0 so it is updated as we go along, allowing us to
        # use this function multiple times.
        xx = self.x0
        D = xx.size

        if burn is None:
            # In case we are sampling again there is no need for burn-in.
            if self.func_count > 0:
                burn = 0
            else:
                burn = round(N / 3)

        # Sanity checks
        if not np.isscalar(thin) or thin <= 0:
            raise ValueError(
                "The thinning factor option needs to be a positive integer."
            )

        if not np.isscalar(burn) or burn < 0:
            raise ValueError(
                "The burn-in samples option needs to be a non-negative "
                "integer."
            )

        if (
            burn == 0
            and self.base_widths is None
            and self.adaptive
            and self.func_count == 0
        ):
            self.logger.warning(
                "WIDTHS not specified and adaptation is ON (OPTIONS."
                "Adaptive == 1), but OPTIONS.Burnin is set to 0. "
                "SLICESAMPLEBND will attempt to use default values for "
                "WIDTHS."
            )

        # Effective samples
        eff_N = N + (N - 1) * (thin - 1)

        samples = np.zeros((N, D))
        xx_sum = np.zeros((D,))
        xx_sq_sum = np.zeros((D,))

        log_dist = self.__log_pdf_bound
        log_Px, f_val, log_prior = log_dist(xx)
        log_priors = np.zeros((N,))
        f_vals = np.zeros((N, np.size(f_val)))

        if np.any(~np.isfinite(log_Px)):
            raise ValueError(
                "The initial starting point X0 needs to evaluate to a "
                "real number (not Inf or NaN)."
            )

        # Force xx into vector for ease of use
        xx_shape = xx.shape
        xx = xx.ravel()
        logdist_vec = lambda x: log_dist(np.reshape(x, xx_shape))

        self.logger.debug(
            " Iteration     f-count       log p(x)                   Action"
        )
        display_format = " %7.0f     %8.0f    %12.6g    %26s"

        # Main loop
        perm = np.array(range(D))
        for i in range(0, eff_N + burn):
            if i == burn:
                action = "start recording"
                self.logger.debug(
                    display_format,
                    i - burn + 1,
                    self.func_count,
                    log_Px,
                    action,
                )

            # Metropolis step (optional)
            if self.metropolis_flag:
                xx, log_Px, f_val, log_prior = self.__metropolis_step(
                    xx, logdist_vec, log_Px, f_val, log_prior
                )

            ## Slice sampling step.
            x_l = xx.copy()
            x_r = xx.copy()
            xprime = xx.copy()

            # Random scan through axes
            np.random.shuffle(perm)
            for dd in perm:
                # Skip fixed dimensions.
                if self.LB[dd] == self.UB[dd]:
                    continue

                log_uprime = log_Px + np.log(np.random.rand())
                # Create a horizontal interval (x_l, x_r) enclosing xx
                rr = np.random.rand()
                x_l[dd] -= rr * self.widths[dd]
                x_r[dd] += (1 - rr) * self.widths[dd]

                # Adjust interval to outside bounds for bounded problems.
                x_l[dd] = np.fmax(x_l[dd], self.LB_out[dd])
                x_r[dd] = np.fmin(x_r[dd], self.UB_out[dd])

                if self.step_out:
                    steps = 0
                    # Typo in early book editions: said compare to u
                    # should be u'
                    while logdist_vec(x_l)[0] > log_uprime:
                        x_l[dd] -= self.widths[dd]
                        steps += 1
                    while logdist_vec(x_r)[0] > log_uprime:
                        x_r[dd] += self.widths[dd]
                        steps += 1
                    if steps >= 10:
                        action = (
                            "step-out dim "
                            + str(dd)
                            + " ("
                            + str(steps)
                            + " steps)"
                        )
                        self.logger.debug(
                            display_format,
                            i - burn + 1,
                            self.func_count,
                            log_Px,
                            action,
                        )

                # Inner loop:
                # Propose xprimes and shrink interval until good one found
                shrink = 0
                while True:
                    shrink += 1
                    xprime[dd] = (
                        np.random.rand() * (x_r[dd] - x_l[dd]) + x_l[dd]
                    )
                    log_Px, f_val, log_prior = logdist_vec(xprime)
                    if log_Px > log_uprime:
                        break  # this is the only way to leave the while loop

                    # Shrink in
                    if xprime[dd] > xx[dd]:
                        x_r[dd] = xprime[dd]
                    elif xprime[dd] < xx[dd]:
                        x_l[dd] = xprime[dd]
                    else:
                        # Maybe even raise an exception?
                        self.logger.warning(
                            "WARNING: Shrunk to current position and still "
                            " not acceptable!"
                        )
                        break

                # Width adaptation (only during burn-in, might break
                # detailed balance)
                if i < burn and self.adaptive:
                    delta = self.UB[dd] - self.LB[dd]
                    if shrink > 3:
                        if np.isfinite(delta):
                            # take absolute value to make sure we don't have
                            # issues with np.spacing returning negative values
                            self.widths[dd] = np.maximum(
                                self.widths[dd] / 1.1,
                                np.abs(np.spacing(delta)),
                            )
                        else:
                            self.widths[dd] = np.maximum(
                                self.widths[dd] / 1.1, np.spacing(1)
                            )
                    elif shrink < 2:
                        self.widths[dd] = np.minimum(
                            self.widths[dd] * 1.2, delta
                        )

                if shrink >= 10:
                    action = (
                        "shrink dim "
                        + str(dd)
                        + " ("
                        + str(shrink)
                        + " steps)"
                    )
                    self.logger.debug(
                        display_format,
                        i - burn + 1,
                        self.func_count,
                        log_Px,
                        action,
                    )

                xx[dd] = xprime[dd]

            # Metropolis step (optional)
            if self.metropolis_flag:
                xx, log_Px, f_val, log_prior = self.__metropolis_step(
                    xx, logdist_vec, log_Px, f_val, log_prior
                )

            # Record samples and miscellaneous bookkeeping.
            record = i >= burn and np.mod(i - burn, thin) == 0
            if record:
                i_smpl = (i - burn) // thin
                samples[i_smpl, :] = xx
                f_vals[i_smpl, :] = f_val
                log_priors[i_smpl] = log_prior

            # Store summary statistics starting half.way into burn-in.
            if burn / 2 <= i < burn:
                xx_sum += xx
                xx_sq_sum += xx ** 2

                # End of burn-in, update widths if using adaptive method.
                if i == burn - 1 and self.adaptive:
                    burn_stored = np.floor(burn / 2)
                    # There can be numerical error here but then width
                    # has already shrunk to 0?
                    new_widths = np.fmin(
                        5
                        * np.sqrt(
                            np.maximum(
                                xx_sq_sum / burn_stored
                                - (xx_sum / burn_stored) ** 2,
                                0,
                            )
                        ),
                        self.UB_out - self.LB_out,
                    )
                    if not np.all(np.isreal(new_widths)):
                        new_widths = self.widths
                    if self.base_widths is None:
                        self.widths = new_widths
                    else:
                        # Max between new widths and geometric mean with
                        # user-supplied widths (i.e. bias towards keeping
                        # larger widths)
                        self.widths = np.maximum(
                            new_widths, np.sqrt(new_widths * self.base_widths)
                        )

            if i < burn:
                action = "burn"
            elif not record:
                action = "thin"
            else:
                action = "record"

            self.logger.debug(
                display_format, i - burn + 1, self.func_count, log_Px, action
            )

        if thin > 1:
            thin_msg = "   and keeping 1 sample every " + str(thin) + ", "
        else:
            thin_msg = "   "
        thin_msg += "for a total of %d function evaluations."
        self.logger.info("\nSampling terminated: ")
        self.logger.info(
            " * %d samples obtained after a burn-in period of %d samples",
            N,
            burn,
        )
        self.logger.info(thin_msg, self.func_count)

        R = eff_N = None
        exit_flag = 0
        if self.diagnostics:
            exit_flag, R, eff_N = self.__diagnose(samples)
            diag_msg = ""
            if exit_flag in (-2, -3):
                diag_msg = (
                    " * Try sampling for longer, by increasing N "
                    " or the thinning factor"
                )
            elif exit_flag == -1:
                diag_msg = (
                    " * Try increasing thinning factor to obtain "
                    "more uncorrelated samples"
                )
            elif exit_flag == 0:
                diag_msg = (
                    " * No violations of convergence have been "
                    "detected (this does NOT guarantee convergence)"
                )

            if diag_msg != "":
                self.logger.info(diag_msg)

        sampling_result = {
            "samples": samples,
            "exit_flag": exit_flag,
            "f_vals": f_vals,
            "log_priors": log_priors,
            "R": R,
            "eff_N": eff_N,
        }

        return sampling_result

    def __diagnose(self, samples: np.ndarray):
        """Performs a quick and dirty diagnosis of convergence."""
        N = samples.shape[0]
        # split psrf
        split_samples = np.array(
            [
                samples[0 : math.floor(N / 2), :],
                samples[math.floor(N / 2) : 2 * math.floor(N / 2)],
            ]
        )
        R = self.__gelman_rubin(split_samples)
        eff_N = self.__effective_n(split_samples)

        diag_msg = None
        exit_flag = 0
        if np.any(R > 1.5):
            diag_msg = (
                " * Detected lack of convergence! (max R = %.2f >> 1"
                ", mean R = %.2f)" % (np.max(R), np.mean(R))
            )
            exit_flag = -3
        elif np.any(R > 1.1):
            diag_msg = (
                " * Detected probable lack of convergence! (max R = %.2f"
                " > 1, mean R = %.2f)" % (np.max(R), np.mean(R))
            )
            exit_flag = -2

        if np.any(eff_N < N / 10.0):
            diag_msg = (
                " * Low number of effective samples! (min eff_N = %.1f"
                ", mean eff_N = %.1f, requested N = %d)"
                % (np.min(eff_N), np.mean(eff_N), N)
            )
            if exit_flag == 0:
                exit_flag = -1

        if diag_msg is None and exit_flag == 0:
            exit_flag = 1

        if diag_msg is not None:
            self.logger.info(diag_msg)

        return exit_flag, R, eff_N

    def __log_pdf_bound(self, x):
        """Evaluate log pdf with bounds and prior."""
        y = f_val = log_prior = None

        if np.any(x < self.LB) or np.any(x > self.UB):
            y = -np.inf
        else:
            if self.log_prior is not None:
                log_prior = self.log_prior(x)
                if np.isnan(log_prior):
                    y = -np.inf
                    self.logger.warning(
                        "Prior density function returned NaN. "
                        "Trying to continue."
                    )
                    return y, f_val, log_prior

                if not np.isfinite(log_prior):
                    y = -np.inf
                    return y, f_val, log_prior
            else:
                log_prior = 0

            f_val = self.log_f(x)
            self.func_count += 1

            if np.any(np.isnan(f_val)):
                self.logger.warning(
                    "Target density function returned NaN. Trying to continue."
                )
                y = -np.inf
            else:
                y = np.sum(f_val) + log_prior

        return y, f_val, log_prior

    def __metropolis_step(self, x, log_f, log_Px, f_val, log_prior):
        """Metropolis step."""
        xx_new = self.metropolis_rnd()
        log_Px_new, f_val_new, log_prior_new = log_f(xx_new)

        # Acceptance rate
        a = np.exp(log_Px_new - log_Px) * (
            self.metropolis_pdf(x) / self.metropolis_pdf(xx_new)
        )

        # Accept proposal?
        if np.random.rand() < a:
            return xx_new, log_Px_new, f_val_new, log_prior_new

        return x, log_Px, f_val, log_prior

    def __gelman_rubin(self, x, return_var=False):
        """Returns estimate of R for a set of traces.

        Parameters
        ----------
        x : ndarray, shape (m, n, k)
          An array containing the 2 or more traces of a stochastic parameter.
          Here m is the number of traces, n the number of samples, and k
          the dimension of the stochastic.

        return_var : bool
          Flag for returning the marginal posterior variance instead of R-hat.

        Returns
        -------
        Rhat : float
          Return the potential scale reduction factor, :math:`\\hat{R}`

        Raises
        ------
        ValueError
            Raised when `x` only contains one trace of a stochastic parameter.
            As the Gelman-Rubin diagnostic requires multiple chains of the same
            length.

        Notes
        -----

        The Gelman-Rubin diagnostic tests for lack of convergence by comparing
        the variance between multiple chains to the variance within each chain.
        If convergence has been achieved, the between-chain and within-chain
        variances should be identical. To be most effective in detecting
        evidence for nonconvergence, each chain should have been initialized
        to starting values that are dispersed relative to the target
        distribution.

        """
        if np.shape(x) < (2,):
            raise ValueError(
                "Gelman-Rubin diagnostic requires multiple chains of the "
                "same length."
            )

        try:
            m, n = np.shape(x)
        except ValueError:
            return np.array(
                [self.__gelman_rubin(np.transpose(y)) for y in np.transpose(x)]
            )

        # Calculate between-chain variance
        B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)

        # Calculate within-chain variances
        W = np.sum(
            [(x[i] - xbar) ** 2 for i, xbar in enumerate(np.mean(x, 1))]
        ) / (m * (n - 1))

        # (over) estimate of variance
        s2 = W * (n - 1) / n + B_over_n

        if return_var:
            return s2

        # Pooled posterior variance estimate
        # It seems that the part in the comment is not in the definition of
        # this diagnostic test.
        V = s2  # + B_over_n / m

        # Calculate PSRF
        R = V / W

        return np.sqrt(R)

    def __effective_n(self, x):
        """Returns estimate of the effective sample size of a set of traces.

        Parameters
        ----------
        x : ndarray, shape (m, n, k)
          An array containing the 2 or more traces of a stochastic parameter.
          Here m is the number of traces, n the number of samples, and k the
          dimension of the stochastic.

        Returns
        -------
        n_eff : float
          Return the effective sample size, :math:`\\hat{n}_{eff}`

        Raises
        ------
        ValueError
            Raised when `x` only contains one trace of a stochastic parameter.
            As the calculation of effective sample size requires multiple
            chains of the same length.
        """
        if np.shape(x) < (2,):
            raise ValueError(
                "Calculation of effective sample size requires multiple "
                "chains of the same length."
            )

        try:
            m, n = np.shape(x)
        except ValueError:
            return np.array(
                [self.__effective_n(np.transpose(y)) for y in np.transpose(x)]
            )

        s2 = self.__gelman_rubin(x, return_var=True)

        negative_autocorr = False
        t = 1

        variogram = lambda t: (
            sum(
                sum((x[j][i] - x[j][i - t]) ** 2 for i in range(t, n))
                for j in range(m)
            )
            / (m * (n - t))
        )
        rho = np.ones(n)
        # Iterate until the sum of consecutive estimates of autocorrelation
        # is negative
        while not negative_autocorr and (t < n):
            rho[t] = 1.0 - variogram(t) / (2.0 * s2)

            if t % 2:
                negative_autocorr = sum(rho[t - 1 : t + 1]) < 0

            t += 1

        # This part in the original code was slightly different, along with
        # the modulo check above.
        # However, looking at definitions this seems like the correct way.
        return m * n / (-1 + 2 * rho[0 : t - 2].sum())
