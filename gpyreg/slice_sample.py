import math
import logging

import numpy as np

class SliceSampler:
    '''This is the form of a docstring.

    It can be spread over several lines.
    
    References
    ----------
    -- R. Neal (2003), Slice Sampling, Annals of Statistics, 31(3), p705-67.
    
    -- D. J. MacKay (2003), Information theory, inference and learning 
       algorithms, Cambridge university press, p374-7.
    -- S. Watanabe (2010), Asymptotic equivalence of Bayes cross validation
       and widely applicable information criterion in singular learning 
       theory, The Journal of Machine Learning Research, 11, p3571-94.
    -- A. Gelman, et al (2013), Bayesian data analysis. Vol. 2. Boca Raton, 
       FL, USA: Chapman & Hall/CRC.
       
    '''
    
    def __init__(self, log_f, x0, widths=None, LB=None, UB=None, options={}):
        D = x0.size
        self.log_f = log_f
        self.x0 = x0
        
        if LB is None:
            self.LB = np.tile(-np.inf, D)
            self.LB_out = np.tile(-np.inf, D)
        else:
            self.LB = LB
            if np.size(LB) == 1:
                self.LB = np.tile(LB, D)
        self.LB_out = self.LB + np.spacing(self.LB)
            
        if UB is None:
            self.UB = np.tile(np.inf, D)
            self.UB_out = np.tile(np.inf, D)
        else:
            self.UB = UB
            if np.size(UB) == 1:
                self.UB = np.tile(UB, D)
        self.UB_out = self.UB + np.spacing(self.UB)
            
        if widths is None:
            self.widths = (self.UB - self.LB) / 2
            self.base_widths = widths
        else:
            if np.size(widths) == 1:
                self.widths = np.tile(widths, D)
            else:
                self.widths = widths
            self.base_widths = self.widths.copy()
            
        self.widths[np.isinf(self.widths)] = 10
        self.widths[self.LB == self.UB] = 1 # Widths is irrelevant when LB == UB, set to 1

        self.func_count = 0
            
        # Default options
        self.thin = options.get("thin", 1)
        self.burn = options.get('burn_in', None) 
        self.step_out = options.get("step_out", False)
        self.display = options.get("display", "full")
        self.adaptive = options.get("adaptive", True)
        self.log_prior = options.get("log_prior", None)
        self.diagnostics = options.get("diagnostics", True)
        self.metropolis_pdf = options.get("metropolis_pdf", None) 
        self.metropolis_rnd = options.get("metopolis_rnd", None)
        self.metropolis_flag = self.metropolis_pdf is not None and self.metropolis_rnd is not None
        
        # Logging
        self.logger = logging.getLogger("SliceSampler")
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)
        if self.display == 'off':
            self.logger.setLevel(logging.WARN)
        elif self.display == 'summary':
            self.logger.setLevel(logging.INFO)
        elif self.display == 'full':
            self.logger.setLevel(logging.DEBUG)
        
    def sample(self, N):
        '''Description here
        
        Parameters
        ----------
        N : int
            The number of samples to return.
            
        '''
        xx = self.x0
        D = xx.size
        if self.burn is None:
            self.burn = round(N/3)
            
        if self.burn == 0 and self.base_widths is None and self.adaptive:
            self.logger.warning('WIDTHS not specified and adaptation is ON (OPTIONS.Adaptive == 1), but OPTIONS.Burnin is set to 0. SLICESAMPLEBND will attempt to use default values for WIDTHS.')
            
        # Sanity checks
        assert np.ndim(self.x0) <= 1, 'The initial point X0 needs to be a scalar or a 1D array.'
        assert np.shape(self.LB) == np.shape(self.x0) and np.shape(self.UB) == np.shape(self.x0), 'LB and UB need to be None, scalars, or 1D arrays of the same size as X0.'
        assert np.all(self.UB >= self.LB), 'All upper bounds UB need to be equal or greater than lower bounds LB.'
        assert np.all(self.widths > 0) and np.all(np.isfinite(self.widths)) and np.all(np.isreal(self.widths)), 'The widths vector needs to be all positive real numbers.'
        assert np.all(self.x0 >= self.LB) and np.all(self.x0 <= self.UB), 'The initial starting point X0 is outside the bounds.'
        assert np.isscalar(self.thin) and self.thin > 0, 'The thinning factor option needs to be a positive integer.'
        assert np.isscalar(self.burn) and self.burn >= 0, 'The burn-in samples option needs to be a non-negative integer.'

        # Effective samples
        eff_N = N + (N-1) * (self.thin - 1)
        
        samples = np.zeros((N, D))
        xx_sum = np.zeros((D,))
        xx_sq_sum = np.zeros((D,))
        
        log_dist = lambda xx_ : self.__log_pdf_bound(xx_)
        log_Px, f_val, log_prior = log_dist(xx)
        log_priors = np.zeros((N,))
        f_vals = np.zeros((N, np.size(f_val)))
        
        assert np.all(np.isfinite(log_Px)), 'The initial starting point X0 needs to evaluate to a real number (not Inf or NaN).'
        
        # Force xx into vector for ease of use
        xx_shape = xx.shape
        xx = xx.ravel()
        logdist_vec = lambda x: log_dist(np.reshape(x, xx_shape))
        
        self.logger.debug(' Iteration     f-count       log p(x)                   Action')
        display_format = ' %7.0f     %8.0f    %12.6g    %26s'
        
        # Main loop
        perm = np.array(range(D))
        for i in range(0, eff_N + self.burn):
            if i == self.burn:
                action = 'start recording'
                self.logger.debug(display_format % (i-self.burn+1, self.func_count, log_Px, action))
        
            # Metropolis step (optional)
            if self.metropolis_flag:
                xx, log_Px, f_val, log_prior = self.__metropolis_step(xx, logdist_vec, log_Px, f_val, log_prior)
  
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
                x_l[dd] -= rr*self.widths[dd]
                x_r[dd] += (1-rr)*self.widths[dd]
                
                # Adjust interval to outside bounds for bounded problems.
                x_l[dd] = np.fmax(x_l[dd], self.LB_out[dd])
                x_r[dd] = np.fmin(x_r[dd], self.UB_out[dd])
                     
                if self.step_out:
                    steps = 0
                    # Typo in early book editions: said compare to u, should be u'
                    while logdist_vec(x_l)[0] > log_uprime:
                        x_l[dd] -= self.widths[dd]
                        steps += 1
                    while logdist_vec(x_r)[0] > log_uprime:
                        x_r[dd] += self.widths[dd]
                        steps += 1
                    if steps >= 10:
                        action = 'step-out dim ' + str(dd) + ' (' + str(steps) + ' steps)' 
                        self.logger.debug(display_format % (i-self.burn+1, self.func_count, log_Px, action))        
   
                # Inner loop:
                # Propose xprimes and shrink interval until good one found
                shrink = 0
                while True:
                    shrink += 1
                    xprime[dd] = np.random.rand()*(x_r[dd] - x_l[dd]) + x_l[dd]
                    log_Px, f_val, log_prior = logdist_vec(xprime)
                    if log_Px > log_uprime:
                        break # this is the only way to leave the while loop
                    else:
                        # Shrink in
                        if xprime[dd] > xx[dd]:
                            x_r[dd] = xprime[dd]
                        elif xprime[dd] < xx[dd]:
                            x_l[dd] = xprime[dd]
                        else:
                            self.logger.warning('WARNING: Shrunk to current position and still not acceptable!')
                        #    raise Exception('BUG DETECTED: Shrunk to current '
                        #        + 'position and still not acceptable.')
                            break
                            
                # Width adaptation (only during burn-in, might break detailed balance)
                if i < self.burn and self.adaptive:
                    delta = self.UB[dd] - self.LB[dd]
                    if shrink > 3:
                        if np.isfinite(delta):
                            self.widths[dd] = np.maximum(self.widths[dd]/1.1, np.spacing(delta))
                        else:
                            self.widths[dd] = np.maximum(self.widths[dd]/1.1, np.spacing(1))
                    elif shrink < 2:
                        self.widths[dd] = np.minimum(self.widths[dd]*1.2, delta)
                        
                if shrink >= 10:
                    action = 'shrink dim ' + str(dd) + ' (' + str(shrink) + ' steps)' 
                    self.logger.debug(display_format % (i-self.burn+1, self.func_count, log_Px, action))

                xx[dd] = xprime[dd]

            # Metropolis step (optional)
            if self.metropolis_flag:
                xx, log_Px, f_val, log_prior = self.__metropolis_step(xx, logdist_vec, log_Px, f_val, log_prior)
                
            # Record samples and miscellaneous bookkeeping.
            record = i >= self.burn and np.mod(i - self.burn, self.thin) == 0
            if record:
                i_smpl = (i - self.burn) // self.thin
                samples[i_smpl, :] = xx
                f_vals[i_smpl, :] = f_val
                log_priors[i_smpl] = log_prior
            
            # Store summary statistics starting half.way into burn-in.
            if i < self.burn and i >= self.burn / 2:
                xx_sum += xx 
                xx_sq_sum += xx**2
                
                # End of burn-in, update widths if using adaptive method.
                if i == self.burn - 1 and self.adaptive:
                    burn_stored = np.floor(self.burn / 2)
                    # There can be numerical error here but then width has already shrunk to 0?
                    new_widths = np.fmin(5 * np.sqrt(xx_sq_sum / burn_stored - (xx_sum / burn_stored)**2), self.UB_out - self.LB_out)
                    if not np.all(np.isreal(new_widths)):
                        new_widths = self.widths
                    if self.base_widths is None:
                        self.widths = new_widths
                    else:
                        # Max between new widths and geometric mean with user-supplied
                        # widths (i.e. bias towards keeping larger widths)
                        self.widths = np.maximum(new_widths, np.sqrt(new_widths * self.base_widths))
                        
            if i < self.burn:
                action = 'burn'
            elif not record:
                action = 'thin'
            else:
                action = 'record' 
            self.logger.debug(display_format % (i-self.burn+1, self.func_count, log_Px, action))

        if self.thin > 1:
           thin_msg = '   and keeping 1 sample every ' + str(self.thin) + ', '
        else:
           thin_msg = '   '
        self.logger.info('\nSampling terminated: ')
        self.logger.info(' * %d samples obtained after a burn-in period of %d samples' % (N, self.burn))
        self.logger.info(thin_msg + ('for a total of %d function evaluations.' % self.func_count))
        
        R = eff_N = None
        exit_flag = 0
        if self.diagnostics:
            exit_flag, R, eff_N = self.__diagnose(samples)
            diag_msg = ''
            if exit_flag == -2 or exit_flag == -3:
                diag_msg = ' * Try sampling for longer, by increasing N or the thinning factor'
            elif exit_flag == -1:
                diag_msg = ' * Try increasing thinning factor to obtain more uncorrelated samples'
            elif exit_flag == 0:
                diag_msg = ' * No violations of convergence have been detected (this does NOT guarantee convergence)'
                
            if diag_msg != '':
                self.logger.info(diag_msg)

        return SamplingResult(samples, f_vals, exit_flag, self.widths.copy(), log_priors, self.func_count, R, eff_N)
        
    def __diagnose(self, samples):
        N = samples.shape[0]
        split_samples = np.array([samples[0:math.floor(N/2), :], samples[math.floor(N/2):2*math.floor(N/2)]])
        R = self.__gelman_rubin(split_samples)
        eff_N = self.__effective_n(split_samples)
        
        diag_msg = None
        exit_flag = 0
        if np.any(R > 1.5):
            diag_msg = ' * Detected lack of convergence! (max R = %.2f >> 1, mean R = %.2f)' % (np.max(R), np.mean(R)) 
            exit_flag = -3
        elif np.any(R > 1.1):
           diag_msg = ' * Detected probable lack of convergence! (max R = %.2f > 1, mean R = %.2f)' % (np.max(R), np.mean(R)) 
           exit_flag = -2
           
        if np.any(eff_N < N/10.0):
            diag_msg = ' * Low number of effective samples! (min eff_N = %.1f, mean eff_N = %.1f, requested N = %d)' % (np.min(eff_N), np.mean(eff_N), N)
            if exit_flag == 0:
                exit_flag = -1
        
        if diag_msg is None and exit_flag == 0:
            exit_flag == 1
        
        if diag_msg is not None:
            self.logger.info(diag_msg)
            
        return exit_flag, R, eff_N
    
    # Evaluate log pdf with bounds and prior.
    def __log_pdf_bound(self, x):
        y = f_val = log_prior = None
        
        if np.any(x < self.LB) or np.any(x > self.UB):
            y = -np.inf
        else:
            if self.log_prior is not None:
                log_prior = self.log_prior(x)
                if np.isnan(log_prior):
                    y = -np.inf
                    self.logger.warning('Prior density function returned NaN. Trying to continue.')
                    return y, f_val, log_prior
                elif not np.isfinite(log_prior):
                    y = -np.inf
                    return y, f_val, log_prior
            else:
                log_prior = 0
                
            f_val = self.log_f(x)
            self.func_count += 1
            
            if np.any(np.isnan(f_val)):
                self.logger.warning('Target density function returned NaN. Trying to continue.')
                y = -np.inf
            else:
                y = np.sum(f_val) + log_prior
                
        return y, f_val, log_prior
        
    def __metropolis_step(self, x, log_f, log_Px, f_val, log_prior):
        xx_new = self.metropolis_rnd()
        log_Px_new, f_val_new, log_prior_new = log_f(xx_new)
        
        # Acceptance rate
        a = np.exp(log_Px_new - log_Px) * (self.metropolis_pdf(x) / self.metropolis_pdf(xx_new))
        
        # Accept proposal?
        if np.random.rand() < a:
            return xx_new, log_Px_new, f_val_new, log_prior_new
        else:
            return x, log_Px, f_val, log_prior
        
    def __gelman_rubin(self, x, return_var = False):
        '''Returns estimate of R for a set of traces.
        
        The Gelman-Rubin diagnostic tests for lack of convergence by comparing
        the variance between multiple chains to the variance within each chain.
        If convergence has been achieved, the between-chain and within-chain
        variances should be identical. To be most effective in detecting evidence
        for nonconvergence, each chain should have been initialized to starting
        values that are dispersed relative to the target distribution.
        
        Parameters
        ----------
        x : array-like
          An array containing the 2 or more traces of a stochastic parameter. 
          That is, an array of dimension m x n x k, where m is the number of traces, n the number of samples, and k the dimension of the stochastic.
          
        return_var : bool
          Flag for returning the marginal posterior variance instead of R-hat (defaults of False).
          
        Returns
        -------
        Rhat : float
          Return the potential scale reduction factor, :math:`\hat{R}`
          
        '''
        if np.shape(x) < (2,):
            raise ValueError(
                'Gelman-Rubin diagnostic requires multiple chains of the same length.')

        try:
            m, n = np.shape(x)
        except ValueError:
            return np.array([self.__gelman_rubin(np.transpose(y)) for y in np.transpose(x)])

        # Calculate between-chain variance
        B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)

        # Calculate within-chain variances
        W = np.sum(
            [(x[i] - xbar) ** 2 for i,
             xbar in enumerate(np.mean(x,
                                       1))]) / (m * (n - 1))

        # (over) estimate of variance
        s2 = W * (n - 1) / n + B_over_n

        if return_var:
            return s2

        # Pooled posterior variance estimate
        V = s2 # + B_over_n / m

        # Calculate PSRF
        R = V / W

        return np.sqrt(R)

    def __effective_n(self, x):
        '''Returns estimate of the effective sample size of a set of traces.
    
        Parameters
        ----------
        x : array-like
          An array containing the 2 or more traces of a stochastic parameter. 
          That is, an array of dimension m x n x k, where m is the number of traces, n the number of samples, and k the dimension of the stochastic.
        
        Returns
        -------
        n_eff : float
          Return the effective sample size, :math:`\hat{n}_{eff}`

        '''       
        if np.shape(x) < (2,):
            raise ValueError(
                'Calculation of effective sample size requires multiple chains of the same length.')

        try:
            m, n = np.shape(x)
        except ValueError:
            return np.array([self.__effective_n(np.transpose(y)) for y in np.transpose(x)])
            
        s2 = self.__gelman_rubin(x, return_var=True)
        
        negative_autocorr = False
        t = 1
        
        variogram = lambda t: (sum(sum((x[j][i] - x[j][i-t])**2 for i in range(t,n)) for j in range(m)) 
                                    / (m*(n - t)))
        rho = np.ones(n)
        # Iterate until the sum of consecutive estimates of autocorrelation is negative
        while not negative_autocorr and (t < n):
            
            rho[t] = 1. - variogram(t)/(2.*s2)
 
            if t % 2:
                negative_autocorr = sum(rho[t-1:t+1]) < 0
            
            t += 1

        return m*n / (-1 + 2*rho[0:t-2].sum())
        
class SamplingResult:
    def __init__(self, samples, f_vals, exit_flag, widths, log_priors, func_count, R = None, eff_N = None):
        self.samples = samples
        self.f_vals = f_vals
        self.exit_flag = exit_flag
        self.widths = widths
        self.log_priors = log_priors
        self.func_count = func_count
        self.R = R
        self.eff_N = eff_N
