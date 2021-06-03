import math
import numpy as np

class SliceSampler:
    def __init__(self, log_f, x0, widths, LB, UB, options={}):
        np.random.seed(1234)
        D = x0.size
        self.log_f = log_f
        self.x0 = x0
        self.LB = LB
        self.UB = UB
        if np.size(LB) == 1:
            self.LB = np.tile(LB, D)
        if np.size(UB) == 1:
            self.UB = np.tile(UB, D)
        self.LB_out = LB + np.spacing(LB)
        self.UB_out = UB + np.spacing(UB)
        
        if np.size(widths) == 1:
            widths = np.tile(widths, D)
        self.widths = widths
        self.base_widths = widths.copy()
        
        if self.widths is None:
            self.widths = (self.UB - self.LB) / 2
        self.widths[np.isinf(self.widths)] = 10
        self.widths[self.LB == self.UB] = 1 # Widths is irrelevant when LB == UB, set to 1
        
        # Default options
        self.thin = options.get("thin", 1)
        self.burn = options.get('burn_in', None) 
        self.step_out = options.get("step_out", False)
        self.display = options.get("display", "notify")
        self.adaptive = options.get("adaptive", True)
        self.log_prior = options.get("log_prior", None)
        self.diagnostics = options.get("diagnostics", True)
        self.metropolis_pdf = options.get("metropolis_pdf", None)
        self.metropolis_rnd = options.get("metopolis_rnd", None)
        
    def sample(self, N):
        xx = self.x0
        D = xx.size
        if self.burn is None:
            self.burn = round(N/3)

        # Sanity checks
        assert(np.ndim(self.x0) <= 1)
        assert(np.shape(self.LB) == np.shape(self.x0) and np.shape(self.UB) == np.shape(self.x0))
        assert(np.all(self.UB >= self.LB))
        assert(np.all(self.widths > 0) and np.all(np.isfinite(self.widths)) and np.all(np.isreal(self.widths)))
        assert(np.all(self.x0 >= self.LB) and np.all(self.x0 <= self.UB))
        assert(np.isscalar(self.thin) and self.thin > 0)
        assert(np.isscalar(self.burn) and self.burn >= 0)

        # Effective samples
        eff_N = N + (N-1) * (self.thin - 1)
        
        samples = np.zeros((N, D))
        xx_sum = np.zeros((D,))
        xx_sq_sum = np.zeros((D,))
        
        log_dist = lambda xx_ : self.__log_pdf_bound(xx_, False)[0]
        xx_shape = xx.shape
        logdist_vec = lambda x: log_dist(np.reshape(x, xx_shape))
        log_Px = log_dist(xx)
        
        # Main loop
        for i in range(0, eff_N + self.burn):
            # Metropolis step (optional)
  
            ## Slice sampling step.
            perm = np.array(range(D))
            # Force xx into vector for ease of use:
            xx = xx.ravel()
            x_l = xx.copy()
            x_r = xx.copy()
            xprime = xx.copy()

            # Random scan through axes
            perm = rand_perm(D) # np.random.shuffle(perm)
            for dd in perm:
                log_uprime = log_Px + np.log(np.random.rand())
                # Create a horizontal interval (x_l, x_r) enclosing xx
                rr = np.random.rand()
                x_l[dd] = xx[dd] - rr*self.widths[dd]
                x_r[dd] = xx[dd] + (1-rr)*self.widths[dd]
                
                # Adjust interval to outside bounds for bounded problems.
                if np.isfinite(self.LB[dd]) or self.isfinite(self.UB[dd]):
                    if x_l[dd] < self.LB_out[dd]:
                        delta = self.LB_out[dd] - x_l[dd]
                        x_l[dd] += delta
                        x_r[dd] += delta
                    if x_r[dd] > self.UB_out[dd]:
                        delta = x_r[dd] - self.UB_out[dd]
                        x_l[dd] -= delta
                        x_r[dd] -= delta
                    x_l[dd] = np.maximum(x_l[dd], self.LB_out[dd])
                    x_r[dd] = np.minimum(x_r[dd], self.UB_out[dd])

                if self.step_out:
                    # Typo in early book editions: said compare to u, should be u'
                    while logdist_vec(x_l) > log_uprime:
                        x_l[dd] = x_l[dd] - self.widths[dd]
                    while logdist_vec(x_r) > log_uprime:
                        x_r[dd] = x_r[dd] + self.widths[dd]
                        
                # Inner loop:
                # Propose xprimes and shrink interval until good one found
                shrink = 0
                while True:
                    shrink += 1
                    xprime[dd] = np.random.rand()*(x_r[dd] - x_l[dd]) + x_l[dd]
                    log_Px = float(logdist_vec(xprime))
                    if log_Px > log_uprime:
                        break # this is the only way to leave the while loop
                    else:
                        # Shrink in
                        if xprime[dd] > xx[dd]:
                            x_r[dd] = xprime[dd]
                        elif xprime[dd] < xx[dd]:
                            x_l[dd] = xprime[dd]
                        else:
                            if warnings:
                                print('WARNING: Shrunk to current '
                                    + 'position and still not acceptable.')
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

                xx[dd] = xprime[dd]

            # Metropolis step (optional)
                
            # Record samples and miscellaneous bookkeeping.
            record = i >= self.burn and np.mod(i - self.burn, self.thin) == 0
            if record:
                i_smpl = (i - self.burn) // self.thin
                samples[i_smpl, :] = xx
            
            # Store summary statistics starting half.way into burn-in.
            if i < self.burn and i >= self.burn / 2:
                xx_sum += xx 
                xx_sq_sum += xx**2
                
                # End of burn-in, update widths if using adaptive method.
                if i == self.burn - 1 and self.adaptive:
                    burn_stored = np.floor(self.burn / 2)
                    new_widths = np.minimum(5 * np.sqrt(xx_sq_sum / burn_stored - (xx_sum / burn_stored)**2), self.UB_out - self.LB_out)
                    if not np.all(np.isreal(new_widths)):
                        new_widths = self.widths
                    if self.base_widths is None:
                        self.widths = new_widths
                    else:
                        # Max between new widths and geometric mean with user-supplied
                        # widths (i.e. bias towards keeping larger widths)
                        self.widths = np.maximum(new_widths, np.sqrt(new_widths * self.base_widths))

        split_samples = np.array([samples[0:math.floor(N/2), :], samples[math.floor(N/2):2*math.floor(N/2)]])
        print(self.__gelman_rubin(split_samples))
        print(self.__effective_n(split_samples))
        return samples
    
    # Evaluate log pdf with bounds and prior.
    def __log_pdf_bound(self, x, do_prior):
        y = f_val = log_prior = None
        
        if np.any(x < self.LB) or np.any(x > self.UB):
            y = -np.inf
        else:
            if do_prior:
                assert(False)
            else:
                log_prior = 0
                
            f_val = self.log_f(x)
            if np.any(np.isnan(f_val)):
                y = -np.inf
            else:
                y = np.sum(f_val) + log_prior
                
        return y, f_val, log_prior
        
    def __gelman_rubin(self, x, return_var = False):
        if np.shape(x) < (2,):
            raise ValueError(
                'Gelman-Rubin diagnostic requires multiple chains of the same length.')

        try:
            m, n = np.shape(x)
        except ValueError:
            return [self.__gelman_rubin(np.transpose(y)) for y in np.transpose(x)]

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
        V = s2 + B_over_n / m

        # Calculate PSRF
        R = V / W

        return np.sqrt(R)

    def __effective_n(self, x):       
        if np.shape(x) < (2,):
            raise ValueError(
                'Calculation of effective sample size requires multiple chains of the same length.')

        try:
            m, n = np.shape(x)
        except ValueError:
            return [self.__effective_n(np.transpose(y)) for y in np.transpose(x)]
            
        s2 = self.__gelman_rubin(x, return_var=True)
        
        negative_autocorr = False
        t = 1
        
        variogram = lambda t: (sum(sum((x[j][i] - x[j][i-t])**2 for i in range(t,n)) for j in range(m)) 
                                    / (m*(n - t)))
        rho = np.ones(n)
        # Iterate until the sum of consecutive estimates of autocorrelation is negative
        while not negative_autocorr and (t < n):
            
            rho[t] = 1. - variogram(t)/(2.*s2)
            
            if not t % 2:
                negative_autocorr = sum(rho[t-1:t+1]) < 0
            
            t += 1
            
        return int(m*n / (1 + 2*rho[1:t].sum()))

def rand_int(hi):
    proportion = 1.0 / hi
    tmp = np.random.rand()
    res = lo
    while res * proportion < tmp:
        res += 1 
    return res

def fisher_yates_shuffle(a):
    b = a.copy()
    left = b.size
    
    while left > 1:
        i = int(np.floor(np.random.rand() * left))
        left -= 1
        b[i], b[left] = b[left], b[i]
    return b
    
def rand_perm(n):
    return fisher_yates_shuffle(np.array(range(0, n)))
