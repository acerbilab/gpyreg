import math
import numpy as np

def slice_sample(log_f, x0, N, widths, LB, UB):
    ## Default options
    thin = 1
    burn = 50 # round(N/3)
    step_out = False
    display = 'notify'
    adaptive = True
    log_prior = None
    diagnostics = True
    metropolis_pdf = None
    metropolis_rnd = None
    
    ## Startup and initial checks.
    D = x0.size   
    if np.size(LB) == 1:
        LB = np.tile(LB, D)
    if np.size(UB) == 1:
        UB = np.tile(UB, D)
    if np.size(widths) == 1:
        widths = np.tile(widths, D)
    LB_out = LB + np.spacing(LB)
    UB_out = UB + np.spacing(UB)
    base_widths = widths.copy()
    xx = x0

    if widths is None:
        widths = (UB - LB) / 2
    widths[np.isinf(widths)] = 10
    widths[LB == UB] = 1 # Widths is irrelevant when LB == UB, set to 1
     
    samples = np.zeros((N, D))
    
    # Sanity checks
    assert(np.ndim(x0) <= 1)
    assert(np.shape(LB) == np.shape(x0) and np.shape(UB) == np.shape(x0))
    assert(np.all(UB >= LB))
    assert(np.all(widths > 0) and np.all(np.isfinite(widths)) and np.all(np.isreal(widths)))
    assert(np.all(x0 >= LB) and np.all(x0 <= UB))
    assert(np.isscalar(thin) and thin > 0)
    assert(np.isscalar(burn) and burn >= 0)

    # Effective samples
    eff_N = N + (N-1) * (thin - 1)
    
    xx_sum = np.zeros((D,))
    xx_sq_sum = np.zeros((D,))
    
    log_dist = lambda xx_ : log_pdf_bound(log_f, xx_, LB, UB, False)[0]
    log_Px = log_dist(xx)
   
    # Main loop
    for i in range(0, eff_N + burn):
        # Metropolis step (optional)
        
        # Slice sampling step.
        xx, widths = slice_sweep(xx, log_dist, widths, step_out, width_adapt = i < burn and adaptive, LB = LB, UB=UB)
        
        # Metropolis step (optional)
            
        # Record samples and miscellaneous bookkeeping.
        record = i >= burn and np.mod(i - burn, thin) == 0
        if record:
            i_smpl = (i - burn) // thin
            samples[i_smpl, :] = xx
        
        # Store summary statistics starting half.way into burn-in.
        if i < burn and i > burn / 2:
            xx_sum += xx 
            xx_sq_sum += xx**2
            
            # End of burn-in, update widths if using adaptive method.
            if i == burn - 1 and adaptive:
                burn_stored = np.floor(burn / 2)
                new_widths = np.minimum(5 * np.sqrt(xx_sq_sum / burn_stored - (xx_sum / burn_stored)**2), UB_out - LB_out)
                if not np.all(np.isreal(new_widths)):
                    new_widths = widths
                if base_widths is None:
                    widths = new_widths
                else:
                    # Max between new widths and geometric mean with user-supplied
                    # widths (i.e. bias towards keeping larger widths)
                    widths = np.maximum(new_widths, np.sqrt(new_widths * base_widths))

                
    return samples
    
# Evaluate log pdf with bounds and prior.
def log_pdf_bound(log_f, x, LB, UB, do_prior):
    y = f_val = log_prior = None
    
    if np.any(x < LB) or np.any(x > UB):
        y = -np.inf
    else:
        if do_prior:
            assert(False)
        else:
            log_prior = 0
            
        f_val = log_f(x)
        if np.any(np.isnan(f_val)):
            y = -np.inf
        else:
            y = np.sum(f_val) + log_prior
            
    return y, f_val, log_prior

def slice_sweep(xx, logdist, widths=1.0, step_out=True, Lp=None, width_adapt=True, LB=None, UB=None, warnings=True):
    """simple axis-aligned slice sampling sweep
         xx_next = slice_sweep(xx, logdist)
     Inputs:
                xx  D,  initial state (or array with D elements)
           logdist  fn  function: log of unnormalized probability of xx
            widths  D,  or 1x1, step sizes for slice sampling (default 1.0)
          step_out bool set to True (default) if widths sometimes far too small
                Lp  1,  Optional: logdist(xx) if have already evaluated it
          warnings bool print warnings if slice falls back to point (default True)
     Outputs:
                xx  D,  final state (same shape as at start)
     If Lp was provided as an input, then return tuple with second element:
                Lp  1,  final log-prob, logdist(xx)
    """
    # Iain Murray 2004, 2009, 2010, 2013, 2016
    # Luigi Acerbi 2021
    # Algorithm orginally by Radford Neal, e.g., Annals of Statistic (2003)
    # See also pseudo-code in David MacKay's text book p375
    
    # startup stuff
    D = xx.size
    widths = np.array(widths)
    if widths.size == 1:
        widths = np.tile(widths, D)
    output_Lp = Lp is not None
    if Lp is None:
        log_Px = logdist(xx)
    else:
        log_Px = Lp
    perm = np.array(range(D))
    # Force xx into vector for ease of use:
    xx_shape = xx.shape
    logdist_vec = lambda x: logdist(np.reshape(x, xx_shape))
    xx = xx.ravel().copy()
    x_l = xx.copy()
    x_r = xx.copy()
    xprime = xx.copy()

    # Random scan through axes
    np.random.shuffle(perm)
    for dd in perm:
        log_uprime = log_Px + np.log(np.random.rand())
        # Create a horizontal interval (x_l, x_r) enclosing xx
        rr = np.random.rand()
        x_l[dd] = xx[dd] - rr*widths[dd]
        x_r[dd] = xx[dd] + (1-rr)*widths[dd]
        if step_out:
            # Typo in early book editions: said compare to u, should be u'
            while logdist_vec(x_l) > log_uprime:
                x_l[dd] = x_l[dd] - widths[dd]
            while logdist_vec(x_r) > log_uprime:
                x_r[dd] = x_r[dd] + widths[dd]
                
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
        if width_adapt:
            delta = UB[dd] - LB[dd]
            if shrink > 3:
                if np.isfinite(delta):
                    widths[dd] = np.maximum(widths[dd]/1.1, np.spacing(delta))
                else:
                    widths[dd] = np.maximum(widths[dd]/1.1, np.spacing(1))
            elif shrink < 2:
                widths[dd] = np.minimum(widths[dd]*1.2, delta)

        xx[dd] = xprime[dd]

    if output_Lp:
        return xx, log_Px, widths
    else:
        return xx, widths
