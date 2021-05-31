import numpy as np
import scipy as sp

from gpyreg.sobol import i4_sobol_generate

def f_min_fill(f, x0, LB, UB, PLB, PUB, tprior):
    N0 = 1
    N = 1024
    n_vars = np.max([x0.shape[0], np.size(LB), np.size(UB), np.size(PLB), np.size(PUB)])

    # Force provided points to be inside bounds
    x0 = np.clip(x0, LB, UB)
    
    if N > N0:
       # First test hyperparameters on a space-filling initial design
       S = None
       
       if n_vars <= 40: # Sobol generator handles up to 40 variables
           max_seed = 997
           seed = 477 # np.random.default_rng().integers(1, max_seed)
           S = i4_sobol_generate(n_vars, N - N0 + 1, seed).T[1:, :]
           # np.random.shuffle(S.T) # Randomly permute columns
           
       if S is None: # just use random sampling
           S = np.random.random_sample((N-N0, nvars))
           
       sX = np.zeros((N-N0, n_vars))
       
       for i in range(0, n_vars):
           mu = tprior.mu[i]
           sigma = tprior.sigma[i]
           
           if not np.isfinite(mu) or not np.isfinite(sigma): # Uniform distribution?
               if np.isfinite(LB[i]) and np.isfinite(UB[i]):
                   # Mixture of uniforms (full bounds and plausible bounds)
                   w = 0.5**(1/n_vars) # Half of all starting points from inside the plausible box
                   sX[:, i] = uuinv(S[:, i], [LB[i], PLB[i], PUB[i], UB[i]], w)
               else:
                   # All starting points from inside the plausible box
                   sX[:, i] = S[:, i] * (PUB[i] - PLB[i]) + PLB[i]
           else: # Student's t prior
               df = tprior.df[i]
               # Force fat tails
               if not np.isfinite(df):
                   df = 3
               df = np.minimum(df, 3)
               if df == 0:
                   df = np.inf
                   
               tcdf_lb = sp.stats.t.cdf((LB[i] - mu) / sigma, df)
               tcdf_ub = sp.stats.t.cdf((UB[i] - mu) / sigma, df)
               S_scaled = tcdf_lb + (tcdf_ub - tcdf_lb) * S[:, i]
               sX[:, i] = sp.stats.t.ppf(S_scaled, df) * sigma + mu

    X = np.concatenate([[x0], sX])
    y = np.full((N,), np.inf)
    for i in range(0, N):
        y[i] = f(X[i, :])
        
    order = np.argsort(y)
    
    return X[order, :], y[order]
      
# p = percentile
# B = bounds
# w = width?
def uuinv(p, B, w):
    x = np.zeros(p.shape)
    L1 = B[3] - B[0]
    L2 = B[2] - B[1]
    
    # First step
    i1 = p <= (1 - w) * (B[1] - B[0]) / L1
    x[i1] = B[0] + p[i1] * L1 / (1 - w)
    
    # Second step
    i2 = (p <= (1 - w) * (B[2] - B[0]) / L1 + w) & ~i1
    x[i2] = (p[i2] * L1 * L2 + B[0] * (1 - w) * L2 + w * B[1] * L1) / (L1 * w + L2 * (1-w))
    
    # Third step
    i3 = p > (1 - w) * (B[2] - B[0]) / L1 + w
    x[i3] = (p[i3] - w + B[0]*(1-w)/L1) * L1/(1-w)
    
    x[p < 0] = np.nan
    x[p > 1] = np.nan
    
    return x
