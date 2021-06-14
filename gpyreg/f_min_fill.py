import time

import numpy as np
import scipy as sp

from gpyreg.doe_lhs import lhs

def f_min_fill(f, x0, LB, UB, PLB, PUB, tprior):
    N0 = 1
    N = 1024
    n_vars = np.max([x0.shape[0], np.size(LB), np.size(UB), np.size(PLB), np.size(PUB)])

    # Force provided points to be inside bounds
    x0 = np.clip(x0, LB, UB)
    
    if N > N0:
       # First test hyperparameters on a space-filling initial design
       S = lhs(n_vars, samples = N - N0, criterion = 'maximin')
       np.random.shuffle(S.T) # Randomly permute columns

       sX = np.zeros((N-N0, n_vars))
       
       for i in range(0, n_vars):
           mu = tprior.mu[i]
           sigma = tprior.sigma[i]
           a = tprior.a[i]
           b = tprior.b[i]
           
           if not np.isfinite(mu) and not np.isfinite(sigma): # Uniform distribution?
               if np.isfinite(LB[i]) and np.isfinite(UB[i]):
                   # Mixture of uniforms (full bounds and plausible bounds)
                   w = 0.5**(1/n_vars) # Half of all starting points from inside the plausible box
                   sX[:, i] = __uuinv(S[:, i], [LB[i], PLB[i], PUB[i], UB[i]], w)
               else:
                   # All starting points from inside the plausible box
                   sX[:, i] = S[:, i] * (PUB[i] - PLB[i]) + PLB[i]
           elif np.isfinite(a) and np.isfinite(b): # Smooth box student's t prior
               df = tprior.df[i]
               # Force fat tails
               if not np.isfinite(df):
                   df = 3
               df = np.minimum(df, 3)
               if df == 0:
                   cdf_lb = smoothbox_cdf(LB[i], sigma, a, b)
                   cdf_ub = smoothbox_cdf(UB[i], sigma, a, b)
                   S_scaled = cdf_lb + (cdf_ub - cdf_lb) * S[:, i]
                   for j in range(0, (N-N0)):
                       sX[j, i] = smoothbox_ppf(S_scaled[j], sigma, a, b)
               else:           
                   tcdf_lb = smoothbox_student_t_cdf(LB[i], df, sigma, a, b)
                   tcdf_ub = smoothbox_student_t_cdf(UB[i], df, sigma, a, b)
                   S_scaled = tcdf_lb + (tcdf_ub - tcdf_lb) * S[:, i]
                   for j in range(0, (N-N0)):
                       sX[j, i] = smoothbox_student_t_ppf(S_scaled[j], df, sigma, a, b)
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

    t_s = time.time()
    X = np.concatenate([[x0], sX])
    y = np.full((N,), np.inf)
    for i in range(0, N):
        y[i] = f(X[i, :])
     
    order = np.argsort(y)
    # print(time.time() - t_s)
    
    return X[order, :], y[order]

# Inverse of mixture of uniform cumulative distribution function.
def __uuinv(p, B, w):
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
    
def smoothbox_cdf(x, sigma, a, b):
    # Normalization constant so that integral over pdf is 1.
    C = 1.0 + (b - a) / (sigma * np.sqrt(2 * np.pi))
            
    if x < a:
        return sp.stats.norm.cdf(x, loc=a, scale=sigma)/C
    elif x >= a and x <= b:
        return (0.5 + (x-a)/(sigma*np.sqrt(2 * np.pi)))/C
    else:
        return (C - 1.0 + sp.stats.norm.cdf(x, loc=b, scale=sigma))/C
        
def smoothbox_student_t_cdf(x, df, sigma, a, b):
    # Normalization constant so that integral over pdf is 1.
    c = sp.special.gamma(0.5*(df+1)) / (sp.special.gamma(0.5*df) * sigma * np.sqrt(df * np.pi))
    C = 1.0 + (b - a) * c
                    
    if x < a:
        return sp.stats.t.cdf(x, df, loc=a, scale=sigma)/C
    elif x >= a and x <= b:
        return (0.5 + (x-a)*c)/C
    else:
        return (C - 1.0 + sp.stats.t.cdf(x, df, loc=b, scale=sigma))/C
        
def smoothbox_ppf(q, sigma, a, b):
    # Normalization constant so that integral over pdf is 1.
    C = 1.0 + (b - a) / (sigma * np.sqrt(2 * np.pi))
    
    if q < 0.5/C:
        return sp.stats.norm.ppf(C*q, loc=a, scale=sigma)
    elif q >= 0.5/C and q <= (C-0.5)/C:
        return (q*C - 0.5)*sigma * np.sqrt(2 * np.pi) + a
    else:
        return sp.stats.norm.ppf(C*q-(C-1), loc=b, scale=sigma)
        
def smoothbox_student_t_ppf(q, df, sigma, a, b):
    # Normalization constant so that integral over pdf is 1.
    c = sp.special.gamma(0.5*(df+1)) / (sp.special.gamma(0.5*df) * sigma * np.sqrt(df * np.pi))
    C = 1.0 + (b - a) * c
    
    if q < 0.5/C:
        return sp.stats.t.ppf(C*q, df, loc=a, scale=sigma)
    elif q >= 0.5/C and q <= (C-0.5)/C:
        return (q*C - 0.5) / c + a
    else:
        return sp.stats.t.ppf(C*q-(C-1), df, loc=b, scale=sigma)
