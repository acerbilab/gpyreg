import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

class SquaredExponential:
    def __init__(self):
        pass
        
    def hyperparameter_count(self, d):
        return d + 1
        
    def get_info(self, X, y):
        return CovarianceInfo(self, X, y)
    
    def compute(self, hyp, X, X_star = None, compute_grad = False):
       N, D = X.shape
       cov_N = self.hyperparameter_count(D)
       hyp_N = hyp.size

       assert(hyp_N == cov_N)
       assert(hyp.ndim == 1)
       
       ell = np.exp(hyp[0:D])
       sf2 = np.exp(2*hyp[D])
       
       tmp = None
       if X_star is None:
           tmp = sq_dist(np.diag(1 / ell) @ X.T)
       elif isinstance(X_star, str):
           tmp = np.zeros((X.shape[0], 1))
       else:
           tmp = sq_dist(np.diag(1 / ell) @ X.T, np.diag(1 / ell) @ X_star.T)
           
       K = sf2 * np.exp(-tmp/2)
       
       if compute_grad:
           assert(False)

       return K

class Matern:
    def __init__(self, degree):
        self.degree = degree
        
    def hyperparameter_count(self, d):
        return d + 1
        
    def get_info(self, X, y):
        return CovarianceInfo(self, X, y)
        
    def compute(self, hyp, X, X_star = None, compute_grad = False):
       N, D = X.shape
       cov_N = self.hyperparameter_count(D)
       hyp_N = hyp.size

       assert(hyp_N == cov_N)
       assert(hyp.ndim == 1)
       
       ell = np.exp(hyp[0:D])
       sf2 = np.exp(2*hyp[D])
       
       f = df = tmp = None
       if self.degree == 1:
           f = lambda t : 1
           df = lambda t : 1 / t
       elif self.degree == 3:
           f = lambda t : 1 + t
           df = lambda t : 1
       elif self.degree == 5:
           f = lambda t : 1 + t * (1+ t / 3)
           df = lambda t : (1+t)/3
       else:
           assert(False)

       if X_star is None:
           # tmp = squareform(pdist(np.diag(np.sqrt(self.degree) / ell) @ X.T))
           tmp = np.sqrt(sq_dist(np.diag(np.sqrt(self.degree) / ell) @ X.T))
       elif isinstance(X_star, str):
           tmp = np.zeros((X.shape[0], 1))
       else:
           a = np.diag(np.sqrt(self.degree) / ell) @ X.T
           b = np.diag(np.sqrt(self.degree) / ell) @ X_star.T
           # tmp = squareform(cdist(a, b))
           tmp = np.sqrt(sq_dist(a, b))
           
       K = sf2 * f(tmp) * np.exp(-tmp)
       
       if compute_grad:
           assert(False)
        
       return K
    
def sq_dist(a, b=None):
    D, n = a.shape
    d = m = None

    # Computation of a^2 - 2*a*b + b^2 is less stable than (a-b)^2 because numerical precision 
    # can be lost when both a and b have very large absolute value and the same sign.
    # For that reason, we subtract the mean from the data beforehand to stabilise the computations.
    # This is OK because the squared error is independent of the mean.
    if b is None:
        mu = np.reshape(np.mean(a, 1), (-1, 1))
        a = a - mu
        b = a
        m = n
    else:
        d, m = b.shape
        assert(d == D)
        
        mu = np.reshape((m/(n+m))*np.mean(b, 1) + (n/(n+m))*np.mean(a, 1), (-1, 1))
        a = a - mu
        b = b - mu
    
    # Compute squared distances.
    # TODO: annoying reshapes
    C = np.reshape(np.sum(a * a, 0), (1, -1)).T + np.reshape(np.sum(b * b, 0), (1, -1)) - 2 * a.T @ b
 
    # Numerical noise can cause C to go negative, i.e. C > -1e-14
    return np.maximum(C, 0)
    
class CovarianceInfo:
    def __init__(self, gp, X, y):
        N, D = X.shape
        cov_N = gp.hyperparameter_count(D)
        tol = 1e-6
        self.LB = np.full((cov_N,), -np.inf)
        self.UB = np.full((cov_N,), np.inf)
        self.PLB = np.full((cov_N,), -np.inf)
        self.PUB = np.full((cov_N,), np.inf)
        self.x0 = np.full((cov_N,), np.nan)
        
        width = np.max(X, axis=0) - np.min(X, axis=0)
        if np.size(y) <= 1:
            y = np.array([0, 1])
        height = np.max(y) - np.min(y)    

        self.LB[0:D] = np.log(width) + np.log(tol)
        self.UB[0:D] = np.log(width * 10)
        self.PLB[0:D] = np.log(width) + 0.5 * np.log(tol)
        self.PUB[0:D] = np.log(width)
        self.x0[0:D] = np.log(np.std(X, ddof=1))
        
        self.LB[D] = np.log(height) + np.log(tol)
        self.UB[D] = np.log(height * 10)
        self.PLB[D] = np.log(height) + 0.5 * np.log(tol)
        self.PUB[D] = np.log(height)
        self.x0[D] = np.log(np.std(y, ddof=1))
        
        # Plausible starting point
        i_nan = np.isnan(self.x0)
        self.x0[i_nan] = 0.5 * (self.PLB[i_nan] + self.PUB[i_nan])
